import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from .agent_workflow import create_graph_app

# LlamaIndex 核心组件
from llama_index.core import Settings, VectorStoreIndex, PropertyGraphIndex
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.retrievers import BaseRetriever

# 数据库组件
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
import qdrant_client

# 1. 加载环境变量
load_dotenv()

# 配置参数
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "edu_matrix_v2"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or ""

# 全局引擎容器
rag_engine = {}

# ==========================================
# 🔧 工具类定义
# ==========================================

class HybridRetriever(BaseRetriever):
    """
    混合检索器：同时从 Vector (Qdrant) 和 Graph (Neo4j) 检索，并合并结果。
    """
    def __init__(self, vector_retriever, graph_retriever):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        super().__init__()
    
    def _retrieve(self, query_bundle):
        # 1. 并行检索
        nodes_vect = self.vector_retriever.retrieve(query_bundle)
        nodes_graph = self.graph_retriever.retrieve(query_bundle)
        
        # 2. 合并去重 (基于 node_id)
        combined_dict = {n.node.node_id: n for n in (nodes_vect + nodes_graph)}
        return list(combined_dict.values())

# ==========================================
# 🚀 生命周期管理 (初始化核心)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [Startup] 正在初始化 EduMatrix 引擎...")
    
    try:
        # 1. 初始化模型 (Embedding + LLM)
        print("🧠 加载模型 (Embedding: BGE-M3, LLM: Qwen)...")
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            trust_remote_code=True,
            local_files_only=True,
            device="cpu",
        )
        Settings.embed_model = embed_model

        llm = DashScope(
            model_name=os.getenv("DASHSCOPE_MODEL_NAME"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.1,
        )
        rag_engine["llm"] = llm

        # 2. 连接 Qdrant (负责原文检索)
        print("🔌 连接 Qdrant (Vector Store)...")
        qdrant_client_obj = qdrant_client.QdrantClient(url=QDRANT_URL)
        vector_store = QdrantVectorStore(
            collection_name=QDRANT_COLLECTION,
            client=qdrant_client_obj,
        )
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # 3. 连接 Neo4j (负责关系检索)
        print("🔌 连接 Neo4j (Graph Store)...")
        graph_store = Neo4jPropertyGraphStore(
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            url=NEO4J_URL,
        )
        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            llm=llm
        )

        rag_engine["graph_store"] = graph_store

        # 4. 构建混合检索策略
        # A. 向量检索工具
        vector_tool = vector_index.as_retriever(similarity_top_k=5)
        
        # B. 图谱检索工具 (使用 VectorContextRetriever 进行定位 + 扩散)
        from llama_index.core.retrievers import VectorContextRetriever
        sub_retriever = VectorContextRetriever(
            graph_store=graph_store,
            similarity_top_k=5,
            path_depth=3 # 抓取 3 跳邻居
        )
        graph_tool = graph_index.as_retriever(
            sub_retrievers=[sub_retriever]
        )

        # C. 组装混合检索器
        hybrid_retriever = HybridRetriever(vector_tool, graph_tool)

        # 5. 🔥 构建 Agent (替换原来的 ChatEngine)
        print("🤖 构建 LangGraph Agent...")
        # 把 llm 和 retriever 传进去
        graph_app = create_graph_app(hybrid_retriever, llm)

        rag_engine["graph_app"] = graph_app
        print("✅ 引擎初始化完成！等待请求...")

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        raise e

    yield 
    print("👋 [Shutdown] 服务器已关闭")

# ==========================================
# 📡 API 接口定义
# ==========================================

app = FastAPI(title="EduMatrix API", lifespan=lifespan)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not rag_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    # 1. 解析请求
    last_message = request.messages[-1].content
    print(f"📩 收到问题: {last_message}")

    # 2. 准备历史记录
    chat_history = [
        ChatMessage(role=m.role, content=m.content)
        for m in request.messages[:-1]
    ]

    inputs = {
    "question": last_message,
    "original_question": last_message, # ✅ 新增这个
    "chat_history": chat_history,
    "retrieved_nodes": [],
    "grade_status": "",
    "retry_count": 0, # ✅ 初始化计数器
    "final_response": ""
}

    # 运行图谱，直到结束
    # 注意：我们的 generate_node 返回的是一个 stream iterator 对象
    result = await rag_engine["graph_app"].ainvoke(inputs)

    streaming_response = result["final_response"]

    # 4. 生成流式响应
    async def response_generator():
        # A. 吐出 AI 回答
        # situation A: 如果是普通字符串 (来自 Apologize Node)
        if isinstance(streaming_response, str):
            yield streaming_response
            
        # situation B: 如果是流式响应对象 (来自 Generate Node)
        elif hasattr(streaming_response, "async_response_gen"):
            async for token in streaming_response.async_response_gen():
                yield token.delta
        
        # situation C: 兜底 (有些版本的 LlamaIndex 返回的是直接的 AsyncGenerator)
        else:
            try:
                async for token in streaming_response:
                    yield token.delta
            except Exception as e:
                yield f"❌ 响应解析错误: {str(e)}"

        
        # B. 吐出参考来源 (如果有)
        nodes = result.get("retrieved_nodes", [])
        if nodes:
            yield "\n\n---\n**🧠 思考路径：**\n"
            yield f"- 检索到 {len(nodes)} 个知识片段\n"
            yield "- 正在基于 Graph + Vector 进行推理...\n"
                
            yield "\n**📚 参考来源：**\n"
            seen = set()
            for n in nodes:
                txt = n.get_content()[:50].replace('\n', ' ')
                if txt not in seen:
                    yield f"> {txt}...\n"
                    seen.add(txt)
            
    return StreamingResponse(response_generator(), media_type="text/plain")

@app.get("/")
def read_root():
    return {"message": "EduMatrix API is running! Go to /docs for Swagger UI."}
# 确保在文件顶部有这个导入
from neo4j import GraphDatabase

@app.post("/api/graph")
async def get_graph(request: ChatRequest):
    # 默认返回值
    result_data = {"links": []}
    
    try:
        # 1. 安全获取关键词
        if not request.messages or not request.messages[-1].content:
            print("⚠️ [Graph API] 收到空消息")
            return result_data
            
        user_query = request.messages[-1].content
        print(f"📩 [Graph API] 用户原始提问: {user_query}")

        # 定义提取 Prompt，强制要求格式简洁
        extract_prompt = (
            "你是一个不仅懂中文，还懂计算机科学的实体提取助手。\n"
            "请从用户的提问中提取出 1 到 3 个最核心的【实体关键词】，用于在知识图谱中检索。\n"
            "要求：\n"
            "1. 只返回关键词，用逗号 ',' 分隔。\n"
            "2. 去掉所有修饰词（如'请问'、'是什么'、'介绍一下'）。\n"
            "3. 如果没有明显实体，提取最关键的名词。\n"
            "4. 不要输出任何其他废话。\n"
            "\n"
            f"用户提问：{user_query}\n"
            "关键词："
        )

        response = await rag_engine["llm"].acomplete(extract_prompt)
        llm_output = response.text.strip()

        # 清洗 LLM 输出：按逗号分割 -> 去空 -> 去重
        keywords = [k.strip() for k in llm_output.split(',') if k.strip()]
        
        # 再次兜底：如果 LLM 啥都没吐出来
        if not keywords:
            keywords = [user_query]
        
        print(f"🔍 [Graph API] LLM 提取的关键词: {keywords}")

        # Cypher 解释：
        # ANY(k IN $keywords WHERE ...) : 只要节点名字包含列表里的任意一个词，就匹配
        # toLower(...) : 忽略大小写
        # type(r) <> 'MENTIONS' : 过滤掉那些没有语义的引用连线
        cypher_sql = """
        MATCH (n)-[r]->(m)
        WHERE (
            ANY(k IN $keywords WHERE toLower(n.id) CONTAINS toLower(k)) 
            OR 
            ANY(k IN $keywords WHERE toLower(m.id) CONTAINS toLower(k))
        )
        AND type(r) <> 'MENTIONS'
        RETURN n.id AS source, type(r) AS label, m.id AS target
        LIMIT 30
        """
        
        # 3. 连接数据库
        # 请确保 NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD 变量已定义
        driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session() as session:
            result = session.run(cypher_sql, keywords=keywords)
            records = [record.data() for record in result]
            print(f"✅ Neo4j 查询成功，找到 {len(records)} 条关系")
            result_data["links"] = records
            
        driver.close()

    except Exception as e:
        # 🔥 关键：打印详细错误，方便你在终端看到
        import traceback
        traceback.print_exc() 
        print(f"❌ Neo4j 查询发生严重错误: {str(e)}")
        # 即使出错，result_data 也是 {"links": []}，不会是 None

    # 🔥 关键：无论 try 里发生了什么，这里一定会返回一个字典
    return result_data