import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

# LlamaIndex 核心组件
from llama_index.core import Settings, VectorStoreIndex, PropertyGraphIndex
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
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
NEO4J_PASSWORD = "password123"

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

        # 4. 构建混合检索策略
        # A. 向量检索工具
        vector_tool = vector_index.as_retriever(similarity_top_k=5)
        
        # B. 图谱检索工具 (使用 VectorContextRetriever 进行定位 + 扩散)
        from llama_index.core.retrievers import VectorContextRetriever
        sub_retriever = VectorContextRetriever(
            graph_store=graph_store,
            similarity_top_k=5,
            path_depth=2 # 抓取 2 跳邻居
        )
        graph_tool = graph_index.as_retriever(
            sub_retrievers=[sub_retriever]
        )
        
        # C. 组装混合检索器
        hybrid_retriever = HybridRetriever(vector_tool, graph_tool)

        # 5. 构建智能对话引擎 (ChatEngine)
        print("🤖 构建 ContextChatEngine...")
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        chat_engine = ContextChatEngine.from_defaults(
            retriever=hybrid_retriever,
            memory=memory,
            llm=llm,
            system_prompt="""
            你是一名专业的计算机课程助教 (EduMatrix)。
            
            【你的资源】：
            1. **对话历史**：用户之前的提问和你之前的回答。
            2. **背景知识**：检索到的教材原文(Qdrant)和图谱关系(Neo4j)。

            【回答策略】：
            1. 🚨 **最高优先级**：如果用户问**“你刚才说的”**、**“上一次回答”**等历史相关问题，请**务必优先基于【对话历史】**回答，不要重新检索或编造。
            2. 对于知识性问题，请基于【背景知识】回答，并尝试理清概念间的关系。
            3. 如果背景知识不足，请诚实告知。
            """
        )

        rag_engine["chat_engine"] = chat_engine
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

    # 3. 调用引擎 (流式)
    streaming_response = rag_engine["chat_engine"].stream_chat(
        last_message,
        chat_history=chat_history,
    )

    # 4. 生成流式响应
    def response_generator():
        # A. 吐出 AI 回答
        for token in streaming_response.response_gen:
            yield token
        
        # B. 吐出参考来源 (如果有)
        if streaming_response.source_nodes:
            yield "\n\n---\n**📚 参考来源：**\n"
            seen_sources = set()
            for node in streaming_response.source_nodes:
                # 简单去重和清洗
                clean_text = node.text[:100].replace('\n', ' ')
                if clean_text not in seen_sources:
                    yield f"- {clean_text}...\n"
                    seen_sources.add(clean_text)
            
    return StreamingResponse(response_generator(), media_type="text/plain")

@app.get("/")
def read_root():
    return {"message": "EduMatrix API is running! Go to /docs for Swagger UI."}