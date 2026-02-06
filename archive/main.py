import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

load_dotenv()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, PropertyGraphIndex
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.llms import ChatMessage
from llama_index.core.retrievers import VectorContextRetriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine # 这个类偏向于底层，而as_chat_engine是封装完毕的高级一些的接口，底层用的仍然是ContextChatEngine
from llama_index.core.chat_engine.types import ChatMode
import qdrant_client

rag_engine = {}
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "edu_matrix_v2"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [Startup] 正在初始化 EduMatrix 引擎...")
    
    try:
        # 1. 初始化 Embedding 模型 (本地)
        print("🧠 加载 Embedding 模型...")
        embed_model = HuggingFaceEmbedding( # from llama_index.embeddings.huggingface import ...
            model_name="BAAI/bge-m3",
            trust_remote_code=True,
            local_files_only=True,
            device="cpu",
        )
        Settings.embed_model = embed_model

        # 2. 初始化 LLM (阿里云 Qwen)
        print("🤖 连接 DashScope LLM...")
        llm = DashScope( # from llama_index.llms.dashscope import ...
            model_name=os.getenv("DASHSCOPE_MODEL_NAME"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.1,
        )

        # 3. 连接向量数据库 (Qdrant)
        print("🔌 连接 Qdrant...")
        client = qdrant_client.QdrantClient( # 这是qdrant的原生库，连接qdrant客户端
            url=QDRANT_URL,
        )
        vector_store = QdrantVectorStore( # 让llama-index能够使用qdrant，把llamaindex的指令翻译成qdrant能听懂的原生指令
            collection_name=QDRANT_COLLECTION,
            client=client,
        ) # 不仅负责存储，还负责找数据

        # index:是静态的，不负责查询检索动作，而是负责维护数据结构
        #       如果是from_documents：负责把乱七八糟的文档整理成有序的向量或图谱（建数据库）
        #       如果是from_existing：代表整个数据库的访问句柄，知道数据库中有啥内容。
        #       from_vector_store原理与from_existing是一样的。
        # 与retriever的关系：它手里握着数据库连接，它能生产出各种各样的工具，其中一个工具就是 Retriever。

        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
        )

        # 4. 连接图数据库 (Neo4j)
        print("🔌 连接 Neo4j...") # property graph：带属性的图
        graph_store = Neo4jPropertyGraphStore( # 物理存储
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            url=NEO4J_URL,
        )
        graph_index = PropertyGraphIndex.from_existing( # .from_existing：读取模式，从现有的数据库中读取
            property_graph_store=graph_store,
            # embed_model=embed_model,
            llm=llm
        ) # .from_documents：从文档创建，进行向量化与图谱化

        # rag_engine["vector_retriever"] = vector_index.as_retriever(similarity_top_k=4)

        sub_retriever = VectorContextRetriever( # 这一步包含把用户问题转成向量，去neo4j中查询向量相似的节点，然后抓取子图
            graph_store=graph_store,
            # embed_model=embed_model,
            similarity_top_k=5,
            path_depth=2 # 多跳
        )

        # rag_engine["graph_retriever"] = graph_index.as_retriever(
        #     sub_retrievers=[sub_retriever,],
        #     include_text=True,
        # )

        # rag_engine["llm"] = llm

        vector_tool = vector_index.as_retriever(similarity_top_k=5)
        graph_tool = graph_index.as_retriever(
            sub_retrievers=[sub_retriever,]
        )

        from llama_index.core.retrievers import BaseRetriever

        class HybridRetriever(BaseRetriever):
            def __init__(self, vector_retriever, graph_retriever):
                self.vector_retriever = vector_retriever
                self.graph_retriever = graph_retriever
                super().__init__()
            
            def _retrieve(self, query_bundle):
                nodes_vect = self.vector_retriever.retrieve(query_bundle)
                nodes_graph = self.graph_retriever.retrieve(query_bundle)

                return list({n.node.node_id: n for n in (nodes_vect + nodes_graph)}.values())
        
        hybrid_retriever = HybridRetriever(vector_tool, graph_tool)

        # 定义内存缓冲区
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        # 这里我们用 ContextChatEngine，配合我们的混合检索器
        chat_engine = ContextChatEngine.from_defaults(
            retriever=hybrid_retriever,
            memory=memory,
            llm=llm,
            system_prompt="""
            你是一名专业的计算机课程助教 (EduMatrix)。
            
            【你的能力】：
            1. 你拥有【对话历史】（用户和你之前的聊天记录）。
            2. 你拥有【背景知识】（检索到的教材原文和图谱关系）。

            【回答策略】：
            1. 🚨 **最高优先级**：如果用户的问题是关于**“你刚才说的”、“上一次回答”**或**“之前的对话”**（例如：“你刚刚列出的第一点是什么？”），请**务必优先基于【对话历史】**进行回答，忽略与之冲突的检索结果。
            2. 对于其他知识性问题（例如：“什么是Word2Vec？”），请基于检索到的【背景知识】回答。
            3. 结合原文和图谱关系，使得回答既准确又有逻辑。
            """,
        )

        # system_prompt = """
        #     你是一名专业的计算机课程助教 (EduMatrix)。
        #     你的知识库包含了教材原文（向量）和概念关系图谱（结构化知识）。
            
        #     请综合利用检索到的【背景知识】回答用户问题。
        #     回答时，请尝试理清概念之间的关系（例如：A是B的组成部分，C导致了D）。
        #     如果背景知识不足，请诚实地说不知道。
        #     """

        # 什么是ChatEngine?它是LlamaIndex的高级封装。工作流程：用户提问 -> 看历史记录 -> 重写问题 -> 去向量库检索 -> 给LLM
        # chat_engine = vector_index.as_chat_engine( # 使用对话引擎，记载历史对话
        #     chat_mode=ChatMode.CONTEXT, # 每次回答，都会先去检索相关文档
        #     memory=memory,
        #     system_prompt=system_prompt,
        #     llm=llm,
        #     similarity_top_k=5,
        # )

        # chat_engine = graph_index.as_chat_engine(
        #     chat_mode=ChatMode.CONTEXT,
        #     memory=memory,
        #     llm=llm,
        #     system_prompt=system_prompt,
        #     retriever_kwargs={
        #         "sub_retrievers": [sub_retriever],
        #         "include_text": True,
        #     }
        # )

        rag_engine["chat_engine"] = chat_engine

        print("✅ 引擎初始化完成！等待请求...")

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        raise e

    yield # 服务器开始运行

    # 服务器关闭时的清理工作 (这里暂时不需要)
    print("👋 [Shutdown] 服务器已关闭")

app = FastAPI(title="EduMatrix API", lifespan=lifespan)


# class ChatRequest(BaseModel):
#     query: str

# class ChatResponse(BaseModel):
#     answer: str
#     sources: list[str]

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]


# @app.post("/api/chat")
# async def chat_endpoint(request: ChatRequest):
#     if not rag_engine:
#         raise HTTPException(status_code=500, detail="Engine not initialized")
    
#     query = request.query
#     print(f"📩 收到提问: {query}")

#     # 1. 执行混合检索
#     vector_nodes = rag_engine["vector_retriever"].retrieve(query)
#     graph_nodes = rag_engine["graph_retriever"].retrieve(query)

#     # 2. 整理上下文
#     all_nodes = vector_nodes + graph_nodes
#     context_str = "\n".join([n.text for n in all_nodes])
    
#     # 收集来源信息 (为了展示给用户看)
#     source_texts = [n.text[:150].replace('\n', ' ') + "..." for n in all_nodes]

#     # 3. 构造 Prompt
#     prompt = f"""
#     你是一名专业的计算机课程助教。请基于以下【背景知识】回答用户的【问题】。
#     如果背景知识不足以回答，请诚实地说不知道，不要编造。

#     【问题】：{query}

#     【背景知识】：
#     {context_str}
#     """

#     # 4. 调用 LLM
#     # DashScope 的 chat 接口
#     messages = [
#         ChatMessage(role="system", content="你是一个乐于助人的助教。"),
#         ChatMessage(role="user", content=prompt)
#     ]
#     streaming_response = rag_engine["llm"].stream_chat(messages)

#     def response_generator():
#         # A. 先把 LLM 生成的答案，一个字一个字吐给前端
#         for token in streaming_response:
#             # 1. 尝试取增量文本 (Standard LlamaIndex streaming)
#             if hasattr(token, 'delta') and token.delta:
#                 yield token.delta
#             # 2. 如果没有 delta，尝试取 message.content
#             elif hasattr(token, 'message') and token.message.content:
#                 yield token.message.content
#             # 3. 如果它本身就是字符串 (防御性编程)
#             elif isinstance(token, str):
#                 yield token
#             # 4. 实在不行，强转字符串 (虽然可能会带上格式噪音，但至少不报错)
#             else:
#                 yield str(token)

#         # B. 答案吐完了，我们在最后追加"参考来源"
#         # 这样用户最后能看到引用了哪些书
#         if source_texts:
#             yield "\n\n---\n**📚 参考来源：**\n"
#             for src in source_texts:
#                 yield f"- {src}\n"
    
#     return StreamingResponse(response_generator(), media_type="text/plain")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not rag_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    # 获取用户提的问题
    last_message = request.messages[-1].content
    print(f"📩 收到问题: {last_message}")

    # 将剩余的消息充当历史消息
    chat_history = [
        ChatMessage(role=m.role, content=m.content)
        for m in request.messages[:-1] # 去除最后一条消息
    ]

    # 调用引擎
    streaming_response = rag_engine["chat_engine"].stream_chat(
        last_message,
        chat_history=chat_history,
    )

    # 输出
    def response_generator():
        for token in streaming_response.response_gen: # 搜索引擎的流式输出才有response_gen，而原来的检索器检索（as_retriever）检索时直接用stream_response就行了
            yield token
        
        if streaming_response.source_nodes:
            yield "\n\n---\n**📚 参考来源：**\n"
            for node in streaming_response.source_nodes:
                # 这里的 node.text 就是检索到的教材片段
                clean_text = node.text[:100].replace('\n', ' ')
                yield f"- {clean_text}...\n"
            
    return StreamingResponse(response_generator(), media_type="text/plain")

@app.get("/")
def read_root():
    return {"message": "EduMatrix API is running! Go to /docs for Swagger UI."}