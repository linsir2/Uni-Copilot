import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
import qdrant_client
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, PropertyGraphIndex
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.llms import ChatMessage

load_dotenv()

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "edu_matrix_v2" # 对应 ingest_pipeline.py 里的名字

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or ""

def main():
    print("🚀 [Hybrid Search] 正在初始化检索引擎...")

    # -------------------------------------------------------
    # 1. 准备模型 (Embedding + LLM)
    # -------------------------------------------------------
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        trust_remote_code=True,
        device="cpu",
        local_files_only=True,
    )
    Settings.embed_model = embed_model

    llm = DashScope(
        model_name=DashScopeGenerationModels.QWEN_MAX,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.4,
    )
    Settings.llm = llm

    client = qdrant_client.QdrantClient(url=QDRANT_URL,)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
    )
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )

    vector_retriever = vector_index.as_retriever(similarity_top_k=3)

    print("🔌 连接 Neo4j (图谱推理)...")
    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USER,
        url=NEO4J_URL,
        password=NEO4J_PASSWORD,
    )
    graph_index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_model=embed_model
    )
    
    graph_retriever = graph_index.as_retriever(include_text=True,)

    query = "神经网络包含什么"
    print(f"\n 测试提问：{query}")

    print("\n🔍 --- Vector Search Result (Qdrant) ---")
    vector_nodes = vector_retriever.retrieve(query)
    for i, node in enumerate(vector_nodes):
        clean_text = node.text[:100].replace('\n', ' ')
        print(f"[{i+1}] {clean_text}...")

    print("\n🔍 --- Graph Search Result (Neo4j) ---")
    graph_nodes = graph_retriever.retrieve(query)
    for i, node in enumerate(graph_nodes):
        clean_text = node.text[:100].replace('\n', ' ')
        print(f"[{i+1}] {clean_text}...")

    context_str = "\n".join([n.text for n in vector_nodes + graph_nodes])
    prompt = f"基于以下背景知识回答问题：'{query}'。\n\n背景知识：\n{context_str}"
    
    print("\n🤖 --- LLM Final Answer ---")
    messages = [
        ChatMessage(role="system", content="你是一个乐于助人的计算机课程助教。"),
        ChatMessage(role="user", content=prompt)
    ]
    response = llm.chat(messages)
    print(response.message.content)

if __name__ == "__main__":
    main()