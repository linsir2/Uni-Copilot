import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from pathlib import Path

from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings, PropertyGraphIndex
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor     # from llama_index.core.indices.property_graph import SchemaLLMPathExtractor 千问与openai格式不同，不兼容

load_dotenv()

PDF_PATH = Path(__file__).resolve().parents[1] / "data" / "深度学习进阶_自然语言处理_斋藤康毅.pdf"

NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"
NEO4J_URI = "bolt://localhost:7687"

TEST_MODE = True

def main():
    print(f"🚀 [Graph] 准备构建知识图谱...")

    print(f"🤖 初始化 LLM: {os.getenv('DASHSCOPE_MODEL_NAME')}...")
    llm = OpenAILike(
        model=os.getenv("DASHSCOPE_MODEL_NAME") or "",
        api_base=os.getenv("DASHSCOPE_BASE_URL"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        is_chat_model=True, # 告诉系统这是一个对话模型
        context_window=32000, # 模型一次读取的字数
    )
    Settings.llm = llm

    loader = PyMuPDFReader()
    documents = loader.load_data(file_path=PDF_PATH)
    
    """
    # ⚠️ 测试模式截断
    if TEST_MODE:
        print("⚡️ [测试模式] 仅处理前 20 页数据...")
        documents = documents[30:45]
    """

    graph_store = Neo4jPropertyGraphStore(
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI
    )

    print("🧠 开始提取知识实体与关系 (Graph Extraction)...")
    print("☕️ 这步比较慢，Qwen 正在阅读并整理知识点，请耐心等待...")

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        trust_remote_code=True,
        local_files_only=True,
    )

    kg_extractor = SimpleLLMPathExtractor(
        llm=llm,
        max_paths_per_chunk=15, # 每段文本最多提取15条关系，防止幻觉
        num_workers=4
    )

    index = PropertyGraphIndex.from_documents(  # 把文档，提取器，存储器串联起来，执行流水线工作
        documents=documents,
        kg_extractors=[kg_extractor],
        embed_model=embed_model, # 既可以做图搜索，也可以对图上的节点做向量搜索
        property_graph_store=graph_store,
        show_progress=True,
    )

    print("\n🎉 ================= Success ================= 🎉")
    print("知识图谱构建完成！")
    print(f"数据已存入 Neo4j。")
    print("下一步: 打开浏览器 http://localhost:7474 查看你的图谱！")
    print("推荐查询语句: MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 50")

if __name__ == "__main__":
    main()