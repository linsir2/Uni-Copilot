import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
import nest_asyncio
import asyncio
from pathlib import Path
load_dotenv()
nest_asyncio.apply() # 运行嵌套使用asyncio循环

from llama_parse import LlamaParse, ResultType
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

PDF_PATH = Path(__file__).resolve().parents[1] / "data" / "深度学习进阶_自然语言处理_斋藤康毅.pdf"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "edu_matrix_v2"

async def main():
    print(f"🚀 [Async] 开始解析文件: {PDF_PATH} ...")

    # 初始化解析器
    parser = LlamaParse(
        result_type=ResultType.MD, # 输出格式为markdown
        verbose=True, # 在终端打印详细的进度条和日志
        language="ch_sim",
        num_workers=4,
        api_key=os.getenv("LLAMACLOUD_API_KEY") or "",
        fast_mode=False,
        system_prompt="""
        这是一个计算机科学教材。
        1. 请精确保留所有的数学公式（使用 LaTeX 格式）。
        2. 不要输出页眉和页脚的页码信息。
        3. 如果遇到表格，请将其转换为 Markdown 表格。
        4. 保持正文的连贯性。不要输出 'Here are some facts' 这类无关文字。
        """,
    )

    print("⏳ 正在请求 LlamaCloud API 进行云端解析（这可能需要几十秒）...")
    documents = await parser.aload_data(str(PDF_PATH))

    # print("\n--- [Preview] Markdown 源码预览 ---")
    # print(documents[0].text[:500])

    # 构建parent-child索引策略
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[1000, 200], # 父块以及字块各自的tokens数
    )

    print("✂️  正在执行本地切分 (Parent-Child Strategy)...")
    # 这一步是在本地 CPU 运行的，速度很快
    nodes = node_parser.get_nodes_from_documents(documents)

    # 获取所有的“叶子节点”（也就是最底层的 Child Chunk，那 200 tokens 的块）
    leaf_nodes = get_leaf_nodes(nodes)

    print(f"✅ 数据治理完成！")
    print(f"📊 统计数据:")
    print(f"  - 总节点数 (Parent + Child): {len(nodes)}")
    print(f"  - 待存入向量库的子节点数 (Child Nodes): {len(leaf_nodes)}")

    client = qdrant_client.QdrantClient(url=QDRANT_URL)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("🧠 正在加载 BGE-M3 嵌入模型 (首次运行会自动下载)...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        trust_remote_code=True
    )
    Settings.embed_model = embed_model # 全局默认使用该模型嵌入向量

    index = VectorStoreIndex( # 自动调用之前设置的转换向量模型把文本块转换成向量
        leaf_nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    print("\n🎉 ================= Success ================= 🎉")
    print(f"数据已成功注入 EduMatrix！")
    print(f"  - 向量数据库: Qdrant")
    print(f"  - 集合名称: {COLLECTION_NAME}")
    print(f"  - 嵌入模型: BGE-M3")
    print("下一步: 你可以去 Qdrant 的 Dashboard (http://localhost:6333/dashboard) 查看数据了！")

if __name__ == "__main__":
    asyncio.run(main())