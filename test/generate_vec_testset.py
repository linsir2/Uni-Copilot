import os
import json
import pandas as pd
from qdrant_client import QdrantClient
from langchain_core.documents import Document as LCDocument
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# 1. 配置
COLLECTION_NAME = "edu_matrix_chunks"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
TESTSET_SIZE = 15  # 生成多少道题

def fetch_docs_from_qdrant():
    print(f"🔌 连接数据库: {QDRANT_URL} ...")
    client = QdrantClient(url=QDRANT_URL)
    
    # 使用 Scroll 遍历数据
    # 这里我们取前 300 个 chunk 作为出题素材（避免 Token 爆炸）
    # 如果是生产环境，建议随机采样
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=200, 
        with_payload=True,
        with_vectors=False
    )
    
    docs = []
    print(f"📥 从 Qdrant 读取到 {len(records)} 个片段...")

    for r in records:
        payload = r.payload
        
        # 🚫 关键过滤：不要基于“全文摘要”出题
        # 摘要包含全书内容，出的题太宏观，容易导致检索评估不准
        if payload.get("is_global_summary") == "true":
            continue

        # 提取内容 (兼容不同存储格式)
        content = payload.get("text")
        if not content and "_node_content" in payload:
            try:
                content = json.loads(payload["_node_content"]).get("text")
            except:
                pass
        
        if content:
            # 转换为 LangChain Document 给 Ragas 用
            docs.append(LCDocument(
                page_content=content,
                metadata={
                    "filename": payload.get("file_name", "unknown"),
                    "page_label": payload.get("page_label", "?")
                }
            ))
    
    print(f"✅ 筛选后有效出题片段: {len(docs)} 个")
    return docs

def main():
    # 1. 获取真实数据
    documents = fetch_docs_from_qdrant()
    
    if not documents:
        print("❌ 数据库为空，无法生成试题！请先运行 worker.py")
        return

    # 2. 初始化出题模型 (建议用最强的模型出题，如 qwen-max)
    llm = ChatOpenAI(
        model="qwen-plus-2025-09-11", 
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.0,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    embeddings = DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )

    # 3. Ragas 生成器
    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embedding_model=embeddings,
        llm_context=(
        "你是一个信息抽取模型，而不是讲解模型。\n"
        "⚠️ 你必须严格、只、且仅输出 JSON。\n"
        "⚠️ 不允许输出任何解释性文字、段落、说明。\n"
        "⚠️ 不允许出现非 JSON 内容。\n"
        "⚠️ JSON 必须是单个对象，而不是数组。\n"
        "所有字段值必须是字符串。\n"
        "所有内容使用【简体中文】。\n"
        "⚠️ 严禁使用 LaTeX 语法（如 \\( \\)、$ $）。\n"
        "⚠️ 严禁在字符串中出现反斜杠 \\ 。\n"
        "如涉及数学公式，请用自然语言描述，不要写公式。\n"
        )
    )

    run_config = RunConfig(
        max_workers=3,
        max_retries=5,
        timeout=120
    )

    print("🧠 正在根据数据库内容生成考题 (这需要一点时间)...")
    dataset = generator.generate_with_langchain_docs(
        documents,
        testset_size=TESTSET_SIZE,
        run_config=run_config,
    )

    # 4. 导出 CSV
    df = dataset.to_pandas()
    
    # 映射列名以适配你的 test.py
    if "user_input" in df.columns:
        df = df.rename(columns={"user_input": "question", "reference": "ground_truth"})
    
    # 这里的 ground_truth 是 Ragas 生成的标准答案
    # 这里的 question 是 Ragas 生成的问题
    

    output_file = Path(__file__).resolve().parent / "my_golden_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"🎉 考卷生成完毕！已保存至 {output_file}")
    print(df[["question", "ground_truth"]].head(2))

if __name__ == "__main__":
    main()