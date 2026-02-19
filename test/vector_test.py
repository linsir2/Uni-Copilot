import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
from backend.app.core.edu_parser.base import MultimodalAgenticRAGPack
from llama_index.llms.dashscope import DashScope
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    _faithfulness,
    _context_recall,
    _context_precision,
)
from dotenv import load_dotenv
from pathlib import Path
import asyncio

load_dotenv()
test_csv = Path(__file__).resolve().parent / "my_golden_dataset.csv"
df = pd.read_csv(test_csv)

rag_pack = MultimodalAgenticRAGPack(
    qdrant_url=os.getenv("QDRANT_URL") or "",
    neo4j_password="password123",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
)
async def main():
    results = []

    for index, row in df.iterrows():
        result = await rag_pack.run(query=row["question"])
        streaming_response = result["final_response"]
        text = ""

        async for token in streaming_response:
            text += token.delta
        
        results.append({
            "question": row["question"],
            "answer": text,
            "retrieved_contexts": [node["text"] for node in result["retrieved_nodes"]],
            "ground_truth": row["ground_truth"],
        })
    
    eval_dataset = Dataset.from_list(results)
    from langchain_community.chat_models import ChatTongyi
    llm = ChatTongyi(model="qwen-plus-2025-09-11", api_key=os.getenv("DASHSCOPE_API_KEY"))
    
    print("⚖️ 正在阅卷中，请稍候...")
    score = evaluate(
        eval_dataset,
        metrics=[
        _faithfulness,
        _context_recall,
        _context_precision,
        ],
        llm=llm
    )
    print("\n📊 RAGAS 评估结果:")
    print(score)
    df_score = score.to_pandas()

    output_file = Path(__file__).resolve().parent / "final_eval_report.csv"
    df_score.to_csv(output_file, index=False)

if __name__=="__main__":
    asyncio.run(main())