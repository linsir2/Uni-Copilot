import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from pathlib import Path
import re

from llama_index.llms.dashscope import DashScope
from llama_index.core import Settings, PropertyGraphIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor # from llama_index.core.indices.property_graph import SchemaLLMPathExtractor 千问与openai格式不同，不兼容
from llama_parse import LlamaParse, ResultType
load_dotenv()

def clean_text(text: str) -> str:
    """清除PDF提取文本中的噪声"""
    # 去掉页码
    text = re.sub(r'^\s*\d+\s*$', ' ', text, flags=re.MULTILINE)

    text = text.replace('深度学习进阶：自然语言处理', '')

    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()

# 1. 定义一个自定义的解析函数，专门处理 "实体 | 关系 | 实体" 这种格式
def custom_parse_triplets(llm_output: str):
    """
    手动解析 LLM 输出，避免逗号干扰。
    期望格式: 实体1 | 关系 | 实体2
    """
    triplets = []
    lines = llm_output.strip().split("\n")
    for line in lines:
        # 跳过空行或过短的行
        if len(line) < 5: 
            continue
            
        # 使用 | 进行切分
        parts = line.split("|")
        if len(parts) == 3:
            subj = parts[0].strip()
            pred = parts[1].strip()
            obj = parts[2].strip()
            
            # 🧹 数据清洗：如果实体是纯数字、单字母变量(x, y)，或者看起来像乱码，直接丢弃
            # 这里用正则过滤掉 "0", "1", "t", "(0,0,1)" 这种垃圾实体
            if len(subj) < 2 or len(obj) < 2:
                continue
            if re.match(r'^[\d\(\)\[\],.=\s]+$', subj): # 过滤纯数字符号组合
                continue
                
            triplets.append((subj, pred, obj))
    return triplets



PDF_PATH = Path(__file__).resolve().parents[1] / "data" # / "深度学习进阶_自然语言处理_斋藤康毅.pdf"
# PDF_PATH = "../data/深度学习进阶_自然语言处理_斋藤康毅.pdf"

NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"
NEO4J_URI = "bolt://localhost:7687"

TEST_MODE = True

def split_markdown_semantic(md_text: str):
    """
    将 LlamaParse 输出的 Markdown 拆成【纯正文 chunk】
    - 丢弃表格
    - 丢弃过短噪声
    """
    chunks = []
    buffer = []

    for line in md_text.splitlines():
        line = line.strip()

        # 丢弃表格
        if line.startswith("|") or re.match(r"^\|?[-: ]+\|?$", line):
            continue

        # 标题：切 chunk
        if line.startswith("#"):
            if buffer:
                chunk = "\n".join(buffer).strip()
                if len(chunk) > 80:
                    chunks.append(chunk)
                buffer = []
            continue

        if line:
            buffer.append(line)

    if buffer:
        chunk = "\n".join(buffer).strip()
        if len(chunk) > 80:
            chunks.append(chunk)

    return chunks

def main():
    print(f"🚀 [Graph] 准备构建知识图谱...")

    print(f"🤖 初始化 LLM: {os.getenv('DASHSCOPE_MODEL_NAME')}...")
    llm = DashScope(
            model_name=os.getenv("DASHSCOPE_MODEL_NAME"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.1,
        )
    Settings.llm = llm

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

    file_extractor = {".pdf": parser}

    print("⏳ 正在请求 LlamaCloud API 进行云端解析（这可能需要几十秒）...")
    # documents = parser.load_data(PDF_PATH)

    # SimpleDirectoryReader中加入参数recursive=True，可以让这个reader读取填入的路径下的子文件夹
    raw_docs = SimpleDirectoryReader(input_dir=PDF_PATH, file_extractor=file_extractor).load_data() # pyright: ignore[reportArgumentType]
    # for doc in documents:
    #     # 获取原始内容
    #     original_text = doc.get_content() # 或者 doc.text
    
    #     # 清洗
    #     cleaned_text = clean_text(original_text)
    
    #     # ✅ 使用 set_content 替代 doc.text = ... 以消除 Pylance 报错
    #     doc.set_content(cleaned_text)

    # print(f"🧹 已清洗 {len(documents)} 页文档的噪声数据。")
    
    """
    # ⚠️ 测试模式截断
    if TEST_MODE:
        print("⚡️ [测试模式] 仅处理前 20 页数据...")
        documents = documents[30:45]
    """

    raw_docs = raw_docs[30:60]

    documents = []

    for doc in raw_docs:
        md_text = clean_text(doc.get_content())
        chunks = split_markdown_semantic(md_text)

        for chunk in chunks:
            documents.append(
                doc.__class__(
                    text=chunk,
                    metadata={
                        "source": "raw_text",
                        "type": "text"
                    }
                )
            )

    print(f"✅ 结构化完成，最终入库 chunk 数量: {len(documents)}")
    
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

    # kg_extractor = SimpleLLMPathExtractor(
    #     llm=llm,
    #     max_paths_per_chunk=15, # 每段文本最多提取15条关系，防止幻觉
    #     num_workers=4
    # )

    # kg_prompt_template = (
    #     "你是一个知识图谱提取专家。\n"
    #     "请从以下文本中提取实体和关系，格式为 (实体1, 关系, 实体2)。\n"
    #     "不要输出任何介绍性文字（如'Here are some facts...'）。\n"
    #     "---------------------\n"
    #     "{text}\n"
    #     "---------------------\n"
    # )

#     kg_prompt_template = """
# 你是一名【中文计算机教材】知识图谱构建专家。

# 请从下列教材文本中，尽可能多地提取【有意义的实体关系三元组】。

# 要求：
# 1. 每条输出为一行
# 2. 格式为：实体1, 关系, 实体2
# 3. 实体请使用教材中的原始中文术语
# 4. 关系请使用简短英文动词或动词短语（如 IS_A, USES, PART_OF, APPLIED_TO 等）
# 5. 如果关系在语义上成立，即可输出，不必过度保守，但也不能随便创建关系
# 6. 不要输出任何解释性文字

# 教材文本：
# {text}
# """

#     kg_extractor = SimpleLLMPathExtractor(
#         llm=llm,
#         extract_prompt=kg_prompt_template,
#         max_paths_per_chunk=15,
#         num_workers=4
#     )

    kg_prompt_template = """
你是一名【计算机科学】知识图谱构建专家。
请从下列教材文本中提取【核心概念】及其【关系】，构建知识三元组。

### 严格约束：
1. **格式**：每行一个三元组，使用 "|" 分隔，格式为：`实体1 | 关系 | 实体2`
2. **拒绝数学符号**：不要提取纯数字（如 "0", "1"）、单字母变量（如 "x", "t"）或公式片段作为实体。
3. **实体要求**：实体必须是具有独立语义的名词（如“交叉熵误差”、“Softmax函数”、“神经网络”）。
4. **关系要求**：关系必须是动词或动词短语（如“计算”、“属于”、“包含”、“用于”）。
5. **语言**：保持实体为中文（除非原文是专有名词英文）。

### 错误示例（绝对不要输出）：
- 0 | 0 | 1  (禁止纯数字)
- t | 等于 | (0,0,1) (禁止公式)
- 这里的 | 是 | 标签 (禁止无意义文本)

### 正确示例：
- Softmax层 | 输出 | 概率分布
- 交叉熵误差 | 用于衡量 | 损失
- 神经网络 | 包含 | 隐藏层

### 待处理文本：
{text}
"""

    # 3. 实例化 Extractor 时，传入 parse_fn
    kg_extractor = SimpleLLMPathExtractor(
        llm=llm,
        extract_prompt=kg_prompt_template,
        max_paths_per_chunk=20, # 稍微调大一点
        num_workers=4,
        parse_fn=custom_parse_triplets  # <--- 关键：注入我们自定义的解析器
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