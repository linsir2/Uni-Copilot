import os
import sys
import re
import asyncio
import json
from typing import List, Set, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import nest_asyncio
import qdrant_client
from qdrant_client.http import models as rest

# 添加项目根目录到 path 以便导入 etl 模块
sys.path.append(str(Path(__file__).resolve().parents[1]))

# 导入你的 Parser (v1.5.1 / v1.6)
from etl.local_parser import LocalPDFParser

# 环境配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
load_dotenv()
nest_asyncio.apply()

from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import Settings, VectorStoreIndex, StorageContext, PropertyGraphIndex
from llama_index.core.graph_stores.types import EntityNode, Relation, KG_NODES_KEY, KG_RELATIONS_KEY
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.llms.dashscope import DashScope
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import TransformComponent, BaseNode
from neo4j import GraphDatabase

# --- 配置 ---
PDF_PATH = Path(__file__).resolve().parents[1] / "data" / "深度学习进阶_自然语言处理_斋藤康毅.pdf"

QDRANT_URL = "http://localhost:6333"
CHUNK_COLLECTION = "edu_matrix_chunks"   
ENTITY_COLLECTION = "edu_matrix_entities" 
EMBEDDING_DIM = 1024 

NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or "password"
NEO4J_URI = "bolt://localhost:7687"

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "parser_cache"
SIDECAR_FILE = CACHE_DIR / "page_heavy_data.json"
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# --- 提示词 (Prompts) ---
# LlamaIndex 会自动将 node.text 填入 {text} 占位符
KG_EXTRACTION_PROMPT = """
你是一名专门从事【计算机科学与工程教育】的知识图谱构建专家。
你的任务是从给定的教材页面内容中提取核心概念及其逻辑关系。

### 内容背景：
输入的文本经过了多模态预处理，包含：
1. [SECTION]: 当前所属的章节标题，提供了宏观语境。
2. [KEYWORDS]: 页面高频术语，暗示了核心实体。
3. === 插图描述 ===: 由视觉模型(VLM)生成的图像语义，包含图中特有的组件和逻辑流。

### 提取规则：
1. **格式约束**：每行仅输出一个三元组，格式为：`实体1 | 关系 | 实体2`。
2. **实体粒度**：
   - 优先提取专业术语（如“反向传播”、“流水线冒险”、“虚拟地址空间”）。
   - 必须包含插图描述中提到的关键组件（如“ALU”、“寄存器文件”）。
   - 禁止提取纯数字、单字母变量（x, y, i）或无意义的代词（作者、本文、下图）。
3. **关系类型**：
   - 使用具体的动词短语描述逻辑：`包含`, `属于`, `实现`, `解决`, `导致`, `数据流向`, `控制信号`, `计算`, `优化`。
4. **语言要求**：保持实体名称与原文一致（中英文混排）。
5. **拒绝幻觉**：仅根据提供的文本提取，不要引入外部知识或输出解释性文字。

### 正确示例：
- 梯度消失 | 导致 | 权重更新缓慢
- ReLU函数 | 解决 | 梯度消失问题
- 插图1 | 展示了 | MIPS五级流水线结构
- 译码阶段 | 生成 | 控制信号

### 待处理文本：
{text}
"""

# ==========================================
# 🛠️ 工具函数：Qdrant 集合检查与创建 (含索引优化)
# ==========================================
def check_and_create_collection(
    client: qdrant_client.QdrantClient, 
    collection_name: str, 
    vector_size: int,
    hnsw_config: Optional[rest.HnswConfigDiff] = None,
    recreate: bool = False
):
    """
    检查集合。如果 recreate=True，则先删除旧集合再创建，防止数据重复叠加。
    """
    if recreate and client.collection_exists(collection_name):
        print(f"♻️  正在删除旧集合 {collection_name} (Recreate Mode)...")
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        print(f"⚠️ 集合 {collection_name} 不存在，正在创建 (Dim: {vector_size})...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
                on_disk=True, # [优化] 开启磁盘存储
            ),
            hnsw_config=hnsw_config 
        )
        
        # [优化] 创建 Payload 索引加速过滤
        client.create_payload_index(collection_name, "page", rest.PayloadSchemaType.INTEGER)
        client.create_payload_index(collection_name, "chunk_type", rest.PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name, "section_title", rest.PayloadSchemaType.TEXT)
        
        print(f"✅ 集合 {collection_name} 创建并初始化完成")
    else:
        print(f"✅ 集合 {collection_name} 已存在 (跳过创建)")

# ==========================================
# 🧠 Sidecar Aware Extractor (修复 Pydantic Init 报错)
# ==========================================
class MetadataGraphExtractor(TransformComponent):
    # 显式声明字段类型
    sidecar_data: Dict[str, Any] = {}

    def __init__(self, sidecar_path: Path, **kwargs):
        # 1. 先调用父类初始化 (不带 sidecar_data，防止 Pylance 报错)
        super().__init__(**kwargs)
        
        # 2. 手动加载数据并赋值给字段
        data = {}
        if sidecar_path.exists():
            try:
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"⚠️ 读取 Sidecar 文件失败: {e}")
        
        # 直接赋值，Pydantic 会处理
        self.sidecar_data = data

    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        for node in nodes:
            page_id_key = node.metadata.get("page_id_key")
            if not page_id_key: continue

            # 从 sidecar 获取图谱数据
            heavy_data = self.sidecar_data.get(page_id_key, {})
            graph_data = heavy_data.get("graph_data", {})
            
            if not graph_data: continue

            # [优化] 使用 list() 复制防止引用污染，增加 set() 去重防止重复添加
            existing_nodes = list(node.metadata.get(KG_NODES_KEY, []))
            existing_relations = list(node.metadata.get(KG_RELATIONS_KEY, []))
            
            seen_nodes: Set[str] = set(n.name for n in existing_nodes)
            seen_rels: Set[str] = set(f"{r.source_id}-{r.target_id}-{r.label}" for r in existing_relations)

            # 注入 VLM 识别的实体
            for entity in graph_data.get("entities", []):
                name = entity.get("name")
                label = entity.get("category", "Concept")
                if name and name not in seen_nodes:
                    existing_nodes.append(EntityNode(name=name, label=label, properties=entity))
                    seen_nodes.add(name)

            # 注入 VLM 识别的关系
            for rel in graph_data.get("relations", []):
                src = rel.get("source")
                tgt = rel.get("target")
                label = rel.get("relation", "RELATED_TO")
                rel_key = f"{src}-{tgt}-{label}"
                
                if src and tgt and rel_key not in seen_rels:
                    existing_relations.append(Relation(source_id=src, target_id=tgt, label=label, properties=rel))
                    seen_rels.add(rel_key)

            node.metadata[KG_NODES_KEY] = existing_nodes
            node.metadata[KG_RELATIONS_KEY] = existing_relations
            
        return nodes

def custom_parse_triplets(llm_output: str):
    """清洗 LLM 提取的三元组"""
    triplets = []
    lines = llm_output.strip().split("\n")
    for line in lines:
        if len(line) < 5: continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) == 3:
            subj, pred, obj = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if len(subj) < 2 or len(obj) < 2: continue
            if re.match(r'^[\d\(\)\[\],.=\s%<>\-\+\*\/\\a-zA-Z0-9]+$', subj) and len(subj) < 4: continue
            if re.match(r'^[\d\(\)\[\],.=\s%<>\-\+\*\/\\a-zA-Z0-9]+$', obj) and len(obj) < 4: continue
            if "here are" in subj.lower() or "example" in subj.lower(): continue
            triplets.append((subj, pred, obj))
    return triplets

async def main():
    print(f"🚀 [Async] 开始 Pipeline: {PDF_PATH}")
    
    # 1. 解析 (使用 v1.5+ Sidecar 模式 Parser)
    parser = LocalPDFParser(
        pdf_path=PDF_PATH,
        image_output_dir=str(CACHE_DIR / "images"),
        cache_file=str(CACHE_DIR / "vlm_cache.json"),
        hash_record_file=str(CACHE_DIR / "processed_hashes.json"),
        sidecar_file=str(SIDECAR_FILE),
        use_vlm=True,
        max_concurrency=5
    )
    documents = await parser.parse()
    # documents = documents[10:15] # 调试切片
    print(f"✅ 解析完成，获得 {len(documents)} 个页面级文档")

    # 2. Parent-Child 切分
    # [优化] Chunk Size: [800, 200] 适配教材密度
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[800, 200])
    nodes = node_parser.get_nodes_from_documents(documents)
    
    leaf_nodes = get_leaf_nodes(nodes)
    parent_nodes = [n for n in nodes if n.node_id not in set(x.node_id for x in leaf_nodes)]

    for n in parent_nodes:
        n.metadata["chunk_type"] = "parent"
        if n.metadata.get("page_label"): n.metadata["page"] = int(n.metadata["page_label"])

    for n in leaf_nodes:
        n.metadata["chunk_type"] = "leaf"
        if n.metadata.get("page_label"): n.metadata["page"] = int(n.metadata["page_label"])
    
    print(f"📊 切分统计: 父节点 {len(parent_nodes)} 个, 叶子节点 {len(leaf_nodes)} 个")

    # 3. 初始化 & 集合管理
    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    
    # [关键] Recreate=True 防止重复数据堆叠 (第一次跑或全量更新时开启)
    FORCE_RECREATE = True 
    
    check_and_create_collection(client, CHUNK_COLLECTION, EMBEDDING_DIM, recreate=FORCE_RECREATE)
    
    # [优化] 实体集合使用内存优化版 HNSW
    entity_hnsw = rest.HnswConfigDiff(m=16, ef_construct=64)
    check_and_create_collection(client, ENTITY_COLLECTION, EMBEDDING_DIM, hnsw_config=entity_hnsw, recreate=FORCE_RECREATE)
    
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", trust_remote_code=True)
    Settings.embed_model = embed_model
    llm = DashScope(model_name=os.getenv("DASHSCOPE_MODEL_NAME", "qwen-plus"), api_key=os.getenv("DASHSCOPE_API_KEY"), temperature=0.1)
    Settings.llm = llm

    # 4. 构建 Chunk 向量索引
    print(f"\n🧠 [Step 1/2] 构建 Chunk 索引 ({CHUNK_COLLECTION})...")
    vector_store_chunks = QdrantVectorStore(client=client, collection_name=CHUNK_COLLECTION)
    storage_context_chunks = StorageContext.from_defaults(vector_store=vector_store_chunks)
    
    # 只为 Leaf Nodes 建立向量索引
    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context_chunks,
        show_progress=True,
    )

    # 5. 构建 Graph
    print(f"\n🕸️ [Step 2/2] 构建知识图谱 ({ENTITY_COLLECTION})...")
    
    # 如果强制重建，顺便清空 Neo4j (开发模式安全措施)
    graph_store = Neo4jPropertyGraphStore(username=NEO4J_USER, password=NEO4J_PASSWORD, url=NEO4J_URI)
    if FORCE_RECREATE:
        print("⚠️ [DEV] 正在清空 Neo4j 数据库...")
        try:
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                driver.execute_query("MATCH (n) DETACH DELETE n")
            print("✅ Neo4j 清空完成")
        except Exception as e:
            print(f"Neo4j 清空失败 (可能为空): {e}")

    vector_store_entities = QdrantVectorStore(client=client, collection_name=ENTITY_COLLECTION)

    # 抽取器配置
    llm_extractor = SimpleLLMPathExtractor(
        llm=llm,
        extract_prompt=KG_EXTRACTION_PROMPT, # 使用新的详细提示词 (LlamaIndex 会自动填充 {text})
        max_paths_per_chunk=10, # [成本控制] 适度降低每块提取数量
        num_workers=4,
        parse_fn=custom_parse_triplets
    )
    
    # 传入 Sidecar 路径初始化 MetadataGraphExtractor
    metadata_extractor = MetadataGraphExtractor(sidecar_path=SIDECAR_FILE)

    PropertyGraphIndex(
        nodes=parent_nodes,
        kg_extractors=[metadata_extractor, llm_extractor],
        llm=llm,
        embed_model=embed_model,
        property_graph_store=graph_store,
        vector_store=vector_store_entities,
        # [优化] 显式开启 KG Embedding
        embed_kg_nodes=True, 
        show_progress=True,
    )

    print("\n🎉 ================= Pipeline Completed ================= 🎉")
    print(f"✅ Chunk Size: 800/200 optimized")
    print(f"✅ Entity HNSW: Memory optimized")
    print(f"✅ Graph Embedding: Enabled")
    print(f"✅ Dual-Source Fusion: Active")

if __name__ == "__main__":
    asyncio.run(main())