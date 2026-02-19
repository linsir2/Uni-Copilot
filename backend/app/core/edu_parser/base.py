import os
import json
import hashlib
import nest_asyncio
import asyncio
from typing import Any, Dict, List, Optional
from pathlib import Path
from pydantic import PrivateAttr
from enum import Enum

# LlamaIndex Core
from llama_index.core.llama_pack import BaseLlamaPack
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    PropertyGraphIndex,
)
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import (
    TransformComponent,
    BaseNode,
    TextNode,
    NodeWithScore,
)
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
)
from llama_index.core.retrievers import BaseRetriever, VectorContextRetriever
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM

# LlamaIndex Integrations
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.fastembed import FastEmbedEmbedding
# Qdrant & Neo4j Drivers
import qdrant_client
from qdrant_client.http import models as rest
from neo4j import GraphDatabase

# Local Imports
from .parser import EduMatrixPDFReader
from .workflow import EduMatrixWorkflow
from .entity_upsert import upsert_entities

# Apply nest_asyncio
nest_asyncio.apply()


def embed_text_wrapper(text: str) -> List[float]:
    """Wraps the global embedding model for the upsert utility."""
    return Settings.embed_model.get_text_embedding(text)


# --- Helper: Hybrid Retriever ---
class HybridRetriever(BaseRetriever):
    """
    Custom Hybrid Retriever: Queries both Vector Index and Property Graph Index,
    deduplicates based on content hash, and re-ranks by score.
    """

    def __init__(self, vector_retriever: BaseRetriever, graph_retriever: BaseRetriever, chunk_store: QdrantVectorStore):
        self.vector = vector_retriever
        self.graph = graph_retriever
        self.chunk_store = chunk_store
        super().__init__()

    def _retrieve(self, query_bundle) -> List[NodeWithScore]:
        # 1. Parallel Retrieval
        vec_nodes = self.vector.retrieve(query_bundle)
        graph_nodes = self.graph.retrieve(query_bundle)
        
        # RRF Algorithm
        rrf_k = 60
        merged_scores = {}
        node_map = {}

        def process_nodes(nodes):
            for rank, n in enumerate(nodes):
                if n.node.metadata.get("is_global_summary") == "true":
                    continue

                node_type = type(n.node).__name__
                node_key = f"{node_type}:{n.node.node_id}"
                if node_key not in node_map:
                    node_map[node_key] = n.node
                
                current_score = merged_scores.get(node_key, 0.0)
                merged_scores[node_key] = current_score + 1 / (rrf_k + rank + 1)
        
        process_nodes(vec_nodes)
        process_nodes(graph_nodes)

        sorted_candidates = sorted(
            merged_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        final_nodes: List[NodeWithScore] = []
        seen_hashes: set[str] = set()

        for node_id, score in sorted_candidates:
            node = node_map[node_id]
            content = node.get_content()

            if not content:
                continue

            norm_text = (content[:200] + content[-200:]).strip().lower()
            content_hash = hashlib.md5(norm_text.encode("utf-8")).hexdigest()

            if content_hash in seen_hashes:
                continue

            final_nodes.append(
                NodeWithScore(node=node, score=score)
            )
            seen_hashes.add(content_hash)

        return final_nodes

    async def fetch_global_summary(self) -> List[NodeWithScore]:
        """
        🛑 Fallback Method:
        Directly fetch the global summary node from Qdrant using metadata filter.
        Bypasses vector similarity search entirely.
        """
        try:
            client = self.chunk_store.client

            scroll_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="is_global_summary",
                        match=rest.MatchValue(value="true")
                    )
                ]
            )

            if asyncio.iscoroutinefunction(client.scroll):
                points, _ = await client.scroll(
                    collection_name=self.chunk_store.collection_name,
                    scroll_filter=scroll_filter,
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
            else:
                points, _ = client.scroll(
                    collection_name=self.chunk_store.collection_name,
                    scroll_filter=scroll_filter,
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )

            if not points:
                print("⚠️ No Global Summary found in DB.")
                return []
            
            payload = points[0].payload
            text = payload.get("text", "") or payload.get("_node_content", "{}")

            if isinstance(text, str) and text.startswith("{"):
                try:
                    content_dict = json.loads(text)
                    text = content_dict.get("text", "")
                except:
                    pass
            
            node = TextNode(text=text, metadata=payload)
            return [NodeWithScore(node=node, score=0.01)]
        
        except Exception as e:
            print(f"❌ Failed to fetch global summary: {e}")
            return []
            
# --- Helper: Sidecar Extractor ---
class MetadataGraphExtractor(TransformComponent):
    """
    Injects Graph Data from the Sidecar JSON into LlamaIndex Nodes
    for PropertyGraphIndex ingestion.
    """

    _sidecar_data: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _sidecar_path: Path = PrivateAttr()

    def __init__(self, sidecar_path: str, **kwargs):
        super().__init__(**kwargs)
        # Assign to private attributes using the underscore prefix
        self._sidecar_path = Path(sidecar_path)
        if self._sidecar_path.exists():
            try:
                with open(self._sidecar_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._sidecar_data = data
                print(
                    f"📂 [Sidecar] Loaded metadata for {len(self._sidecar_data)} pages."
                )
            except Exception as e:
                print(f"⚠️ Failed to read Sidecar file: {e}")
        else:
            print(f"⚠️ Sidecar file not found: {sidecar_path}")

    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        for node in nodes:
            page_id_key = node.metadata.get("page_id_key")
            if not page_id_key:
                continue

            # Access the private data attribute
            heavy_data = self._sidecar_data.get(page_id_key, {})
            if "images" in heavy_data:
                node.metadata["image_path"] = heavy_data["images"]

            evidence = heavy_data.get("evidence_images", [])
            if evidence:
                # format: [[x0, y0, x1, y1], [x0, y0, x1, y1]]
                image_bboxes = [img["bbox"] for img in evidence if "bbox" in img]

                if image_bboxes:
                    if node.metadata.get("chunk_type") == "parent":
                        node.metadata["bbox"] = json.dumps(image_bboxes)

            graph_data = heavy_data.get("graph_data", {})
            if not graph_data:
                continue

            existing_nodes = node.metadata.get(KG_NODES_KEY, [])
            existing_relations = node.metadata.get(KG_RELATIONS_KEY, [])

            seen_nodes = {n.name for n in existing_nodes}
            seen_rels = {
                f"{r.source_id}-{r.target_id}-{r.label}" for r in existing_relations
            }

            for entity in graph_data.get("entities", []):
                name = entity.get("name")
                label = entity.get("category", "Concept")
                if name and name not in seen_nodes:
                    props = {
                        k: v for k, v in entity.items() if k not in ["name", "category"]
                    }
                    existing_nodes.append(
                        EntityNode(name=name, label=label, properties=props)
                    )
                    seen_nodes.add(name)

            for rel in graph_data.get("relations", []):
                src = rel.get("source")
                tgt = rel.get("target")
                label = rel.get("relation", "RELATED_TO")
                if src and tgt:
                    rel_key = f"{src}-{tgt}-{label}"
                    if rel_key not in seen_rels:
                        props = {
                            k: v
                            for k, v in rel.items()
                            if k not in ["source", "target", "relation"]
                        }
                        existing_relations.append(
                            Relation(
                                source_id=src,
                                target_id=tgt,
                                label=label,
                                properties=props,
                            )
                        )
                        seen_rels.add(rel_key)

            node.metadata[KG_NODES_KEY] = existing_nodes
            node.metadata[KG_RELATIONS_KEY] = existing_relations
        return nodes


# ================= MAIN PACK CLASS =================


class MultimodalAgenticRAGPack(BaseLlamaPack):
    """
    Multimodal Agentic RAG Pack.
    """

    def __init__(
        self,
        qdrant_url: str,
        neo4j_url: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "password",
        dashscope_api_key: Optional[str] = None,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        qdrant_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        chunk_collection: str = "edu_matrix_chunks",
        entity_collection: str = "edu_matrix_entities",
        data_dir: str = "./data_sidecar",
        force_recreate: bool = False,
    ) -> None:
        self.dashscope_api_key = dashscope_api_key
        self.neo4j_creds = {
            "url": neo4j_url,
            "user": neo4j_username,
            "password": neo4j_password,
        }
        self.qdrant_config = {"url": qdrant_url, "api_key": qdrant_api_key}
        self.collections = {"chunks": chunk_collection, "entities": entity_collection}
        self.data_dir = Path(data_dir)
        self.force_recreate = force_recreate

        # 1. Setup Models
        if llm is not None:
            self.llm = llm
        elif dashscope_api_key:
            self.llm = DashScope(
                model_name="qwen-flash-2025-07-28", api_key=dashscope_api_key, temperature=0.1
            )
        else:
            raise ValueError(
                "❌ Init Error: Please provide either an 'llm' object or a 'dashscope_api_key'."
            )

        if embed_model is not None:
            self.embed_model = embed_model
        else:
            print("ℹ️ No embed_model provided, falling back to default embed_model")
            self.embed_model = FastEmbedEmbedding(
                model_name="BAAI/bge-small-zh-v1.5",
                threads=None,
                cache_dir="./local_cache",
            )

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # 2. Setup Clients
        self.qdrant_client = qdrant_client.QdrantClient(**self.qdrant_config)

        # 3. Setup Stores
        self.vector_store_chunks = QdrantVectorStore(
            client=self.qdrant_client, collection_name=chunk_collection
        )
        self.vector_store_entities = QdrantVectorStore(
            client=self.qdrant_client, collection_name=entity_collection
        )
        self.graph_store = Neo4jPropertyGraphStore(
            username=neo4j_username, password=neo4j_password, url=neo4j_url
        )

        # 4. Initialize Workflow
        self.workflow = EduMatrixWorkflow(
            retriever=None,
            llm=self.llm,
            tavily_api_key=tavily_api_key,
        )

        try:
            self._refresh_retriever()
        except Exception:
            print("⚠️ Initial retriever load skipped. Run ingestion to populate DB.")

    def get_modules(self) -> Dict[str, Any]:
        return {
            "workflow": self.workflow,
            "llm": self.llm,
            "graph_store": self.graph_store,
            "vector_store_chunks": self.vector_store_chunks,
        }

    def _check_and_create_collection(
        self, name: str, vector_size: int = 1024, is_entity: bool = False
    ):
        if self.force_recreate and self.qdrant_client.collection_exists(name):
            self.qdrant_client.delete_collection(name)
        if not self.qdrant_client.collection_exists(name):
            hnsw = rest.HnswConfigDiff(m=16, ef_construct=64) if is_entity else None
            self.qdrant_client.create_collection(
                collection_name=name,
                vectors_config=rest.VectorParams(
                    size=vector_size, distance=rest.Distance.COSINE, on_disk=True
                ),
                hnsw_config=hnsw,
            )
            self.qdrant_client.create_payload_index(
                name, "page", rest.PayloadSchemaType.INTEGER
            )
            if is_entity:
                self.qdrant_client.create_payload_index(
                    name, "name", rest.PayloadSchemaType.KEYWORD
                )

    def _clear_neo4j(self):
        if not self.force_recreate:
            return
        try:
            with GraphDatabase.driver(
                self.neo4j_creds["url"],
                auth=(self.neo4j_creds["user"], self.neo4j_creds["password"]),
            ) as driver:
                with driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")
                    session.run(
                        "CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)"
                    )
                    session.run(
                    """
                    CREATE CONSTRAINT entity_norm_name IF NOT EXISTS
                    FOR (e:Entity)
                    REQUIRE e.normalized_name IS UNIQUE
                    """
                    )
        except Exception:
            pass

    async def run_ingestion(self, pdf_path: str):
        if not self.dashscope_api_key:
            raise ValueError(
                "❌ Ingestion Error: 'dashscope_api_key' is required for PDF parsing (VLM), "
                "even if you use a different LLM for retrieval."
            )

        print(f"🚀 Starting ingestion for {pdf_path}...")

        # 0. Prepare Paths & Environment
        safe_name = Path(pdf_path).stem.replace(" ", "_")
        cache_dir = self.data_dir / "parser_cache" / safe_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        sidecar_file = cache_dir / "page_heavy_data.json"

        test_embed = await self.embed_model.aget_text_embedding("hello")
        current_dim = len(test_embed)
        print(f"📏 Detected Embedding Dimension: {current_dim}")

        self._check_and_create_collection(self.collections["chunks"], vector_size=current_dim, is_entity=False)
        self._check_and_create_collection(self.collections["entities"], vector_size=current_dim, is_entity=True)
        self._clear_neo4j()

        # 1. Parse PDF
        os.environ["DASHSCOPE_API_KEY"] = self.dashscope_api_key
        reader = EduMatrixPDFReader(
            pdf_path=pdf_path,
            image_output_dir=str(cache_dir / "images"),
            cache_file=str(cache_dir / "vlm_cache.json"),
            hash_record_file=str(cache_dir / "processed_hashes.json"),
            sidecar_file=str(sidecar_file),
            embedding_cache_file=str(cache_dir / "embedding_cache.json"),
            alias_map_file=str(cache_dir / "global_alias_map.json"),
        )
        documents = await reader.parse()

        print("📝 Generating Global Summary...")

        # Limit context to avoid token overflow (30k chars is approx 8k-10k tokens)
        full_text = "\n".join([d.text for d in documents])[:30000]

        summary_prompt = (
            "You are an expert technical researcher. "
            "Please generate a comprehensive summary of the following document content. "
            "The summary must cover the core technologies, main arguments, and key conclusions.\n"
            "**IMPORTANT**: Detect the dominant language of the provided text and generate the summary **IN THAT SAME LANGUAGE**.\n\n"
            f"--- DOCUMENT CONTENT ---\n{full_text}"
        )
        summary_res = await asyncio.wait_for(self.llm.acomplete(summary_prompt), timeout=30)
        global_summary = summary_res.text

        summary_node = TextNode(
            text=f"【Global Document Summary / 全文摘要】\n{global_summary}",
            metadata={
                "file_name": safe_name,
                "is_global_summary": "true",
                "page": "all",
                "section_title": "summary",
            },
        )

        print("⚡ Generating embedding for summary node...")
        summary_embedding = await self.embed_model.aget_text_embedding(
            summary_node.get_content()
        )
        summary_node.embedding = summary_embedding

        self.vector_store_chunks.add([summary_node])
        print(f"✅ Global Summary stored: {global_summary[:50]}...")
        
        print(f"🔄 准备从 Sidecar 加载实体: {sidecar_file}")
        all_entities = []
        
        if sidecar_file.exists():
            # 1. 直接读取，不再静默失败
            with open(sidecar_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    print(f"📄 Sidecar JSON 读取成功，共包含 {len(data)} 页")
                except json.JSONDecodeError as e:
                    print(f"❌ Sidecar JSON 格式错误: {e}")
                    data = {}

            # 2. 遍历提取，打印每页的实体数量
            for page_key, page_val in data.items():
                # 兼容可能的结构差异
                graph_data = page_val.get("graph_data")
                if not graph_data:
                    print(f"⚠️  Page {page_key}: 缺少 'graph_data' 字段，跳过")
                    continue
                
                ents = graph_data.get("entities", [])
                if ents:
                    print(f"   - Page {page_key}: 提取到 {len(ents)} 个实体")
                    all_entities.extend(ents)
                else:
                    print(f"   - Page {page_key}: 'entities' 列表为空")
        else:
            print(f"❌ 严重错误: Sidecar 文件未找到! 路径: {sidecar_file}")

        print(f"📊 实体汇总: 总计 {len(all_entities)} 个实体准备入库")

        if all_entities:
            try:
                print("🚀 开始执行 upsert_entities...")
                upsert_stats = upsert_entities(
                    self.qdrant_client,
                    self.collections["entities"],
                    self.neo4j_creds["url"],
                    self.neo4j_creds["user"],
                    self.neo4j_creds["password"],
                    all_entities,
                    embed_text_wrapper,
                )
                print(f"✅ 实体入库成功! 统计: {upsert_stats}")
            except Exception as e:
                print(f"❌ 入库过程发生异常: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️ 警告: 实体列表为空，跳过数据库写入步骤。")

        # 3. Chunking & Vector Indexing
        print("💾 Indexing Chunks...")
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 256])
        nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(nodes)
        parent_nodes = [
            n for n in nodes if n.node_id not in {x.node_id for x in leaf_nodes}
        ]

        for n in parent_nodes:
            n.metadata["chunk_type"] = "parent"
        for n in leaf_nodes:
            n.metadata["chunk_type"] = "leaf"

        print("💉 Injecting Sidecar Metadata (BBox & Images)...")
        metadata_extractor = MetadataGraphExtractor(sidecar_path=str(sidecar_file))

        all_nodes = leaf_nodes + parent_nodes
        all_nodes = metadata_extractor(all_nodes)

        storage_context_chunks = StorageContext.from_defaults(
            vector_store=self.vector_store_chunks
        )
        VectorStoreIndex(
            all_nodes,
            storage_context=storage_context_chunks,
            embed_model=self.embed_model,
            show_progress=True,
        )

        # 4. Build Property Graph Index (Sidecar + LLM)
        print("🕸️ Building Knowledge Graph Index (Dual Extraction)...")

        class Relations(str, Enum):
            CONTAINS = "CONTAINS"
            BELONGS_TO = "BELONGS_TO"
            IMPLEMENTS = "IMPLEMENTS"
            SOLVES = "SOLVES"
            CAUSES = "CAUSES"
            DATA_FLOWS_TO = "DATA_FLOWS_TO"
            CONTROLS = "CONTROLS"
            CALCULATES = "CALCULATES"
            OPTIMIZES = "OPTIMIZES"
            PART_OF = "PART_OF"
            RELATED_TO = "RELATED_TO"

        llm_extractor = SchemaLLMPathExtractor(
            llm=self.llm,
            possible_relations=Relations,
            strict=False,
            num_workers=4,
        )

        # Combine both extractors
        kg_extractors: List[TransformComponent] = [metadata_extractor, llm_extractor]

        PropertyGraphIndex(
            nodes=parent_nodes,
            kg_extractors=kg_extractors,
            llm=self.llm,
            embed_model=self.embed_model,
            property_graph_store=self.graph_store,
            vector_store=self.vector_store_entities,
            embed_kg_nodes=True,
            show_progress=True,
        )

        print("🎉 Ingestion Pipeline Completed!")

    def _refresh_retriever(self):
        """Re-initializes the Hybrid Retriever (Vector + Graph)"""
        print("🔄 Refreshing Hybrid Retriever...")

        chunk_index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store_chunks, embed_model=self.embed_model
        )
        vector_retriever = chunk_index.as_retriever(similarity_top_k=8)

        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store,
            vector_store=self.vector_store_entities,
            llm=self.llm,
            embed_model=self.embed_model,
        )

        sub_retriever = VectorContextRetriever(
            graph_store=self.graph_store,
            embed_model=self.embed_model,
            vector_store=self.vector_store_entities,
            similarity_top_k=8,
            path_depth=2,
        )

        graph_retriever = graph_index.as_retriever(sub_retrievers=[sub_retriever])

        self.workflow.retriever = HybridRetriever(
            vector_retriever=vector_retriever, 
            graph_retriever=graph_retriever,
            chunk_store=self.vector_store_chunks,
        )
        print("✅ Hybrid Retriever (Vector + Graph) is active!")

    async def run(self, query: str, **kwargs: Any) -> Any:
        if not self.workflow.retriever:
            self._refresh_retriever()
        return await self.workflow.run(question=query, **kwargs)