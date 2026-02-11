import os
import hashlib
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from llama_index.core import Settings, VectorStoreIndex, PropertyGraphIndex
from llama_index.core.retrievers import BaseRetriever, VectorContextRetriever
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore

from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
import qdrant_client
from neo4j import GraphDatabase

from .agent_workflow import create_graph_app

load_dotenv()

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "edu_matrix_chunks"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

rag_engine = {}

# [优化 2] 排序增强的 Hybrid Retriever
class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, graph_retriever):
        self.vector = vector_retriever
        self.graph = graph_retriever
        super().__init__()
    
    def _retrieve(self, query_bundle) -> list[NodeWithScore]:
        vec_nodes = self.vector.retrieve(query_bundle)
        graph_nodes = self.graph.retrieve(query_bundle)
        
        combined = []
        seen_hashes = set()
        
        # 合并策略：优先向量（通常分高），再补图谱
        for n in (vec_nodes + graph_nodes):
            content = n.node.get_content()
            if not content: continue
            norm_text = content[:200].strip().lower()
            h = hashlib.md5(norm_text.encode('utf-8')).hexdigest()
            
            if h not in seen_hashes:
                combined.append(n)
                seen_hashes.add(h)
        
        # [关键] 按分数降序排列，确保高质量内容在前
        combined.sort(key=lambda x: x.score or 0.0, reverse=True)
        return combined

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [Startup] Initializing Engine...")
    try:
        # Models
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", trust_remote_code=True, device="cpu")
        Settings.embed_model = embed_model
        llm = DashScope(model_name="qwen-plus", api_key=os.getenv("DASHSCOPE_API_KEY"))
        rag_engine["llm"] = llm

        # Stores
        q_client = qdrant_client.QdrantClient(url=QDRANT_URL)
        v_store = QdrantVectorStore(client=q_client, collection_name=QDRANT_COLLECTION)
        v_index = VectorStoreIndex.from_vector_store(vector_store=v_store)
        
        g_store = Neo4jPropertyGraphStore(username=NEO4J_USER, password=NEO4J_PASSWORD, url=NEO4J_URL)
        g_index = PropertyGraphIndex.from_existing(property_graph_store=g_store, llm=llm, embed_model=embed_model)

        # Retrievers
        v_tool = v_index.as_retriever(similarity_top_k=8)
        sub_retriever = VectorContextRetriever(graph_store=g_store, embed_model=embed_model, similarity_top_k=8, path_depth=2)
        g_tool = g_index.as_retriever(sub_retrievers=[sub_retriever])
        
        hybrid_retriever = HybridRetriever(v_tool, g_tool)

        # Workflow
        workflow = create_graph_app(hybrid_retriever, llm)
        rag_engine["graph_app"] = workflow
        
        rag_engine["neo4j_driver"] = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("✅ Engine Ready!")
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        traceback.print_exc()

    yield

    print("👋 [Shutdown] Cleaning up...")
    if "graph_app" in rag_engine:
        await rag_engine["graph_app"].aclose()
    if "neo4j_driver" in rag_engine:
        rag_engine["neo4j_driver"].close()

app = FastAPI(title="EduMatrix API", lifespan=lifespan)

# Static Files
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
if not DATA_DIR.exists(): os.makedirs(DATA_DIR, exist_ok=True)

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

class ChatRequest(BaseModel):
    messages: list[dict]

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if "graph_app" not in rag_engine:
        raise HTTPException(status_code=503, detail="Service not ready")

    last_msg = request.messages[-1]["content"]
    history = [ChatMessage(role=m["role"], content=m["content"]) for m in request.messages[:-1]]

    try:
        result = await rag_engine["graph_app"].run(question=last_msg, chat_history=history)
        streaming_response = result["final_response"]
        retrieved_nodes = result["retrieved_nodes"]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    async def response_generator():
        iterator = streaming_response
        if hasattr(streaming_response, "async_response_gen"):
            iterator = streaming_response.async_response_gen()
        
        async for token in iterator:
            text = getattr(token, "delta", None) or getattr(token, "text", None) or str(token)
            yield text

        if retrieved_nodes:
            yield "\n\n---\n**🧠 Thinking Process:**\n"
            seen_texts = set()
            seen_images = set()
            
            for n in retrieved_nodes:
                # 兼容 Dict
                if isinstance(n, dict):
                    meta = n.get("metadata", {})
                    content = n.get("text", "")
                else:
                    meta = n.node.metadata
                    content = n.node.get_content()

                preview = content[:50].replace('\n', ' ')
                if preview not in seen_texts:
                    yield f"> **[{meta.get('file_name','Doc')} P{meta.get('page','?')}]**: {preview}...\n"
                    seen_texts.add(preview)

                fname = meta.get("file_name", "")
                if fname not in ["Web", "Textbook"]:
                    try:
                        safe_name = Path(fname).stem.replace(" ", "_")
                        page = meta.get("page", "?")
                        img_dir = DATA_DIR / "parser_cache" / safe_name / "images"
                        if img_dir.exists():
                            prefix = f"p{page}_"
                            for f in os.listdir(img_dir):
                                if f.startswith(prefix) and f.lower().endswith(('.jpg', '.png')):
                                    img_url = f"{API_BASE_URL}/static/parser_cache/{safe_name}/images/{f}"
                                    if img_url not in seen_images:
                                        yield f"\n![Page {page}]({img_url})\n"
                                        seen_images.add(img_url)
                    except Exception:
                        pass

    return StreamingResponse(response_generator(), media_type="text/plain")

@app.post("/api/graph")
async def get_graph(request: ChatRequest):
    result_data = {"links": []}
    if not request.messages: return result_data
    
    query = request.messages[-1]["content"]
    try:
        prompt = f"Extract 1-3 technical entities from '{query}', comma separated."
        res = await rag_engine["llm"].acomplete(prompt)
        keywords = [k.strip() for k in res.text.split(',') if k.strip()] or [query]

        cypher = """
        MATCH (n)-[r]->(m)
        WHERE (ANY(k IN $kw WHERE toLower(n.name) CONTAINS toLower(k)) 
            OR ANY(k IN $kw WHERE toLower(m.name) CONTAINS toLower(k)))
        RETURN n.name as source, type(r) as label, m.name as target LIMIT 30
        """
        
        driver = rag_engine.get("neo4j_driver")
        if driver:
            with driver.session() as session:
                res = session.run(cypher, kw=keywords)
                result_data["links"] = [r.data() for r in res]
                
    except Exception as e:
        print(f"Graph Error: {e}")
        
    return result_data