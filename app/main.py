import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import hashlib
import traceback
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

# LlamaIndex Core
from llama_index.core import Settings, VectorStoreIndex, PropertyGraphIndex
from llama_index.core.retrievers import BaseRetriever, VectorContextRetriever
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore

# Integrations
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
import qdrant_client
from neo4j import GraphDatabase

# Import your Agent Workflow
from .agent_workflow import create_graph_app

# 1. Environment Setup
load_dotenv()

# Configuration
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "edu_matrix_chunks" # Ensure this matches ingestion
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Global Engine Container
rag_engine = {}

# ==========================================
# 🔧 Robust Hybrid Retriever
# ==========================================
class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever with Content-Based Deduplication.
    Combines Vector Search (Dense) + Graph Search (Relationship-based).
    """
    def __init__(self, vector_retriever, graph_retriever):
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        super().__init__()
    
    def _retrieve(self, query_bundle) -> list[NodeWithScore]:
        # 1. Parallel Retrieval
        nodes_vect = self.vector_retriever.retrieve(query_bundle)
        nodes_graph = self.graph_retriever.retrieve(query_bundle)
        
        # 2. Robust Deduplication (Content Hash)
        # Why? node_id can be unstable across different retrievers.
        # Content hash ensures we don't show the same text twice.
        combined = []
        seen_hashes = set()
        
        # Merge lists (Vector first usually gives better semantic matches)
        for n in (nodes_vect + nodes_graph):
            # Safe access to node content
            content = n.node.get_content()
            if not content: continue
            
            # Create a hash of the first 200 chars (sufficient for uniqueness)
            content_hash = hashlib.md5(content[:200].encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                combined.append(n)
                seen_hashes.add(content_hash)
                
        return combined

# ==========================================
# 🚀 Lifecycle Management
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [Startup] Initializing EduMatrix Engine...")
    
    try:
        # 1. Initialize Models
        print("🧠 Loading Models (BGE-M3 + Qwen)...")
        # [Optimization] Use 'cuda' if available, else 'cpu'
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            trust_remote_code=True,
            # local_files_only=True, # Uncomment if models are pre-downloaded
            device="cpu", # Change to "cuda" for GPU
        )
        Settings.embed_model = embed_model

        llm = DashScope(
            model_name=os.getenv("DASHSCOPE_MODEL_NAME", "qwen-plus"),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.1,
        )
        rag_engine["llm"] = llm

        # 2. Connect Qdrant (Chunks)
        print(f"🔌 Connecting Qdrant ({QDRANT_COLLECTION})...")
        qdrant_client_obj = qdrant_client.QdrantClient(url=QDRANT_URL)
        vector_store = QdrantVectorStore(
            collection_name=QDRANT_COLLECTION,
            client=qdrant_client_obj,
        )
        vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # 3. Connect Neo4j (Graph)
        print("🔌 Connecting Neo4j...")
        graph_store = Neo4jPropertyGraphStore(
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            url=NEO4J_URL,
        )
        # Note: We use from_existing because ingestion is done separately
        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            llm=llm,
            embed_model=embed_model # Ensure consistency
        )

        rag_engine["graph_store"] = graph_store

        # 4. Build Retrievers
        # A. Vector Retriever (Dense)
        vector_tool = vector_index.as_retriever(similarity_top_k=8)
        
        # B. Graph Retriever (Contextual)
        # [Fix] Explicitly pass embed_model to avoid default fallback issues
        sub_retriever = VectorContextRetriever(
            graph_store=graph_store,
            embed_model=embed_model, 
            similarity_top_k=8,
            path_depth=2 # Fetch 2-hop neighbors (CPU -> ALU -> ControlUnit)
        )
        graph_tool = graph_index.as_retriever(
            sub_retrievers=[sub_retriever]
        )

        # C. Hybrid Retriever
        hybrid_retriever = HybridRetriever(vector_tool, graph_tool)

        # 5. Build LangGraph Agent
        print("🤖 Building LangGraph Workflow...")
        graph_app = create_graph_app(hybrid_retriever, llm)

        rag_engine["graph_app"] = graph_app
        print("✅ Engine Initialized Successfully!")

    except Exception as e:
        print(f"❌ Startup Failed: {traceback.format_exc()}")
        # We don't raise here to allow the server to start (and return 500s), 
        # but in production, you might want to raise.
        
    yield 
    print("👋 [Shutdown] EduMatrix Engine stopped.")

# ==========================================
# 📡 API Endpoints
# ==========================================

app = FastAPI(title="EduMatrix API", lifespan=lifespan)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not rag_engine.get("graph_app"):
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
    
    # 1. Parse Input
    last_message = request.messages[-1].content
    print(f"📩 Query: {last_message}")

    # 2. Prepare History
    chat_history = [
        ChatMessage(role=m.role, content=m.content)
        for m in request.messages[:-1]
    ]

    inputs = {
        "question": last_message,
        "original_question": last_message,
        "chat_history": chat_history,
        "retrieved_nodes": [],
        "grade_status": "",
        "retry_count": 0,
        "final_response": ""
    }

    # 3. Execute Graph
    try:
        result = await rag_engine["graph_app"].ainvoke(inputs)
        streaming_response = result["final_response"]
    except Exception as e:
        print(f"❌ Graph Execution Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 4. Stream Response
    async def response_generator():
        # A. AI Content
        if isinstance(streaming_response, str):
            yield streaming_response
        elif hasattr(streaming_response, "async_response_gen"):
            async for token in streaming_response.async_response_gen():
                yield token.delta
        else:
            # [Fix] Safer iteration for unknown stream types
            try:
                async for token in streaming_response:
                    # Robust attribute access
                    text = getattr(token, "delta", None) or getattr(token, "text", "")
                    yield text
            except Exception as e:
                yield f" [Error reading stream: {e}]"

        # B. Citations / Sources
        nodes = result.get("retrieved_nodes", [])
        if nodes:
            yield "\n\n---\n**🧠 Thinking Process:**\n"
            yield f"- Retrieved {len(nodes)} fragments (Vector + Graph)\n"
            yield "\n**📚 Sources:**\n"
            
            seen_texts = set()
            for n in nodes:
                # [Fix] Access metadata via n.node.metadata
                meta = n.node.metadata or {}
                page = meta.get("page", "?")
                # Fallback to 50 chars preview if file_name missing
                source_name = meta.get("file_name", "Textbook")
                
                # Robust content access
                content_preview = n.node.get_content()[:50].replace('\n', ' ')
                
                if content_preview not in seen_texts:
                    yield f"> **[{source_name} P{page}]**: {content_preview}...\n"
                    seen_texts.add(content_preview)
            
    return StreamingResponse(response_generator(), media_type="text/plain")

@app.post("/api/graph")
async def get_graph(request: ChatRequest):
    """
    Fetch subgraph for visualization based on user query.
    """
    result_data = {"links": []}
    
    try:
        if not request.messages:
            return result_data
            
        user_query = request.messages[-1].content
        print(f"🕸️ [Graph API] Query: {user_query}")

        # 1. Extract Keywords (Simple LLM Call)
        # Using a direct prompt to get cleaner keywords
        prompt = (
            f"Extract 1-3 core technical entities from this query: '{user_query}'. "
            "Output ONLY the keywords separated by commas. No other text."
        )
        response = await rag_engine["llm"].acomplete(prompt)
        keywords = [k.strip() for k in response.text.split(',') if k.strip()]
        
        if not keywords: 
            keywords = [user_query]
        
        print(f"   Keywords: {keywords}")

        # 2. Cypher Query
        # [Fix] Use 'name' instead of 'id' because ingestion uses 'name'.
        # [Fix] Use 'CONTAINS' and 'toLower' for fuzzy matching.
        cypher_sql = """
        MATCH (n)-[r]->(m)
        WHERE (
            ANY(k IN $keywords WHERE toLower(n.name) CONTAINS toLower(k)) 
            OR 
            ANY(k IN $keywords WHERE toLower(m.name) CONTAINS toLower(k))
        )
        RETURN n.name AS source, type(r) AS label, m.name AS target
        LIMIT 30
        """
        
        # 3. Execute
        driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run(cypher_sql, keywords=keywords)
            records = [record.data() for record in result]
            result_data["links"] = records
            print(f"   Found {len(records)} relations.")
            
        driver.close()

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Graph API Error: {e}")

    return result_data

@app.get("/")
def read_root():
    return {"status": "active", "model": "EduMatrix V2"}