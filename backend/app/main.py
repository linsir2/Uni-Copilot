import os
import time
import traceback
import hashlib
import json
import gzip
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from neo4j import GraphDatabase
from redis import asyncio as aioredis
from redis.asyncio import Redis
from litellm import aembedding
from langfuse import observe, get_client
from core.edu_parser.base import MultimodalAgenticRAGPack
from worker import celery_app

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
if not DATA_DIR.exists(): 
    os.makedirs(DATA_DIR, exist_ok=True)

print(f"📂 [Config] DATA_DIR set to: {DATA_DIR}")

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

rag_pack: MultimodalAgenticRAGPack | None = None
redis_client: Redis | None = None
neo4j_driver = None  # Kept separate for the raw graph visualization endpoint
qdrant_client = None

SEMANTIC_COLLECTION = "graph_query_cache"

@observe(name="Semantic_Cache_Embedding", as_type="span")
async def get_query_embedding(text: str, trace_id: str):
    try:
        resp = await aembedding(
            model="text-embedding-3-large", # 确保与 config.yaml 对应
            api_base="http://localhost:4000",
            api_key="sk-anything",
            input=[text],
            encoding_format="float",
            # 🔥 注入 Langfuse 追踪
            metadata={
                "trace_id": trace_id,
                "generation_name": "LiteLLM_Cache_Embedding"
            }
        )
        return resp.data[0]["embedding"]
    except Exception as e:
        print(f"⚠️ Embedding Error: {e}")
    return []

async def send_celery_task(file_path):
    loop = asyncio.get_running_loop()
    # 把同步 send_task 调用放到线程池，不阻塞 event loop
    task = await loop.run_in_executor(
        None,
        lambda: celery_app.send_task("parse_pdf_task", args=[str(file_path)])
    )
    return task

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [Startup] Initializing Retrieval RAG Pack...")
    global redis_client, neo4j_driver, rag_pack, qdrant_client

    try:
        rag_pack = MultimodalAgenticRAGPack(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            neo4j_url=NEO4J_URL,
            neo4j_username=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            data_dir=str(DATA_DIR),
        )

        redis_client = await aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

        neo4j_driver = GraphDatabase.driver(
            NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_client = QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=False)

        if not qdrant_client.collection_exists(SEMANTIC_COLLECTION):
            print(f"📦 Creating Semantic Cache Collection: {SEMANTIC_COLLECTION}")
            test_vec = await get_query_embedding("test", trace_id="startup_test")
            cache_dim = len(test_vec) if test_vec else 1024
            print(f"📏 Initializing Semantic Cache with Dimension: {cache_dim}")

            qdrant_client.create_collection(
                collection_name=SEMANTIC_COLLECTION,
                vectors_config=rest.VectorParams(
                    size=cache_dim, 
                    distance=rest.Distance.COSINE
                )
            )

        print("✅ Retrieval Engine Ready!")

    except Exception as e:
        print(f"❌ Startup Error: {e}")
        traceback.print_exc()

    yield

    if neo4j_driver:
        neo4j_driver.close()
    if redis_client:
        await redis_client.aclose()
    if qdrant_client:
        qdrant_client.close()

app = FastAPI(title="EduMatrix API", lifespan=lifespan)

# Static Files Setup
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

class ChatRequest(BaseModel):
    messages: list[dict]

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    上传 PDF -> Celery Ingestion
    """
    print("ROUTE HIT", flush=True)

    # 1. 确保数据目录存在
    upload_dir = DATA_DIR / "uploads"
    if not upload_dir.exists():
        upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / (file.filename or "")

    hasher = hashlib.md5()

    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
                buffer.write(chunk)
        print("保存成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")
    
    file_hash = hasher.hexdigest()
    cache_key = f"pdf:task:{file_hash}"

    # if redis_client:
    #     existing_task_id = await redis_client.get(cache_key)
    #     if existing_task_id:
    #         return {
    #             "status": "cached",
    #             "task_id": existing_task_id,
    #             "message": f"⚡ 秒传成功！{file.filename} 之前已上传过。",
    #         }

    # 3. 发送给Celery
    try:
        task = await send_celery_task(file_path)
    except Exception as e:
        print("❌ send_task failed:", e)

    if redis_client:
        await redis_client.set(
            cache_key,
            task.id,
            ex=24 * 3600,
        )
    return {
        "status": "queued",
        "task_id": task.id,
        "message": f"{file.filename} 已加入解析队列"
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    last_msg = request.messages[-1]["content"]
    
    # Run the Pack (it handles retrieval -> grading -> generation internally)
    try:
        result = await rag_pack.run(query=last_msg)
        streaming_response = result["final_response"]
        retrieved_nodes = result["retrieved_nodes"]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    async def response_generator():
        try:
            # 1. Stream the LLM response
            iterator = streaming_response
            if hasattr(streaming_response, "async_response_gen"):
                iterator = streaming_response.async_response_gen()

            async for token in iterator:
                text = getattr(token, "delta", None) or getattr(token, "text", None)
                if text:
                    yield text

            # 2. Append Citations & Images (Frontend logic)
            if retrieved_nodes:
                yield "\n\n---\n**🧠 Thinking Process:**\n"
                seen_texts = set()
                seen_images = set()
            
                for n in retrieved_nodes:
                    meta = n.get("metadata", {})
                    content = n.get("text", "")
                    
                    # Text Citation
                    preview = content[:50].replace('\n', ' ')
                    if preview not in seen_texts:
                        yield f"> **[{meta.get('file_name','Doc')} P{meta.get('page_label', meta.get('page','?'))}]**: {preview}...\n"
                        seen_texts.add(preview)

                    # Image Citation
                    fname = meta.get("file_name", "")
                    if fname not in ["Web", "Textbook"]:
                        try:
                            safe_name = Path(fname).stem.replace(" ", "_")
                            page_idx = meta.get("page_label", meta.get("page", "?"))
                        
                            # Check local storage for images associated with this page
                            img_dir = DATA_DIR / "parser_cache" / safe_name / "images"
                            prefix = f"p{page_idx}_"

                            print(f"🔍 [ImgDebug] 寻找图片: {prefix} in {img_dir}")
                            if img_dir.exists():
                                for f in os.listdir(img_dir):
                                    if f.startswith(prefix) and f.lower().endswith(('.jpg', '.png', '.jpeg')):
                                        img_url = f"{API_BASE_URL}/static/parser_cache/{safe_name}/images/{f}"
                                        if img_url not in seen_images:
                                            print(f"✅ [ImgDebug] 找到图片: {img_url}")
                                            yield f"\n![Page {page_idx}]({img_url})\n"
                                            seen_images.add(img_url)
                            else:
                                 print(f"❌ [ImgDebug] 目录不存在: {img_dir}")

                        except Exception as e:
                            print(f"⚠️ Image Error: {e}")
                            traceback.print_exc()
        except Exception as e:
            yield "\n\n⚠️ 出现错误，请稍后再试"
            traceback.print_exc()

    return StreamingResponse(response_generator(), media_type="text/plain")

@app.post("/api/graph")
@observe(name="Graph_Visualization_Endpoint")
async def get_graph(request: ChatRequest):
    """
    Extracts keywords using the Pack's LLM and queries Neo4j for visualization.
    """
    result_data = {"links": []}
    if not request.messages: return result_data
    
    query = request.messages[-1]["content"]

    client = get_client()
    trace_id = client.get_current_trace_id()
    client.update_current_trace(session_id="web_ui")

    query_hash = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()
    CACHE_VERSION = "v1"
    cache_key = f"graph:{CACHE_VERSION}:{query_hash}"

    try:
        query_vec = await get_query_embedding(query, trace_id=trace_id)

        if query_vec and qdrant_client:
            version_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="cache_version",
                        match=rest.MatchValue(value=CACHE_VERSION)
                    ),
                    rest.FieldCondition(
                        key="model",
                        match=rest.MatchValue(value="dashscope-v2")
                    )
                ]
            )

            search_res = qdrant_client.query_points(
                collection_name=SEMANTIC_COLLECTION,
                query=query_vec,
                limit=1,
                score_threshold=0.86,
                query_filter=version_filter,
            )

            if search_res and hasattr(search_res, "points") and len(search_res.points) > 0:
                hit = search_res.points[0]
                payload_data = hit.payload or {}
                cached_graph = payload_data.get("graph_data")
                original_q = payload_data.get("query")
                
                if cached_graph:
                    print(f"⚡ [Semantic Hit] '{query}' ≈ '{original_q}' (sim: {hit.score:.4f})")
                    if redis_client:
                        query_hash = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()
                        cache_key = f"graph:v1:{query_hash}"
                        # 压缩并写回 Redis
                        compressed = gzip.compress(json.dumps(cached_graph).encode("utf-8"))
                        await redis_client.set(cache_key, compressed, ex=3600)
                    return cached_graph
            else:
                print(f"ℹ️ [Semantic Cache] No similar query found above 0.86 threshold.")
    except Exception as e:
        print(f"⚠️ Semantic Cache Check Failed: {e}")
  
    if redis_client:
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            try:
                decompressed = gzip.decompress(cached_data).decode("utf-8")
                print(f"⚡ [Cache Hit] Graph served from Redis for: {query}")
                return json.loads(decompressed)
            except Exception:
                pass
        
    print(f"🐢 [Cache Miss] Generating Graph for: {query}")

    try:
        # Use the LLM instance directly from the Pack
        prompt = f"Extract 1-3 technical entities from '{query}', comma separated. No intro/outro."
        res = await rag_pack.llm.acomplete(
            prompt,
        )
        keywords = [k.strip() for k in res.text.split(',') if k.strip()] or [query]

        cypher = """
        MATCH (n)-[r]->(m)
        WHERE ANY(k IN $kw WHERE toLower(COALESCE(n.name, n.text, '')) CONTAINS toLower(k))
           OR ANY(k IN $kw WHERE toLower(COALESCE(m.name, m.text, '')) CONTAINS toLower(k))

        RETURN 
            COALESCE(
                n.name, 
                CASE WHEN n.file_name IS NOT NULL THEN n.file_name + ' (P' + toString(n.page_label) + ')' END, 
                'Unknown Node'
            ) AS source,
    
            type(r) AS label,
    
            COALESCE(
                m.name, 
                CASE WHEN m.file_name IS NOT NULL THEN m.file_name + ' (P' + toString(m.page_label) + ')' END, 
                'Unknown Node'
            ) AS target
        LIMIT 30
        """
        
        if neo4j_driver:
            with neo4j_driver.session() as session:
                result_data["links"] = session.execute_read(
                    lambda tx: [r.data() for r in tx.run(cypher, kw=keywords)]
                )
        
        if result_data["links"]:
            # 1. 存入 Redis (短期精确缓存)
            if redis_client:
                compressed = gzip.compress(json.dumps(result_data).encode("utf-8"))
                await redis_client.set(cache_key, compressed, ex=3600)
            
            semantic_id = hashlib.md5(
                (query.strip().lower() + "|dashscope-v2" + CACHE_VERSION).encode("utf-8")
            ).hexdigest()

            # 2. 存入 Qdrant (长期语义缓存)
            if query_vec and qdrant_client:
                qdrant_client.upsert(
                    collection_name=SEMANTIC_COLLECTION,
                    points=[
                        rest.PointStruct(
                            id=semantic_id,
                            vector=query_vec,
                            payload={
                                "query": query,
                                "graph_data": result_data,
                                "created_at": time.time(),
                                "model": "dashscope-v2",
                                "cache_version": CACHE_VERSION,
                            }
                        )
                    ]
                )
                
    except Exception as e:
        print(f"Graph Error: {e}")
        
    return result_data