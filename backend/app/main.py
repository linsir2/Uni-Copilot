import os
import traceback
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from neo4j import GraphDatabase

from core.edu_parser.base import MultimodalAgenticRAGPack
from worker import parse_pdf

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
if not DATA_DIR.exists(): 
    os.makedirs(DATA_DIR, exist_ok=True)

print(f"📂 [Config] DATA_DIR set to: {DATA_DIR}")

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

rag_pack: MultimodalAgenticRAGPack
neo4j_driver = None  # Kept separate for the raw graph visualization endpoint

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [Startup] Initializing Retrieval RAG Pack...")
    global neo4j_driver, rag_pack

    try:
        rag_pack = MultimodalAgenticRAGPack(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            neo4j_url=NEO4J_URL,
            neo4j_username=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            data_dir=str(DATA_DIR),
        )

        neo4j_driver = GraphDatabase.driver(
            NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

        print("✅ Retrieval Engine Ready!")

    except Exception as e:
        print(f"❌ Startup Error: {e}")
        traceback.print_exc()

    yield

    if neo4j_driver:
        neo4j_driver.close()

app = FastAPI(title="EduMatrix API", lifespan=lifespan)

# Static Files Setup
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")

class ChatRequest(BaseModel):
    messages: list[dict]

@app.post("/api/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    上传 PDF -> Celery Ingestion
    """
    # 1. 确保数据目录存在
    upload_dir = DATA_DIR / "uploads"
    if not upload_dir.exists():
        upload_dir.mkdir(parents=True, exist_ok=True)

    # 2. 保存文件到本地
    file_path = upload_dir / (file.filename or "")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    # 3. 发送给Celery
    task = parse_pdf.delay(str(file_path))

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

    return StreamingResponse(response_generator(), media_type="text/plain")

@app.post("/api/graph")
async def get_graph(request: ChatRequest):
    """
    Extracts keywords using the Pack's LLM and queries Neo4j for visualization.
    """
    result_data = {"links": []}
    if not request.messages: return result_data
    
    query = request.messages[-1]["content"]
    try:
        # Use the LLM instance directly from the Pack
        prompt = f"Extract 1-3 technical entities from '{query}', comma separated. No intro/outro."
        res = await rag_pack.llm.acomplete(prompt)
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
                res = session.run(cypher, kw=keywords)
                result_data["links"] = [r.data() for r in res]
                
    except Exception as e:
        print(f"Graph Error: {e}")
        
    return result_data