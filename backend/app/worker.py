import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
from celery import Celery
from celery.signals import worker_process_init
from core.edu_parser.base import MultimodalAgenticRAGPack
from dotenv import load_dotenv
from pathlib import Path
import asyncio


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
if not DATA_DIR.exists(): 
    os.makedirs(DATA_DIR, exist_ok=True)

celery_app = Celery(
    "edumatrix_worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    )

celery_app.conf.update(
    task_serializer="json",
    accept_content=['json'],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
)

rag_pack: MultimodalAgenticRAGPack | None = None

def get_rag_pack():
    global rag_pack
    if rag_pack is None:
        print("🧠 Initializing RAG-PACK inside task...")
        rag_pack = MultimodalAgenticRAGPack(
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            data_dir=str(DATA_DIR),
            force_recreate=True,
        )
        print("✅ RAG-PACK ready")
    return rag_pack

@worker_process_init.connect
def init_worker(**kwargs):
    global rag_pack
    rag_pack = get_rag_pack() 


@celery_app.task(bind=True, name="parse_pdf_task")
def parse_pdf(self, pdf_path):
    try:
        print("🔥 INGEST TASK STARTED:", pdf_path)

        asyncio.run(rag_pack.run_ingestion(pdf_path))

        print("✅ INGEST TASK FINISHED:", pdf_path)
        return {"status": "ok", "pdf_path": pdf_path}

    except Exception as e:
        print("❌ INGEST FAILED:", e)
        raise self.retry(exc=e, countdown=5)
