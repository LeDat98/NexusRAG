"""
JobNexus — Job Recommendation Engine powered by NexusRAG pipeline.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging

from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine, Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting JobNexus API...")
    import os
    auto_create = os.environ.get("AUTO_CREATE_TABLES", "true").lower() == "true"
    if auto_create:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Auto-migrate: add new columns if missing
            await conn.execute(
                text("ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS system_prompt TEXT")
            )
            # Ensure chat_messages table + indexes exist (idempotent)
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    workspace_id INTEGER NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                    message_id VARCHAR(50) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    sources JSON,
                    related_entities JSON,
                    image_refs JSON,
                    thinking TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_chat_messages_workspace_id ON chat_messages(workspace_id)"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_chat_messages_message_id ON chat_messages(message_id)"
            ))
            await conn.execute(text(
                "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS ratings JSON"
            ))
            await conn.execute(text(
                "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS agent_steps JSON"
            ))
        logger.info("Database tables created/verified")
    else:
        logger.info("AUTO_CREATE_TABLES=false — skipping auto-migration")
    yield
    logger.info("Shutting down...")
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    description="JobNexus — Job Recommendation Engine with semantic matching, knowledge graph, and LLM chat",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    redirect_slashes=False,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    return {"status": "ready"}


# API routes
from app.api.router import api_router  # noqa: E402

app.include_router(api_router, prefix="/api/v1")

# Static files — document images extracted by NexusRAG (Docling)
_docling_data = Path(__file__).resolve().parent.parent / "data" / "docling"
_docling_data.mkdir(parents=True, exist_ok=True)
app.mount("/static/doc-images", StaticFiles(directory=str(_docling_data)), name="static_doc_images")

# Import models so SQLAlchemy registers them
from app.models import knowledge_base, document, chat_message  # noqa: E402, F401
from app.models import candidate, job, match  # noqa: E402, F401
