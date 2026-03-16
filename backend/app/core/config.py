from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

# Find .env file - check project root first, fallback for Docker
_candidate = Path(__file__).resolve().parent.parent.parent.parent / ".env"
ENV_FILE = str(_candidate) if _candidate.exists() else ".env"


class Settings(BaseSettings):
    # App
    APP_NAME: str = "JobNexus"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Base directory (backend folder)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Database
    DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5433/nexusrag")

    # LLM Provider: "gemini" | "ollama"
    LLM_PROVIDER: str = Field(default="gemini")

    # Google AI
    GOOGLE_AI_API_KEY: str = Field(default="")

    # Ollama
    OLLAMA_HOST: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="gemma3:12b")
    OLLAMA_ENABLE_THINKING: bool = Field(default=False)

    # LLM (fast model for chat + KG extraction — used when provider=gemini)
    LLM_MODEL_FAST: str = Field(default="gemini-2.5-flash")

    # Thinking level for Gemini 3.x+ models: "minimal" | "low" | "medium" | "high"
    # Gemini 2.5 uses thinking_budget_tokens instead (auto-detected)
    LLM_THINKING_LEVEL: str = Field(default="medium")

    # Max output tokens for LLM chat responses (includes thinking tokens)
    # Gemini 3.1 Flash-Lite supports up to 65536
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=8192)

    # KG Embedding provider (can differ from LLM provider)
    KG_EMBEDDING_PROVIDER: str = Field(default="gemini")
    KG_EMBEDDING_MODEL: str = Field(default="gemini-embedding-001")
    KG_EMBEDDING_DIMENSION: int = Field(default=3072)

    # ChromaDB
    CHROMA_HOST: str = Field(default="localhost")
    CHROMA_PORT: int = Field(default=8002)

    # NexusRAG Pipeline
    NEXUSRAG_ENABLED: bool = True
    NEXUSRAG_ENABLE_KG: bool = True
    NEXUSRAG_ENABLE_IMAGE_EXTRACTION: bool = True
    NEXUSRAG_ENABLE_IMAGE_CAPTIONING: bool = True
    NEXUSRAG_ENABLE_TABLE_CAPTIONING: bool = True
    NEXUSRAG_MAX_TABLE_MARKDOWN_CHARS: int = 8000
    NEXUSRAG_CHUNK_MAX_TOKENS: int = 512
    NEXUSRAG_KG_QUERY_TIMEOUT: float = 30.0
    NEXUSRAG_KG_CHUNK_TOKEN_SIZE: int = 1200
    NEXUSRAG_KG_LANGUAGE: str = "Vietnamese"
    NEXUSRAG_KG_ENTITY_TYPES: list[str] = [
        "Skill", "JobTitle", "Company", "Industry", "Certification",
        "University", "Location", "Technology", "Person",
    ]
    NEXUSRAG_DEFAULT_QUERY_MODE: str = "hybrid"
    NEXUSRAG_DOCLING_IMAGES_SCALE: float = 2.0
    NEXUSRAG_MAX_IMAGES_PER_DOC: int = 50
    NEXUSRAG_ENABLE_FORMULA_ENRICHMENT: bool = True

    # NexusRAG Retrieval Quality
    NEXUSRAG_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    NEXUSRAG_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    NEXUSRAG_VECTOR_PREFETCH: int = 20
    NEXUSRAG_RERANKER_TOP_K: int = 8
    NEXUSRAG_MIN_RELEVANCE_SCORE: float = 0.15

    # JobNexus Matching Weights
    MATCHING_SEMANTIC_WEIGHT: float = 0.50
    MATCHING_SKILL_WEIGHT: float = 0.25
    MATCHING_EXPERIENCE_WEIGHT: float = 0.10
    MATCHING_LOCATION_WEIGHT: float = 0.10
    MATCHING_SALARY_WEIGHT: float = 0.05

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5174", "http://localhost:3000"]

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
