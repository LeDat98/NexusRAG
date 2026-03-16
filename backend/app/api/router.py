"""
JobNexus API router — aggregates all endpoint groups.
"""
from fastapi import APIRouter

from app.api.workspaces import router as workspaces_router
from app.api.documents import router as documents_router
from app.api.rag import router as rag_router
from app.api.config import router as config_router
from app.api.candidates import router as candidates_router
from app.api.jobs import router as jobs_router
from app.api.matching import router as matching_router

api_router = APIRouter()

# Original NexusRAG endpoints (kept for backward compatibility)
api_router.include_router(workspaces_router)
api_router.include_router(documents_router)
api_router.include_router(rag_router)
api_router.include_router(config_router)

# JobNexus endpoints
api_router.include_router(candidates_router)
api_router.include_router(jobs_router)
api_router.include_router(matching_router)
