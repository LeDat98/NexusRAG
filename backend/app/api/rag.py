"""
RAG API endpoints for document querying and retrieval.
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.deps import get_db
from app.core.exceptions import NotFoundError
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document, DocumentImage, DocumentStatus
import logging

from app.schemas.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievedChunkResponse,
    CitationResponse,
    DocumentImageResponse,
    DocumentProcessRequest,
    DocumentProcessResponse,
    BatchProcessRequest,
    ProjectRAGStatsResponse,
    KGEntityResponse,
    KGRelationshipResponse,
    KGGraphResponse,
    KGGraphNodeResponse,
    KGGraphEdgeResponse,
    KGAnalyticsResponse,
    DocumentBreakdownItem,
    ProjectAnalyticsResponse,
    ChatRequest,
    ChatResponse,
    ChatSourceChunk,
    ChatImageRef,
    PersistedChatMessage,
    ChatHistoryResponse,
    LLMCapabilitiesResponse,
    DebugRetrievedSource,
    DebugChatResponse,
    RateSourceRequest,
)

logger = logging.getLogger(__name__)
import string, random
from app.services.rag_service import get_rag_service

# ---------------------------------------------------------------------------
# Citation ID generation — 4-char alphanumeric IDs matching PageIndex format
# ---------------------------------------------------------------------------
_CITATION_ID_CHARS = string.ascii_lowercase + string.digits


def _generate_citation_id(existing: set[str]) -> str:
    """Generate a unique 4-char alphanumeric citation ID.

    Always contains at least one letter so it cannot be confused with
    old-style numeric indices (e.g. "1", "23").
    """
    while True:
        cid = "".join(random.choices(_CITATION_ID_CHARS, k=4))
        if any(c.isalpha() for c in cid) and cid not in existing:
            return cid

router = APIRouter(prefix="/rag", tags=["rag"])

UPLOAD_DIR = "uploads"

# Prompt constants — see chat_prompt.py for full documentation
from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT


async def verify_workspace_access(
    workspace_id: int,
    db: AsyncSession,
) -> KnowledgeBase:
    """Verify knowledge base exists."""
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    kb = result.scalar_one_or_none()

    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)

    return kb


@router.post("/query/{workspace_id}", response_model=RAGQueryResponse)
async def query_documents(
    workspace_id: int,
    request: RAGQueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Query indexed documents using semantic search (+ optional KG)."""
    await verify_workspace_access(workspace_id, db)

    rag_service = get_rag_service(db, workspace_id)

    # Try deep query if available
    from app.services.nexus_rag_service import NexusRAGService
    if isinstance(rag_service, NexusRAGService) and request.mode != "vector_only":
        result = await rag_service.query_deep(
            question=request.question,
            top_k=request.top_k,
            document_ids=request.document_ids,
            mode=request.mode,
            metadata_filter=request.metadata_filter,
        )

        chunks_response = []
        for i, chunk in enumerate(result.chunks):
            citation = result.citations[i] if i < len(result.citations) else None
            citation_resp = None
            if citation:
                citation_resp = CitationResponse(
                    source_file=citation.source_file,
                    document_id=citation.document_id,
                    page_no=citation.page_no,
                    heading_path=citation.heading_path,
                    formatted=citation.format(),
                )
            chunks_response.append(RetrievedChunkResponse(
                content=chunk.content,
                chunk_id=f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}",
                score=0.0,
                metadata={
                    "source": chunk.source_file,
                    "page_no": chunk.page_no,
                    "heading_path": " > ".join(chunk.heading_path),
                },
                citation=citation_resp,
            ))

        image_refs = [
            DocumentImageResponse(
                image_id=img.image_id,
                document_id=img.document_id,
                page_no=img.page_no,
                caption=img.caption,
                width=img.width,
                height=img.height,
                url=f"/static/doc-images/kb_{workspace_id}/images/{img.image_id}.png",
            )
            for img in result.image_refs
        ]

        citations = [
            CitationResponse(
                source_file=c.source_file,
                document_id=c.document_id,
                page_no=c.page_no,
                heading_path=c.heading_path,
                formatted=c.format(),
            )
            for c in result.citations
        ]

        return RAGQueryResponse(
            query=result.query,
            chunks=chunks_response,
            context=result.context,
            total_chunks=len(result.chunks),
            knowledge_graph_summary=result.knowledge_graph_summary,
            citations=citations,
            image_refs=image_refs,
        )

    # Fallback: legacy sync query
    result = rag_service.query(
        question=request.question,
        top_k=request.top_k,
        document_ids=request.document_ids
    )

    return RAGQueryResponse(
        query=result.query,
        chunks=[
            RetrievedChunkResponse(
                content=chunk.content,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
                metadata=chunk.metadata
            )
            for chunk in result.chunks
        ],
        context=result.context,
        total_chunks=len(result.chunks)
    )


@router.post("/process/{document_id}", response_model=DocumentProcessResponse)
async def process_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Trigger document processing (parsing + indexing) as a background task."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if document.status in (DocumentStatus.PROCESSING, DocumentStatus.PARSING, DocumentStatus.INDEXING):
        # Check if stale (exceeded processing timeout) — auto-recover
        from datetime import datetime, timedelta
        from app.core.config import settings
        timeout = settings.NEXUSRAG_PROCESSING_TIMEOUT_MINUTES
        cutoff = datetime.utcnow() - timedelta(minutes=timeout)
        if document.updated_at < cutoff:
            # Stale — reset to allow re-processing
            document.status = DocumentStatus.FAILED
            document.error_message = f"Processing timeout ({timeout}min). Retrying..."
            await db.commit()
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document is already being analyzed"
            )

    if document.status == DocumentStatus.INDEXED:
        return DocumentProcessResponse(
            document_id=document_id,
            status=document.status.value,
            chunk_count=document.chunk_count,
            message="Document is already indexed"
        )

    from pathlib import Path
    file_path = Path(UPLOAD_DIR) / document.filename

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found on disk"
        )

    # Mark as processing immediately so UI updates
    document.status = DocumentStatus.PROCESSING
    document.error_message = None
    await db.commit()

    # Launch background task
    from app.api.documents import process_document_background
    import asyncio
    asyncio.get_event_loop().create_task(
        process_document_background(document_id, str(file_path), document.workspace_id)
    )

    return DocumentProcessResponse(
        document_id=document_id,
        status="processing",
        chunk_count=0,
        message="Processing started. Document will be parsed and indexed in the background."
    )


@router.post("/process-batch")
async def process_batch(
    request: BatchProcessRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Process multiple documents sequentially in the background.
    Marks all as PROCESSING immediately, then processes one-by-one to avoid
    resource contention (each doc uses Docling + embeddings + KG ingest).
    """
    from pathlib import Path as _P

    accepted_ids = []
    skipped_ids = []

    for doc_id in request.document_ids:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc is None:
            skipped_ids.append(doc_id)
            continue

        # Skip documents already being processed or already indexed
        if doc.status in (
            DocumentStatus.PROCESSING, DocumentStatus.PARSING, DocumentStatus.INDEXING,
        ):
            skipped_ids.append(doc_id)
            continue

        file_path = _P(UPLOAD_DIR) / doc.filename
        if not file_path.exists():
            skipped_ids.append(doc_id)
            continue

        # Mark as processing immediately so UI updates
        doc.status = DocumentStatus.PROCESSING
        doc.error_message = None
        accepted_ids.append((doc_id, str(file_path), doc.workspace_id))

    await db.commit()

    if accepted_ids:
        import asyncio
        asyncio.get_event_loop().create_task(
            _process_batch_background(accepted_ids)
        )

    return {
        "message": f"Processing {len(accepted_ids)} document(s)",
        "accepted": [aid[0] for aid in accepted_ids],
        "skipped": skipped_ids,
    }


async def _process_batch_background(
    items: list[tuple[int, str, int]],
):
    """Process documents sequentially to avoid resource contention."""
    from app.api.documents import process_document_background

    for doc_id, file_path, workspace_id in items:
        try:
            await process_document_background(doc_id, file_path, workspace_id)
            logger.info(f"Batch: document {doc_id} processed")
        except Exception as e:
            logger.error(f"Batch: document {doc_id} failed: {e}")


@router.post("/reindex/{document_id}", response_model=DocumentProcessResponse)
async def reindex_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Re-process an existing document through the NexusRAG pipeline."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if document.status in (DocumentStatus.PROCESSING, DocumentStatus.PARSING, DocumentStatus.INDEXING):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is currently being processed"
        )

    from pathlib import Path
    file_path = Path(UPLOAD_DIR) / document.filename

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found on disk"
        )

    rag_service = get_rag_service(db, document.workspace_id)

    # Delete existing data first
    try:
        await rag_service.delete_document(document_id)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to delete old data for reindex: {e}")

    # Reset document metadata
    document.status = DocumentStatus.PENDING
    document.chunk_count = 0
    document.markdown_content = None
    document.image_count = 0
    document.table_count = 0
    document.parser_version = None
    document.error_message = None
    await db.commit()

    try:
        chunk_count = await rag_service.process_document(
            document_id=document_id,
            file_path=str(file_path)
        )
        return DocumentProcessResponse(
            document_id=document_id,
            status=DocumentStatus.INDEXED.value,
            chunk_count=chunk_count,
            message=f"Re-indexed with NexusRAG: {chunk_count} chunks created"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reindex document: {str(e)}"
        )


@router.post("/reindex-workspace/{workspace_id}")
async def reindex_workspace(
    workspace_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Reindex ALL documents in a workspace.
    Deletes the old vector collection (handles embedding dimension changes)
    and re-processes every document through the NexusRAG pipeline.
    Runs in background — returns immediately with document count.
    """
    await verify_workspace_access(workspace_id, db)

    # Find all documents in this workspace
    result = await db.execute(
        select(Document).where(
            Document.workspace_id == workspace_id,
            Document.status.notin_([
                DocumentStatus.PROCESSING,
                DocumentStatus.PARSING,
                DocumentStatus.INDEXING,
            ]),
        )
    )
    documents = list(result.scalars().all())

    if not documents:
        return {"message": "No documents to reindex", "document_count": 0}

    # Delete old vector collection (required when embedding dimensions change)
    try:
        from app.services.vector_store import get_vector_store
        vs = get_vector_store(workspace_id)
        vs.delete_collection()
        logger.info(f"Deleted old vector collection for workspace {workspace_id}")
    except Exception as e:
        logger.warning(f"Failed to delete old collection: {e}")

    async def _reindex_all(doc_ids: list[int], ws_id: int):
        """Background task: reindex each document sequentially."""
        from app.core.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            rag_service = get_rag_service(session, ws_id)
            for did in doc_ids:
                try:
                    res = await session.execute(
                        select(Document).where(Document.id == did)
                    )
                    doc = res.scalar_one_or_none()
                    if not doc:
                        continue

                    from pathlib import Path
                    file_path = Path(UPLOAD_DIR) / doc.filename
                    if not file_path.exists():
                        logger.warning(f"Skipping doc {did}: file not found")
                        continue

                    # Delete old chunk data for this document
                    try:
                        await rag_service.delete_document(did)
                    except Exception:
                        pass

                    # Reset metadata
                    doc.status = DocumentStatus.PENDING
                    doc.chunk_count = 0
                    doc.image_count = 0
                    doc.error_message = None
                    await session.commit()

                    # Re-process
                    await rag_service.process_document(
                        document_id=did, file_path=str(file_path)
                    )
                    logger.info(f"Reindexed document {did} in workspace {ws_id}")
                except Exception as e:
                    logger.error(f"Failed to reindex document {did}: {e}")

    doc_ids = [d.id for d in documents]
    background_tasks.add_task(_reindex_all, doc_ids, workspace_id)

    return {
        "message": f"Reindexing {len(doc_ids)} documents in background",
        "document_count": len(doc_ids),
        "document_ids": doc_ids,
    }


@router.get("/stats/{workspace_id}", response_model=ProjectRAGStatsResponse)
async def get_workspace_rag_stats(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get RAG statistics for a knowledge base."""
    await verify_workspace_access(workspace_id, db)

    total_result = await db.execute(
        select(func.count(Document.id)).where(Document.workspace_id == workspace_id)
    )
    total_documents = total_result.scalar() or 0

    indexed_result = await db.execute(
        select(func.count(Document.id)).where(
            Document.workspace_id == workspace_id,
            Document.status == DocumentStatus.INDEXED
        )
    )
    indexed_documents = indexed_result.scalar() or 0

    # Count NexusRAG documents (parser_version = 'docling')
    nexusrag_result = await db.execute(
        select(func.count(Document.id)).where(
            Document.workspace_id == workspace_id,
            Document.parser_version == "docling"
        )
    )
    nexusrag_documents = nexusrag_result.scalar() or 0

    # Count total images
    image_result = await db.execute(
        select(func.count(DocumentImage.id))
        .join(Document, DocumentImage.document_id == Document.id)
        .where(Document.workspace_id == workspace_id)
    )
    image_count = image_result.scalar() or 0

    rag_service = get_rag_service(db, workspace_id)
    try:
        total_chunks = rag_service.get_chunk_count()
    except Exception:
        total_chunks = 0

    return ProjectRAGStatsResponse(
        workspace_id=workspace_id,
        total_documents=total_documents,
        indexed_documents=indexed_documents,
        total_chunks=total_chunks,
        image_count=image_count,
        nexusrag_documents=nexusrag_documents,
    )


@router.get("/chunks/{document_id}")
async def get_document_chunks(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get all chunks for a specific document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if document.status != DocumentStatus.INDEXED:
        return {
            "document_id": document_id,
            "status": document.status.value,
            "chunks": [],
            "message": "Document is not yet indexed"
        }

    rag_service = get_rag_service(db, document.workspace_id)

    chunk_ids = [f"doc_{document_id}_chunk_{i}" for i in range(document.chunk_count)]

    try:
        results = rag_service.vector_store.get_by_ids(chunk_ids)

        chunks = []
        for i in range(len(results.get("ids", []))):
            chunks.append({
                "chunk_id": results["ids"][i],
                "content": results["documents"][i] if results.get("documents") else None,
                "metadata": results["metadatas"][i] if results.get("metadatas") else {}
            })

        return {
            "document_id": document_id,
            "status": document.status.value,
            "chunk_count": document.chunk_count,
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunks: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Knowledge Graph exploration endpoints (Phase 9)
# ---------------------------------------------------------------------------

async def _get_kg_service(workspace_id: int):
    """Get KnowledgeGraphService for a knowledge base (if NexusRAG is active)."""
    from app.services.knowledge_graph_service import KnowledgeGraphService
    return KnowledgeGraphService(workspace_id)


@router.get("/entities/{workspace_id}", response_model=list[KGEntityResponse])
async def get_kg_entities(
    workspace_id: int,
    search: str | None = None,
    entity_type: str | None = None,
    limit: int = 200,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List entities in the workspace's knowledge graph."""
    await verify_workspace_access(workspace_id, db)
    kg = await _get_kg_service(workspace_id)
    try:
        entities = await kg.get_entities(
            search=search, entity_type=entity_type, limit=limit, offset=offset
        )
        return [KGEntityResponse(**e) for e in entities]
    except Exception as e:
        logger.error(f"Failed to get KG entities for workspace {workspace_id}: {e}")
        return []


@router.get("/relationships/{workspace_id}", response_model=list[KGRelationshipResponse])
async def get_kg_relationships(
    workspace_id: int,
    entity: str | None = None,
    limit: int = 500,
    db: AsyncSession = Depends(get_db),
):
    """List relationships in the workspace's knowledge graph."""
    await verify_workspace_access(workspace_id, db)
    kg = await _get_kg_service(workspace_id)
    try:
        rels = await kg.get_relationships(entity_name=entity, limit=limit)
        return [KGRelationshipResponse(**r) for r in rels]
    except Exception as e:
        logger.error(f"Failed to get KG relationships for workspace {workspace_id}: {e}")
        return []


@router.get("/graph/{workspace_id}", response_model=KGGraphResponse)
async def get_kg_graph(
    workspace_id: int,
    center: str | None = None,
    max_depth: int = 3,
    max_nodes: int = 150,
    db: AsyncSession = Depends(get_db),
):
    """Export knowledge graph data for frontend visualization."""
    await verify_workspace_access(workspace_id, db)
    kg = await _get_kg_service(workspace_id)
    try:
        data = await kg.get_graph_data(
            center_entity=center, max_depth=max_depth, max_nodes=max_nodes
        )
        return KGGraphResponse(
            nodes=[KGGraphNodeResponse(**n) for n in data["nodes"]],
            edges=[KGGraphEdgeResponse(**e) for e in data["edges"]],
            is_truncated=data.get("is_truncated", False),
        )
    except Exception as e:
        logger.error(f"Failed to export KG graph for workspace {workspace_id}: {e}")
        return KGGraphResponse()


@router.get("/analytics/{workspace_id}", response_model=ProjectAnalyticsResponse)
async def get_workspace_analytics(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get extended analytics for a knowledge base (stats + KG + per-doc breakdown)."""
    await verify_workspace_access(workspace_id, db)

    # Base stats
    total_result = await db.execute(
        select(func.count(Document.id)).where(Document.workspace_id == workspace_id)
    )
    total_documents = total_result.scalar() or 0

    indexed_result = await db.execute(
        select(func.count(Document.id)).where(
            Document.workspace_id == workspace_id,
            Document.status == DocumentStatus.INDEXED,
        )
    )
    indexed_documents = indexed_result.scalar() or 0

    nexusrag_result = await db.execute(
        select(func.count(Document.id)).where(
            Document.workspace_id == workspace_id,
            Document.parser_version == "docling",
        )
    )
    nexusrag_documents = nexusrag_result.scalar() or 0

    image_result = await db.execute(
        select(func.count(DocumentImage.id))
        .join(Document, DocumentImage.document_id == Document.id)
        .where(Document.workspace_id == workspace_id)
    )
    image_count = image_result.scalar() or 0

    rag_service = get_rag_service(db, workspace_id)
    try:
        total_chunks = rag_service.get_chunk_count()
    except Exception:
        total_chunks = 0

    stats = ProjectRAGStatsResponse(
        workspace_id=workspace_id,
        total_documents=total_documents,
        indexed_documents=indexed_documents,
        total_chunks=total_chunks,
        image_count=image_count,
        nexusrag_documents=nexusrag_documents,
    )

    # KG analytics (optional — only if NexusRAG active)
    kg_analytics = None
    if nexusrag_documents > 0:
        try:
            kg = await _get_kg_service(workspace_id)
            analytics_data = await kg.get_analytics()
            kg_analytics = KGAnalyticsResponse(
                entity_count=analytics_data["entity_count"],
                relationship_count=analytics_data["relationship_count"],
                entity_types=analytics_data["entity_types"],
                top_entities=[KGEntityResponse(**e) for e in analytics_data["top_entities"]],
                avg_degree=analytics_data["avg_degree"],
            )
        except Exception as e:
            logger.warning(f"Failed to get KG analytics for workspace {workspace_id}: {e}")

    # Per-document breakdown
    doc_result = await db.execute(
        select(Document)
        .where(Document.workspace_id == workspace_id)
        .order_by(Document.created_at.desc())
    )
    documents = doc_result.scalars().all()
    breakdown = [
        DocumentBreakdownItem(
            document_id=d.id,
            filename=d.original_filename,
            chunk_count=d.chunk_count,
            image_count=d.image_count or 0,
            page_count=d.page_count or 0,
            file_size=d.file_size,
            status=d.status.value if hasattr(d.status, "value") else str(d.status),
        )
        for d in documents
    ]

    return ProjectAnalyticsResponse(
        stats=stats,
        kg_analytics=kg_analytics,
        document_breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Chat history persistence
# ---------------------------------------------------------------------------

@router.get("/chat/{workspace_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Load persisted chat history for a workspace."""
    await verify_workspace_access(workspace_id, db)

    from app.models.chat_message import ChatMessage as ChatMessageModel
    result = await db.execute(
        select(ChatMessageModel)
        .where(ChatMessageModel.workspace_id == workspace_id)
        .order_by(ChatMessageModel.created_at.asc())
    )
    messages = result.scalars().all()

    return ChatHistoryResponse(
        workspace_id=workspace_id,
        messages=[
            PersistedChatMessage(
                id=m.id,
                message_id=m.message_id,
                role=m.role,
                content=m.content,
                sources=m.sources,
                related_entities=m.related_entities,
                image_refs=m.image_refs,
                thinking=m.thinking,
                agent_steps=m.agent_steps,
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m in messages
        ],
        total=len(messages),
    )


@router.delete("/chat/{workspace_id}/history")
async def delete_chat_history(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Clear all chat history for a workspace."""
    await verify_workspace_access(workspace_id, db)

    from app.models.chat_message import ChatMessage as ChatMessageModel
    from sqlalchemy import delete
    await db.execute(
        delete(ChatMessageModel).where(ChatMessageModel.workspace_id == workspace_id)
    )
    await db.commit()
    return {"status": "cleared", "workspace_id": workspace_id}


# ---------------------------------------------------------------------------
# Source Rating endpoint
# ---------------------------------------------------------------------------

@router.post("/chat/{workspace_id}/rate")
async def rate_source(
    workspace_id: int,
    body: RateSourceRequest,
    db: AsyncSession = Depends(get_db),
):
    """Rate a source citation in a chat message."""
    await verify_workspace_access(workspace_id, db)

    from app.models.chat_message import ChatMessage as ChatMessageModel

    result = await db.execute(
        select(ChatMessageModel).where(
            ChatMessageModel.workspace_id == workspace_id,
            ChatMessageModel.message_id == body.message_id,
        )
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found",
        )

    current_ratings = row.ratings or {}
    current_ratings[body.source_index] = body.rating
    row.ratings = current_ratings
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(row, "ratings")
    await db.commit()

    return {"success": True, "message_id": body.message_id, "ratings": current_ratings}


# ---------------------------------------------------------------------------
# Chat endpoint — LLM-powered document Q&A via NexusRAG
# ---------------------------------------------------------------------------
# SSE Streaming chat endpoint
# ---------------------------------------------------------------------------

@router.post("/chat/{workspace_id}/stream")
async def chat_stream(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """SSE streaming chat with semi-agentic retrieval."""
    from app.api.chat_agent import chat_stream_endpoint
    return await chat_stream_endpoint(workspace_id, request, db)


# ---------------------------------------------------------------------------

@router.post("/chat/{workspace_id}", response_model=ChatResponse)
async def chat_with_documents(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Chat with documents using NexusRAG retrieval + LLM answer generation."""
    kb = await verify_workspace_access(workspace_id, db)

    rag_service = get_rag_service(db, workspace_id)

    # -- 1. Retrieve relevant chunks via NexusRAG --
    chunks = []
    citations = []
    kg_summary = ""

    from app.services.nexus_rag_service import NexusRAGService
    if isinstance(rag_service, NexusRAGService):
        result = await rag_service.query_deep(
            question=request.message,
            top_k=8,
            document_ids=request.document_ids,
            mode="hybrid",
            include_images=False,  # No longer need separate image lookup
        )
        chunks = result.chunks
        citations = result.citations
        kg_summary = result.knowledge_graph_summary
    else:
        # Fallback: legacy vector-only
        legacy = rag_service.query(
            question=request.message,
            top_k=5,
            document_ids=request.document_ids,
        )
        for i, c in enumerate(legacy.chunks):
            from types import SimpleNamespace
            chunks.append(SimpleNamespace(
                content=c.content,
                document_id=int(c.metadata.get("document_id", 0)),
                chunk_index=i,
                page_no=int(c.metadata.get("page_no", 0)),
                heading_path=str(c.metadata.get("heading_path", "")).split(" > ") if c.metadata.get("heading_path") else [],
                source_file=str(c.metadata.get("source", "")),
                image_refs=[],
            ))

    # -- 2. Build sources list --
    # Source labels use "Source [XXXX]" format (4-char alphanumeric IDs).
    # Never put extra text inside brackets — LLMs copy that format.
    used_ids: set[str] = set()
    sources = []
    context_parts = []
    for i, chunk in enumerate(chunks):
        citation = citations[i] if i < len(citations) else None
        cid = _generate_citation_id(used_ids)
        used_ids.add(cid)
        sources.append(ChatSourceChunk(
            index=cid,
            chunk_id=f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}",
            content=chunk.content,
            document_id=chunk.document_id,
            page_no=chunk.page_no,
            heading_path=chunk.heading_path,
            score=0.0,
            source_type="vector",
        ))
        # Build metadata line (filename, page, heading) — OUTSIDE brackets
        meta_parts = []
        if citation:
            meta_parts.append(citation.source_file)
            if citation.page_no:
                meta_parts.append(f"page {citation.page_no}")
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else ""
        if heading:
            meta_parts.append(heading)
        meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""

        context_parts.append(f"Source [{cid}]{meta_line}:\n{chunk.content}")

    # NOTE: KG summary is NOT added as a citable source.
    # LightRAG's query() can hallucinate data that doesn't exist in documents.
    # If we make it a citable [N] source, the LLM faithfully cites fabricated data.
    # Instead, KG summary is only used as background context (no source number).
    context = "\n\n---\n\n".join(context_parts)

    # -- 2b. Build image references (chunk metadata → fallback: page-based) --
    from pathlib import Path as _P
    from app.core.config import settings

    # Strategy 1: collect image_ids from chunk metadata (image-aware chunks)
    seen_image_ids: set[str] = set()
    chunk_image_ids: list[str] = []
    for c in chunks:
        for iid in getattr(c, "image_refs", []) or []:
            if iid and iid not in seen_image_ids:
                seen_image_ids.add(iid)
                chunk_image_ids.append(iid)

    # Look up DocumentImage rows for these IDs
    resolved_images: list[DocumentImage] = []
    if chunk_image_ids:
        img_result = await db.execute(
            select(DocumentImage).where(DocumentImage.image_id.in_(chunk_image_ids))
        )
        resolved_images = list(img_result.scalars().all())

    # Strategy 2 (fallback): if chunk metadata yielded no images, try
    # page-based lookup from the retrieved chunks' page numbers.
    if not resolved_images:
        source_pages = {
            (getattr(c, "document_id", 0), getattr(c, "page_no", 0))
            for c in chunks
            if getattr(c, "page_no", 0) > 0
        }
        if source_pages:
            from sqlalchemy import or_, and_
            page_filters = [
                and_(
                    DocumentImage.document_id == doc_id,
                    DocumentImage.page_no == page_no,
                )
                for doc_id, page_no in source_pages
            ]
            img_result = await db.execute(
                select(DocumentImage).where(or_(*page_filters))
            )
            resolved_images = list(img_result.scalars().all())
            # Deduplicate
            seen = set()
            deduped = []
            for img in resolved_images:
                if img.image_id not in seen:
                    seen.add(img.image_id)
                    deduped.append(img)
            resolved_images = deduped

    chat_image_refs: list[ChatImageRef] = []
    image_context_parts: list[str] = []
    image_parts = []  # genai.types.Part for multimodal

    MAX_VISION_IMAGES = 3  # Limit images to control token cost
    for idx, img in enumerate(resolved_images[:MAX_VISION_IMAGES]):
        img_ref_id = _generate_citation_id(used_ids)
        used_ids.add(img_ref_id)
        img_url = f"/static/doc-images/kb_{workspace_id}/images/{img.image_id}.png"
        chat_image_refs.append(ChatImageRef(
            ref_id=img_ref_id,
            image_id=img.image_id,
            document_id=img.document_id,
            page_no=img.page_no,
            caption=img.caption or "",
            url=img_url,
            width=img.width,
            height=img.height,
        ))
        # Image caption for text context — [IMG-XXXX] format
        cap = f'"{img.caption}"' if img.caption else "no caption"
        image_context_parts.append(
            f"- [IMG-{img_ref_id}] Page {img.page_no}: {cap}"
        )
        # Read actual image file for Gemini Vision
        img_path = _P(img.file_path)
        if img_path.exists():
            try:
                img_bytes = img_path.read_bytes()
                mime = img.mime_type or "image/png"
                image_parts.append({
                    "inline_data": {"mime_type": mime, "data": img_bytes},
                    "page_no": img.page_no,
                    "caption": img.caption or "",
                    "img_ref_id": img_ref_id,
                })
            except Exception as e:
                logger.warning(f"Failed to read image {img.image_id}: {e}")

    # -- 3. Call LLM with context + images --
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMImagePart, LLMMessage, LLMResult

    provider = get_llm_provider()

    # ── Prompt architecture for local models (gemma3, etc.) ──────────
    # Key insight: Local models ignore system prompts when context is long.
    # Solution: SHORT system prompt + sources/rules in USER MESSAGE.
    # The model pays most attention to the user message.

    system_prompt = (kb.system_prompt or DEFAULT_SYSTEM_PROMPT) + HARD_SYSTEM_PROMPT

    # ── Build user message: sources + rules + question ──────────────
    # Structure: CONTEXT → RULES → QUESTION (model reads context first)

    user_parts: list[str] = []

    # 1. Document sources (the model reads this first)
    user_parts.append("I have retrieved the following document sources for you.\n")
    user_parts.append("=== DOCUMENT SOURCES ===")
    user_parts.append(context)
    user_parts.append("=== END SOURCES ===\n")

    # 2. Image references (if any)
    if image_context_parts:
        user_parts.append("Document Images:")
        user_parts.extend(image_context_parts)
        user_parts.append("")

    # 3. Contextual rules (only things not covered by system prompt)
    user_parts.append(
        "IMPORTANT:\n"
        "- Read EVERY source above carefully. Answers often require "
        "combining data from MULTIPLE sources.\n"
        "- TABLE DATA: Sources may contain table data as 'Key, Year = Value' pairs. "
        "Example: 'ROE, 2023 = 12,8%' means ROE was 12.8% in 2023. "
        "Extract and report these values.\n"
        "- If no source contains relevant information, say: "
        "\"Tài liệu không chứa thông tin này.\"\n"
    )

    # 4. Conversation context recap (if history exists)
    if request.history:
        last_exchange = request.history[-2:]  # last Q+A pair
        recap_parts = []
        for msg in last_exchange:
            prefix = "User" if msg.role == "user" else "Assistant"
            recap_parts.append(f"{prefix}: {msg.content[:300]}")
        user_parts.append(
            "CONVERSATION CONTEXT (previous exchange):\n"
            + "\n".join(recap_parts) + "\n"
        )

    # 5. The actual question (last = highest attention position)
    user_parts.append(f"My question: {request.message}")

    user_content = "\n".join(user_parts)

    messages: list[LLMMessage] = []
    for msg in request.history[-10:]:  # Keep last 10 messages for context
        role = "user" if msg.role == "user" else "assistant"
        messages.append(LLMMessage(role=role, content=msg.content))

    # Attach images to user message (for multimodal models)
    user_images: list[LLMImagePart] = []
    if image_parts:
        for img_data in image_parts:
            user_content += f"\n[IMG-{img_data['img_ref_id']}] (page {img_data['page_no']}):"
            user_images.append(LLMImagePart(
                data=img_data["inline_data"]["data"],
                mime_type=img_data["inline_data"]["mime_type"],
            ))

    messages.append(LLMMessage(role="user", content=user_content, images=user_images))

    thinking_text: str | None = None
    try:
        result = await provider.acomplete(
            messages,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=4096,
            think=request.enable_thinking,
        )
        if isinstance(result, LLMResult):
            answer = result.content
            thinking_text = result.thinking or None
        else:
            answer = result
        if not answer:
            answer = "Unable to generate a response."
        # Strip Gemini token artifacts (e.g. <unused778>:)
        import re
        answer = re.sub(r'<unused\d+>:?\s*', '', answer).strip()
    except Exception as e:
        logger.error(f"LLM chat error: {e}")
        answer = f"Sorry, I encountered an error generating the response: {str(e)}"

    # -- 4. Extract related entities from KG --
    related_entities: list[str] = []
    if kg_summary:
        try:
            kg = await _get_kg_service(workspace_id)
            entities = await kg.get_entities(limit=200)
            entity_names = {e["name"].lower(): e["name"] for e in entities}
            answer_lower = answer.lower()
            context_lower = context.lower()
            for lower_name, original_name in entity_names.items():
                if len(lower_name) >= 2 and (lower_name in answer_lower or lower_name in context_lower):
                    related_entities.append(original_name)
        except Exception as e:
            logger.warning(f"Failed to extract related entities: {e}")

    # -- 5. Persist messages to DB (best-effort) --
    try:
        import uuid
        from app.models.chat_message import ChatMessage as ChatMessageModel

        user_row = ChatMessageModel(
            workspace_id=workspace_id,
            message_id=str(uuid.uuid4()),
            role="user",
            content=request.message,
        )
        db.add(user_row)

        assistant_row = ChatMessageModel(
            workspace_id=workspace_id,
            message_id=str(uuid.uuid4()),
            role="assistant",
            content=answer,
            sources=[s.model_dump() for s in sources] if sources else None,
            related_entities=related_entities[:30] if related_entities else None,
            image_refs=[img.model_dump() for img in chat_image_refs] if chat_image_refs else None,
            thinking=thinking_text,
        )
        db.add(assistant_row)
        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to persist chat messages: {e}")
        await db.rollback()

    return ChatResponse(
        answer=answer,
        sources=sources,
        related_entities=related_entities[:30],  # Cap at 30
        kg_summary=kg_summary or None,
        image_refs=chat_image_refs,
        thinking=thinking_text,
    )


# ---------------------------------------------------------------------------
# LLM Capabilities endpoint
# ---------------------------------------------------------------------------

@router.get("/capabilities", response_model=LLMCapabilitiesResponse)
async def get_llm_capabilities():
    """Check LLM provider capabilities (thinking, vision)."""
    from app.services.llm import get_llm_provider
    from app.core.config import settings

    provider = get_llm_provider()
    provider_name = settings.LLM_PROVIDER.lower()

    # Per-provider thinking default:
    # Gemini: thinking ON by default (fast, cloud-based)
    # Ollama: thinking OFF by default (slow on local hardware), configurable via OLLAMA_ENABLE_THINKING
    if provider_name == "ollama":
        thinking_default = settings.OLLAMA_ENABLE_THINKING
    else:
        thinking_default = provider.supports_thinking()

    return LLMCapabilitiesResponse(
        provider=settings.LLM_PROVIDER,
        model=settings.OLLAMA_MODEL if provider_name == "ollama" else settings.LLM_MODEL_FAST,
        supports_thinking=provider.supports_thinking(),
        supports_vision=provider.supports_vision(),
        thinking_default=thinking_default,
    )


# ---------------------------------------------------------------------------
# Debug endpoint — inspect retrieval + LLM answer quality
# ---------------------------------------------------------------------------

@router.post("/debug-chat/{workspace_id}", response_model=DebugChatResponse)
async def debug_chat(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Debug version of chat — returns retrieval details + system prompt + answer
    so you can inspect what the LLM received vs what it answered.
    """
    kb = await verify_workspace_access(workspace_id, db)

    rag_service = get_rag_service(db, workspace_id)

    # -- 1. Retrieve --
    chunks = []
    citations = []
    kg_summary = ""

    from app.services.nexus_rag_service import NexusRAGService
    if isinstance(rag_service, NexusRAGService):
        result = await rag_service.query_deep(
            question=request.message,
            top_k=8,
            document_ids=request.document_ids,
            mode="hybrid",
            include_images=False,
        )
        chunks = result.chunks
        citations = result.citations
        kg_summary = result.knowledge_graph_summary

    # -- 2. Build sources + context (same logic as chat endpoint) --
    debug_used_ids: set[str] = set()
    debug_sources: list[DebugRetrievedSource] = []
    context_parts = []
    for i, chunk in enumerate(chunks):
        citation = citations[i] if i < len(citations) else None
        cid = _generate_citation_id(debug_used_ids)
        debug_used_ids.add(cid)
        debug_sources.append(DebugRetrievedSource(
            index=cid,
            document_id=chunk.document_id,
            page_no=chunk.page_no,
            heading_path=chunk.heading_path,
            source_file=citation.source_file if citation else "",
            content_preview=chunk.content[:500],
            score=0.0,
            source_type="vector",
        ))
        meta_parts = []
        if citation:
            meta_parts.append(citation.source_file)
            if citation.page_no:
                meta_parts.append(f"page {citation.page_no}")
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else ""
        if heading:
            meta_parts.append(heading)
        meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""
        context_parts.append(f"Source [{cid}]{meta_line}:\n{chunk.content}")

    # NOTE: KG summary NOT added as citable source (can contain hallucinated data)
    context = "\n\n---\n\n".join(context_parts)

    # -- 3. Build prompt (same architecture as chat endpoint) --
    # SHORT system prompt + sources/rules in USER MESSAGE
    sys_prompt = (kb.system_prompt or DEFAULT_SYSTEM_PROMPT) + HARD_SYSTEM_PROMPT

    # Build user message: CONTEXT → RULES → QUESTION
    user_parts: list[str] = []
    user_parts.append("I have retrieved the following document sources for you.\n")
    user_parts.append("=== DOCUMENT SOURCES ===")
    user_parts.append(context)
    user_parts.append("=== END SOURCES ===\n")

    user_parts.append(
        "IMPORTANT INSTRUCTIONS:\n"
        "- CRITICAL: Read EVERY source carefully before answering. The answer often "
        "requires combining data from MULTIPLE sources. Do NOT skip any source.\n"
        "- TABLE DATA: Sources contain table data as 'Key, Year = Value' pairs. "
        "You MUST extract the actual values. "
        "Example: 'ROE, 2023 = 12,8%. ROE, 2024 = 15,6%' means ROE was 12.8% in 2023 "
        "and 15.6% in 2024. Report these numbers in your answer.\n"
        "- Use the DOCUMENT SOURCES above to answer. Do NOT add outside knowledge.\n"
        "- You MAY compare, synthesize, and reason across multiple sources.\n"
        "- Cite every fact using the source IDs shown in brackets, e.g. [a3x9][b2m7] — one ID per bracket.\n"
        "- For images: [IMG-p4f2][IMG-q7r3] — use the IDs shown in the image list.\n"
        "- NEVER say 'không có thông tin' or 'no information' for data that IS present "
        "in any source. If a source contains 'Key = Value', report that value.\n"
        "- Only say information is unavailable when you have checked ALL sources "
        "and none contains the answer.\n"
        "- If no source is relevant at all, say: "
        "\"Tài liệu không chứa thông tin này.\" without any citations.\n"
        "- Answer in the same language as my question.\n"
    )

    # Conversation context recap (if history exists)
    if request.history:
        last_exchange = request.history[-2:]
        recap_parts = []
        for msg in last_exchange:
            prefix = "User" if msg.role == "user" else "Assistant"
            recap_parts.append(f"{prefix}: {msg.content[:300]}")
        user_parts.append(
            "CONVERSATION CONTEXT (previous exchange):\n"
            + "\n".join(recap_parts) + "\n"
        )

    user_parts.append(f"My question: {request.message}")
    user_content = "\n".join(user_parts)

    # -- 4. Call LLM --
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMMessage, LLMResult

    provider = get_llm_provider()

    messages: list[LLMMessage] = []
    for msg in request.history[-10:]:
        role = "user" if msg.role == "user" else "assistant"
        messages.append(LLMMessage(role=role, content=msg.content))
    messages.append(LLMMessage(role="user", content=user_content))

    answer = ""
    thinking_text: str | None = None
    try:
        llm_result = await provider.acomplete(
            messages,
            system_prompt=sys_prompt,
            temperature=0.1,
            max_tokens=4096,
            think=request.enable_thinking,
        )
        if isinstance(llm_result, LLMResult):
            answer = llm_result.content
            thinking_text = llm_result.thinking or None
        else:
            answer = llm_result
        # Strip Gemini token artifacts (e.g. <unused778>:)
        import re
        answer = re.sub(r'<unused\d+>:?\s*', '', answer).strip()
    except Exception as e:
        answer = f"LLM error: {e}"

    from app.core.config import settings as _s
    return DebugChatResponse(
        question=request.message,
        workspace_id=workspace_id,
        retrieved_sources=debug_sources,
        kg_summary=kg_summary,
        total_sources=len(debug_sources),
        system_prompt=f"[SYSTEM]: {sys_prompt}\n\n[USER MESSAGE]:\n{user_content}",
        answer=answer,
        thinking=thinking_text,
        image_count=0,
        provider=_s.LLM_PROVIDER,
        model=_s.OLLAMA_MODEL if _s.LLM_PROVIDER == "ollama" else _s.LLM_MODEL_FAST,
    )
