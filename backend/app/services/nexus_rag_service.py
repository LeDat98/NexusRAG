"""
Deep RAG Service
=================

Orchestrator for the NexusRAG pipeline:
  Document → Docling Parse → ChromaDB Index + LightRAG KG → Hybrid Retrieval

Backward-compatible: exposes the same `process_document()`, `query()`,
`delete_document()`, `get_chunk_count()` interface as legacy RAGService.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.core.config import settings
from app.models.document import Document, DocumentImage, DocumentTable, DocumentStatus
from app.services.document_parser import get_document_parser
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.deep_retriever import DeepRetriever
from app.services.embedder import EmbeddingService, get_embedding_service
from app.services.vector_store import VectorStore, get_vector_store
from app.services.reranker import get_reranker_service
from app.services.rag_service import RAGQueryResult, RetrievedChunk
from app.services.models.parsed_document import DeepRetrievalResult
from app.services.chunk_dedup import deduplicate_chunks

logger = logging.getLogger(__name__)


class NexusRAGService:
    """
    Full NexusRAG pipeline orchestrator.

    Phases:
      1. PARSING  — Docling parse → markdown + chunks + images
      2. INDEXING — Embed chunks → ChromaDB + ingest markdown → LightRAG KG
      3. INDEXED  — Update document metadata in DB

    Query:
      - query()       — backward-compatible sync vector-only search
      - query_deep()  — full async hybrid retrieval (KG + vector + images)
    """

    def __init__(
        self,
        db: AsyncSession,
        workspace_id: int,
        kg_language: str | None = None,
        kg_entity_types: list[str] | None = None,
    ):
        self.db = db
        self.workspace_id = workspace_id

        # Services
        self.parser = get_document_parser(workspace_id=workspace_id)
        self.embedder = get_embedding_service()
        self.vector_store = get_vector_store(workspace_id)

        # KG service (optional, gated by config)
        self.kg_service: Optional[KnowledgeGraphService] = None
        if settings.NEXUSRAG_ENABLE_KG:
            self.kg_service = KnowledgeGraphService(
                workspace_id=workspace_id,
                kg_language=kg_language,
                kg_entity_types=kg_entity_types,
            )

        # Retriever (with cross-encoder reranker)
        self.retriever = DeepRetriever(
            workspace_id=workspace_id,
            kg_service=self.kg_service,
            vector_store=self.vector_store,
            embedder=self.embedder,
            db=db,
            reranker=get_reranker_service(),
        )

    # ------------------------------------------------------------------
    # Document Processing
    # ------------------------------------------------------------------

    async def process_document(self, document_id: int, file_path: str) -> int:
        """
        Process a document through the full NexusRAG pipeline.

        Returns:
            Number of chunks created
        """
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        if document is None:
            raise ValueError(f"Document {document_id} not found")

        start_time = time.time()

        try:
            # Phase 1: PARSING
            document.status = DocumentStatus.PARSING
            await self.db.commit()

            import asyncio
            parsed = await asyncio.to_thread(
                self.parser.parse,
                file_path=file_path,
                document_id=document_id,
                original_filename=document.original_filename,
            )

            # Save markdown + images to DB
            document.markdown_content = parsed.markdown
            document.page_count = parsed.page_count
            document.table_count = parsed.tables_count
            document.parser_version = self.parser.parser_name
            await self.db.commit()

            # Clean up old image records before saving new ones (handles re-processing)
            await self.db.execute(
                delete(DocumentImage).where(DocumentImage.document_id == document_id)
            )
            await self.db.commit()

            # Save extracted images to DB
            for img in parsed.images:
                db_image = DocumentImage(
                    document_id=document_id,
                    image_id=img.image_id,
                    page_no=img.page_no,
                    file_path=img.file_path,
                    caption=img.caption,
                    width=img.width,
                    height=img.height,
                    mime_type=img.mime_type,
                )
                self.db.add(db_image)
            if parsed.images:
                document.image_count = len(parsed.images)
                await self.db.commit()

            # Clean up old table records before saving new ones (handles re-processing)
            await self.db.execute(
                delete(DocumentTable).where(DocumentTable.document_id == document_id)
            )
            await self.db.commit()

            # Save extracted tables to DB
            for tbl in parsed.tables:
                db_table = DocumentTable(
                    document_id=document_id,
                    table_id=tbl.table_id,
                    page_no=tbl.page_no,
                    content_markdown=tbl.content_markdown,
                    caption=tbl.caption,
                    num_rows=tbl.num_rows,
                    num_cols=tbl.num_cols,
                )
                self.db.add(db_table)
            if parsed.tables:
                await self.db.commit()

            # Phase 1.5: PRE-INGESTION DEDUP
            if parsed.chunks:
                parsed.chunks, dedup_stats = deduplicate_chunks(parsed.chunks)
                if dedup_stats["input"] != dedup_stats["output"]:
                    logger.info(
                        f"Dedup for doc {document_id}: "
                        f"{dedup_stats['input']}→{dedup_stats['output']} chunks "
                        f"(noise={dedup_stats['noise_removed']}, "
                        f"exact={dedup_stats['exact_removed']}, "
                        f"near={dedup_stats['near_removed']})"
                    )

            # Phase 2: INDEXING
            document.status = DocumentStatus.INDEXING
            await self.db.commit()

            chunk_count = 0
            if parsed.chunks:
                def _index_sync():
                    # Embed and store in ChromaDB
                    chunk_texts = [c.content for c in parsed.chunks]
                    embeddings = self.embedder.embed_texts(chunk_texts)

                    ids = [
                        f"doc_{document_id}_chunk_{i}"
                        for i in range(len(parsed.chunks))
                    ]
                    # Build image_id→URL lookup for metadata
                    _img_url_map = {
                        img.image_id: f"/static/doc-images/kb_{self.workspace_id}/images/{img.image_id}.png"
                        for img in parsed.images
                    }

                    metadatas = []
                    for c in parsed.chunks:
                        meta = {
                            "document_id": document_id,
                            "chunk_index": c.chunk_index,
                            "source": c.source_file,
                            "file_type": document.file_type,
                            "page_no": c.page_no,
                            "heading_path": " > ".join(c.heading_path) if c.heading_path else "",
                            "has_table": c.has_table,
                            "has_code": c.has_code,
                            # Image-aware metadata: pipe-separated IDs and URLs
                            "image_ids": "|".join(c.image_refs) if c.image_refs else "",
                            "table_ids": "|".join(c.table_refs) if c.table_refs else "",
                            "image_urls": "|".join(
                                _img_url_map.get(iid, "") for iid in c.image_refs
                            ) if c.image_refs else "",
                        }
                        if document.custom_metadata:
                            meta.update(document.custom_metadata)
                        metadatas.append(meta)

                    self.vector_store.add_documents(
                        ids=ids,
                        embeddings=embeddings,
                        documents=chunk_texts,
                        metadatas=metadatas,
                    )
                await asyncio.to_thread(_index_sync)
                chunk_count = len(parsed.chunks)

            # KG ingest (async, non-blocking failure)
            if self.kg_service and parsed.markdown:
                try:
                    await self.kg_service.ingest(parsed.markdown)
                except Exception as e:
                    logger.error(
                        f"KG ingest failed for document {document_id}, "
                        f"continuing without KG: {e}"
                    )

            # Phase 3: INDEXED
            elapsed_ms = int((time.time() - start_time) * 1000)
            document.status = DocumentStatus.INDEXED
            document.chunk_count = chunk_count
            document.processing_time_ms = elapsed_ms
            await self.db.commit()

            logger.info(
                f"NexusRAG processed document {document_id}: "
                f"{chunk_count} chunks, {len(parsed.images)} images, "
                f"{parsed.tables_count} tables in {elapsed_ms}ms"
            )
            return chunk_count

        except Exception as e:
            logger.error(f"NexusRAG failed for document {document_id}: {e}")
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)[:500]
            await self.db.commit()
            raise

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        document_ids: Optional[list[int]] = None,
        metadata_filter: dict | None = None,
    ) -> RAGQueryResult:
        """
        Backward-compatible sync query (vector-only).
        Returns same RAGQueryResult as legacy RAGService.
        """
        query_embedding = self.embedder.embed_query(question)

        # Merge metadata_filter and document_ids
        where = metadata_filter.copy() if metadata_filter else {}
        if document_ids:
            where["document_id"] = {"$in": document_ids}
            
        if not where:
            where = None

        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )

        chunks = []
        for i, doc in enumerate(results.get("documents", [])):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            chunks.append(RetrievedChunk(
                content=doc,
                metadata=meta,
                score=results["distances"][i] if results.get("distances") else 0.0,
                chunk_id=results["ids"][i] if results.get("ids") else "",
            ))

        chunks.sort(key=lambda x: x.score)

        # Assemble context with citations
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page_no", 0)
            heading = chunk.metadata.get("heading_path", "")
            citation = source
            if page:
                citation += f" | p.{page}"
            if heading:
                citation += f" | {heading}"
            context_parts.append(f"[{i + 1}] {citation}\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)

        return RAGQueryResult(
            chunks=chunks,
            context=context,
            query=question,
        )

    async def query_deep(
        self,
        question: str,
        top_k: int = 5,
        document_ids: Optional[list[int]] = None,
        mode: str = "hybrid",
        include_images: bool = True,
        metadata_filter: dict | None = None,
    ) -> DeepRetrievalResult:
        """
        Full async hybrid retrieval with KG + vector + images + citations.
        """
        return await self.retriever.query(
            question=question,
            mode=mode,
            top_k=top_k,
            document_ids=document_ids,
            include_images=include_images,
            metadata_filter=metadata_filter,
        )

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    async def delete_document(self, document_id: int) -> None:
        """Delete a document's data from vector store and KG."""
        self.vector_store.delete_by_document_id(document_id)

        # Delete images from DB (cascade handles it, but clean up files)
        result = await self.db.execute(
            select(DocumentImage).where(DocumentImage.document_id == document_id)
        )
        for img in result.scalars().all():
            from pathlib import Path
            img_path = Path(img.file_path)
            if img_path.exists():
                img_path.unlink()

        logger.info(f"Deleted document {document_id} from NexusRAG stores")

    def get_chunk_count(self) -> int:
        """Return total number of chunks in the knowledge base's vector store."""
        return self.vector_store.count()
