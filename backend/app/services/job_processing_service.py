"""
Job Processing Service
=======================

Orchestrates ingestion for both CVs and Job Descriptions:
  CV/JD file → Docling Parse → LLM Structured Extraction → Embed → ChromaDB + LightRAG KG

Reuses NexusRAG pipeline components:
  - DeepDocumentParser (Docling + HybridChunker)
  - EmbeddingService (bge-m3)
  - VectorStore (ChromaDB)
  - KnowledgeGraphService (LightRAG)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.models.candidate import CandidateCV, CVStatus, Candidate
from app.models.job import JobPosting, JobStatus
from app.services.deep_document_parser import DeepDocumentParser
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.embedder import EmbeddingService, get_embedding_service
from app.services.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

CV_COLLECTION = "cv_chunks"
JD_COLLECTION = "jd_chunks"


class _NamedVectorStore(VectorStore):
    """VectorStore subclass that uses a fixed collection name instead of kb_{id}."""

    def __init__(self, name: str):
        self.workspace_id = 0
        self.collection_name = name
        self._collection = None


def _get_cv_vector_store() -> VectorStore:
    """ChromaDB collection for CV chunks (shared across all candidates)."""
    return _NamedVectorStore(CV_COLLECTION)


def _get_jd_vector_store() -> VectorStore:
    """ChromaDB collection for JD chunks (shared across all jobs)."""
    return _NamedVectorStore(JD_COLLECTION)


# Shared KG workspace ID for the job platform skills graph
_KG_WORKSPACE_ID = 9999


class JobProcessingService:
    """
    Processes CVs and JDs through the NexusRAG pipeline with job-specific
    structured extraction.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.parser = DeepDocumentParser(workspace_id=_KG_WORKSPACE_ID)
        self.embedder = get_embedding_service()
        self.cv_store = _get_cv_vector_store()
        self.jd_store = _get_jd_vector_store()

        self.kg_service: Optional[KnowledgeGraphService] = None
        if settings.NEXUSRAG_ENABLE_KG:
            self.kg_service = KnowledgeGraphService(workspace_id=_KG_WORKSPACE_ID)

    # ------------------------------------------------------------------
    # CV Processing
    # ------------------------------------------------------------------

    async def process_cv(self, cv_id: int, file_path: str) -> int:
        """
        Process a CV through the full pipeline.

        Returns number of chunks created.
        """
        result = await self.db.execute(
            select(CandidateCV).where(CandidateCV.id == cv_id)
        )
        cv = result.scalar_one_or_none()
        if cv is None:
            raise ValueError(f"CandidateCV {cv_id} not found")

        start_time = time.time()

        try:
            # Phase 1: PARSING
            cv.status = CVStatus.PARSING
            await self.db.commit()

            parsed = self.parser.parse(
                file_path=file_path,
                document_id=cv_id,
                original_filename=cv.original_filename,
            )

            cv.markdown_content = parsed.markdown
            cv.page_count = parsed.page_count
            await self.db.commit()

            # Phase 2: LLM Structured Extraction
            profile = await self._extract_candidate_profile(parsed.markdown)
            cv.skills_extracted = profile.get("skills", [])
            cv.experience_extracted = profile.get("experience", [])
            cv.education_extracted = profile.get("education", [])
            cv.summary_extracted = profile.get("summary", "")

            # Also update candidate record with extracted data
            cand_result = await self.db.execute(
                select(Candidate).where(Candidate.id == cv.candidate_id)
            )
            candidate = cand_result.scalar_one_or_none()
            if candidate and profile:
                if profile.get("desired_role") and not candidate.desired_role:
                    candidate.desired_role = profile["desired_role"]
                if profile.get("experience_years") and not candidate.experience_years:
                    candidate.experience_years = profile["experience_years"]
                if profile.get("education_level") and not candidate.education_level:
                    candidate.education_level = profile["education_level"]
                if profile.get("location") and not candidate.location:
                    candidate.location = profile["location"]
            await self.db.commit()

            # Phase 3: INDEXING — embed chunks into cv_chunks collection
            cv.status = CVStatus.INDEXING
            await self.db.commit()

            chunk_count = 0
            if parsed.chunks:
                chunk_texts = [c.content for c in parsed.chunks]
                embeddings = self.embedder.embed_texts(chunk_texts)

                ids = [f"cv_{cv_id}_chunk_{i}" for i in range(len(parsed.chunks))]
                metadatas = [
                    {
                        "cv_id": cv_id,
                        "candidate_id": cv.candidate_id,
                        "chunk_index": c.chunk_index,
                        "source": c.source_file,
                        "page_no": c.page_no,
                        "heading_path": " > ".join(c.heading_path) if c.heading_path else "",
                        "section_type": self._detect_section_type(c.heading_path, c.content),
                        "skills": ",".join(cv.skills_extracted or []),
                    }
                    for c in parsed.chunks
                ]

                self.cv_store.add_documents(
                    ids=ids,
                    embeddings=embeddings,
                    documents=chunk_texts,
                    metadatas=metadatas,
                )
                chunk_count = len(parsed.chunks)

            # Phase 4: KG ingest
            if self.kg_service and parsed.markdown:
                try:
                    await self.kg_service.ingest(parsed.markdown)
                except Exception as e:
                    logger.error(f"KG ingest failed for CV {cv_id}: {e}")

            # Done
            elapsed_ms = int((time.time() - start_time) * 1000)
            cv.status = CVStatus.INDEXED
            cv.chunk_count = chunk_count
            cv.processing_time_ms = elapsed_ms
            await self.db.commit()

            logger.info(
                f"Processed CV {cv_id}: {chunk_count} chunks, "
                f"skills={cv.skills_extracted}, in {elapsed_ms}ms"
            )
            return chunk_count

        except Exception as e:
            logger.error(f"Failed to process CV {cv_id}: {e}")
            cv.status = CVStatus.FAILED
            cv.error_message = str(e)[:500]
            await self.db.commit()
            raise

    # ------------------------------------------------------------------
    # JD Processing
    # ------------------------------------------------------------------

    async def process_job(self, job_id: int, file_path: str | None = None) -> int:
        """
        Process a job description. If file_path is provided, parse the file;
        otherwise use the description_text field directly.

        Returns number of chunks created.
        """
        result = await self.db.execute(
            select(JobPosting).where(JobPosting.id == job_id)
        )
        job = result.scalar_one_or_none()
        if job is None:
            raise ValueError(f"JobPosting {job_id} not found")

        start_time = time.time()

        try:
            markdown = ""

            if file_path and DeepDocumentParser.is_docling_supported(file_path):
                parsed = self.parser.parse(
                    file_path=file_path,
                    document_id=job_id,
                    original_filename=job.original_filename or f"job_{job_id}",
                )
                markdown = parsed.markdown
                chunks_list = parsed.chunks
            else:
                # Text-based JD — use description_text directly
                text = job.description_text or ""
                if not text.strip():
                    job.error_message = "No description text provided"
                    await self.db.commit()
                    return 0

                markdown = text
                # Create simple chunks from text using legacy chunker
                from app.services.chunker import DocumentChunker
                chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
                raw_chunks = chunker.split_text(
                    text=text,
                    source=job.title,
                    extra_metadata={"job_id": job_id},
                )
                from types import SimpleNamespace
                chunks_list = [
                    SimpleNamespace(
                        content=c.content,
                        chunk_index=c.chunk_index,
                        source_file=job.title,
                        document_id=job_id,
                        page_no=0,
                        heading_path=[],
                        image_refs=[],
                        table_refs=[],
                        has_table=False,
                        has_code=False,
                    )
                    for c in raw_chunks
                ]

            job.markdown_content = markdown

            # LLM structured extraction for JD
            jd_profile = await self._extract_job_profile(markdown)
            if jd_profile.get("skills_required") and not job.skills_required:
                job.skills_required = jd_profile["skills_required"]
            if jd_profile.get("skills_nice_to_have") and not job.skills_nice_to_have:
                job.skills_nice_to_have = jd_profile["skills_nice_to_have"]
            if jd_profile.get("experience_required") and not job.experience_required:
                job.experience_required = jd_profile["experience_required"]
            if jd_profile.get("location") and not job.location:
                job.location = jd_profile["location"]
            await self.db.commit()

            # Embed and index into jd_chunks
            chunk_count = 0
            if chunks_list:
                chunk_texts = [c.content for c in chunks_list]
                embeddings = self.embedder.embed_texts(chunk_texts)

                ids = [f"jd_{job_id}_chunk_{i}" for i in range(len(chunks_list))]
                metadatas = [
                    {
                        "job_id": job_id,
                        "company_id": job.company_id,
                        "chunk_index": getattr(c, "chunk_index", i),
                        "source": getattr(c, "source_file", job.title),
                        "page_no": getattr(c, "page_no", 0),
                        "heading_path": " > ".join(getattr(c, "heading_path", []) or []),
                        "skills": ",".join(job.skills_required or []),
                    }
                    for i, c in enumerate(chunks_list)
                ]

                self.jd_store.add_documents(
                    ids=ids,
                    embeddings=embeddings,
                    documents=chunk_texts,
                    metadatas=metadatas,
                )
                chunk_count = len(chunks_list)

            # KG ingest
            if self.kg_service and markdown:
                try:
                    await self.kg_service.ingest(markdown)
                except Exception as e:
                    logger.error(f"KG ingest failed for job {job_id}: {e}")

            elapsed_ms = int((time.time() - start_time) * 1000)
            job.chunk_count = chunk_count
            job.processing_time_ms = elapsed_ms
            job.status = JobStatus.OPEN
            await self.db.commit()

            logger.info(
                f"Processed JD {job_id}: {chunk_count} chunks, "
                f"skills={job.skills_required}, in {elapsed_ms}ms"
            )
            return chunk_count

        except Exception as e:
            logger.error(f"Failed to process JD {job_id}: {e}")
            job.error_message = str(e)[:500]
            await self.db.commit()
            raise

    # ------------------------------------------------------------------
    # LLM Structured Extraction
    # ------------------------------------------------------------------

    async def _extract_candidate_profile(self, markdown: str) -> dict:
        """Use LLM to extract structured fields from a CV."""
        from app.services.llm import get_llm_provider
        from app.services.llm.types import LLMMessage

        provider = get_llm_provider()

        prompt = (
            "You are a CV/resume parser. Extract the following from this CV and return ONLY valid JSON.\n\n"
            "Required JSON structure:\n"
            "{\n"
            '  "skills": ["skill1", "skill2", ...],\n'
            '  "experience": [{"company": "...", "role": "...", "duration": "...", "description": "..."}],\n'
            '  "education": [{"school": "...", "degree": "...", "field": "...", "year": "..."}],\n'
            '  "summary": "Brief 2-3 sentence professional summary",\n'
            '  "desired_role": "Most likely target job title based on experience",\n'
            '  "experience_years": 0,\n'
            '  "education_level": "Bachelor/Master/PhD/Other",\n'
            '  "location": "City or region if mentioned"\n'
            "}\n\n"
            "CV Content:\n"
            f"{markdown[:8000]}"
        )

        try:
            result = await provider.acomplete(
                [LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=2048,
            )
            return self._parse_json_response(result)
        except Exception as e:
            logger.error(f"LLM extraction failed for CV: {e}")
            return {}

    async def _extract_job_profile(self, markdown: str) -> dict:
        """Use LLM to extract structured fields from a JD."""
        from app.services.llm import get_llm_provider
        from app.services.llm.types import LLMMessage

        provider = get_llm_provider()

        prompt = (
            "You are a job description parser. Extract the following from this job posting and return ONLY valid JSON.\n\n"
            "Required JSON structure:\n"
            "{\n"
            '  "skills_required": ["skill1", "skill2", ...],\n'
            '  "skills_nice_to_have": ["skill1", "skill2", ...],\n'
            '  "experience_required": 0,\n'
            '  "location": "City or region",\n'
            '  "salary_range": "if mentioned",\n'
            '  "employment_type": "full_time/part_time/contract/internship/remote"\n'
            "}\n\n"
            "Job Description:\n"
            f"{markdown[:8000]}"
        )

        try:
            result = await provider.acomplete(
                [LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=1024,
            )
            return self._parse_json_response(result)
        except Exception as e:
            logger.error(f"LLM extraction failed for JD: {e}")
            return {}

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Extract JSON from LLM response (handles markdown code fences)."""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1]
            if "```" in text:
                text = text.split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1]
            if "```" in text:
                text = text.split("```", 1)[0]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {}

    @staticmethod
    def _detect_section_type(heading_path: list[str], content: str) -> str:
        """Heuristic to classify chunk section type for metadata filtering."""
        text = (" ".join(heading_path) + " " + content[:200]).lower()
        if any(kw in text for kw in ["experience", "kinh nghiệm", "work history", "employment"]):
            return "experience"
        if any(kw in text for kw in ["education", "học vấn", "degree", "university", "trường"]):
            return "education"
        if any(kw in text for kw in ["skill", "kỹ năng", "technology", "tool", "framework"]):
            return "skills"
        if any(kw in text for kw in ["summary", "tóm tắt", "objective", "mục tiêu", "profile"]):
            return "summary"
        if any(kw in text for kw in ["project", "dự án"]):
            return "projects"
        if any(kw in text for kw in ["certificate", "chứng chỉ", "certification"]):
            return "certifications"
        return "other"

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_cv_chunks(self, cv_id: int) -> None:
        """Delete CV chunks from vector store."""
        try:
            self.cv_store.collection.delete(where={"cv_id": cv_id})
        except Exception as e:
            logger.warning(f"Failed to delete CV chunks for cv_id={cv_id}: {e}")

    def delete_job_chunks(self, job_id: int) -> None:
        """Delete JD chunks from vector store."""
        try:
            self.jd_store.collection.delete(where={"job_id": job_id})
        except Exception as e:
            logger.warning(f"Failed to delete JD chunks for job_id={job_id}: {e}")
