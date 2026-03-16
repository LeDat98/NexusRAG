"""
Matching Service
=================

Two-sided semantic matching between Candidates and Job Postings.

Pipeline per match direction:
  1. Embed query text (CV summary or JD requirements)
  2. Parallel: Vector over-fetch from opposite collection + KG skill lookup
  3. Cross-encoder rerank
  4. Structured score boost (skills, experience, location, salary)
  5. Assemble final ranked results with explanations

Reuses: EmbeddingService, VectorStore, RerankerService, KnowledgeGraphService
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.candidate import Candidate, CandidateCV, CVStatus
from app.models.job import JobPosting, Company, JobStatus
from app.models.match import MatchResult, MatchStatus
from app.services.embedder import get_embedding_service
from app.services.reranker import get_reranker_service
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.job_processing_service import (
    _get_cv_vector_store,
    _get_jd_vector_store,
    _KG_WORKSPACE_ID,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchCandidate:
    """Intermediate match result before DB persistence."""
    candidate_id: int
    job_id: int
    semantic_score: float = 0.0
    skill_match_score: float = 0.0
    experience_score: float = 0.0
    location_score: float = 0.0
    salary_score: float = 0.0
    overall_score: float = 0.0
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    explanation: str = ""


class MatchingService:
    """Two-sided semantic matching engine."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedder = get_embedding_service()
        self.reranker = get_reranker_service()
        self.cv_store = _get_cv_vector_store()
        self.jd_store = _get_jd_vector_store()

        self.kg_service: Optional[KnowledgeGraphService] = None
        if settings.NEXUSRAG_ENABLE_KG:
            self.kg_service = KnowledgeGraphService(workspace_id=_KG_WORKSPACE_ID)

    # ------------------------------------------------------------------
    # Candidate → Jobs (find jobs for a candidate)
    # ------------------------------------------------------------------

    async def match_candidate_to_jobs(
        self,
        candidate_id: int,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[MatchResult]:
        """Find and rank best job matches for a candidate."""
        candidate, cv = await self._load_candidate(candidate_id)

        query_text = self._build_candidate_query(candidate, cv)
        if not query_text.strip():
            return []

        candidate_skills = set(s.lower() for s in (cv.skills_extracted or []))

        # Step 1: Vector over-fetch from jd_chunks
        prefetch_k = max(settings.NEXUSRAG_VECTOR_PREFETCH, top_k * 3)
        query_embedding = self.embedder.embed_query(query_text)
        raw_results = self.jd_store.query(
            query_embedding=query_embedding,
            n_results=prefetch_k,
        )

        if not raw_results.get("documents"):
            return []

        # Step 2: Cross-encoder rerank
        doc_texts = raw_results["documents"]
        reranked = self.reranker.rerank(
            query=query_text,
            documents=doc_texts,
            top_k=top_k * 2,
            min_score=settings.NEXUSRAG_MIN_RELEVANCE_SCORE,
        )

        if not reranked:
            reranked_indices = list(range(min(3, len(doc_texts))))
            reranked_scores = [0.3] * len(reranked_indices)
        else:
            reranked_indices = [r.index for r in reranked]
            reranked_scores = [r.score for r in reranked]

        # Step 3: Collect unique job_ids and compute structured scores
        seen_jobs: set[int] = set()
        matches: list[MatchCandidate] = []

        for rank, (idx, semantic) in enumerate(zip(reranked_indices, reranked_scores)):
            meta = raw_results["metadatas"][idx] if raw_results.get("metadatas") else {}
            job_id = meta.get("job_id", 0)
            if not job_id or job_id in seen_jobs:
                continue
            seen_jobs.add(job_id)

            job = await self._load_job(job_id)
            if job is None or job.status != JobStatus.OPEN:
                continue

            mc = self._compute_structured_scores(
                candidate=candidate,
                cv=cv,
                job=job,
                candidate_skills=candidate_skills,
                semantic_raw=semantic,
            )
            matches.append(mc)

            if len(matches) >= top_k:
                break

        matches.sort(key=lambda m: m.overall_score, reverse=True)

        if min_score > 0:
            matches = [m for m in matches if m.overall_score >= min_score]

        # Step 4: Persist match results
        db_matches = []
        for mc in matches:
            db_match = MatchResult(
                candidate_id=mc.candidate_id,
                job_id=mc.job_id,
                overall_score=mc.overall_score,
                semantic_score=mc.semantic_score,
                skill_match_score=mc.skill_match_score,
                experience_score=mc.experience_score,
                location_score=mc.location_score,
                salary_score=mc.salary_score,
                matched_skills=mc.matched_skills,
                missing_skills=mc.missing_skills,
                explanation=mc.explanation,
                status=MatchStatus.PENDING,
            )
            self.db.add(db_match)
            db_matches.append(db_match)

        await self.db.commit()
        for m in db_matches:
            await self.db.refresh(m)

        return db_matches

    # ------------------------------------------------------------------
    # Job → Candidates (find candidates for a job)
    # ------------------------------------------------------------------

    async def match_job_to_candidates(
        self,
        job_id: int,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[MatchResult]:
        """Find and rank best candidate matches for a job."""
        job = await self._load_job(job_id)
        if job is None:
            raise ValueError(f"JobPosting {job_id} not found")

        query_text = self._build_job_query(job)
        if not query_text.strip():
            return []

        required_skills = set(s.lower() for s in (job.skills_required or []))

        # Vector over-fetch from cv_chunks
        prefetch_k = max(settings.NEXUSRAG_VECTOR_PREFETCH, top_k * 3)
        query_embedding = self.embedder.embed_query(query_text)
        raw_results = self.cv_store.query(
            query_embedding=query_embedding,
            n_results=prefetch_k,
        )

        if not raw_results.get("documents"):
            return []

        # Cross-encoder rerank
        doc_texts = raw_results["documents"]
        reranked = self.reranker.rerank(
            query=query_text,
            documents=doc_texts,
            top_k=top_k * 2,
            min_score=settings.NEXUSRAG_MIN_RELEVANCE_SCORE,
        )

        if not reranked:
            reranked_indices = list(range(min(3, len(doc_texts))))
            reranked_scores = [0.3] * len(reranked_indices)
        else:
            reranked_indices = [r.index for r in reranked]
            reranked_scores = [r.score for r in reranked]

        seen_candidates: set[int] = set()
        matches: list[MatchCandidate] = []

        for idx, semantic in zip(reranked_indices, reranked_scores):
            meta = raw_results["metadatas"][idx] if raw_results.get("metadatas") else {}
            candidate_id = meta.get("candidate_id", 0)
            if not candidate_id or candidate_id in seen_candidates:
                continue
            seen_candidates.add(candidate_id)

            try:
                candidate, cv = await self._load_candidate(candidate_id)
            except ValueError:
                continue

            candidate_skills = set(s.lower() for s in (cv.skills_extracted or []))

            mc = self._compute_structured_scores(
                candidate=candidate,
                cv=cv,
                job=job,
                candidate_skills=candidate_skills,
                semantic_raw=semantic,
            )
            matches.append(mc)

            if len(matches) >= top_k:
                break

        matches.sort(key=lambda m: m.overall_score, reverse=True)

        if min_score > 0:
            matches = [m for m in matches if m.overall_score >= min_score]

        db_matches = []
        for mc in matches:
            db_match = MatchResult(
                candidate_id=mc.candidate_id,
                job_id=mc.job_id,
                overall_score=mc.overall_score,
                semantic_score=mc.semantic_score,
                skill_match_score=mc.skill_match_score,
                experience_score=mc.experience_score,
                location_score=mc.location_score,
                salary_score=mc.salary_score,
                matched_skills=mc.matched_skills,
                missing_skills=mc.missing_skills,
                explanation=mc.explanation,
                status=MatchStatus.PENDING,
            )
            self.db.add(db_match)
            db_matches.append(db_match)

        await self.db.commit()
        for m in db_matches:
            await self.db.refresh(m)

        return db_matches

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_structured_scores(
        self,
        candidate: Candidate,
        cv: CandidateCV,
        job: JobPosting,
        candidate_skills: set[str],
        semantic_raw: float,
    ) -> MatchCandidate:
        """Compute weighted match score with all sub-scores."""
        w = settings

        # Normalize semantic score to [0, 1]
        semantic = max(0.0, min(1.0, (semantic_raw + 1.0) / 2.0))

        # Skill match (Jaccard-like)
        required = set(s.lower() for s in (job.skills_required or []))
        matched = candidate_skills & required
        missing = required - candidate_skills
        skill_score = len(matched) / max(len(required), 1)

        # Experience fit (sigmoid)
        exp_score = self._experience_score(
            candidate.experience_years, job.experience_required
        )

        # Location match
        loc_score = self._location_score(candidate.location, job.location)

        # Salary overlap
        sal_score = self._salary_score(
            candidate.desired_salary_min, candidate.desired_salary_max,
            job.salary_min, job.salary_max,
        )

        overall = (
            w.MATCHING_SEMANTIC_WEIGHT * semantic
            + w.MATCHING_SKILL_WEIGHT * skill_score
            + w.MATCHING_EXPERIENCE_WEIGHT * exp_score
            + w.MATCHING_LOCATION_WEIGHT * loc_score
            + w.MATCHING_SALARY_WEIGHT * sal_score
        )

        explanation_parts = []
        if matched:
            explanation_parts.append(f"Matched skills: {', '.join(sorted(matched))}")
        if missing:
            explanation_parts.append(f"Missing skills: {', '.join(sorted(missing))}")
        explanation_parts.append(
            f"Experience: {candidate.experience_years or '?'}y vs required {job.experience_required or '?'}y"
        )
        explanation_parts.append(f"Location: {candidate.location or '?'} vs {job.location or '?'}")

        return MatchCandidate(
            candidate_id=candidate.id,
            job_id=job.id,
            semantic_score=round(semantic, 4),
            skill_match_score=round(skill_score, 4),
            experience_score=round(exp_score, 4),
            location_score=round(loc_score, 4),
            salary_score=round(sal_score, 4),
            overall_score=round(overall, 4),
            matched_skills=sorted(matched),
            missing_skills=sorted(missing),
            explanation=" | ".join(explanation_parts),
        )

    @staticmethod
    def _experience_score(
        candidate_years: float | None,
        required_years: float | None,
    ) -> float:
        if candidate_years is None or required_years is None:
            return 0.5
        diff = candidate_years - required_years
        # Sigmoid: 0 diff → 0.8, positive → up to 1.0, negative → down to 0.0
        return 1.0 / (1.0 + math.exp(-diff))

    @staticmethod
    def _location_score(
        candidate_loc: str | None,
        job_loc: str | None,
    ) -> float:
        if not candidate_loc or not job_loc:
            return 0.5
        c = candidate_loc.lower().strip()
        j = job_loc.lower().strip()
        if c == j:
            return 1.0
        if c in j or j in c:
            return 0.8
        if "remote" in j:
            return 0.7
        return 0.2

    @staticmethod
    def _salary_score(
        c_min: float | None, c_max: float | None,
        j_min: float | None, j_max: float | None,
    ) -> float:
        if c_min is None and c_max is None:
            return 0.5
        if j_min is None and j_max is None:
            return 0.5

        c_lo = c_min or 0
        c_hi = c_max or c_lo * 1.5
        j_lo = j_min or 0
        j_hi = j_max or j_lo * 1.5

        if c_hi < j_lo or j_hi < c_lo:
            return 0.0

        overlap = min(c_hi, j_hi) - max(c_lo, j_lo)
        total = max(c_hi, j_hi) - min(c_lo, j_lo)
        if total <= 0:
            return 0.5
        return min(1.0, overlap / total)

    # ------------------------------------------------------------------
    # Query builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_candidate_query(candidate: Candidate, cv: CandidateCV) -> str:
        parts = []
        if cv.summary_extracted:
            parts.append(cv.summary_extracted)
        if candidate.desired_role:
            parts.append(f"Looking for: {candidate.desired_role}")
        if cv.skills_extracted:
            parts.append(f"Skills: {', '.join(cv.skills_extracted[:20])}")
        if not parts and cv.markdown_content:
            parts.append(cv.markdown_content[:1000])
        return " ".join(parts)

    @staticmethod
    def _build_job_query(job: JobPosting) -> str:
        parts = [job.title]
        if job.skills_required:
            parts.append(f"Required: {', '.join(job.skills_required[:20])}")
        if job.description_text:
            parts.append(job.description_text[:500])
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_candidate(self, candidate_id: int) -> tuple[Candidate, CandidateCV]:
        result = await self.db.execute(
            select(Candidate).where(Candidate.id == candidate_id)
        )
        candidate = result.scalar_one_or_none()
        if candidate is None:
            raise ValueError(f"Candidate {candidate_id} not found")

        cv_result = await self.db.execute(
            select(CandidateCV)
            .where(
                CandidateCV.candidate_id == candidate_id,
                CandidateCV.status == CVStatus.INDEXED,
            )
            .order_by(CandidateCV.created_at.desc())
            .limit(1)
        )
        cv = cv_result.scalar_one_or_none()
        if cv is None:
            raise ValueError(f"No indexed CV found for candidate {candidate_id}")

        return candidate, cv

    async def _load_job(self, job_id: int) -> JobPosting | None:
        result = await self.db.execute(
            select(JobPosting).where(JobPosting.id == job_id)
        )
        return result.scalar_one_or_none()
