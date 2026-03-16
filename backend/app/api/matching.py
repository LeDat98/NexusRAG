"""
Matching API endpoints — explicit match triggers and explanation.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.deps import get_db
from app.models.match import MatchResult
from app.schemas.matching import (
    MatchRequest,
    MatchResultResponse,
    ScoreBreakdown,
    MatchExplanation,
    MatchExplainResponse,
    CandidateToJobsResponse,
    JobToCandidatesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/match", tags=["matching"])


@router.post("/candidate-to-jobs/{candidate_id}", response_model=CandidateToJobsResponse)
async def match_candidate_to_jobs(
    candidate_id: int,
    body: MatchRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Find best job matches for a candidate."""
    from app.services.matching_service import MatchingService

    req = body or MatchRequest()
    svc = MatchingService(db)
    try:
        matches = await svc.match_candidate_to_jobs(
            candidate_id, top_k=req.top_k, min_score=req.min_score,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return CandidateToJobsResponse(
        candidate_id=candidate_id,
        total=len(matches),
        matches=[_to_response(m) for m in matches],
    )


@router.post("/job-to-candidates/{job_id}", response_model=JobToCandidatesResponse)
async def match_job_to_candidates(
    job_id: int,
    body: MatchRequest | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Find best candidate matches for a job."""
    from app.services.matching_service import MatchingService

    req = body or MatchRequest()
    svc = MatchingService(db)
    try:
        matches = await svc.match_job_to_candidates(
            job_id, top_k=req.top_k, min_score=req.min_score,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return JobToCandidatesResponse(
        job_id=job_id,
        total=len(matches),
        matches=[_to_response(m) for m in matches],
    )


@router.get("/explain/{match_id}", response_model=MatchExplainResponse)
async def explain_match(
    match_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get detailed explanation of a match."""
    result = await db.execute(
        select(MatchResult).where(MatchResult.id == match_id)
    )
    match = result.scalar_one_or_none()
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")

    # Parse explanation string into structured format
    parts = (match.explanation or "").split(" | ")
    exp_fit = ""
    sal_fit = ""
    loc_fit = ""
    for p in parts:
        if p.startswith("Experience:"):
            exp_fit = p
        elif p.startswith("Salary"):
            sal_fit = p
        elif p.startswith("Location:"):
            loc_fit = p

    return MatchExplainResponse(
        match_id=match.id,
        candidate_id=match.candidate_id,
        job_id=match.job_id,
        overall_score=match.overall_score,
        scores=ScoreBreakdown(
            semantic_score=match.semantic_score,
            skill_match_score=match.skill_match_score,
            experience_score=match.experience_score,
            location_score=match.location_score,
            salary_score=match.salary_score,
        ),
        explanation=MatchExplanation(
            matched_skills=match.matched_skills or [],
            missing_skills=match.missing_skills or [],
            experience_fit=exp_fit,
            salary_fit=sal_fit,
            location_fit=loc_fit,
        ),
    )


@router.get("/history/{candidate_id}")
async def match_history(
    candidate_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get all past match results for a candidate."""
    result = await db.execute(
        select(MatchResult)
        .where(MatchResult.candidate_id == candidate_id)
        .order_by(MatchResult.overall_score.desc())
    )
    matches = result.scalars().all()
    return {
        "candidate_id": candidate_id,
        "total": len(matches),
        "matches": [_to_response(m) for m in matches],
    }


def _to_response(m: MatchResult) -> MatchResultResponse:
    return MatchResultResponse(
        id=m.id,
        candidate_id=m.candidate_id,
        job_id=m.job_id,
        overall_score=m.overall_score,
        scores=ScoreBreakdown(
            semantic_score=m.semantic_score,
            skill_match_score=m.skill_match_score,
            experience_score=m.experience_score,
            location_score=m.location_score,
            salary_score=m.salary_score,
        ),
        matched_skills=m.matched_skills,
        missing_skills=m.missing_skills,
        explanation=m.explanation,
        status=m.status,
        created_at=m.created_at,
    )
