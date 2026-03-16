"""
Candidate & CV API endpoints.
"""
from __future__ import annotations

import os
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.deps import get_db
from app.models.candidate import Candidate, CandidateCV, CandidateStatus, CVStatus
from app.schemas.candidate import (
    CandidateCreate,
    CandidateUpdate,
    CandidateResponse,
    CandidateDetailResponse,
    CVResponse,
    CVUploadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/candidates", tags=["candidates"])

UPLOAD_DIR = settings.BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE = 50 * 1024 * 1024


# ------------------------------------------------------------------
# Candidate CRUD
# ------------------------------------------------------------------

@router.post("", response_model=CandidateResponse, status_code=status.HTTP_201_CREATED)
async def create_candidate(
    body: CandidateCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new candidate profile."""
    candidate = Candidate(
        name=body.name,
        email=body.email,
        phone=body.phone,
        location=body.location,
        desired_role=body.desired_role,
        desired_salary_min=body.desired_salary_min,
        desired_salary_max=body.desired_salary_max,
        experience_years=body.experience_years,
        education_level=body.education_level,
    )
    db.add(candidate)
    await db.commit()
    await db.refresh(candidate)
    return candidate


@router.get("", response_model=list[CandidateResponse])
async def list_candidates(
    db: AsyncSession = Depends(get_db),
    status_filter: CandidateStatus | None = None,
):
    """List all candidates."""
    query = select(Candidate).order_by(Candidate.created_at.desc())
    if status_filter:
        query = query.where(Candidate.status == status_filter)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{candidate_id}", response_model=CandidateDetailResponse)
async def get_candidate(
    candidate_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get candidate detail with CVs."""
    result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    candidate = result.scalar_one_or_none()
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")

    cv_result = await db.execute(
        select(CandidateCV)
        .where(CandidateCV.candidate_id == candidate_id)
        .order_by(CandidateCV.created_at.desc())
    )
    cvs = cv_result.scalars().all()

    return CandidateDetailResponse(
        **{c.key: getattr(candidate, c.key) for c in Candidate.__table__.columns},
        cvs=[CVResponse.model_validate(cv) for cv in cvs],
    )


@router.put("/{candidate_id}", response_model=CandidateResponse)
async def update_candidate(
    candidate_id: int,
    body: CandidateUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update candidate profile."""
    result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    candidate = result.scalar_one_or_none()
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(candidate, field, value)

    await db.commit()
    await db.refresh(candidate)
    return candidate


@router.delete("/{candidate_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_candidate(
    candidate_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a candidate and all associated CVs/chunks."""
    result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    candidate = result.scalar_one_or_none()
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Clean up vector store chunks for all CVs
    cv_result = await db.execute(
        select(CandidateCV).where(CandidateCV.candidate_id == candidate_id)
    )
    for cv in cv_result.scalars().all():
        try:
            from app.services.job_processing_service import JobProcessingService
            svc = JobProcessingService(db)
            svc.delete_cv_chunks(cv.id)
        except Exception:
            pass
        file_path = UPLOAD_DIR / cv.filename
        if file_path.exists():
            os.remove(file_path)

    await db.delete(candidate)
    await db.commit()


# ------------------------------------------------------------------
# CV Upload & Processing
# ------------------------------------------------------------------

@router.post("/{candidate_id}/upload-cv", response_model=CVUploadResponse)
async def upload_cv(
    candidate_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a CV file for a candidate."""
    result = await db.execute(
        select(Candidate).where(Candidate.id == candidate_id)
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Candidate not found")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    filename = f"{uuid.uuid4()}{ext}"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        f.write(content)

    cv = CandidateCV(
        candidate_id=candidate_id,
        filename=filename,
        original_filename=file.filename,
        file_type=ext[1:],
        file_size=len(content),
        status=CVStatus.PENDING,
    )
    db.add(cv)
    await db.commit()
    await db.refresh(cv)

    return CVUploadResponse(
        id=cv.id,
        candidate_id=candidate_id,
        filename=cv.original_filename,
        status=cv.status,
        message="CV uploaded. Trigger processing to extract and index.",
    )


@router.post("/{candidate_id}/process/{cv_id}")
async def process_cv(
    candidate_id: int,
    cv_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Trigger CV processing (parse + extract + index)."""
    result = await db.execute(
        select(CandidateCV).where(
            CandidateCV.id == cv_id,
            CandidateCV.candidate_id == candidate_id,
        )
    )
    cv = result.scalar_one_or_none()
    if cv is None:
        raise HTTPException(status_code=404, detail="CV not found")

    if cv.status == CVStatus.INDEXED:
        return {"status": "already_indexed", "cv_id": cv_id, "chunk_count": cv.chunk_count}

    if cv.status in (CVStatus.PARSING, CVStatus.INDEXING):
        raise HTTPException(status_code=400, detail="CV is already being processed")

    file_path = UPLOAD_DIR / cv.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="CV file not found on disk")

    # Background processing
    import asyncio
    from app.core.database import async_session_maker

    async def _process():
        async with async_session_maker() as session:
            from app.services.job_processing_service import JobProcessingService
            svc = JobProcessingService(session)
            await svc.process_cv(cv_id, str(file_path))

    asyncio.get_event_loop().create_task(_process())

    return {"status": "processing", "cv_id": cv_id, "message": "CV processing started"}


@router.get("/{candidate_id}/recommendations")
async def get_candidate_recommendations(
    candidate_id: int,
    top_k: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """Get job recommendations for a candidate (triggers fresh matching)."""
    from app.services.matching_service import MatchingService

    svc = MatchingService(db)
    try:
        matches = await svc.match_candidate_to_jobs(candidate_id, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "candidate_id": candidate_id,
        "total": len(matches),
        "matches": [
            {
                "id": m.id,
                "job_id": m.job_id,
                "overall_score": m.overall_score,
                "semantic_score": m.semantic_score,
                "skill_match_score": m.skill_match_score,
                "experience_score": m.experience_score,
                "location_score": m.location_score,
                "salary_score": m.salary_score,
                "matched_skills": m.matched_skills,
                "missing_skills": m.missing_skills,
                "explanation": m.explanation,
                "status": m.status.value,
            }
            for m in matches
        ],
    }
