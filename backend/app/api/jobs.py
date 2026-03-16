"""
Company & Job Posting API endpoints.
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
from app.models.job import Company, JobPosting, JobStatus, EmploymentType
from app.schemas.job import (
    CompanyCreate,
    CompanyResponse,
    JobPostingCreate,
    JobPostingUpdate,
    JobPostingResponse,
    JobPostingDetailResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

UPLOAD_DIR = settings.BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


# ------------------------------------------------------------------
# Company CRUD
# ------------------------------------------------------------------

@router.post("/companies", response_model=CompanyResponse, status_code=status.HTTP_201_CREATED)
async def create_company(
    body: CompanyCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new company."""
    company = Company(
        name=body.name,
        industry=body.industry,
        location=body.location,
        size=body.size,
        description=body.description,
    )
    db.add(company)
    await db.commit()
    await db.refresh(company)
    return company


@router.get("/companies", response_model=list[CompanyResponse])
async def list_companies(db: AsyncSession = Depends(get_db)):
    """List all companies."""
    result = await db.execute(select(Company).order_by(Company.name))
    return result.scalars().all()


@router.get("/companies/{company_id}", response_model=CompanyResponse)
async def get_company(
    company_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Company).where(Company.id == company_id))
    company = result.scalar_one_or_none()
    if company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return company


# ------------------------------------------------------------------
# Job Posting CRUD
# ------------------------------------------------------------------

@router.post("", response_model=JobPostingResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    body: JobPostingCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new job posting with structured fields."""
    # Verify company exists
    result = await db.execute(select(Company).where(Company.id == body.company_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Company not found")

    job = JobPosting(
        company_id=body.company_id,
        title=body.title,
        description_text=body.description_text,
        location=body.location,
        salary_min=body.salary_min,
        salary_max=body.salary_max,
        experience_required=body.experience_required,
        skills_required=body.skills_required,
        skills_nice_to_have=body.skills_nice_to_have,
        employment_type=body.employment_type,
        status=JobStatus.DRAFT,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


@router.get("", response_model=list[JobPostingResponse])
async def list_jobs(
    db: AsyncSession = Depends(get_db),
    status_filter: JobStatus | None = None,
    location: str | None = None,
):
    """List job postings with optional filters."""
    query = select(JobPosting).order_by(JobPosting.created_at.desc())
    if status_filter:
        query = query.where(JobPosting.status == status_filter)
    if location:
        query = query.where(JobPosting.location.ilike(f"%{location}%"))
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{job_id}", response_model=JobPostingDetailResponse)
async def get_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get job posting detail with company info."""
    result = await db.execute(select(JobPosting).where(JobPosting.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    comp_result = await db.execute(
        select(Company).where(Company.id == job.company_id)
    )
    company = comp_result.scalar_one_or_none()

    return JobPostingDetailResponse(
        **{c.key: getattr(job, c.key) for c in JobPosting.__table__.columns},
        company=CompanyResponse.model_validate(company) if company else None,
    )


@router.put("/{job_id}", response_model=JobPostingResponse)
async def update_job(
    job_id: int,
    body: JobPostingUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a job posting."""
    result = await db.execute(select(JobPosting).where(JobPosting.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(job, field, value)

    await db.commit()
    await db.refresh(job)
    return job


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a job posting and its chunks."""
    result = await db.execute(select(JobPosting).where(JobPosting.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        from app.services.job_processing_service import JobProcessingService
        svc = JobProcessingService(db)
        svc.delete_job_chunks(job_id)
    except Exception:
        pass

    if job.filename:
        file_path = UPLOAD_DIR / job.filename
        if file_path.exists():
            os.remove(file_path)

    await db.delete(job)
    await db.commit()


# ------------------------------------------------------------------
# JD Upload & Processing
# ------------------------------------------------------------------

@router.post("/{job_id}/upload-jd")
async def upload_jd(
    job_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a JD file for a job posting."""
    result = await db.execute(select(JobPosting).where(JobPosting.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type {ext} not allowed")

    content = await file.read()
    filename = f"{uuid.uuid4()}{ext}"
    file_path = UPLOAD_DIR / filename
    with open(file_path, "wb") as f:
        f.write(content)

    job.filename = filename
    job.original_filename = file.filename
    job.file_type = ext[1:]
    job.file_size = len(content)
    await db.commit()

    return {"status": "uploaded", "job_id": job_id, "filename": file.filename}


@router.post("/{job_id}/process")
async def process_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Trigger JD processing (parse + extract + index)."""
    result = await db.execute(select(JobPosting).where(JobPosting.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    file_path = None
    if job.filename:
        fp = UPLOAD_DIR / job.filename
        if fp.exists():
            file_path = str(fp)

    if not file_path and not job.description_text:
        raise HTTPException(
            status_code=400,
            detail="Job has no file and no description text to process",
        )

    import asyncio
    from app.core.database import async_session_maker

    async def _process():
        async with async_session_maker() as session:
            from app.services.job_processing_service import JobProcessingService
            svc = JobProcessingService(session)
            await svc.process_job(job_id, file_path)

    asyncio.get_event_loop().create_task(_process())

    return {"status": "processing", "job_id": job_id, "message": "JD processing started"}


@router.get("/{job_id}/candidates")
async def get_job_candidates(
    job_id: int,
    top_k: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """Get candidate recommendations for a job (triggers fresh matching)."""
    from app.services.matching_service import MatchingService

    svc = MatchingService(db)
    try:
        matches = await svc.match_job_to_candidates(job_id, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "job_id": job_id,
        "total": len(matches),
        "matches": [
            {
                "id": m.id,
                "candidate_id": m.candidate_id,
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
