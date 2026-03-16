"""
Candidate & CV schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from datetime import datetime

from app.models.candidate import CandidateStatus, CVStatus


class CandidateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    desired_role: str | None = None
    desired_salary_min: float | None = None
    desired_salary_max: float | None = None
    experience_years: float | None = None
    education_level: str | None = None


class CandidateUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    desired_role: str | None = None
    desired_salary_min: float | None = None
    desired_salary_max: float | None = None
    experience_years: float | None = None
    education_level: str | None = None
    status: CandidateStatus | None = None


class CandidateResponse(BaseModel):
    id: int
    name: str
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    desired_role: str | None = None
    desired_salary_min: float | None = None
    desired_salary_max: float | None = None
    experience_years: float | None = None
    education_level: str | None = None
    status: CandidateStatus
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CVUploadResponse(BaseModel):
    id: int
    candidate_id: int
    filename: str
    status: CVStatus
    message: str


class CVResponse(BaseModel):
    id: int
    candidate_id: int
    original_filename: str
    file_type: str
    file_size: int
    status: CVStatus
    chunk_count: int = 0
    page_count: int = 0
    skills_extracted: list | None = None
    experience_extracted: list | None = None
    education_extracted: list | None = None
    summary_extracted: str | None = None
    processing_time_ms: int = 0
    error_message: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class CandidateDetailResponse(CandidateResponse):
    cvs: list[CVResponse] = []
