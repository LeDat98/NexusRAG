"""
Company & JobPosting schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from datetime import datetime

from app.models.job import EmploymentType, JobStatus


class CompanyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    industry: str | None = None
    location: str | None = None
    size: str | None = None
    description: str | None = None


class CompanyResponse(BaseModel):
    id: int
    name: str
    industry: str | None = None
    location: str | None = None
    size: str | None = None
    description: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class JobPostingCreate(BaseModel):
    company_id: int
    title: str = Field(..., min_length=1, max_length=255)
    description_text: str | None = None
    location: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    experience_required: float | None = None
    skills_required: list[str] | None = None
    skills_nice_to_have: list[str] | None = None
    employment_type: EmploymentType = EmploymentType.FULL_TIME


class JobPostingUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description_text: str | None = None
    location: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    experience_required: float | None = None
    skills_required: list[str] | None = None
    skills_nice_to_have: list[str] | None = None
    employment_type: EmploymentType | None = None
    status: JobStatus | None = None


class JobPostingResponse(BaseModel):
    id: int
    company_id: int
    title: str
    description_text: str | None = None
    location: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    experience_required: float | None = None
    skills_required: list[str] | None = None
    skills_nice_to_have: list[str] | None = None
    employment_type: EmploymentType
    status: JobStatus
    chunk_count: int = 0
    processing_time_ms: int = 0
    original_filename: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class JobPostingDetailResponse(JobPostingResponse):
    company: CompanyResponse | None = None
    markdown_content: str | None = None
