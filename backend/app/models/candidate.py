"""
Candidate and CV models for the JobNexus recommendation engine.
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import String, ForeignKey, DateTime, Integer, Text, Enum, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class CandidateStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class CVStatus(str, enum.Enum):
    PENDING = "pending"
    PARSING = "parsing"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    desired_role: Mapped[str | None] = mapped_column(String(255), nullable=True)
    desired_salary_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    desired_salary_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    experience_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    education_level: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[CandidateStatus] = mapped_column(
        Enum(CandidateStatus), default=CandidateStatus.ACTIVE
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    cvs: Mapped[list["CandidateCV"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )
    matches: Mapped[list["MatchResult"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan",
        foreign_keys="MatchResult.candidate_id",
    )


class CandidateCV(Base):
    __tablename__ = "candidate_cvs"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    original_filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(50))
    file_size: Mapped[int] = mapped_column(Integer)
    status: Mapped[CVStatus] = mapped_column(
        Enum(CVStatus), default=CVStatus.PENDING
    )
    markdown_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)

    skills_extracted: Mapped[list | None] = mapped_column(JSON, nullable=True)
    experience_extracted: Mapped[list | None] = mapped_column(JSON, nullable=True)
    education_extracted: Mapped[list | None] = mapped_column(JSON, nullable=True)
    summary_extracted: Mapped[str | None] = mapped_column(Text, nullable=True)

    processing_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    candidate: Mapped["Candidate"] = relationship(back_populates="cvs")
