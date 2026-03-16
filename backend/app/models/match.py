"""
MatchResult model — stores computed job-candidate matches with score breakdown.
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import String, ForeignKey, DateTime, Integer, Text, Enum, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class MatchStatus(str, enum.Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class MatchResult(Base):
    __tablename__ = "match_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidates.id", ondelete="CASCADE"), index=True
    )
    job_id: Mapped[int] = mapped_column(
        ForeignKey("job_postings.id", ondelete="CASCADE"), index=True
    )

    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    semantic_score: Mapped[float] = mapped_column(Float, default=0.0)
    skill_match_score: Mapped[float] = mapped_column(Float, default=0.0)
    experience_score: Mapped[float] = mapped_column(Float, default=0.0)
    location_score: Mapped[float] = mapped_column(Float, default=0.0)
    salary_score: Mapped[float] = mapped_column(Float, default=0.0)

    matched_skills: Mapped[list | None] = mapped_column(JSON, nullable=True)
    missing_skills: Mapped[list | None] = mapped_column(JSON, nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[MatchStatus] = mapped_column(
        Enum(MatchStatus), default=MatchStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    candidate: Mapped["Candidate"] = relationship(
        back_populates="matches", foreign_keys=[candidate_id]
    )
    job: Mapped["JobPosting"] = relationship(
        back_populates="matches", foreign_keys=[job_id]
    )
