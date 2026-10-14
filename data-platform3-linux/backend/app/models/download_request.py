"""A saved download query and its explicitly confirmed generation jobs."""
from datetime import datetime
from sqlalchemy import DateTime, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class DownloadRequest(Base):
    __tablename__ = 'download_requests'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_by: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    query_json: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_file_ids: Mapped[str] = mapped_column(Text, default='[]')
    matched_count: Mapped[int] = mapped_column(Integer, default=0)
    generation_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
