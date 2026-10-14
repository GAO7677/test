"""少样本生成任务；输入快照、训练和批次溯源由任务目录保存。"""
from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class FewshotJob(Base):
    __tablename__ = "fewshot_jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100))
    project: Mapped[str] = mapped_column(String(100), default="")
    scene: Mapped[str] = mapped_column(String(100), default="")
    secret_level: Mapped[str] = mapped_column(String(10), default="内部")
    status: Mapped[str] = mapped_column(String(30), default="draft", index=True)
    options: Mapped[str] = mapped_column(Text, default="{}")
    groups: Mapped[str] = mapped_column(Text, default="[]")
    job_dir: Mapped[str] = mapped_column(String(500))
    run_dir: Mapped[str] = mapped_column(String(500), default="")
    batch_dir: Mapped[str] = mapped_column(String(500), default="")
    source_data_id: Mapped[Optional[int]] = mapped_column(Integer)
    reuse_job_id: Mapped[Optional[int]] = mapped_column(Integer)
    download_request_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    result_data_id: Mapped[Optional[int]] = mapped_column(Integer)
    dataset_id: Mapped[Optional[int]] = mapped_column(Integer)
    version_id: Mapped[Optional[int]] = mapped_column(Integer)
    label_task_id: Mapped[Optional[int]] = mapped_column(Integer)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    label_errors: Mapped[int] = mapped_column(Integer, default=0)
    empty_labels: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    smoke: Mapped[bool] = mapped_column(default=False)
    created_by: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
