"""数据获取层：五类数据表（voice/radar/image/video/text）+ 通用文件子表 data_files + 结果回写"""
from __future__ import annotations
from typing import Optional
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class _DataCommon:
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(20), default="")
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[str] = mapped_column(String(500), default="")
    label_file_path: Mapped[str] = mapped_column(String(500), default="")
    project: Mapped[str] = mapped_column(String(100), default="")
    scene: Mapped[str] = mapped_column(String(100), default="")
    source_type: Mapped[str] = mapped_column(String(20), default="真实采集")  # 真实采集/仿真/AIGC
    secret_level: Mapped[str] = mapped_column(String(10), default="内部")  # 公开/受控/内部/秘密/机密/绝密
    status: Mapped[str] = mapped_column(String(20), default="未标注")  # 未标注/待标注/已标注
    label_type: Mapped[str] = mapped_column(String(50), default="")
    label_progress: Mapped[int] = mapped_column(Integer, default=0)
    remark: Mapped[str] = mapped_column(String(500), default="")
    created_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class VoiceData(_DataCommon, Base):
    __tablename__ = "voice_data"
    duration_seconds: Mapped[Optional[float]] = mapped_column(nullable=True)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class RadarData(_DataCommon, Base):
    __tablename__ = "radar_data"
    point_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    scan_mode: Mapped[str] = mapped_column(String(50), default="")


class ImageData(_DataCommon, Base):
    __tablename__ = "image_data"
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class VideoData(_DataCommon, Base):
    __tablename__ = "video_data"
    duration_seconds: Mapped[Optional[float]] = mapped_column(nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(nullable=True)


class TextData(_DataCommon, Base):
    __tablename__ = "text_data"
    line_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    word_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    encoding: Mapped[str] = mapped_column(String(20), default="UTF-8")


class Project(Base):
    """项目表：数据归属项目，真实落库（可新建/可复选）。"""
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String(500), default="")
    created_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


DATA_TYPE_MODELS = {
    "voice": VoiceData,
    "radar": RadarData,
    "image": ImageData,
    "video": VideoData,
    "text": TextData,
}


class DataFile(Base):
    """通用文件表：五类数据共用，一条记录 = 数据集内一个文件（决策①）"""
    __tablename__ = "data_files"
    __table_args__ = (Index("idx_dataset", "data_type", "dataset_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False)
    data_type: Mapped[str] = mapped_column(String(10), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_name: Mapped[str] = mapped_column(String(255), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    chunk_total: Mapped[int] = mapped_column(Integer, default=1)
    chunk_received: Mapped[int] = mapped_column(Integer, default=0)
    file_hash: Mapped[str] = mapped_column(String(64), default="")
    stage: Mapped[str] = mapped_column(String(10), default="raw")  # raw/label
    status: Mapped[str] = mapped_column(String(20), default="complete")  # uploading/complete/failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class WritebackResult(Base):
    """结果回写（5.4）：version 为模型结果自由文本版本号，与数据集版本机制独立"""
    __tablename__ = "writeback_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    result_type: Mapped[str] = mapped_column(String(50), default="")
    original_data_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    original_data_type: Mapped[str] = mapped_column(String(10), default="")
    file_path: Mapped[str] = mapped_column(String(500), default="")
    status: Mapped[str] = mapped_column(String(20), default="成功")
    version: Mapped[str] = mapped_column(String(30), default="")
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    created_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
