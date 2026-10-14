"""数据预处理层：datasets / dataset_versions（版本化核心）/ 各任务表 / workflows"""
from __future__ import annotations
from typing import Optional
from datetime import datetime

from sqlalchemy import (BigInteger, DateTime, ForeignKey, Index, Integer,
                        Numeric, String, Text, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    data_type: Mapped[str] = mapped_column(String(10), nullable=False)
    description: Mapped[str] = mapped_column(String(500), default="")
    data_count: Mapped[int] = mapped_column(Integer, default=0)
    label_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="可用")
    source_data_ids: Mapped[str] = mapped_column(Text, default="")  # JSON 数组
    secret_level: Mapped[str] = mapped_column(String(10), default="内部")  # 公开/受控/内部/秘密/机密/绝密
    current_version_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class DatasetVersion(Base):
    """数据集版本表：版本化管理核心（决策⑫⑬⑭）"""
    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version_no", name="uk_dataset_version"),
        Index("idx_version_parent", "parent_version_id"),
        Index("idx_version_dataset", "dataset_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, nullable=False)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft/published
    stage: Mapped[str] = mapped_column(String(20), default="原始")  # 原始/已标注/清洗后/转换后/集成后/采样后
    file_path: Mapped[str] = mapped_column(String(500), default="")
    data_count: Mapped[int] = mapped_column(Integer, default=0)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    parent_version_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_task_type: Mapped[str] = mapped_column(String(30), default="")
    source_task_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 1), nullable=True)
    quality_evaluated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    description: Mapped[str] = mapped_column(String(500), default="")
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


class _TaskCommon:
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending/running/completed/failed
    result_path: Mapped[str] = mapped_column(String(500), default="")
    input_version_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_version_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class AutoLabelTask(_TaskCommon, Base):
    __tablename__ = "auto_label_tasks"
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    data_type: Mapped[str] = mapped_column(String(10), nullable=False)
    source_ids: Mapped[str] = mapped_column(Text, default="")  # JSON
    algorithm: Mapped[str] = mapped_column(String(50), default="")
    rules_config: Mapped[str] = mapped_column(Text, default="")  # JSON
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_msg: Mapped[str] = mapped_column(Text, default="")


class ManualLabelTask(_TaskCommon, Base):
    __tablename__ = "manual_label_tasks"
    source_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_type: Mapped[str] = mapped_column(String(10), default="")
    label_type: Mapped[str] = mapped_column(String(50), default="")


class HybridLabelTask(_TaskCommon, Base):
    __tablename__ = "hybrid_label_tasks"
    source_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_type: Mapped[str] = mapped_column(String(10), default="")
    template_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class CleanTask(_TaskCommon, Base):
    __tablename__ = "clean_tasks"
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    data_type: Mapped[str] = mapped_column(String(10), nullable=False)
    source_ids: Mapped[str] = mapped_column(Text, default="")
    rules: Mapped[str] = mapped_column(Text, default="")  # JSON
    progress: Mapped[int] = mapped_column(Integer, default=0)
    stats_before: Mapped[str] = mapped_column(Text, default="")
    stats_after: Mapped[str] = mapped_column(Text, default="")


class IntegrateTask(_TaskCommon, Base):
    __tablename__ = "integrate_tasks"
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_ids: Mapped[str] = mapped_column(Text, default="")
    source_types: Mapped[str] = mapped_column(Text, default="")
    combo_type: Mapped[str] = mapped_column(String(20), default="")
    strategy: Mapped[str] = mapped_column(String(30), default="")
    detail: Mapped[str] = mapped_column(Text, default="")  # 其余输入版本ID列表（多输入血缘记录）


class TransformTask(_TaskCommon, Base):
    __tablename__ = "transform_tasks"
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    data_type: Mapped[str] = mapped_column(String(10), nullable=False)
    source_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    transform_type: Mapped[str] = mapped_column(String(30), default="")
    target_format: Mapped[str] = mapped_column(String(20), default="")
    params: Mapped[str] = mapped_column(Text, default="")
    progress: Mapped[int] = mapped_column(Integer, default=0)


class SampleDataset(_TaskCommon, Base):
    __tablename__ = "sample_datasets"
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    data_type: Mapped[str] = mapped_column(String(10), nullable=False)
    source_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    train_ratio: Mapped[int] = mapped_column(Integer, default=70)
    val_ratio: Mapped[int] = mapped_column(Integer, default=10)
    test_ratio: Mapped[int] = mapped_column(Integer, default=20)
    train_count: Mapped[int] = mapped_column(Integer, default=0)
    val_count: Mapped[int] = mapped_column(Integer, default=0)
    test_count: Mapped[int] = mapped_column(Integer, default=0)
    train_path: Mapped[str] = mapped_column(String(500), default="")
    val_path: Mapped[str] = mapped_column(String(500), default="")
    test_path: Mapped[str] = mapped_column(String(500), default="")


class Workflow(_TaskCommon, Base):
    __tablename__ = "workflows"
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    data_type: Mapped[str] = mapped_column(String(10), default="")
    source_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 关联数据集
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_msg: Mapped[str] = mapped_column(Text, default="")
    nodes: Mapped[str] = mapped_column(Text, default="")  # JSON
    edges: Mapped[str] = mapped_column(Text, default="")  # JSON
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


# 任务类型 → 表（版本血缘 source_task_type 取值）
TASK_MODELS = {
    "auto_label": AutoLabelTask,
    "manual_label": ManualLabelTask,
    "hybrid_label": HybridLabelTask,
    "clean": CleanTask,
    "integrate": IntegrateTask,
    "transform": TransformTask,
    "sample": SampleDataset,
    "workflow": Workflow,
}

# 输出版本的阶段标签映射
TASK_STAGE = {
    "init": "原始",
    "auto_label": "已标注",
    "manual_label": "已标注",
    "hybrid_label": "已标注",
    "clean": "清洗后",
    "integrate": "集成后",
    "transform": "转换后",
    "sample": "采样后",
}
