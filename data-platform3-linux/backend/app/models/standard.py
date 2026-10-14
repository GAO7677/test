"""可视化与标准层：templates / semantic_terms / ref_standards / quality_results"""
from __future__ import annotations
from typing import Optional
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Template(Base):
    __tablename__ = "templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    data_type: Mapped[str] = mapped_column(String(10), default="")
    fields: Mapped[str] = mapped_column(Text, default="")  # JSON：模板字段/参数
    description: Mapped[str] = mapped_column(String(500), default="")
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class SemanticTerm(Base):
    __tablename__ = "semantic_terms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    term: Mapped[str] = mapped_column(String(100), nullable=False)
    en_name: Mapped[str] = mapped_column(String(200), default="")
    category: Mapped[str] = mapped_column(String(50), default="")
    definition: Mapped[str] = mapped_column(Text, default="")
    synonyms: Mapped[str] = mapped_column(Text, default="")  # JSON 数组
    unit: Mapped[str] = mapped_column(String(20), default="")
    secret_level: Mapped[str] = mapped_column(String(10), default="内部")
    status: Mapped[str] = mapped_column(String(20), default="现行")  # 现行/待审核/废止
    created_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


class RefStandard(Base):
    __tablename__ = "ref_standards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    standard_type: Mapped[str] = mapped_column(String(30), default="国家标准")
    dimensions: Mapped[str] = mapped_column(String(500), default="")
    scope: Mapped[str] = mapped_column(String(255), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    file_path: Mapped[str] = mapped_column(String(500), default="")
    rules: Mapped[str] = mapped_column(Text, default="")  # JSON：自定义评估规则 [{metric,op,threshold}]
    is_custom: Mapped[bool] = mapped_column(Boolean, default=False)  # 自定义评估标准
    status: Mapped[str] = mapped_column(String(20), default="现行")
    created_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


class QualityResult(Base):
    __tablename__ = "quality_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    data_id: Mapped[int] = mapped_column(Integer, nullable=False)
    data_type: Mapped[str] = mapped_column(String(10), nullable=False)
    completeness: Mapped[float] = mapped_column(Numeric(5, 1), default=0)
    accuracy: Mapped[float] = mapped_column(Numeric(5, 1), default=0)
    consistency: Mapped[float] = mapped_column(Numeric(5, 1), default=0)
    timeliness: Mapped[float] = mapped_column(Numeric(5, 1), default=0)
    standardization: Mapped[float] = mapped_column(Numeric(5, 1), default=0)  # 规范性
    accessibility: Mapped[float] = mapped_column(Numeric(5, 1), default=0)  # 可访问性
    overall_score: Mapped[float] = mapped_column(Numeric(5, 1), default=0)
    details: Mapped[str] = mapped_column(Text, default="")  # JSON
    standards_compliance: Mapped[str] = mapped_column(Text, default="")  # JSON
    version_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # NULL=数据获取层原始数据
    evaluated_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
