"""通用工具：统一错误格式、分页、序列化、日志文件（按天滚动保留30天）。"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from ..config import settings


# ---------- 密级体系（六级，升序） ----------
SECRET_LEVELS = ["公开", "受控", "内部", "秘密", "机密", "绝密"]


def secret_index(level: str) -> int:
    """密级名称 → 级别序号；未知名称按 内部(2) 处理。"""
    try:
        return SECRET_LEVELS.index(level)
    except ValueError:
        return 2


# ---------- 统一错误响应 { code, message, detail } ----------
class ApiError(HTTPException):
    def __init__(self, status_code: int, message: str, detail: Any = None):
        super().__init__(status_code=status_code, detail=detail)
        self.code = status_code
        self.message = message


async def http_exception_handler(request, exc: HTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "请求失败"
    detail = None if isinstance(exc.detail, str) else exc.detail
    return JSONResponse(status_code=exc.status_code,
                        content={"code": exc.status_code, "message": message, "detail": detail})


async def unhandled_exception_handler(request, exc: Exception):
    return JSONResponse(status_code=500,
                        content={"code": 500, "message": "服务器内部错误", "detail": str(exc)})


# ---------- 分页 ----------
def paginate(query, page: int, page_size: int) -> dict:
    """列表统一返回 { total, page, page_size, items }"""
    page = max(int(page), 1)
    page_size = min(max(int(page_size), 1), 100)
    total = query.count()
    items = query.offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size, "items": items}


def apply_sort(query, model, sort: Optional[str], default_col="created_at"):
    """sort 形如 '-created_at'（倒序）或 'name'（正序）；默认按时间倒序。"""
    col_name = (sort or "").lstrip("-") or default_col
    col = getattr(model, col_name, None) or getattr(model, default_col)
    if sort and sort.startswith("-"):
        return query.order_by(col.desc())
    if sort:
        return query.order_by(col.asc())
    return query.order_by(getattr(model, default_col).desc())


# ---------- 序列化 ----------
def to_dict(obj, exclude: tuple = ("password_hash",)) -> dict:
    d = {}
    for c in obj.__table__.columns:
        if c.name in exclude:
            continue
        v = getattr(obj, c.name)
        if isinstance(v, datetime):
            v = v.strftime("%Y-%m-%d %H:%M:%S")
        d[c.name] = v
    return d


def json_default(o: Any):
    """JSON 序列化兜底：数据库 DECIMAL(Numeric) → float，日期时间 → 字符串，其余转字符串。

    质量评估等接口的得分字段在 MySQL 中为 DECIMAL，SQLAlchemy 取出为 Decimal，
    直接 json.dumps 会抛 TypeError: Object of type Decimal is not JSON serializable。
    """
    from decimal import Decimal
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, datetime):
        return o.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(o, date):
        return o.strftime("%Y-%m-%d")
    return str(o)


def parse_json(text: str, default):
    if not text:
        return default
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return default


# ---------- 文件日志（按天滚动，保留30天）----------
_file_logger: Optional[logging.Logger] = None


def get_file_logger() -> logging.Logger:
    global _file_logger
    if _file_logger is None:
        _file_logger = logging.getLogger("data_platform.operation")
        _file_logger.setLevel(logging.INFO)
        handler = TimedRotatingFileHandler(
            settings.log_dir / "operation.log", when="midnight",
            backupCount=30, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        _file_logger.addHandler(handler)
    return _file_logger


def clean_temp_files(days: int = 7) -> int:
    """启动时清理 temp 目录中 N 天前的临时文件，返回清理数量。"""
    temp_dir = settings.upload_dir / "temp"
    if not temp_dir.exists():
        return 0
    cutoff = datetime.now().timestamp() - days * 86400
    removed = 0
    for p in temp_dir.rglob("*"):
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def unique_path(directory: Path, filename: str) -> Path:
    """重名处理：自动加序号 file.jpg → file_1.jpg"""
    target = directory / filename
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    i = 1
    while (directory / f"{stem}_{i}{suffix}").exists():
        i += 1
    return directory / f"{stem}_{i}{suffix}"
