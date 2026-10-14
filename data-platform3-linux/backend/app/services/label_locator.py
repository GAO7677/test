"""标注文件定位器（v3.2 补丁）。

统一标注查找链（按优先级，命中即止）：
1. 数据目录内散放标注（上传带标注场景）：数据文件同目录 {stem}_label.json
2. labels 子目录：fs.dataset_dir(data_type, dataset_id, "labels") / {stem}_label.json
3. 本数据集自己的任务版本目录：DatasetVersion.dataset_id == dataset_id，
   按 version_no 降序，逐个尝试 {v.file_path}/{rel} 与 {v.file_path}/{stem}_label.json
4. 下游数据集版本：Dataset.source_data_ids 包含本数据集 id 的数据集，取其最新版本，
   同上路径规则

Dataset.file_path 恒为 fs.dataset_dir(data_type, id)（以 /data 结尾），
labels 目录用 fs.dataset_dir(..., "labels") 现算，不依赖 file_path 形态。
"""
from __future__ import annotations

import json
import hashlib
from .label_schema import valid_label
from .version_service import version_ready
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from ..models import Dataset, DatasetVersion
from . import file_service as fs


def _read_json(p: Path) -> Optional[dict]:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if valid_label(data) else None
    except (ValueError, OSError, UnicodeDecodeError):
        return None


def find_label_with_path(db: Session, data_type: str, dataset_id: int,
                         data_path: Path,
                         rel_name: Optional[str] = None, cur=None) -> Tuple[Optional[Path], Optional[dict]]:
    """查找标注并同时返回实际文件路径与内容。"""
    stem = data_path.stem
    rel_candidates = [f"{stem}_label.json"]
    if rel_name:
        rel = Path(rel_name)
        rel_candidates.insert(0, str(rel.parent / f"{rel.stem}_label.json"))

    # 1) 数据目录内散放 + 2) labels 子目录
    for cand in [data_path.parent / f"{stem}_label.json",
                 fs.dataset_dir(data_type, dataset_id, "labels") / f"{stem}_label.json"]:
        if cand.exists():
            data = _read_json(cand)
            if data is not None:
                return cand, data

    # 3) 本数据集自己的任务版本 + 4) 下游数据集版本
    versions = []
    from .data_access import excluded_dataset_ids
    excluded=set(excluded_dataset_ids(db,cur)) if cur is not None else set()
    for ds_ in db.query(Dataset).filter(Dataset.data_type == data_type,
                                        Dataset.deleted_at.is_(None)).all():
        if ds_.id in excluded:continue
        try:
            src_ids = json.loads(ds_.source_data_ids or "[]")
        except ValueError:
            src_ids = []
        if dataset_id in src_ids:
            versions += (db.query(DatasetVersion)
                         .filter(DatasetVersion.dataset_id == ds_.id)
                         .order_by(DatasetVersion.version_no.desc()).all())
    versions.sort(key=lambda v: v.id, reverse=True)
    seen = set()
    for v in versions:
        if v.id in seen or not version_ready(db, v):
            continue
        seen.add(v.id)
        root = Path(v.file_path)
        candidates = list(rel_candidates)
        if rel_name:
            candidates.insert(0, str(Path(f'source_{dataset_id}') / Path(rel_name).with_name(stem + '_label.json')))
        for rel in candidates:
            cand = root / rel
            # Only reuse labels when paired data are byte-identical to the requested raw file.
            source = cand.with_name(cand.name[:-len('_label.json')] + data_path.suffix)
            if not source.is_file() or source.stat().st_size != data_path.stat().st_size:
                continue
            def digest(p):
                h = hashlib.sha256()
                with p.open('rb') as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
                return h.digest()
            if digest(source) != digest(data_path):
                continue
            if cand.exists():
                data = _read_json(cand)
                if data is not None:
                    return cand, data
    return None, None


def find_label(db: Session, data_type: str, dataset_id: int,
               data_path: Path, rel_name: Optional[str] = None, cur=None) -> Optional[dict]:
    """查找数据文件对应的标注 JSON，返回 dict / list / None。"""
    return find_label_with_path(db, data_type, dataset_id, data_path, rel_name, cur)[1]


def find_label_path(db: Session, data_type: str, dataset_id: int,
                    data_path: Path, rel_name: Optional[str] = None, cur=None) -> Optional[Path]:
    """查找数据文件对应的标注 JSON 路径。"""
    return find_label_with_path(db, data_type, dataset_id, data_path, rel_name, cur)[0]
