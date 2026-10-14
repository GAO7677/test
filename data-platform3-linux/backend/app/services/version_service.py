"""数据集版本化核心服务（决策⑫⑬⑭）：版本创建/发布/删除/血缘/目录管理。

规则：
- 创建预处理数据集自动生成 V1（draft，source_task_type=init）
- 任务执行成功 → 自动创建新版本（draft），parent=任务 input_version，产物写入
  processed/{dataset_id}/V{version_no}/，datasets.current_version_id 更新为新版本
- draft → published 单向流转；published 只读不可删；draft 可删（无引用且无子版本），
  删除时清理版本目录
- 血缘：parent_version_id 单父链，向上追溯至 V1、向下枚举全部子孙
"""
from __future__ import annotations
from typing import Optional

import shutil
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..models import (AutoLabelTask, CleanTask, Dataset, DatasetVersion,
                      HybridLabelTask, IntegrateTask, ManualLabelTask,
                      SampleDataset, TASK_MODELS, TASK_STAGE, TransformTask,
                      Workflow)
from . import file_service as fs


def get_dataset(db: Session, dataset_id: int, cur=None) -> Dataset:
    ds = db.query(Dataset).filter(Dataset.id == dataset_id,
                                  Dataset.deleted_at.is_(None)).first()
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")
    if cur is not None:
        from .data_access import require_dataset
        require_dataset(db, cur, ds)
    return ds


def get_version(db: Session, dataset_id: int, version_id: int, cur=None) -> DatasetVersion:
    if cur is not None:
        get_dataset(db, dataset_id, cur)
    v = db.query(DatasetVersion).filter(DatasetVersion.id == version_id,
                                        DatasetVersion.dataset_id == dataset_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="版本不存在")
    return v


def version_ready(db, version) -> bool:
    model=TASK_MODELS.get(version.source_task_type)
    if model is None or not version.source_task_id:
        return True
    task=db.get(model,version.source_task_id)
    return task is not None and task.status=='completed'


def require_version_ready(db, version):
    if not version_ready(db,version):
        raise HTTPException(409,'该版本的处理任务尚未成功完成，请选择已完成的版本')
    return version


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def _file_count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file()
               and not p.name.endswith("_label.json")
               and p.name != "_manifest.json") if path.exists() else 0


def next_version_no(db: Session, dataset_id: int) -> int:
    db.query(Dataset).filter(Dataset.id == dataset_id).with_for_update().one()
    last = (db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.version_no.desc()).first())
    return (last.version_no + 1) if last else 1


def create_initial_version(db: Session, dataset: Dataset, user_id: Optional[int],
                           description: str = "初始版本", *, commit: bool = True) -> DatasetVersion:
    """创建数据集时自动生成 V1（draft，init），产物目录 processed/{id}/V1/。"""
    path = fs.settings.upload_dir / "processed" / str(dataset.id) / "V1"
    path.mkdir(parents=True, exist_ok=False)
    v = DatasetVersion(dataset_id=dataset.id, version_no=1, status="draft",
                       stage="原始", file_path=str(path),
                       data_count=_file_count(path), size_bytes=_dir_size(path),
                       parent_version_id=None, source_task_type="init",
                       description=description, created_by=user_id)
    db.add(v)
    db.flush()
    dataset.current_version_id = v.id
    if commit: db.commit()
    return v


def create_output_version(db: Session, dataset: Dataset, *,
                          input_version: DatasetVersion,
                          task_type: str, task_id: int,
                          user_id: Optional[int],
                          description: str = "") -> DatasetVersion:
    """任务执行成功后创建输出版本（draft），更新 current_version_id。

    产物目录 processed/{dataset_id}/V{n}/ 由本函数创建，任务执行器往里写产物。
    """
    version_no = next_version_no(db, dataset.id)
    path = fs.settings.upload_dir / "processed" / str(dataset.id) / f"V{version_no}"
    path.mkdir(parents=True, exist_ok=False)
    v = DatasetVersion(dataset_id=dataset.id, version_no=version_no, status="draft",
                       stage=TASK_STAGE.get(task_type, "原始"), file_path=str(path),
                       parent_version_id=input_version.id,
                       source_task_type=task_type, source_task_id=task_id,
                       description=description or f"由{TASK_STAGE.get(task_type, '')}任务 #{task_id} 生成",
                       created_by=user_id)
    db.add(v)
    db.flush()
    model = TASK_MODELS.get(task_type)
    task = db.get(model, task_id) if model else None
    if task is not None: task.output_version_id = v.id
    db.commit()
    return v


def refresh_version_stats(db: Session, version: DatasetVersion) -> None:
    """任务写完产物后回写文件数与体积。"""
    path = Path(version.file_path)
    version.data_count = _file_count(path)
    version.size_bytes = _dir_size(path)
    ds = db.query(Dataset).filter(Dataset.id == version.dataset_id).first()
    if ds and version_ready(db, version) and version.data_count:
        ds.current_version_id = version.id
        ds.data_count = version.data_count
        data_stems = {p.with_suffix("") for p in path.rglob("*")
                      if p.is_file() and not p.name.endswith("_label.json")
                      and p.name != "_manifest.json"} if path.exists() else set()
        ds.label_count = sum(1 for p in path.rglob("*_label.json") if p.is_file()
                             and p.with_name(p.name[:-len("_label.json")]) in data_stems)
        ds.updated_at = datetime.now()
    db.commit()


def publish_version(db: Session, version: DatasetVersion, description: str = "") -> DatasetVersion:
    """draft → published（单向流转，不可回退）。"""
    if version.status == "published":
        raise HTTPException(status_code=400, detail=f"V{version.version_no} 已是已发布状态")
    require_version_ready(db,version)
    version.status = "published"
    version.published_at = datetime.now()
    if description:
        version.description = description
    db.commit()
    return version


def check_version_deletable(db: Session, version: DatasetVersion) -> None:
    """draft 可删条件：无任何任务的 input_version_id 引用它，且无子版本。"""
    if version.status == "published":
        raise HTTPException(status_code=400, detail="已发布版本只读，不可删除")
    children = db.query(DatasetVersion).filter(DatasetVersion.parent_version_id == version.id).all()
    if children:
        refs = "、".join(f"V{c.version_no}" for c in children)
        raise HTTPException(status_code=400,
                            detail=f"无法删除：存在子版本 {refs} 以该版本为血缘父级")
    for task_type, model in TASK_MODELS.items():
        ref = db.query(model).filter(model.input_version_id == version.id).first()
        if ref:
            name = getattr(ref, "name", f"任务#{ref.id}")
            raise HTTPException(status_code=400,
                                detail=f"无法删除：被{TASK_STAGE.get(task_type, '')}任务「{name}」(id={ref.id}) 引用为输入版本")


def delete_version(db: Session, version: DatasetVersion) -> None:
    """删除 draft 版本并同步清理版本目录。"""
    check_version_deletable(db, version)
    ds = db.query(Dataset).filter(Dataset.id == version.dataset_id).first()
    if ds and version_ready(db, version) and version.data_count:
        ds.current_version_id = version.id
        # 回退到剩余最新版本
        rest = (db.query(DatasetVersion)
                .filter(DatasetVersion.dataset_id == version.dataset_id,
                        DatasetVersion.id != version.id)
                .order_by(DatasetVersion.version_no.desc()).first())
        ds.current_version_id = rest.id if rest else None
    shutil.rmtree(version.file_path, ignore_errors=True)
    db.delete(version)
    db.commit()


def copy_version(db: Session, dataset: Dataset, base_version: DatasetVersion,
                 user_id: Optional[int], description: str = "") -> DatasetVersion:
    """从指定版本复制文件生成新 draft 版本。"""
    require_version_ready(db,base_version)
    version_no = next_version_no(db, dataset.id)
    src = Path(base_version.file_path)
    dst = fs.settings.upload_dir / "processed" / str(dataset.id) / f"V{version_no}"
    shutil.copytree(src, dst)  # exclusive destination; never remove another version
    v = DatasetVersion(dataset_id=dataset.id, version_no=version_no, status="draft",
                       stage=base_version.stage, file_path=str(dst),
                       data_count=_file_count(dst), size_bytes=_dir_size(dst),
                       parent_version_id=base_version.id,
                       source_task_type="copy", source_task_id=None,
                       description=description or f"从 V{base_version.version_no} 复制",
                       created_by=user_id)
    db.add(v)
    db.flush()
    dataset.current_version_id = v.id
    db.commit()
    return v


# ---------------- 血缘 ----------------
def lineage_of(db: Session, version: DatasetVersion) -> dict:
    """单版本血缘：向上至 V1 的祖先链 + 向下全部子孙。"""
    ancestors = []
    cur = version
    seen = set()
    while cur and cur.parent_version_id and cur.parent_version_id not in seen:
        seen.add(cur.id)
        parent = db.query(DatasetVersion).filter(DatasetVersion.id == cur.parent_version_id).first()
        if not parent:
            break
        ancestors.append(parent)
        cur = parent
    descendants = []
    all_versions = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == version.dataset_id).all()
    children_map: dict = {}
    for v in all_versions:
        children_map.setdefault(v.parent_version_id, []).append(v)
    queue = list(children_map.get(version.id, []))
    while queue:
        node = queue.pop(0)
        descendants.append(node)
        queue.extend(children_map.get(node.id, []))
    brief = lambda v: {
        "id": v.id, "version_no": v.version_no, "version": f"V{v.version_no}",
        "stage": v.stage, "status": v.status,
        "source_task_type": v.source_task_type, "source_task_id": v.source_task_id,
        "quality_score": float(v.quality_score) if v.quality_score is not None else None,
        "created_at": v.created_at.strftime("%Y-%m-%d %H:%M:%S") if v.created_at else None,
    }
    return {"current": brief(version),
            "ancestors": [brief(v) for v in ancestors],          # 近→远
            "descendants": [brief(v) for v in descendants]}


def dataset_lineage(db: Session, dataset_id: int) -> dict:
    """数据集全量血缘：全部版本的父子关系列表（前端组链展示）。"""
    versions = (db.query(DatasetVersion)
                .filter(DatasetVersion.dataset_id == dataset_id)
                .order_by(DatasetVersion.version_no).all())
    return {"items": [{
        "id": v.id, "version_no": v.version_no, "version": f"V{v.version_no}",
        "stage": v.stage, "status": v.status,
        "parent_version_id": v.parent_version_id,
        "source_task_type": v.source_task_type, "source_task_id": v.source_task_id,
        "quality_score": float(v.quality_score) if v.quality_score is not None else None,
        "created_at": v.created_at.strftime("%Y-%m-%d %H:%M:%S") if v.created_at else None,
    } for v in versions]}


def validate_input_version(db: Session, dataset: Dataset,
                           input_version_id: Optional[int]) -> DatasetVersion:
    """校验任务输入版本：缺省取数据集 current_version_id；须存在且属于该数据集。"""
    if input_version_id is None:
        if not dataset.current_version_id:
            raise HTTPException(status_code=400, detail="数据集暂无可用版本")
        input_version_id = dataset.current_version_id
    return require_version_ready(db,get_version(db, dataset.id, input_version_id))
