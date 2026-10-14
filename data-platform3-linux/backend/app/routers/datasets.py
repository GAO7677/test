"""数据集管理与版本管理路由（6.7）：datasets CRUD + versions（列表/新建/详情/发布/删除/下载/血缘）。"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Dataset, DatasetVersion
from ..security import CurrentUser, require_login, require_write
from ..services import file_service as fs
from ..services import version_service as vs
from ..services import data_access as access
from ..services.log_service import client_ip, write_log
from ..utils.common import SECRET_LEVELS, apply_sort, paginate, secret_index, to_dict

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


class DatasetBody(BaseModel):
    name: str
    data_type: str
    description: str = ""
    secret_level: str = "内部"
    source_data_ids: List[int] = []


def _zip_version(source: Path, include: str):
    """按内容选项打包版本目录：include 为逗号分隔的 data,labels,manifest。"""
    import io
    import zipfile
    parts = set((include or "data,labels,manifest").split(","))
    import tempfile
    from ..services.file_service import temp_dir
    with tempfile.TemporaryFile(dir=temp_dir()) as buf:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(source.rglob("*")):
                if not p.is_file():
                    continue
                if p.name == "_manifest.json":
                    if "manifest" not in parts:
                        continue
                elif p.name.endswith("_label.json"):
                    if "labels" not in parts:
                        continue
                elif "data" not in parts:
                    continue
                zf.write(p, p.relative_to(source))
                if p.name.endswith('_label.json') and 'labels' in parts:
                    import json
                    from ..services.video_exports import exports
                    try: label = json.loads(p.read_text())
                    except (ValueError, UnicodeError): continue
                    for name, data in exports(label, str(p.relative_to(source))[:-len('_label.json')]).items():
                        zf.writestr(name, data)
        buf.seek(0)
        while chunk := buf.read(1024 * 1024):
            yield chunk


def _version_dict(v: DatasetVersion) -> dict:
    d = to_dict(v)
    d["version"] = f"V{v.version_no}"
    d["quality_score"] = float(v.quality_score) if v.quality_score is not None else None
    return d


def _dataset_dict(db: Session, ds: Dataset) -> dict:
    d = to_dict(ds)
    cur = db.query(DatasetVersion).filter(DatasetVersion.id == ds.current_version_id).first() if ds.current_version_id else None
    d["current_version"] = _version_dict(cur) if cur else None
    d["version_count"] = db.query(DatasetVersion).filter(DatasetVersion.dataset_id == ds.id).count()
    return d


# ---------------- 数据集 CRUD ----------------
@router.get("")
def list_datasets(page: int = 1, page_size: int = 20, data_type: str = "",
                  stage: str = "", search: str = "", sort: str = "",
                  cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    q = db.query(Dataset).filter(Dataset.deleted_at.is_(None))
    if not cur.unrestricted:  # 密级可见性过滤
        q = q.filter(Dataset.secret_level.in_(
            [lv for lv in SECRET_LEVELS if secret_index(lv) <= cur.view_level]))
    q = q.filter(Dataset.id.notin_(access.excluded_dataset_ids(db, cur)))
    if data_type:
        q = q.filter(Dataset.data_type == data_type)
    if search:
        q = q.filter(Dataset.name.like(f"%{search}%"))
    if stage:  # 按当前版本阶段筛选：原始/已标注/清洗后/转换后/集成后/采样后
        sub = db.query(DatasetVersion.stage).filter(DatasetVersion.id == Dataset.current_version_id).scalar_subquery()
        q = q.filter(sub == stage)
    q = apply_sort(q, Dataset, sort)
    result = paginate(q, page, page_size)
    result["items"] = [_dataset_dict(db, x) for x in result["items"]]
    return result


@router.post("")
def create_dataset(body: DatasetBody, request: Request,
                   cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
    if body.data_type not in ("voice", "radar", "image", "video", "text", "mixed"):
        raise HTTPException(status_code=400, detail="数据类型无效")
    if body.secret_level not in SECRET_LEVELS:
        raise HTTPException(status_code=400, detail=f"密级无效（{'/'.join(SECRET_LEVELS)}）")
    from ..models import DATA_TYPE_MODELS
    for sid in body.source_data_ids:
        model = DATA_TYPE_MODELS.get(body.data_type)
        if model is None:
            raise HTTPException(400, "混合数据集请通过版本集成创建")
        access.require_raw(db, cur, body.data_type, db.get(model, sid))
    import json
    ds = Dataset(name=body.name, data_type=body.data_type, description=body.description,
                 secret_level=body.secret_level,
                 source_data_ids=json.dumps(body.source_data_ids), created_by=cur.id)
    db.add(ds)
    db.flush()
    import shutil
    from ..models import DataFile
    dst = None
    try:
        v = vs.create_initial_version(db, ds, cur.id, commit=False)
        dst = Path(v.file_path)
        seen = set()
        for sid in dict.fromkeys(body.source_data_ids):
            files = db.query(DataFile).filter(
                DataFile.data_type == body.data_type, DataFile.dataset_id == sid,
                DataFile.stage == "raw", DataFile.status == "complete",
                DataFile.deleted_at.is_(None)).all()
            for f in files:
                if f.stored_name.endswith('_label.json') or Path(f.stored_name).name == '_manifest.json':
                    continue
                root = fs.dataset_dir(body.data_type, sid).resolve()
                src = (root / f.stored_name).resolve()
                if root not in src.parents or not src.is_file():
                    raise HTTPException(409, f"来源文件缺失或路径无效：{f.filename}")
                rel = src.relative_to(root)
                target = dst / (Path(f"source_{sid}") / rel if len(set(body.source_data_ids)) > 1 else rel)
                # Sidecar naming is stem-based: reject x.jpg/x.png ambiguity too.
                key = target.with_suffix('')
                if key in seen: raise HTTPException(409, f"文件标识冲突：{rel}")
                seen.add(key)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                from ..services.label_locator import find_label_path
                label = find_label_path(db, body.data_type, sid, src, str(rel), cur)
                if label: shutil.copy2(label, target.with_name(target.stem + '_label.json'))
        if body.source_data_ids and not seen:
            raise HTTPException(400, '来源中没有可用数据文件')
        vs.refresh_version_stats(db, v)  # commits only after all copies succeeded
    except Exception:
        db.rollback()
        if dst is not None: shutil.rmtree(dst, ignore_errors=True)
        raise
    write_log(db, user_id=cur.id, username=cur.username, action="创建数据集",
              module="dataset", detail=f"创建预处理数据集 {body.name}（自动生成 V1）",
              ip=client_ip(request))
    return _dataset_dict(db, ds)


@router.get("/{did}")
def get_dataset(did: int, cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    ds = vs.get_dataset(db, did, cur)
    cur.require_view(ds.secret_level)
    return _dataset_dict(db, ds)


@router.delete("/{did}")
def delete_dataset(did: int, request: Request,
                   cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
    ds = vs.get_dataset(db, did, cur)
    ds.deleted_at = datetime.now()  # 软删除，文件保留
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="删除数据集",
              module="dataset", detail=f"软删除数据集 {ds.name}(id={did})", ip=client_ip(request))
    return {"message": "已删除"}


@router.get("/{did}/download")
def download_dataset(did: int, version_id: Optional[int] = None, include: str = "",
                     request: Request = None,
                     cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    """预处理产物 zip 下载：?version_id= 指定版本；?include=data,labels,manifest 选内容。"""
    ds = vs.get_dataset(db, did, cur)
    cur.require_view(ds.secret_level)
    vid = version_id or ds.current_version_id
    v = vs.get_version(db, did, vid, cur)
    vs.require_version_ready(db,v)
    source = Path(v.file_path)
    if not source.exists() or not any(source.iterdir()):
        raise HTTPException(status_code=404, detail="该版本暂无产物文件")
    write_log(db, user_id=cur.id, username=cur.username, action="下载数据集",
              module="dataset",
              detail=f"下载数据集 {ds.name} V{v.version_no}（内容：{include or '全部'}）",
              ip=client_ip(request))
    return StreamingResponse(_zip_version(source, include), media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=dataset_{did}_V{v.version_no}.zip"})


# ---------------- 版本管理 ----------------
@router.get("/{did}/versions")
def list_versions(did: int, cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    cur.require_view(vs.get_dataset(db, did, cur).secret_level)
    rows = (db.query(DatasetVersion).filter(DatasetVersion.dataset_id == did)
            .order_by(DatasetVersion.version_no.desc()).all())
    return {"items": [{**_version_dict(v), "ready": vs.version_ready(db,v)} for v in rows]}


@router.post("/{did}/versions")
def create_version(did: int, body: dict, request: Request,
                   cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
    """新建版本：从指定版本复制文件生成 draft（body: {base_version_id, description}）。"""
    ds = vs.get_dataset(db, did, cur)
    base = vs.get_version(db, did, int(body.get("base_version_id") or ds.current_version_id or 0))
    v = vs.copy_version(db, ds, base, cur.id, body.get("description", ""))
    write_log(db, user_id=cur.id, username=cur.username, action="create",
              module="dataset_version",
              detail=f"数据集 {ds.name} 从 V{base.version_no} 复制新建 V{v.version_no}",
              ip=client_ip(request))
    return _version_dict(v)


@router.get("/{did}/versions/{vid}")
def get_version(did: int, vid: int, cur: CurrentUser = Depends(require_login),
                db: Session = Depends(get_db)):
    v = vs.get_version(db, did, vid, cur)
    d = _version_dict(v)
    path = Path(v.file_path)
    d["files"] = ([{"name": p.name, "size": p.stat().st_size,
                    "path": str(p.relative_to(path))}
                   for p in sorted(path.rglob("*")) if p.is_file()] if path.exists() else [])
    return d


@router.get("/{did}/versions/{vid}/files")
def list_version_files(did: int, vid: int, search: str = "", page: int = 1, page_size: int = 50,
                       cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    """版本内文件分页清单（5000 文件场景）；每项带 label_count（同名 _label.json 的标注条数）。"""
    ds = vs.get_dataset(db, did, cur)
    cur.require_view(ds.secret_level)
    v = vs.get_version(db, did, vid, cur)
    root = Path(v.file_path)
    data, label_paths = [], {}
    if root.exists():
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(root))
            if p.name.endswith("_label.json"):
                label_paths[rel[:-len("_label.json")]] = p
            elif p.name != "_manifest.json":
                data.append({"name": p.name, "path": rel, "size": p.stat().st_size})
    kw = search.strip().lower()
    if kw:
        data = [f for f in data if kw in f["path"].lower()]
    total = len(data)
    page = max(page, 1)
    page_size = min(max(page_size, 1), 500)
    items = data[(page - 1) * page_size: page * page_size]
    import json as _json
    for f in items:
        stem = f["path"].rsplit(".", 1)[0] if "." in f["name"] else f["path"]
        lp = label_paths.get(stem)
        cnt = 0
        if lp:
            try:
                j = _json.loads(lp.read_text(encoding="utf-8"))
                if isinstance(j, list):
                    anns = j
                else:
                    # 兼容各模态标注结构：图像/视频 boxes·items，语音 segments，雷达 detections，
                    # 文本 entities·keywords，通用 annotations/labels
                    anns = []
                    for key in ("annotations", "boxes", "items", "segments", "detections",
                                "entities", "keywords", "labels"):
                        v = j.get(key)
                        if isinstance(v, list) and v:
                            anns = v
                            break
                cnt = len(anns)
            except ValueError:
                cnt = 0
        f["label_count"] = cnt
        f["has_label"] = lp is not None
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("/{did}/versions/{vid}/publish")
def publish_version(did: int, vid: int, body: Optional[dict] = None, request: Request = None,
                    cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
    v = vs.get_version(db, did, vid, cur)
    vs.publish_version(db, v, (body or {}).get("description", ""))
    ds = vs.get_dataset(db, did, cur)
    write_log(db, user_id=cur.id, username=cur.username, action="publish",
              module="dataset_version", detail=f"发布数据集 {ds.name} V{v.version_no}",
              ip=client_ip(request))
    return _version_dict(v)


@router.delete("/{did}/versions/{vid}")
def delete_version(did: int, vid: int, request: Request,
                   cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
    v = vs.get_version(db, did, vid, cur)
    ds = vs.get_dataset(db, did, cur)
    vs.delete_version(db, v)
    write_log(db, user_id=cur.id, username=cur.username, action="delete",
              module="dataset_version",
              detail=f"删除数据集 {ds.name} 草稿版本 V{v.version_no}（已清理版本目录）",
              ip=client_ip(request))
    return {"message": f"V{v.version_no} 已删除"}


@router.get("/{did}/versions/{vid}/download")
def download_version(did: int, vid: int, include: str = "", request: Request = None,
                     cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    ds = vs.get_dataset(db, did, cur)
    cur.require_view(ds.secret_level)
    v = vs.get_version(db, did, vid, cur)
    vs.require_version_ready(db,v)
    source = Path(v.file_path)
    if not source.exists() or not any(source.iterdir()):
        raise HTTPException(status_code=404, detail="该版本暂无产物文件")
    write_log(db, user_id=cur.id, username=cur.username, action="下载版本产物",
              module="dataset_version",
              detail=f"下载数据集版本 V{v.version_no} 产物（内容：{include or '全部'}）",
              ip=client_ip(request))
    return StreamingResponse(_zip_version(source, include), media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=dataset_{did}_V{v.version_no}.zip"})


@router.get("/{did}/versions/{vid}/file")
def download_version_file(did: int, vid: int, path: str, request: Request = None,
                          cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    """版本内单文件下载（path 为版本目录内相对路径）。"""
    from fastapi.responses import FileResponse
    ds = vs.get_dataset(db, did, cur)
    cur.require_view(ds.secret_level)
    v = vs.get_version(db, did, vid, cur)
    root = Path(v.file_path).resolve()
    rel = Path(*[seg for seg in Path(path.replace("\\", "/")).parts if seg not in ("", ".", "..", "/")])
    target = (root / rel).resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target, filename=target.name)


@router.get("/{did}/versions/{vid}/lineage")
def version_lineage(did: int, vid: int, cur: CurrentUser = Depends(require_login),
                    db: Session = Depends(get_db)):
    return vs.lineage_of(db, vs.get_version(db, did, vid, cur))


@router.get("/{did}/lineage")
def full_lineage(did: int, cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    vs.get_dataset(db, did, cur)
    return vs.dataset_lineage(db, did)
