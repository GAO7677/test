"""数据获取层：五类数据结构相同的通用路由工厂（6.5）。

每类挂载：CRUD / 分片上传(init/chunk/complete/progress，断点续传) / 文件清单 /
zip 下载 / 标注上传 / 预览 / 可视化 / 统计。
"""
from __future__ import annotations
from typing import Optional

import json
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import DATA_TYPE_MODELS, DataFile, Dataset, DatasetVersion, Project
from ..security import CurrentUser, require_login, require_write
from ..services import file_service as fs
from ..services import data_access as access
from ..services import version_service as vs
from ..services.label_locator import find_label, find_label_path
from ..services.label_schema import label_names
from ..services.log_service import client_ip, write_log
from ..utils.common import (SECRET_LEVELS, apply_sort, paginate, secret_index,
                            to_dict, unique_path)

TYPE_NAMES = {"voice": "语音", "radar": "雷达", "image": "图像", "video": "视频", "text": "文本"}

projects_router = APIRouter(prefix="/api/projects", tags=["projects"])


@projects_router.get("")
def list_projects(search: str = "", cur: CurrentUser = Depends(require_login),
                  db: Session = Depends(get_db)):
    q = db.query(Project)
    if search:
        q = q.filter(Project.name.like(f"%{search}%"))
    return {"items": [to_dict(p) for p in q.order_by(Project.name).all()]}


@projects_router.post("")
def create_project(body: dict, request: Request,
                   cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
    """新建项目（幂等：名称已存在则直接返回既有项目）。"""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="项目名称不能为空")
    p = db.query(Project).filter(Project.name == name).first()
    if p:
        return to_dict(p)
    p = Project(name=name, description=body.get("description", ""), created_by=cur.id)
    db.add(p)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="新建项目",
              module="project", detail=f"新建项目 {name}", ip=client_ip(request))
    return to_dict(p)


class DataBody(BaseModel):
    name: str
    format: str = ""
    project: str = ""
    scene: str = ""
    source_type: str = "真实采集"
    secret_level: str = "内部"
    status: str = "未标注"
    label_type: str = ""
    remark: str = ""


def _safe_rel_path(rel: str) -> Path:
    """文件夹上传的相对路径消毒：禁止 .. 与绝对路径，返回纯相对 Path。"""
    parts = [seg for seg in Path(rel.replace("\\", "/")).parts if seg not in ("", ".", "..", "/")]
    return Path(*parts) if parts else Path(".")


def _get_or_404(db: Session, model, rid: int, cur=None):
    obj = db.query(model).filter(model.id == rid, model.deleted_at.is_(None)).first()
    if not obj:
        raise HTTPException(status_code=404, detail="数据不存在")
    if cur is not None:
        data_type = next(key for key,value in DATA_TYPE_MODELS.items() if value is model)
        access.require_raw(db, cur, data_type, obj)
    return obj


def _import_coco_json(db: Session, data_type: str, rid: int, coco: dict) -> dict:
    """COCO 整表 → 逐文件 {stem}_label.json。bbox [x,y,w,h] 转 xyxy；跳过 iscrowd=1 与 ignore/others 类别。"""
    images = {im["id"]: im.get("file_name", "") for im in coco.get("images", [])}
    cats = {ct["id"]: ct.get("name", str(ct["id"])) for ct in coco.get("categories", [])}
    anns = coco.get("annotations", [])
    recs = db.query(DataFile).filter(DataFile.data_type == data_type,
                                     DataFile.dataset_id == rid,
                                     DataFile.stage == "raw",
                                     DataFile.status == "complete",
                                     DataFile.deleted_at.is_(None)).all()
    by_rel, by_base = {}, {}
    for r_ in recs:
        rel = str(Path(_safe_rel_path(r_.stored_name))).replace("\\", "/")
        by_rel[rel] = rel
        by_base.setdefault(Path(rel).name, rel)
    grouped: dict[str, list] = {}
    unmatched = 0
    # 兼容不规范导出（annotations 的 category_id 越出 categories 表）：保留 cat_unknown_{id} 而非吞掉类别
    skipped_ignore = 0
    for a in anns:
        if a.get("iscrowd"):
            continue
        cid = a.get("category_id")
        cname = cats.get(cid)
        if cname is None:
            cname = f"cat_{cid}" if cid is not None else ""
        if cname in ("ignore", "others"):
            skipped_ignore += 1
            continue
        fn = images.get(a.get("image_id"), "")
        rel = by_rel.get(fn.replace("\\", "/")) or by_base.get(Path(fn).name)
        if not rel:
            unmatched += 1
            continue
        x, y, w, h = a.get("bbox", [0, 0, 0, 0])
        grouped.setdefault(rel, []).append({
            "type": "bbox",
            "xyxy": [x, y, x + w, y + h],
            "label": cname or "目标",
        })
    if skipped_ignore:
        pass  # ignore/others 类标注按惯例不导入
    root = fs.dataset_dir(data_type, rid).resolve()
    written = 0
    for rel, items in grouped.items():
        data_path = (root / rel).resolve()
        if root not in data_path.parents:
            continue
        target = data_path.parent / f"{data_path.stem}_label.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump({"annotations": items, "source": "coco_import"}, f, ensure_ascii=False, indent=2)
        written += 1
    return {"matched_files": written, "unmatched_annotations": unmatched,
            "total_annotations": len(anns)}


def _maybe_import_coco(db: Session, data_type: str, rid: int, path: Path,
                       cur: CurrentUser, request) -> None:
    """上传完成的 JSON 若是 COCO 整表（文件名 instances*.json/annotations*.json 且结构吻合）则自动导入。
    返回 True 表示已按整表导入（调用方可移除原始 JSON 文件）。

    COCO 整表仅对图像模态生效，避免其他模态上传的普通 JSON 被误判为标注整表。
    """
    if data_type != "image":
        return False
    low = path.name.lower()
    if not (low.startswith("instances") or low.startswith("annotations")):
        return False
    try:
        coco = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False
    if not (isinstance(coco, dict) and coco.get("images") and coco.get("annotations")):
        return False
    res = _import_coco_json(db, data_type, rid, coco)
    if res["matched_files"]:
        write_log(db, user_id=cur.id, username=cur.username, action="导入标注",
                  module=f"data_{data_type}",
                  detail=f"随文件上传自动导入 COCO 标注 {path.name}：{res['matched_files']} 个文件，"
                         f"{res['unmatched_annotations']} 条未匹配",
                  ip=client_ip(request))
        return True
    return False


def _refresh_aggregates(db: Session, data_type: str, dataset_id: int) -> None:
    """上传完成后回写主表聚合字段（文件数/总大小/元数据）。"""
    model = DATA_TYPE_MODELS[data_type]
    obj = db.query(model).filter(model.id == dataset_id).first()
    files = db.query(DataFile).filter(DataFile.data_type == data_type,
                                      DataFile.dataset_id == dataset_id,
                                      DataFile.stage == "raw",
                                      DataFile.status == "complete",
                                      DataFile.deleted_at.is_(None)).all()
    obj.file_count = len(files)
    obj.size_bytes = sum(f.size_bytes or 0 for f in files)
    if files:
        first = fs.dataset_dir(data_type, dataset_id) / files[0].stored_name
        if first.exists():
            for k, v in fs.extract_metadata(data_type, first).items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
    obj.file_path = str(fs.dataset_dir(data_type, dataset_id))
    obj.updated_at = datetime.now()
    db.commit()


def _annotation_workspace(db: Session, cur: CurrentUser, data_type: str, rid: int):
    candidates = db.query(Dataset).filter(
        Dataset.data_type == data_type, Dataset.created_by == cur.id,
        Dataset.deleted_at.is_(None)).order_by(Dataset.updated_at.desc(), Dataset.id.desc()).all()
    for ds in candidates:
        try:
            sources = json.loads(ds.source_data_ids or '[]')
        except (ValueError, TypeError):
            continue
        if sources != [rid]:
            continue
        access.require_dataset(db, cur, ds, operate=True)
        version = vs.get_version(db, ds.id, ds.current_version_id, cur)
        vs.require_version_ready(db, version)
        if not Path(version.file_path).is_dir():
            raise HTTPException(409, '在线标注工作版本的文件目录不存在')
        return {'dataset_id': ds.id, 'version_id': version.id,
                'file_count': version.data_count, 'size_bytes': version.size_bytes}
    return None


def _annotation_sources(db: Session, data_type: str, rid: int):
    files = db.query(DataFile).filter(
        DataFile.data_type == data_type, DataFile.dataset_id == rid,
        DataFile.stage == 'raw', DataFile.status == 'complete',
        DataFile.deleted_at.is_(None)).all()
    files = [f for f in files if not f.stored_name.endswith('_label.json')
             and Path(f.stored_name).name != '_manifest.json']
    if not files:
        raise HTTPException(400, '该数据暂无可标注文件，请先上传数据')
    root = fs.dataset_dir(data_type, rid).resolve()
    result = []
    for file in files:
        rel = Path(file.stored_name.replace('\\', '/'))
        src = (root / rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or not src.is_relative_to(root):
            raise HTTPException(409, '原始数据文件路径无效')
        if not src.is_file():
            raise HTTPException(409, f'原始数据文件不存在：{file.filename}')
        result.append((file, rel, src))
    return result


def build_data_router(data_type: str) -> APIRouter:
    model = DATA_TYPE_MODELS[data_type]
    type_name = TYPE_NAMES[data_type]
    router = APIRouter(prefix=f"/api/{data_type}", tags=[f"data-{data_type}"])

    # ---------------- 列表 / CRUD ----------------
    @router.get("")
    def list_data(page: int = 1, page_size: int = 20, project: str = "", scene: str = "",
                  source_type: str = "", secret_level: str = "", status: str = "",
                  search: str = "", sort: str = "",
                  cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
        q = db.query(model).filter(model.deleted_at.is_(None),
                                   model.id.notin_(access.excluded_raw_ids(db, cur, data_type)),
                                   model.secret_level.in_([lv for lv in SECRET_LEVELS if cur.can_view(lv)]))
        if project:
            q = q.filter(model.project == project)
        if scene:
            q = q.filter(model.scene == scene)
        if source_type:
            q = q.filter(model.source_type == source_type)
        if secret_level:
            q = q.filter(model.secret_level == secret_level)
        if status:
            q = q.filter(model.status == status)
        if search:
            q = q.filter(model.name.like(f"%{search}%"))
        q = apply_sort(q, model, sort)
        result = paginate(q, page, page_size)
        result["items"] = [to_dict(x) for x in result["items"]]
        return result

    @router.post("")
    def create_data(body: DataBody, request: Request,
                    cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
        if body.secret_level not in SECRET_LEVELS:
            raise HTTPException(status_code=400,
                                detail=f"密级无效（{'/'.join(SECRET_LEVELS)}）")
        # 项目 upsert：填了名称就确保项目表里存在
        if body.project:
            if not db.query(Project).filter(Project.name == body.project).first():
                db.add(Project(name=body.project, created_by=cur.id))
                db.commit()
        obj = model(name=body.name, format=body.format.upper(), project=body.project,
                    scene=body.scene, source_type=body.source_type,
                    secret_level=body.secret_level, status=body.status,
                    label_type=body.label_type, remark=body.remark, created_by=cur.id)
        db.add(obj)
        db.commit()
        fs.dataset_dir(data_type, obj.id)
        fs.dataset_dir(data_type, obj.id, "labels")
        obj.file_path = str(fs.dataset_dir(data_type, obj.id))
        db.commit()
        write_log(db, user_id=cur.id, username=cur.username, action="创建数据",
                  module=f"data_{data_type}", detail=f"创建{type_name}数据 {body.name}",
                  ip=client_ip(request))
        return to_dict(obj)

    @router.get("/stats")
    def stats(cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
        q = db.query(model).filter(model.deleted_at.is_(None),
                                   model.id.notin_(access.excluded_raw_ids(db, cur, data_type)),
                                   model.secret_level.in_([lv for lv in SECRET_LEVELS if cur.can_view(lv)]))
        rows = q.all()
        by_status: dict = {}
        by_secret: dict = {}
        for r in rows:
            by_status[r.status] = by_status.get(r.status, 0) + 1
            by_secret[r.secret_level] = by_secret.get(r.secret_level, 0) + 1
        return {"total": len(rows), "total_size": sum(r.size_bytes or 0 for r in rows),
                "total_files": sum(r.file_count or 0 for r in rows),
                "by_status": by_status, "by_secret_level": by_secret}

    @router.get("/{rid}")
    def get_data(rid: int, cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
        obj = _get_or_404(db, model, rid, cur)
        cur.require_view(obj.secret_level)
        return to_dict(obj)

    @router.put("/{rid}")
    def update_data(rid: int, body: DataBody, request: Request,
                    cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
        obj = _get_or_404(db, model, rid, cur)
        if body.secret_level not in SECRET_LEVELS:
            raise HTTPException(status_code=400,
                                detail=f"密级无效（{'/'.join(SECRET_LEVELS)}）")
        if body.project and not db.query(Project).filter(Project.name == body.project).first():
            db.add(Project(name=body.project, created_by=cur.id))
            db.commit()
        # v3.2：数据来源仅管理员可改（防仿真/AIGC 数据冒充真实采集）
        if body.source_type != obj.source_type and cur.role not in ("system_admin", "admin"):
            raise HTTPException(status_code=403, detail="数据来源仅管理员可修改")
        for f in ("name", "project", "scene", "source_type", "secret_level", "status", "label_type", "remark"):
            setattr(obj, f, getattr(body, f))
        obj.format = body.format.upper()
        obj.updated_at = datetime.now()
        db.commit()
        write_log(db, user_id=cur.id, username=cur.username, action="修改数据",
                  module=f"data_{data_type}", detail=f"修改{type_name}数据 {obj.name}(id={rid})",
                  ip=client_ip(request))
        return to_dict(obj)

    @router.delete("/{rid}")
    def delete_data(rid: int, request: Request,
                    cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
        obj = _get_or_404(db, model, rid, cur)
        obj.deleted_at = datetime.now()  # 软删除，文件保留
        db.commit()
        write_log(db, user_id=cur.id, username=cur.username, action="删除数据",
                  module=f"data_{data_type}", detail=f"软删除{type_name}数据 {obj.name}(id={rid})",
                  ip=client_ip(request))
        return {"message": "已删除"}

    # ---------------- 文件清单 ----------------
    @router.get("/{rid}/files")
    def list_files(rid: int, stage: str = "raw", search: str = "",
                   page: int = 0, page_size: int = 50,
                   cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
        """文件清单：page=0 时返回全量（兼容旧调用）；page>=1 时分页，search 按文件名模糊匹配。"""
        obj = _get_or_404(db, model, rid, cur)
        cur.require_view(obj.secret_level)
        q = db.query(DataFile).filter(DataFile.data_type == data_type,
                                      DataFile.dataset_id == rid,
                                      DataFile.deleted_at.is_(None))
        if stage:
            q = q.filter(DataFile.stage == stage)
        if search.strip():
            q = q.filter(DataFile.filename.like(f"%{search.strip()}%"))
        q = q.order_by(DataFile.id)
        if page and page >= 1:
            total = q.count()
            items = q.offset((page - 1) * page_size).limit(min(page_size, 500)).all()
            return {"items": [to_dict(f) for f in items], "total": total,
                    "page": page, "page_size": page_size}
        return {"items": [to_dict(f) for f in q.all()]}

    # ---------------- 分片上传（8MB 分片 / 断点续传 / 队列轮询） ----------------
    @router.post("/{rid}/upload/init")
    def upload_init(rid: int, body: dict, cur: CurrentUser = Depends(require_write),
                    db: Session = Depends(get_db)):
        _get_or_404(db, model, rid, cur)
        filename = body.get("filename", "")
        file_hash = body.get("file_hash", "")
        size = int(body.get("size", 0))
        chunk_total = max(int(body.get("chunk_total", 1)), 1)
        relative_path = str(_safe_rel_path(body.get("relative_path", "")))
        if not filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        if not fs.allowed_file(data_type, filename):
            raise HTTPException(status_code=400,
                                detail=f"不支持的文件格式，{type_name}数据允许：{'/'.join(sorted(fs.ALLOWED_EXTENSIONS[data_type]))}")
        # 文件夹上传：filename 记录相对路径（如 sub/dir/a.jpg），存储时还原目录结构
        display_name = filename if relative_path in ("", ".") else f"{relative_path}/{filename}"
        # 秒传：同一数据集内已存在相同 hash + 相同相对路径的完整文件时直接复用，避免重复落地副本
        if file_hash:
            done = db.query(DataFile).filter(DataFile.data_type == data_type,
                                            DataFile.dataset_id == rid,
                                            DataFile.file_hash == file_hash,
                                            DataFile.filename == display_name,
                                            DataFile.status == "complete",
                                            DataFile.deleted_at.is_(None)).first()
            if done:
                return {"file_id": done.id, "chunk_received": done.chunk_total,
                        "chunk_total": done.chunk_total, "chunk_size": fs.CHUNK_SIZE,
                        "instant": True, "filename": done.filename,
                        "stored_name": done.stored_name}
        # 断点续传：按 文件hash + 相对路径 查找已收分片
        rec = db.query(DataFile).filter(DataFile.data_type == data_type,
                                        DataFile.dataset_id == rid,
                                        DataFile.file_hash == file_hash,
                                        DataFile.filename == display_name,
                                        DataFile.status == "uploading",
                                        DataFile.deleted_at.is_(None)).first()
        if not rec:
            rec = DataFile(dataset_id=rid, data_type=data_type, filename=display_name,
                           stored_name=display_name, size_bytes=size, chunk_total=chunk_total,
                           chunk_received=0, file_hash=file_hash, status="uploading")
            db.add(rec)
            db.commit()
        return {"file_id": rec.id, "chunk_received": rec.chunk_received,
                "chunk_total": rec.chunk_total, "chunk_size": fs.CHUNK_SIZE}

    @router.post("/{rid}/upload/chunk")
    async def upload_chunk(rid: int, file_id: int = Form(...), chunk_index: int = Form(...),
                           chunk: UploadFile = File(...),
                           cur: CurrentUser = Depends(require_write),
                           db: Session = Depends(get_db)):
        _get_or_404(db, model, rid, cur)
        rec = db.query(DataFile).filter(DataFile.id == file_id,
                                        DataFile.data_type == data_type,
                                        DataFile.dataset_id == rid).first()
        if not rec:
            raise HTTPException(status_code=404, detail="上传任务不存在")
        part_dir = fs.temp_dir() / f"upload_{file_id}"
        part_dir.mkdir(parents=True, exist_ok=True)
        part = part_dir / f"chunk_{chunk_index:06d}"
        if not part.exists():  # 幂等：已收到的分片跳过
            with open(part, "wb") as f:
                while True:
                    data = await chunk.read(1024 * 1024)
                    if not data:
                        break
                    f.write(data)
            rec.chunk_received = min(rec.chunk_received + 1, rec.chunk_total)
            db.commit()
        return {"file_id": rec.id, "chunk_received": rec.chunk_received,
                "chunk_total": rec.chunk_total}

    @router.post("/{rid}/upload/complete")
    def upload_complete(rid: int, body: dict, request: Request,
                        cur: CurrentUser = Depends(require_write),
                        db: Session = Depends(get_db)):
        obj = _get_or_404(db, model, rid, cur)
        rec = db.query(DataFile).filter(DataFile.id == int(body.get("file_id", 0)),
                                        DataFile.data_type == data_type,
                                        DataFile.dataset_id == rid).first()
        if not rec:
            raise HTTPException(status_code=404, detail="上传任务不存在")
        if rec.status == "complete":   # 秒传/重复提交：文件已落地，直接返回，保证接口幂等
            return to_dict(rec)
        part_dir = fs.temp_dir() / f"upload_{rec.id}"
        received = len(list(part_dir.glob("chunk_*"))) if part_dir.exists() else 0
        if received < rec.chunk_total:
            raise HTTPException(status_code=400,
                                detail=f"分片不完整（{received}/{rec.chunk_total}），请继续上传")
        # 合并分片 → 数据目录（还原文件夹结构；重名自动加序号）
        rel = _safe_rel_path(rec.filename)
        target_dir = fs.dataset_dir(data_type, rid)
        if str(rel.parent) not in ("", "."):
            target_dir = target_dir / rel.parent
            target_dir.mkdir(parents=True, exist_ok=True)
        target = unique_path(target_dir, rel.name)
        with open(target, "wb") as out:
            for i in range(rec.chunk_total):
                part = part_dir / f"chunk_{i:06d}"
                with open(part, "rb") as f:
                    shutil.copyfileobj(f, out, 1024 * 1024)
        shutil.rmtree(part_dir, ignore_errors=True)
        rec.stored_name = str(target.relative_to(fs.dataset_dir(data_type, rid)))
        rec.size_bytes = target.stat().st_size
        rec.status = "complete"
        rec.chunk_received = rec.chunk_total
        db.commit()
        # COCO 整表随文件夹一起上传时自动导入为逐文件标注（文件名形如 instances*.json / annotations*.json 且含 COCO 结构）
        if rec.filename.lower().endswith(".json") and rec.filename != "_manifest.json":
            coco_done = False
            try:
                coco_done = bool(_maybe_import_coco(db, data_type, rid, target, cur, request))
            except Exception:
                db.rollback()  # 标注导入失败不影响文件本身
            if coco_done:  # 整表不是数据文件：删除磁盘文件与记录，仅保留拆分后的标注
                try:
                    if target.exists():
                        target.unlink()
                except OSError:
                    pass
                rec.deleted_at = datetime.now()
                db.commit()
        _refresh_aggregates(db, data_type, rid)
        write_log(db, user_id=cur.id, username=cur.username, action="上传文件",
                  module=f"data_{data_type}",
                  detail=f"向 {obj.name}(id={rid}) 上传文件 {rec.filename}（{rec.size_bytes} 字节）",
                  ip=client_ip(request))
        return to_dict(rec)

    @router.get("/{rid}/upload/progress")
    def upload_progress(rid: int, cur: CurrentUser = Depends(require_login),
                        db: Session = Depends(get_db)):
        """进度轮询（1秒间隔，不用 WebSocket）。"""
        rows = db.query(DataFile).filter(DataFile.data_type == data_type,
                                         DataFile.dataset_id == rid,
                                         DataFile.status == "uploading",
                                         DataFile.deleted_at.is_(None)).all()
        return {"items": [{"file_id": r.id, "filename": r.filename,
                           "chunk_received": r.chunk_received, "chunk_total": r.chunk_total,
                           "progress": round(r.chunk_received / max(r.chunk_total, 1) * 100, 1)}
                          for r in rows]}

    # ---------------- 下载 / 标注 / 预览 / 可视化 ----------------
    @router.get("/{rid}/download")
    def download(rid: int, stage: str = "raw", request: Request = None,
                 cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
        obj = _get_or_404(db, model, rid, cur)
        source = fs.dataset_dir(data_type, rid, "labels" if stage == "labeled" else "data")
        if not any(source.iterdir()):
            raise HTTPException(status_code=404, detail="无可下载文件")
        write_log(db, user_id=cur.id, username=cur.username, action="下载数据",
                  module=f"data_{data_type}",
                  detail=f"下载{type_name}数据 {obj.name}(id={rid}) stage={stage}",
                  ip=client_ip(request))
        return StreamingResponse(fs.zip_directory_stream(source), media_type="application/zip",
                                 headers={"Content-Disposition": f"attachment; filename={data_type}_{rid}_{stage}.zip"})

    @router.get("/{rid}/annotation-context")
    def annotation_context(rid: int, cur: CurrentUser = Depends(require_write),
                           db: Session = Depends(get_db)):
        obj = _get_or_404(db, model, rid, cur)
        cur.require_operate(obj.secret_level)
        existing = _annotation_workspace(db, cur, data_type, rid)
        if existing:
            return existing
        files = _annotation_sources(db, data_type, rid)
        return {'dataset_id': None, 'version_id': None, 'file_count': len(files),
                'size_bytes': sum(src.stat().st_size for _, _, src in files)}

    @router.post("/{rid}/annotation-context")
    def prepare_annotation(rid: int, request: Request,
                           cur: CurrentUser = Depends(require_write),
                           db: Session = Depends(get_db)):
        # Serialize creation for this source; a second tab reuses the first copy.
        obj = db.query(model).filter(model.id == rid, model.deleted_at.is_(None)).with_for_update().first()
        access.require_raw(db, cur, data_type, obj, operate=True)
        existing = _annotation_workspace(db, cur, data_type, rid)
        if existing:
            return existing
        files = _annotation_sources(db, data_type, rid)
        destination = None
        created_directory = False
        try:
            ds = Dataset(name=f'在线标注 · {obj.name}'[:255], data_type=data_type,
                         source_data_ids=json.dumps([rid]), secret_level=obj.secret_level,
                         description='从原始数据建立的在线标注工作副本', created_by=cur.id)
            db.add(ds); db.flush()
            destination = settings.upload_dir / 'processed' / str(ds.id) / 'V1'
            destination.mkdir(parents=True, exist_ok=False)
            created_directory = True
            version = DatasetVersion(dataset_id=ds.id, version_no=1, status='draft',
                                     stage='原始', file_path=str(destination),
                                     source_task_type='init', description='在线标注初始副本', created_by=cur.id)
            db.add(version); db.flush()
            ds.current_version_id = version.id
            size = 0; labelled = 0
            for _, rel, src in files:
                target = destination / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                size += target.stat().st_size
                label = find_label_path(db, data_type, rid, src, str(rel), cur=cur)
                if label:
                    label_target = target.with_name(f'{target.stem}_label.json')
                    shutil.copy2(label, label_target)
                    size += label_target.stat().st_size
                    labelled += 1
            ds.data_count = version.data_count = len(files)
            ds.label_count = labelled
            version.size_bytes = size
            db.commit()
        except Exception:
            db.rollback()
            if created_directory:
                shutil.rmtree(destination)
            raise
        write_log(db, user_id=cur.id, username=cur.username, action='建立在线标注副本',
                  module=f'data_{data_type}', detail=f'{obj.name} → 数据集 #{ds.id} V1',
                  ip=client_ip(request))
        return {'dataset_id': ds.id, 'version_id': version.id,
                'file_count': version.data_count, 'size_bytes': version.size_bytes}

    @router.post("/{rid}/label")
    async def upload_label(rid: int, label: UploadFile = File(...), request: Request = None,
                           cur: CurrentUser = Depends(require_write),
                           db: Session = Depends(get_db)):
        """上传标注文件（统一 _label.json 格式）。"""
        obj = _get_or_404(db, model, rid, cur)
        filename = label.filename or ""
        if not filename.endswith("_label.json"):
            stem = Path(filename).stem
            filename = f"{stem}_label.json"
        target = unique_path(fs.dataset_dir(data_type, rid, "labels"), filename)
        with open(target, "wb") as f:
            f.write(await label.read())
        rec = DataFile(dataset_id=rid, data_type=data_type, filename=label.filename or filename,
                       stored_name=target.name, size_bytes=target.stat().st_size,
                       stage="label", status="complete")
        db.add(rec)
        obj.label_file_path = str(target)
        if obj.status in ("未标注", "待标注"):
            obj.status = "已标注"
        obj.label_progress = max(obj.label_progress or 0, 100)
        db.commit()
        write_log(db, user_id=cur.id, username=cur.username, action="上传标注",
                  module=f"data_{data_type}",
                  detail=f"为 {obj.name}(id={rid}) 上传标注文件 {filename}",
                  ip=client_ip(request))
        return to_dict(rec)

    @router.post("/{rid}/labels/import")
    async def import_labels(rid: int, request: Request,
                            cur: CurrentUser = Depends(require_write),
                            db: Session = Depends(get_db)):
        """导入 COCO 风格单文件标注 JSON（images/annotations/categories），
        按 file_name 匹配拆分为每文件 {stem}_label.json 落盘到数据目录。

        COCO 整表是图像标注格式（按 file_name 与图片文件匹配），仅图像模态开放；
        语音/雷达/文本/视频等其他模态统一使用「单文件 _label.json」标注。
        """
        if data_type != "image":
            raise HTTPException(status_code=400,
                                detail="COCO 整表标注仅适用于图像数据；"
                                       "当前模态请使用单文件 _label.json 标注")
        obj = _get_or_404(db, model, rid, cur)
        cur.require_operate(obj.secret_level)
        form = await request.form()
        file = form.get("file")
        if file is None or not getattr(file, "filename", None):
            raise HTTPException(status_code=400, detail="请选择标注 JSON 文件")
        if not file.filename.lower().endswith(".json"):
            raise HTTPException(status_code=400, detail="仅支持 .json 标注文件")
        content = await file.read()
        if len(content) > 200 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="标注文件不能超过 200MB")
        try:
            coco = json.loads(content.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(status_code=400, detail="标注 JSON 解析失败")
        if not coco.get("images") or not coco.get("annotations"):
            raise HTTPException(status_code=400, detail="JSON 中未找到 COCO images/annotations 结构")
        # file_name → 数据目录内实际文件（支持带目录相对路径与纯文件名两种匹配）
        res = _import_coco_json(db, data_type, rid, coco)
        _refresh_aggregates(db, data_type, rid)
        write_log(db, user_id=cur.id, username=cur.username, action="导入标注",
                  module=f"data_{data_type}",
                  detail=f"向 {obj.name}(id={rid}) 导入 COCO 标注：{res['matched_files']} 个文件，{res['unmatched_annotations']} 条未匹配",
                  ip=client_ip(request))
        return res

    @router.get("/{rid}/files/{fid}/download")
    def download_file(rid: int, fid: int, request: Request = None,
                      cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
        """单文件下载。"""
        from fastapi.responses import FileResponse
        obj = _get_or_404(db, model, rid, cur)
        cur.require_view(obj.secret_level)
        rec = db.query(DataFile).filter(DataFile.id == fid, DataFile.data_type == data_type,
                                        DataFile.dataset_id == rid,
                                        DataFile.deleted_at.is_(None)).first()
        if not rec:
            raise HTTPException(status_code=404, detail="文件不存在")
        base = fs.dataset_dir(data_type, rid, "labels" if rec.stage == "label" else "data")
        path = base / _safe_rel_path(rec.stored_name)
        if not path.exists():
            raise HTTPException(status_code=404, detail="文件已被移除")
        return FileResponse(path, filename=path.name)

    @router.get("/{rid}/preview")
    def preview(rid: int, file_id: Optional[int] = None,
                cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
        obj = _get_or_404(db, model, rid, cur)
        cur.require_view(obj.secret_level)
        q = db.query(DataFile).filter(DataFile.data_type == data_type,
                                      DataFile.dataset_id == rid,
                                      DataFile.stage == "raw",
                                      DataFile.status == "complete",
                                      DataFile.deleted_at.is_(None))
        files = q.order_by(DataFile.id).all()
        if not files:
            raise HTTPException(status_code=404, detail="暂无可预览文件")
        first = next((f for f in files if f.id == file_id), None) or files[0]
        path = fs.dataset_dir(data_type, rid) / _safe_rel_path(first.stored_name)
        if not path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        # 标注叠加：优先读数据目录内同 stem 的 _label.json；
        # 无原始标注时回读「以本数据为来源的数据集」最新版本目录中的标注（标注任务产物）
        import json as _json
        label, label_source = None, None
        stem = path.stem
        for cand in (path.parent / f"{stem}_label.json",
                     fs.dataset_dir(data_type, rid, "labels") / f"{stem}_label.json"):
            if cand.exists():
                try:
                    label = _json.loads(cand.read_text(encoding="utf-8"))
                    label_source = "原始标注（随数据上传）"
                except ValueError:
                    label = None
                break
        if label is None:
            from ..models import Dataset, DatasetVersion
            rel_stored = Path(_safe_rel_path(first.stored_name))
            rel_label = str(rel_stored.parent / f"{rel_stored.stem}_label.json")
            flat_label = f"{stem}_label.json"
            dss = db.query(Dataset).filter(Dataset.data_type == data_type,
                                           Dataset.deleted_at.is_(None),
                                           Dataset.id.notin_(access.excluded_dataset_ids(db,cur))).all()
            cand_versions = []
            for ds_ in dss:
                try:
                    src_ids = _json.loads(ds_.source_data_ids or "[]")
                except ValueError:
                    src_ids = []
                if rid in src_ids:
                    vers = (db.query(DatasetVersion)
                            .filter(DatasetVersion.dataset_id == ds_.id)
                            .order_by(DatasetVersion.version_no.desc()).all())
                    for v_ in vers:
                        cand_versions.append((ds_, v_))
            for ds_, v_ in cand_versions:
                vroot = Path(v_.file_path)
                for rel in (rel_label, flat_label):
                    cand = vroot / rel
                    if cand.exists():
                        try:
                            label = _json.loads(cand.read_text(encoding="utf-8"))
                            label_source = f"数据集「{ds_.name}」V{v_.version_no}"
                        except ValueError:
                            label = None
                        break
                if label is not None:
                    break
        return {"file": first.stored_name, "type": data_type, "file_id": first.id,
                "files": [{"id": f.id, "name": f.filename} for f in files],
                "label": label, "label_source": label_source,
                "data": fs.preview_data(data_type, path)}

    @router.get("/{rid}/viz")
    def viz(rid: int, file_id: Optional[int] = None,
            cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
        obj = _get_or_404(db, model, rid, cur)
        cur.require_view(obj.secret_level)
        q = db.query(DataFile).filter(DataFile.data_type == data_type,
                                      DataFile.dataset_id == rid,
                                      DataFile.stage == "raw",
                                      DataFile.status == "complete",
                                      DataFile.deleted_at.is_(None))
        rec = q.filter(DataFile.id == file_id).first() if file_id else q.order_by(DataFile.id).first()
        if not rec:
            raise HTTPException(status_code=404, detail="暂无可视化文件")
        path = fs.dataset_dir(data_type, rid) / _safe_rel_path(rec.stored_name)
        if not path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        files = q.order_by(DataFile.id).all()
        return {"file": rec.stored_name, "type": data_type, "file_id": rec.id,
                "files": [{"id": f.id, "name": f.filename} for f in files],
                "data": fs.viz_data(data_type, path)}

    @router.get("/{rid}/label-stats")
    def label_stats(rid: int, cur: CurrentUser = Depends(require_login),
                    db: Session = Depends(get_db)):
        """数据集总览：标签类别分布 / 类别×文件覆盖 / 标注源构成（v3.2）。

        通过统一标注定位器读取原始标注、标注目录及本数据集/下游数据集版本产物。
        """
        obj = _get_or_404(db, model, rid, cur)
        cur.require_view(obj.secret_level)
        files = (db.query(DataFile)
                 .filter(DataFile.data_type == data_type, DataFile.dataset_id == rid,
                         DataFile.stage == "raw", DataFile.status == "complete",
                         DataFile.deleted_at.is_(None))
                 .order_by(DataFile.id).all())
        label_counts: dict = {}
        coverage: dict = {}
        source_counts: dict = {}
        for rec in files:
            path = fs.dataset_dir(data_type, rid) / _safe_rel_path(rec.stored_name)
            if not path.exists():
                continue
            label = find_label(db, data_type, rid, path, _safe_rel_path(rec.stored_name), cur)
            src = "无标注"
            labs = set()
            if isinstance(label, dict) and label:
                src = (f"自动（{label['algorithm']}）" if label.get("algorithm")
                       else "手动/导入")
            elif isinstance(label, list):
                src = "手动/导入"
            for lb in label_names(label):
                labs.add(lb)
                label_counts[lb] = label_counts.get(lb, 0) + 1
            source_counts[src] = source_counts.get(src, 0) + 1
            for lb in labs:
                coverage.setdefault(lb, set()).add(rec.id)
        return {"total_files": len(files),
                "labels": [{"label": k, "count": v} for k, v in
                           sorted(label_counts.items(), key=lambda x: -x[1])],
                "coverage": [{"label": k, "file_count": len(v)} for k, v in
                             sorted(coverage.items(), key=lambda x: -len(x[1]))],
                "sources": [{"source": k, "count": v} for k, v in
                            sorted(source_counts.items(), key=lambda x: -x[1])]}

    @router.post("/{rid}/labels/batch-import")
    async def import_labels_batch(rid: int, request: Request,
                                  cur: CurrentUser = Depends(require_write),
                                  db: Session = Depends(get_db)):
        """批量导入标注文件：一次上传多个 `<数据文件名>_label.json`（支持文件夹，保留相对路径）。

        一个数据集包含成百上千个文件时，逐个上传标注不现实；本接口按文件名自动对应：
        1. 标注文件必须命名为 `<数据文件stem>_label.json`；
        2. 优先按「相对路径 + 文件名」匹配（文件夹上传时保留目录结构）；
        3. 仅文件名匹配且同名多个文件时匹配首个，并在返回中给出歧义提示。
        标注写入数据文件同级目录，与 /preview 的标注回读规则一致。
        """
        obj = _get_or_404(db, model, rid, cur)
        cur.require_operate(obj.secret_level)
        form = await request.form()
        files = form.getlist("files")
        rels = [str(x) for x in form.getlist("rel_paths")]
        if not files:
            raise HTTPException(status_code=400, detail="请选择至少一个标注文件")
        if len(files) > 5000:
            raise HTTPException(status_code=400, detail="一次最多导入 5000 个标注文件")

        root = fs.dataset_dir(data_type, rid)
        # 数据集内数据文件索引：stem -> [文件路径]（排除标注文件本身与清单文件）
        data_index: dict = {}
        for p_ in root.rglob("*"):
            if not p_.is_file() or p_.name.endswith("_label.json") or p_.name == "_manifest.json":
                continue
            data_index.setdefault(p_.stem, []).append(p_)

        matched, unmatched, errors = [], [], []
        written = 0
        for i, up in enumerate(files):
            fname = getattr(up, "filename", "") or ""
            if not fname.lower().endswith("_label.json"):
                errors.append(f"{fname}: 标注文件必须以 _label.json 结尾")
                continue
            stem = Path(fname).name[:-len("_label.json")]
            cands = list(data_index.get(stem, []))
            rel = Path(rels[i].replace("\\", "/")) if i < len(rels) and rels[i] else None
            # 文件夹上传时前端给出的是 webkitRelativePath（带所选根目录名），
            # 因此相对目录按「相等 / 互为后缀」匹配，避免根目录名不同导致对不上。
            if rel is not None and len(rel.parts) > 1 and cands:
                want = str(rel.parent).replace("\\", "/").strip("/")
                if want and want != ".":
                    same = []
                    for c in cands:
                        try:
                            d = str(c.parent.relative_to(root)).replace("\\", "/")
                        except ValueError:
                            d = ""
                        if d == want or d.endswith("/" + want) or (d and want.endswith("/" + d)):
                            same.append(c)
                    if same:
                        cands = same
            if not cands:
                unmatched.append(fname)
                continue
            payload = await up.read()
            try:
                json.loads(payload.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                errors.append(f"{fname}: JSON 解析失败")
                continue
            target_data = cands[0]
            target = target_data.parent / f"{target_data.stem}_label.json"
            with open(target, "wb") as f:
                f.write(payload)
            written += 1
            entry = str(target_data.relative_to(root)).replace("\\", "/")
            if len(cands) > 1:
                entry += f"（同名 {len(cands)} 个，已匹配首个）"
            matched.append(entry)

        if written:
            if obj.status in ("未标注", "待标注"):
                obj.status = "已标注"
            total = sum(len(v) for v in data_index.values()) or 1
            obj.label_progress = max(obj.label_progress or 0, min(round(written / total * 100), 100))
            db.commit()
        write_log(db, user_id=cur.id, username=cur.username, action="批量导入标注",
                  module=f"data_{data_type}",
                  detail=f"向 {obj.name}(id={rid}) 批量导入标注 {written} 个"
                         f"（未匹配 {len(unmatched)}，错误 {len(errors)}）",
                  ip=client_ip(request))
        return {"matched": written, "matched_files": matched[:200],
                "unmatched": unmatched[:200], "unmatched_count": len(unmatched),
                "errors": errors[:200], "error_count": len(errors),
                "data_file_count": sum(len(v) for v in data_index.values())}

    return router
