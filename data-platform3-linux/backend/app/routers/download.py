"""数据下载：场景化取数（v3.2）。

端点：
- GET  /api/download/scenes   当前模态场景选项（下载条件选择用）
- GET  /api/labels            跨数据集标签全集（下载条件选择用）
- POST /api/download/query    按模态/场景/标签/项目/密级查询含指定标签的文件清单
- POST /api/download/pack     按文件清单打包 zip 下载（附 _lineage.json 血缘）
- POST /api/aigc/generate     配置补充数量；用户在少样本页确认后才提交训练。
  下载请求保存查询快照并关联少样本任务；产物下载选择具体预处理版本。

密级：全程后端强制 require_view 可见密级过滤，与数据获取层口径一致。
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import DATA_TYPE_MODELS, DataFile, DownloadRequest
from ..security import CurrentUser, require_login, require_task_user
from ..services.label_locator import find_label, find_label_with_path
from ..services.label_schema import label_names
from ..services import data_access as access, download_service as downloads
from ..services.log_service import client_ip, write_log
from ..utils.common import json_default
import time as _time

router = APIRouter(prefix="/api", tags=["download"])

DATA_TYPES = ("voice", "radar", "image", "video", "text")


# ---------------- 下载筛选选项 ----------------

@router.get("/download/scenes")
def list_download_scenes(data_type: str = "", cur: CurrentUser = Depends(require_login),
                         db: Session = Depends(get_db)):
    """返回当前模态下用户可见的数据场景，供下载页搜索选择。"""
    types = (data_type,) if data_type in DATA_TYPES else DATA_TYPES
    scenes = set()
    for t in types:
        model = DATA_TYPE_MODELS[t]
        for ds in db.query(model).filter(model.deleted_at.is_(None), model.id.notin_(access.excluded_raw_ids(db, cur, t))).all():
            scene = (ds.scene or "").strip()
            if not scene:
                continue
            try:
                cur.require_view(ds.secret_level)
            except HTTPException:
                continue
            scenes.add(scene)
    return {"items": sorted(scenes)}


# ---------------- 标注归一 ----------------

def _labels_of(label) -> set:
    """跨标注结构归一提取标签集合：detections/items/segments 用 label，entities 用 type；
    兼容顶层为标注项数组（list）的手写导出格式。"""
    return set(label_names(label))


def _iter_labelled_files(db: Session, data_type: str, cur: CurrentUser,
                         project: str = "", scene: str = "",
                         secret_level: str = "") -> list:
    """遍历指定模态全部可见数据集，产出 (dataset, data_file, labels) 三元组。"""
    model = DATA_TYPE_MODELS[data_type]
    q = db.query(model).filter(model.deleted_at.is_(None), model.id.notin_(access.excluded_raw_ids(db, cur, data_type)))
    if project:
        q = q.filter(model.project == project)
    if scene:
        q = q.filter(model.scene == scene)
    if secret_level:
        q = q.filter(model.secret_level == secret_level)
    rows = []
    for ds in q.all():
        try:
            cur.require_view(ds.secret_level)
        except HTTPException:
            continue  # 密级不可见，静默跳过（与数据获取层列表口径一致）
        root = Path(ds.file_path)
        for p in sorted(root.rglob("*")):
            if p.is_file() and not p.name.endswith("_label.json") and p.name != "_manifest.json":
                try:
                    rel_name = str(p.relative_to(root))
                except ValueError:
                    rel_name = None
                rows.append((ds, p, _labels_of(find_label(db, data_type, ds.id, p, rel_name, cur))))
    return rows


# ---------------- 标签全集 ----------------

_LABELS_CACHE: dict = {}  # (data_type, secret_level) -> (timestamp, labels)；TTL 60s
_LABELS_TTL = 60.0


@router.get("/labels")
def list_labels(data_type: str = "", cur: CurrentUser = Depends(require_login),
                db: Session = Depends(get_db)):
    """跨数据集标签全集（下载条件里"标签"多选下拉的数据源）。

    全量扫标注文件较贵，按 (模态, 用户密级) 做 60s 内存缓存；新增标注后
    最坏 60s 延迟可见，对下载场景可接受。"""
    key = (data_type or "*", getattr(cur, "secret_level", "") or str(cur.id))
    hit = _LABELS_CACHE.get(key)
    if hit and _time.time() - hit[0] < _LABELS_TTL:
        return {"labels": hit[1], "cached": True}
    types = (data_type,) if data_type in DATA_TYPES else DATA_TYPES
    labels = set()
    for t in types:
        for _, _, labs in _iter_labelled_files(db, t, cur):
            labels.update(labs)
    result = sorted(labels)
    _LABELS_CACHE[key] = (_time.time(), result)
    return {"labels": result, "cached": False}


# ---------------- 条件查询 ----------------

class QueryBody(BaseModel):
    data_type: str
    scene: Optional[str] = ""
    labels: List[str] = []
    project: Optional[str] = ""
    secret_level: Optional[str] = ""
    required_count: int = Field(default=100, ge=1, le=100000)


@router.post("/download/query")
def download_query(body: QueryBody, request: Request,
                   cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    """按条件查询含指定标签的文件清单；matched_count 用于与需求数量对比算缺口。"""
    files = query_files(body, cur, db)
    saved = downloads.create_request(db, cur, body.model_dump(), files)
    write_log(db, user_id=cur.id, username=cur.username, action="数据下载查询",
              module="download", detail=f"模态 {body.data_type}，标签 {'/'.join(body.labels)}，命中 {len(files)} 条",
              ip=client_ip(request))
    return {"files": files, "matched_count": len(files), "download_request_id": saved.id}


def query_files(body, cur, db):
    if body.data_type not in DATA_TYPES:
        raise HTTPException(status_code=400, detail="模态无效")
    wanted = set(body.labels)
    files = []
    for ds, p, labs in _iter_labelled_files(db, body.data_type, cur,
                                            project=body.project or "",
                                            scene=body.scene or "",
                                            secret_level=body.secret_level or ""):
        hit = labs & wanted
        if wanted and not hit:
            continue
        try:
            rel_name = str(p.relative_to(Path(ds.file_path)))
        except ValueError:
            rel_name = p.name
        files.append({
            "file_id": _file_id_of(db, body.data_type, ds.id, p, rel_name),
            "file_name": p.name, "dataset_id": ds.id, "dataset_name": ds.name,
            "data_type": body.data_type, "scene": ds.scene,
            "secret_level": ds.secret_level, "source_type": getattr(ds, "source_type", "真实采集"),
            "matched_labels": sorted(hit if wanted else labs),
        })
    return files


@router.get('/download/requests')
def list_download_requests(all_users: bool = False, page: int = 1,
                           cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    if all_users and not cur.unrestricted:
        raise HTTPException(403, '只有管理员可以查看其他用户的下载请求')
    q=db.query(DownloadRequest)
    if not all_users:q=q.filter(DownloadRequest.created_by==cur.id)
    rows=q.order_by(DownloadRequest.id.desc()).offset((max(1,page)-1)*50).limit(50).all()
    return {'items':[downloads.request_summary(db,r) for r in rows]}


@router.get('/download/requests/{rid}')
def download_request_detail(rid: int,cur: CurrentUser=Depends(require_login),db: Session=Depends(get_db)):
    saved=downloads.get_request(db,rid,cur)
    result=downloads.request_detail(db,saved,cur)
    files=query_files(QueryBody(**result['query']),cur,db)
    result.update(files=files,current_matched_count=len(files))
    return result


@router.get('/download/requests/{rid}/candidates/{fid}/thumbnail')
def candidate_thumbnail(rid: int,fid: int,cur: CurrentUser=Depends(require_login),db: Session=Depends(get_db)):
    from PIL import Image,ImageOps
    saved=downloads.get_request(db,rid,cur)
    _,_,path=downloads.resolve_candidate(db,saved,fid,cur)
    if path.stat().st_size>20*1024**2:raise HTTPException(422,'图片超过 20 MB')
    try:
        with Image.open(path) as im:
            if im.width*im.height>40_000_000:raise ValueError('图片尺寸过大')
            im.thumbnail((256,256));buf=io.BytesIO()
            ImageOps.exif_transpose(im).convert('RGB').save(buf,format='JPEG')
    except (OSError,ValueError,Image.DecompressionBombError):raise HTTPException(422,'无法预览图像')
    return Response(buf.getvalue(),media_type='image/jpeg',headers={'Cache-Control':'private, max-age=60'})


def _file_id_of(db: Session, data_type: str, dataset_id: int, p: Path,
                rel_name: Optional[str] = None) -> Optional[int]:
    candidates = []
    if rel_name:
        normalized = str(rel_name).replace("\\", "/")
        candidates.extend([normalized, normalized.replace("/", "\\")])
    candidates.append(p.name)
    for stored_name in dict.fromkeys(candidates):
        rec = (db.query(DataFile)
               .filter(DataFile.data_type == data_type, DataFile.dataset_id == dataset_id,
                      DataFile.stored_name == stored_name, DataFile.deleted_at.is_(None))
               .first())
        if rec:
            return rec.id
    return None


# ---------------- 打包下载 ----------------

class PackBody(BaseModel):
    file_ids: List[int]


def _zip_files(entries: list, lineage: dict):
    """entries: [(arcname, abs_path)]；附 _lineage.json。"""
    import tempfile
    from ..services.file_service import temp_dir
    with tempfile.TemporaryFile(dir=temp_dir()) as buf:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for arc, src in entries:
                zf.write(src, arc)
            zf.writestr("_lineage.json", json.dumps(lineage, ensure_ascii=False, indent=2,
                                                    default=json_default))
        buf.seek(0)
        while chunk := buf.read(1024 * 1024):
            yield chunk


@router.post("/download/pack")
def download_pack(body: PackBody, request: Request,
                  cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    """按 file_id 列表跨数据集打包下载；血缘清单随包附带。"""
    if not body.file_ids:
        raise HTTPException(status_code=400, detail="file_ids 不能为空")
    if len(body.file_ids) > 2000:
        raise HTTPException(status_code=400, detail="单次打包不超过 2000 个文件，请按条件分批下载")
    entries, lineage_files = [], []
    seen_arcs = set()
    missing = 0
    data_count = 0
    label_count = 0
    total_bytes = 0
    PACK_BYTES_LIMIT = 20 * 1024 ** 3  # 20GB：zip 当前为内存缓冲，超限防 OOM
    for fid in body.file_ids:
        rec = db.query(DataFile).filter(DataFile.id == fid, DataFile.deleted_at.is_(None)).first()
        if not rec:
            missing += 1
            continue
        model = DATA_TYPE_MODELS[rec.data_type]
        ds = db.query(model).filter(model.id == rec.dataset_id,
                                    model.deleted_at.is_(None)).first()
        if not ds:
            missing += 1
            continue
        access.require_raw(db, cur, rec.data_type, ds)
        src = Path(ds.file_path) / rec.stored_name
        if not src.exists():
            missing += 1
            continue
        total_bytes += int(getattr(rec, "size_bytes", 0) or 0)
        if total_bytes > PACK_BYTES_LIMIT:
            raise HTTPException(status_code=400, detail="打包总体积超过 20GB 上限，请按条件分批下载")

        stored_rel = str(rec.stored_name).replace("\\", "/").lstrip("/")
        data_arc = f"{ds.name}/data/{stored_rel}"
        if data_arc not in seen_arcs:
            entries.append((data_arc, src))
            seen_arcs.add(data_arc)
            data_count += 1

        label_path, _ = find_label_with_path(db, rec.data_type, ds.id, src, rec.stored_name, cur)
        label_arc = ""
        if label_path:
            stored_path = Path(stored_rel)
            label_rel = (stored_path.parent / f"{stored_path.stem}_label.json").as_posix()
            label_arc = f"{ds.name}/labels/{label_rel}"
            if label_arc not in seen_arcs:
                entries.append((label_arc, label_path))
                seen_arcs.add(label_arc)
                label_count += 1
                total_bytes += label_path.stat().st_size
                if total_bytes > PACK_BYTES_LIMIT:
                    raise HTTPException(status_code=400, detail="打包总体积超过 20GB 上限，请按条件分批下载")

        lineage_files.append({"file_id": rec.id, "file": rec.filename,
                              "dataset_id": ds.id, "dataset_name": ds.name,
                              "data_type": rec.data_type, "secret_level": ds.secret_level,
                              "scene": ds.scene, "source_type": getattr(ds, "source_type", "真实采集"),
                              "data_file": data_arc, "label_file": label_arc,
                              "label_included": bool(label_path)})
    if not entries:
        raise HTTPException(status_code=404, detail="没有可打包的有效文件")
    lineage = {"type": "download_pack", "created_by": cur.username,
               "file_count": data_count, "label_count": label_count,
               "files": lineage_files}
    write_log(db, user_id=cur.id, username=cur.username, action="数据下载打包",
              module="download", detail=f"打包 {data_count} 个数据文件、{label_count} 个标签文件（缺 {missing}）",
              ip=client_ip(request))
    return StreamingResponse(
        _zip_files(entries, lineage),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=download_{cur.id}_{data_count}files.zip"})


# ---------------- AIGC 生成补足（空接口） ----------------

class AigcBody(BaseModel):
    download_request_id: int
    count: int = Field(ge=1,le=1000)


@router.post('/aigc/generate')
def aigc_generate(body: AigcBody,request: Request,
                  cur: CurrentUser=Depends(require_task_user),db: Session=Depends(get_db)):
    saved=downloads.get_request(db,body.download_request_id,cur,own=True)
    query=json.loads(saved.query_json)
    if query['data_type']!='image':
        raise HTTPException(400,'少样本补充目前支持图像数据')
    shortage=max(0,query['required_count']-saved.matched_count)
    if body.count>shortage:raise HTTPException(422,'补充数量不能超过这次查询的缺口')
    saved.generation_count=body.count;db.commit()
    write_log(db,user_id=cur.id,username=cur.username,action='配置 AIGC 补充',module='aigc_generate',
              detail=f'下载请求 #{saved.id}，候选素材需用户确认，尚未提交训练',ip=client_ip(request))
    return {'download_request_id':saved.id,'status':'awaiting_confirmation','estimated_count':body.count}
