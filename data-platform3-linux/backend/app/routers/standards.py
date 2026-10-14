"""可视化与标准层 + 结果回写 + 仪表盘路由（6.6/6.8）。"""
from __future__ import annotations
import json
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import (DATA_TYPE_MODELS, Dataset, DatasetVersion, QualityResult,
                      RefStandard, SemanticTerm, WritebackResult)
from ..security import CurrentUser, require_login, require_task_user, require_write
from ..services import file_service as fs
from ..services import version_service as vs
from ..services import data_access as access
from ..utils.common import SECRET_LEVELS
from ..services.log_service import client_ip, write_log
from ..services.quality_service import evaluate_quality
from ..utils.common import (apply_sort, json_default, paginate, parse_json, to_dict,
                              unique_path)

router = APIRouter(prefix="/api", tags=["standards"])


# ---------------- 结果回写（6.6） ----------------
class WritebackBody(BaseModel):
    model_name: str
    result_type: str = ""
    original_data_id: Optional[int] = None
    original_data_type: str = ""
    version: str = ""


@router.get("/writeback")
def list_writeback(page: int = 1, page_size: int = 20, sort: str = "",
                   cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    q = apply_sort(db.query(WritebackResult), WritebackResult, sort)
    if not cur.unrestricted:q=q.filter(WritebackResult.created_by==cur.id)
    result = paginate(q, page, page_size)
    result["items"] = [to_dict(x) for x in result["items"]]
    return result


@router.post("/writeback")
async def create_writeback(request: Request, cur: CurrentUser = Depends(require_write),
                           db: Session = Depends(get_db)):
    """multipart 上传结果文件 + 元数据字段。"""
    form = await request.form()
    file = form.get("file")
    model_name = form.get("model_name", "")
    if not model_name:
        raise HTTPException(status_code=400, detail="模型名称不能为空")
    saved_path, size = "", 0
    if file is not None and getattr(file, "filename", None):
        target_dir = settings.upload_dir / "writeback"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = unique_path(target_dir, Path(file.filename).name)
        with open(target, "wb") as f:
            f.write(await file.read())
        saved_path, size = str(target), target.stat().st_size
    rec = WritebackResult(model_name=model_name,
                          result_type=form.get("result_type", ""),
                          original_data_id=int(form["original_data_id"]) if form.get("original_data_id") else None,
                          original_data_type=form.get("original_data_type", ""),
                          file_path=saved_path, size_bytes=size,
                          status="成功", version=form.get("version", ""),
                          created_by=cur.id)
    db.add(rec)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="结果回写",
              module="writeback", detail=f"模型 {model_name} 回写结果（{form.get('result_type', '')}）",
              ip=client_ip(request))
    return to_dict(rec)


@router.get("/writeback/{wid}/download")
def download_writeback(wid: int, cur: CurrentUser = Depends(require_login),
                       db: Session = Depends(get_db)):
    rec = db.query(WritebackResult).filter(WritebackResult.id == wid).first()
    if not rec or not rec.file_path or not Path(rec.file_path).exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    if not cur.unrestricted and rec.created_by!=cur.id:
        raise HTTPException(403,"只能访问自己的回写结果")
    return FileResponse(rec.file_path, filename=Path(rec.file_path).name)


# ---------------- 仪表盘（6.8） ----------------
@router.get("/dashboard/stats")
def dashboard_stats(cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    out = {}
    total_size = 0
    for t, model in DATA_TYPE_MODELS.items():
        rows = db.query(model).filter(model.deleted_at.is_(None),
                                    model.id.notin_(access.excluded_raw_ids(db,cur,t)),
                                    model.secret_level.in_([lv for lv in SECRET_LEVELS if cur.can_view(lv)])).all()
        out[t] = {"count": len(rows), "size": sum(r.size_bytes or 0 for r in rows),
                  "labeled": sum(1 for r in rows if r.status == "已标注")}
        total_size += out[t]["size"]
    return {"by_type": out, "total_size": total_size,
            "dataset_count": db.query(Dataset).filter(Dataset.deleted_at.is_(None),Dataset.id.notin_(access.excluded_dataset_ids(db,cur)),Dataset.secret_level.in_([lv for lv in SECRET_LEVELS if cur.can_view(lv)])).count(),
            "version_count": db.query(DatasetVersion).join(Dataset,Dataset.id==DatasetVersion.dataset_id).filter(Dataset.id.notin_(access.excluded_dataset_ids(db,cur)),Dataset.secret_level.in_([lv for lv in SECRET_LEVELS if cur.can_view(lv)])).count(),
            "standard_count": db.query(RefStandard).count(),
            "term_count": db.query(SemanticTerm).count()}


@router.get("/dashboard/trend")
def dashboard_trend(days: int = 14, cur: CurrentUser = Depends(require_login),
                    db: Session = Depends(get_db)):
    """近 N 天各类数据接入趋势。"""
    days = min(max(days, 1), 90)
    start = datetime.now() - timedelta(days=days - 1)
    buckets = {}
    for i in range(days):
        d = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        buckets[d] = {t: 0 for t in DATA_TYPE_MODELS}
    for t, model in DATA_TYPE_MODELS.items():
        rows = db.query(model).filter(model.deleted_at.is_(None),
                                      model.created_at >= start,
                                      model.id.notin_(access.excluded_raw_ids(db,cur,t)),
                                      model.secret_level.in_([lv for lv in SECRET_LEVELS if cur.can_view(lv)])).all()
        for r in rows:
            key = r.created_at.strftime("%Y-%m-%d")
            if key in buckets:
                buckets[key][t] += 1
    return {"days": list(buckets.keys()),
            "series": {t: [buckets[d][t] for d in buckets] for t in DATA_TYPE_MODELS}}


@router.get("/dashboard/distribution")
def dashboard_distribution(cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    """按密级/状态/场景分布。"""
    dist = {"secret_level": {}, "status": {}, "scene": {}}
    for t, model in DATA_TYPE_MODELS.items():
        for r in db.query(model).filter(model.deleted_at.is_(None),
                                      model.id.notin_(access.excluded_raw_ids(db,cur,t)),
                                      model.secret_level.in_([lv for lv in SECRET_LEVELS if cur.can_view(lv)])).all():
            for field, key in (("secret_level", r.secret_level), ("status", r.status), ("scene", r.scene)):
                if key:
                    dist[field][key] = dist[field].get(key, 0) + 1
    return dist


@router.get("/dashboard/quality")
def dashboard_quality(cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    rows = db.query(QualityResult).order_by(QualityResult.created_at.desc()).limit(50).all()
    if not cur.unrestricted:
        allowed=[]
        for row in rows:
            try:check_quality_access(db,cur,row.data_type,row.data_id,row.version_id)
            except HTTPException:continue
            allowed.append(row)
        rows=allowed
    if not rows:
        return {"avg": None, "recent": []}
    dims = ["completeness", "accuracy", "consistency", "timeliness",
            "standardization", "accessibility"]
    avg = {d: round(sum(float(getattr(r, d) or 0) for r in rows) / len(rows), 1) for d in dims}
    avg["overall"] = round(sum(float(r.overall_score or 0) for r in rows) / len(rows), 1)
    return {"avg": avg, "recent": [to_dict(r) for r in rows[:10]]}


# ---------------- 质量评估（8.4） ----------------
def check_quality_access(db,cur,data_type,data_id,version_id=None,operate=False):
    if version_id:
        version=db.get(DatasetVersion,version_id)
        if version is None or version.dataset_id!=data_id:
            raise HTTPException(404,'版本不存在或不属于该数据集')
        ds=vs.get_dataset(db,data_id,cur)
        if data_type!=ds.data_type:raise HTTPException(400,'评估类型与数据集不一致')
        access.require_dataset(db,cur,ds,operate)
    else:
        model=DATA_TYPE_MODELS.get(data_type)
        if model is None:raise HTTPException(400,'数据类型无效')
        access.require_raw(db,cur,data_type,db.get(model,data_id),operate)


class EvaluateBody(BaseModel):
    data_type: str                    # 五类之一，或 "dataset"（预处理数据集版本）
    data_id: int
    version_id: Optional[int] = None     # 评估数据集版本
    standard_ids: Optional[List[int]] = None  # 勾选参与评估的标准（None=全部现行标准）


@router.post("/quality/evaluate")
def quality_evaluate(body: EvaluateBody, request: Request,
                     cur: CurrentUser = Depends(require_task_user), db: Session = Depends(get_db)):
    check_quality_access(db,cur,body.data_type,body.data_id,body.version_id,operate=True)
    result = evaluate_quality(db, body.data_type, body.data_id, body.version_id, cur.id,
                              body.standard_ids)
    write_log(db, user_id=cur.id, username=cur.username, action="质量评估",
              module="quality",
              detail=f"评估 {body.data_type}/{body.data_id}"
                     + (f" 版本#{body.version_id}" if body.version_id else "")
                     + f"，总分 {result['overall_score']}",
              ip=client_ip(request))
    return result


@router.get("/quality/{data_type}/{data_id}")
def quality_query(data_type: str, data_id: int, version_id: Optional[int] = None,
                  cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    check_quality_access(db,cur,data_type,data_id,version_id)
    q = db.query(QualityResult).filter(QualityResult.data_type == data_type,
                                       QualityResult.data_id == data_id)
    q = q.filter(QualityResult.version_id == version_id) if version_id else q.filter(QualityResult.version_id.is_(None))
    rows = q.order_by(QualityResult.created_at.desc()).all()
    items = []
    for r in rows:
        d = to_dict(r)
        d["details"] = parse_json(r.details, {})
        d["standards_compliance"] = parse_json(r.standards_compliance, [])
        items.append(d)
    return {"items": items}


@router.get("/quality/{data_type}/{data_id}/export")
def quality_export(data_type: str, data_id: int, fmt: str = "json",
                   version_id: Optional[int] = None,
                   cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    """质量报告导出 JSON/CSV（不做 PDF）。"""
    check_quality_access(db,cur,data_type,data_id,version_id)
    q = db.query(QualityResult).filter(QualityResult.data_type == data_type,
                                       QualityResult.data_id == data_id)
    q = q.filter(QualityResult.version_id == version_id) if version_id else q.filter(QualityResult.version_id.is_(None))
    r = q.order_by(QualityResult.created_at.desc()).first()
    if not r:
        raise HTTPException(status_code=404, detail="暂无评估结果")
    report = to_dict(r)
    report["details"] = parse_json(r.details, {})
    report["standards_compliance"] = parse_json(r.standards_compliance, [])
    if fmt == "csv":
        import csv, io
        buf = io.StringIO()
        buf.write("﻿")
        w = csv.writer(buf)
        w.writerow(["维度", "得分"])
        for dim, name in (("completeness", "完整性"), ("accuracy", "准确性"),
                          ("consistency", "一致性"), ("timeliness", "时效性"),
                          ("standardization", "规范性"), ("accessibility", "可访问性"),
                          ("overall_score", "综合得分")):
            w.writerow([name, report[dim]])
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": f"attachment; filename=quality_{data_type}_{data_id}.csv"})
    return StreamingResponse(iter([json.dumps(report, ensure_ascii=False, indent=2, default=json_default)]),
                             media_type="application/json",
                             headers={"Content-Disposition": f"attachment; filename=quality_{data_type}_{data_id}.json"})


class TermBody(BaseModel):
    term: str
    en_name: str = ""
    category: str = ""
    definition: str = ""
    synonyms: List[str] = []
    unit: str = ""
    secret_level: str = "内部"
    status: str = "现行"


@router.get("/semantic/terms")
def list_terms(page: int = 1, page_size: int = 20, search: str = "",
               cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    q = db.query(SemanticTerm).filter(SemanticTerm.secret_level.in_([lv for lv in SECRET_LEVELS if cur.can_view(lv)]))
    if search:
        like = f"%{search}%"
        q = q.filter(SemanticTerm.term.like(like) | SemanticTerm.en_name.like(like)
                     | SemanticTerm.synonyms.like(like))
    result = paginate(q.order_by(SemanticTerm.id), page, page_size)
    items = []
    for t in result["items"]:
        d = to_dict(t)
        d["synonyms"] = parse_json(t.synonyms, [])
        items.append(d)
    result["items"] = items
    return result


@router.post("/semantic/terms")
def create_term(body: TermBody, request: Request,
                cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
    if not body.term or not body.definition:
        raise HTTPException(status_code=400, detail="术语名称与定义不能为空")
    t = SemanticTerm(term=body.term, en_name=body.en_name, category=body.category,
                     definition=body.definition,
                     synonyms=json.dumps(body.synonyms, ensure_ascii=False),
                     unit=body.unit, secret_level=body.secret_level,
                     status=body.status, created_by=cur.id)
    db.add(t)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="新增术语",
              module="semantic", detail=f"新增术语 {body.term}", ip=client_ip(request))
    d = to_dict(t)
    d["synonyms"] = body.synonyms
    return d


@router.put("/semantic/terms/{tid}")
def update_term(tid: int, body: TermBody, request: Request,
                cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
    t = db.query(SemanticTerm).filter(SemanticTerm.id == tid).first()
    if not t:
        raise HTTPException(status_code=404, detail="术语不存在")
    t.term, t.en_name, t.category, t.definition = body.term, body.en_name, body.category, body.definition
    t.synonyms = json.dumps(body.synonyms, ensure_ascii=False)
    t.unit, t.secret_level, t.status = body.unit, body.secret_level, body.status
    t.updated_at = datetime.now()
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="编辑术语",
              module="semantic", detail=f"更新术语 {body.term}", ip=client_ip(request))
    d = to_dict(t)
    d["synonyms"] = body.synonyms
    return d


@router.delete("/semantic/terms/{tid}")
def delete_term(tid: int, request: Request,
                cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
    t = db.query(SemanticTerm).filter(SemanticTerm.id == tid).first()
    if not t:
        raise HTTPException(status_code=404, detail="术语不存在")
    db.delete(t)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="删除术语",
              module="semantic", detail=f"删除术语 {t.term}", ip=client_ip(request))
    return {"message": "已删除"}


class SemanticCheckBody(BaseModel):
    text: str


@router.post("/semantic/check")
def semantic_check(body: SemanticCheckBody, cur: CurrentUser = Depends(require_login),
                   db: Session = Depends(get_db)):
    """术语标准化检查：识别文本中命中的同义词，给出标准术语建议。"""
    terms = db.query(SemanticTerm).filter(SemanticTerm.status == "现行", SemanticTerm.secret_level.in_([lv for lv in SECRET_LEVELS if cur.can_view(lv)])).all()
    hits = []
    for t in terms:
        for syn in parse_json(t.synonyms, []):
            if syn and syn in body.text:
                hits.append({"synonym": syn, "standard": t.term,
                             "en_name": t.en_name, "category": t.category,
                             "count": body.text.count(syn)})
    return {"hits": hits, "hit_count": len(hits)}


@router.post("/semantic/replace")
def semantic_replace(body: dict, request: Request,
                     cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
    """一键替换：将文本中的同义词替换为标准术语。"""
    text = body.get("text", "")
    terms = db.query(SemanticTerm).filter(SemanticTerm.status == "现行", SemanticTerm.secret_level.in_([lv for lv in SECRET_LEVELS if cur.can_view(lv)])).all()
    replaced = 0
    for t in terms:
        for syn in parse_json(t.synonyms, []):
            if syn and syn in text:
                replaced += text.count(syn)
                text = text.replace(syn, t.term)
    write_log(db, user_id=cur.id, username=cur.username, action="术语替换",
              module="semantic", detail=f"替换 {replaced} 处非标准用词", ip=client_ip(request))
    return {"text": text, "replaced": replaced}


# ---------------- 参考标准管理 ----------------
class StandardBody(BaseModel):
    code: str
    name: str
    standard_type: str = "国家标准"
    dimensions: str = ""
    scope: str = ""
    content: str = ""
    rules: Optional[List[dict]] = None  # 自定义评估规则 [{metric, op, threshold, name?}]


@router.post("/standards/{sid}/file")
async def upload_standard_file(sid: int, request: Request,
                               cur: CurrentUser = Depends(require_write),
                               db: Session = Depends(get_db)):
    """上传标准原文附件（PDF/Word/图片，≤50MB），下载时优先返回原文。"""
    from fastapi import UploadFile
    from fastapi import File as FFile
    s = db.query(RefStandard).filter(RefStandard.id == sid).first()
    if not s:
        raise HTTPException(status_code=404, detail="标准不存在")
    form = await request.form()
    file = form.get("file")
    if file is None or not getattr(file, "filename", None):
        raise HTTPException(status_code=400, detail="请选择文件")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        raise HTTPException(status_code=400, detail="仅支持 PDF / Word / 图片格式的标准原文")
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="标准原文不能超过 50MB")
    target_dir = settings.upload_dir / "standards"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = unique_path(target_dir, f"{s.code.replace('/', '_')}{suffix}")
    with open(target, "wb") as f:
        f.write(content)
    s.file_path = str(target)
    s.updated_at = datetime.now()
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="上传标准原文",
              module="standard", detail=f"为标准 {s.code} 上传原文 {Path(file.filename).name}",
              ip=client_ip(request))
    return to_dict(s)


@router.get("/standards")
def list_standards(cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    rows = db.query(RefStandard).order_by(RefStandard.id).all()
    items = []
    for r in rows:
        d = to_dict(r)
        d["rules"] = parse_json(r.rules, [])
        d["has_file"] = bool(r.file_path and Path(r.file_path).exists())
        items.append(d)
    return {"items": items}


@router.post("/standards")
def create_standard(body: StandardBody, request: Request,
                    cur: CurrentUser = Depends(require_write), db: Session = Depends(get_db)):
    if db.query(RefStandard).filter(RefStandard.code == body.code).first():
        raise HTTPException(status_code=400, detail="标准编号已存在")
    is_custom = body.standard_type == "自定义标准" or bool(body.rules)
    s = RefStandard(code=body.code, name=body.name, standard_type=body.standard_type,
                    dimensions=body.dimensions, scope=body.scope, content=body.content,
                    rules=json.dumps(body.rules, ensure_ascii=False) if body.rules else "",
                    is_custom=is_custom, created_by=cur.id)
    db.add(s)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="新增参考标准",
              module="standard", detail=f"新增标准 {body.code}", ip=client_ip(request))
    return to_dict(s)


@router.get("/standards/{sid}/file")
def download_standard_file(sid: int, cur: CurrentUser = Depends(require_login),
                           db: Session = Depends(get_db)):
    """下载标准原文附件（必须已上传）。"""
    s = db.query(RefStandard).filter(RefStandard.id == sid).first()
    if not s:
        raise HTTPException(status_code=404, detail="标准不存在")
    if not (s.file_path and Path(s.file_path).exists()):
        raise HTTPException(status_code=404, detail="该标准尚未上传原文文件")
    return FileResponse(s.file_path, filename=Path(s.file_path).name)


@router.get("/standards/{sid}/download")
def download_standard(sid: int, cur: CurrentUser = Depends(require_login),
                      db: Session = Depends(get_db)):
    s = db.query(RefStandard).filter(RefStandard.id == sid).first()
    if not s:
        raise HTTPException(status_code=404, detail="标准不存在")
    if s.file_path and Path(s.file_path).exists():
        return FileResponse(s.file_path, filename=Path(s.file_path).name)
    # 无附件时导出标准信息 JSON
    content = json.dumps(to_dict(s), ensure_ascii=False, indent=2, default=json_default)
    return StreamingResponse(iter([content]), media_type="application/json",
                             headers={"Content-Disposition": f"attachment; filename=standard_{sid}.json"})
