"""数据服务网关（数据应用层对外供数）：按场景聚合原子能力，一次调用返回组合结果。

四个场景（scene）：
- train   训练供数：版本打包 zip 流，zip 内附带 _lineage.json / _quality.json
- analyze 分析供数：版本演化 + 文件清单 + 预览数据，不落地取数
- quality 质量门禁：六维质量评估 + 标准合规 + 发布建议（会写库留痕，需可操作密级）
- graph   图谱供数：文本内容 + 实体标注 + 术语语义 + 血缘（仅文本型数据集）

横切规则：登录认证（Cookie 或 Authorization: Bearer）+ 六级密级后端强制 +
操作审计（OperationLog + 文件日志）。内部仍复用原子 service，不绕开既有校验。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Dataset, DatasetVersion, QualityResult, SemanticTerm
from ..security import CurrentUser, require_login
from ..services import file_service as fs
from ..services import preprocess as pp
from ..services import quality_service as qs
from ..services import version_service as vs
from ..services.log_service import client_ip, write_log
from ..utils.common import json_default

router = APIRouter(prefix="/api/service", tags=["data_service"])

SCENES = ("train", "analyze", "quality", "graph")
GRAPH_TEXT_SUFFIXES = {".txt", ".md", ".json", ".csv", ".xml", ".log"}
GRAPH_FILE_MAX_BYTES = 200 * 1024        # 图谱场景单文件读取上限
GRAPH_SNIPPET_BYTES = 20 * 1024          # with_text 时每文件返回的文本长度


class SupplyBody(BaseModel):
    scene: str                                  # train / analyze / quality / graph
    dataset_id: int
    version_id: Optional[int] = None            # 缺省 = 数据集当前版本
    include: str = "data,labels,manifest"       # train：zip 内容筛选
    standard_ids: List[int] = []                # quality：限定评估标准子集
    file_path: str = ""                         # analyze：版本目录内相对路径，定位预览文件
    with_text: bool = False                     # graph：是否随节点返回文本片段
    max_files: int = 50                         # analyze/graph：清单与节点数量上限（1~500）


# ---------------- 公共工具 ----------------

def _get_input_version(db: Session, ds: Dataset, version_id: Optional[int]) -> DatasetVersion:
    vid = version_id or ds.current_version_id
    if not vid:
        raise HTTPException(status_code=400, detail="数据集暂无可用版本")
    return vs.get_version(db, ds.id, vid)


def _version_brief(v: DatasetVersion) -> dict:
    return {"id": v.id, "version_no": v.version_no, "version": f"V{v.version_no}",
            "stage": v.stage, "status": v.status,
            "quality_score": float(v.quality_score) if v.quality_score is not None else None,
            "data_count": v.data_count, "size_bytes": v.size_bytes,
            "published_at": v.published_at.strftime("%Y-%m-%d %H:%M:%S") if v.published_at else None,
            "created_at": v.created_at.strftime("%Y-%m-%d %H:%M:%S") if v.created_at else None}


def _latest_quality(db: Session, version_id: int) -> dict:
    """最近一次版本评估结果（无则标记未评估）。"""
    rec = (db.query(QualityResult)
           .filter(QualityResult.version_id == version_id)
           .order_by(QualityResult.id.desc()).first())
    if not rec:
        return {"evaluated": False}
    dims = {k: float(getattr(rec, k)) for k in qs.DIM_NAMES}
    return {"evaluated": True, "overall_score": float(rec.overall_score),
            "dimensions": dims, "evaluated_at": rec.created_at.strftime("%Y-%m-%d %H:%M:%S")}


def _resolve_in_root(root: Path, rel: str) -> Path:
    """版本目录内相对路径解析 + 目录穿越防护。"""
    base = root.resolve()
    p = (base / rel).resolve()
    try:
        p.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="file_path 越出版本目录")
    return p


def _zip_supply(source: Path, include: str, lineage: dict, quality: dict):
    """训练供数打包：内容筛选（data/labels/manifest）+ 附带血缘与质量清单。"""
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
            zf.writestr("_lineage.json", json.dumps(lineage, ensure_ascii=False, indent=2, default=json_default))
            zf.writestr("_quality.json", json.dumps(quality, ensure_ascii=False, indent=2, default=json_default))
        buf.seek(0)
        while chunk := buf.read(1024 * 1024):
            yield chunk


def _release_advice(overall: float, compliance: list) -> str:
    """发布建议：总分≥80 且每条标准合规率≥60 视为可发布。"""
    ok = overall >= 80 and all(c.get("compliance_rate", 0) >= 60 for c in compliance)
    return "可发布" if ok else "不建议发布"


# ---------------- 场景实现 ----------------

def _scene_train(db: Session, ds: Dataset, v: DatasetVersion, body: SupplyBody) -> StreamingResponse:
    lineage = vs.lineage_of(db, v)
    quality = _latest_quality(db, v.id)
    source = Path(v.file_path)
    return StreamingResponse(
        _zip_supply(source, body.include, lineage, quality),
        media_type="application/zip",
        headers={"Content-Disposition":
                 f"attachment; filename=dataset_{ds.id}_V{v.version_no}_train.zip"})


def _scene_analyze(db: Session, ds: Dataset, v: DatasetVersion, body: SupplyBody) -> dict:
    root = Path(v.file_path)
    data_files = pp.data_files(root)[: max(1, min(body.max_files, 500))]
    files = []
    for p in data_files:
        rel = str(p.relative_to(root))
        files.append({"path": rel, "size": p.stat().st_size,
                      "has_label": p.with_name(p.stem + "_label.json").exists()})
    preview = None
    if data_files:
        target = None
        if body.file_path:
            p = _resolve_in_root(root, body.file_path)
            if p.is_file() and not p.name.endswith("_label.json"):
                target = p
        if target is None:
            target = data_files[0]
        try:
            preview = {"file": str(target.relative_to(root)),
                       "data": fs.preview_data(ds.data_type, target)}
        except Exception as e:  # 单个文件损坏不应阻断整体分析
            preview = {"file": str(target.relative_to(root)), "error": str(e)}
    versions = (db.query(DatasetVersion)
                .filter(DatasetVersion.dataset_id == ds.id)
                .order_by(DatasetVersion.version_no).all())
    return {"scene": "analyze", "dataset": {"id": ds.id, "name": ds.name,
                                            "data_type": ds.data_type,
                                            "secret_level": ds.secret_level},
            "version": _version_brief(v),
            "versions": [_version_brief(x) for x in versions],
            "quality": _latest_quality(db, v.id),
            "files": files, "preview": preview}


def _scene_quality(db: Session, cur: CurrentUser, ds: Dataset,
                   v: DatasetVersion, body: SupplyBody) -> dict:
    if body.standard_ids:
        found = db.query(RefStandard).filter(RefStandard.id.in_(body.standard_ids)).count()
        if found != len(body.standard_ids):
            raise HTTPException(status_code=400, detail="存在无效的 standard_id")
    result = qs.evaluate_quality(db, ds.data_type, ds.id, v.id, cur.id,
                                 body.standard_ids or None)
    result["scene"] = "quality"
    result["version"] = f"V{v.version_no}"
    result["release_advice"] = _release_advice(result["overall_score"],
                                               result["standards_compliance"])
    return result


def _scene_graph(db: Session, ds: Dataset, v: DatasetVersion, body: SupplyBody, cur=None) -> dict:
    if ds.data_type != "text":
        raise HTTPException(status_code=400,
                            detail="graph 场景仅支持文本型数据集（data_type=text）")
    root = Path(v.file_path)
    data_files = [p for p in pp.data_files(root)
                  if p.suffix.lower() in GRAPH_TEXT_SUFFIXES]
    data_files = data_files[: max(1, min(body.max_files, 500))]
    nodes, edges, entities, snippets = [], [], [], []
    for p in data_files:
        rel = str(p.relative_to(root))
        nodes.append({"id": rel, "type": "file", "name": p.name})
        raw = p.read_bytes()[:GRAPH_FILE_MAX_BYTES]
        text = raw.decode("utf-8", errors="ignore")
        if body.with_text:
            snippets.append({"file": rel, "text": text[:GRAPH_SNIPPET_BYTES]})
        label_path = p.with_name(p.stem + "_label.json")
        if not label_path.exists():
            continue
        try:
            label = json.loads(label_path.read_text(encoding="utf-8", errors="ignore"))
        except (ValueError, OSError):
            continue
        if isinstance(label, dict) and label.get("error"): continue
        for e in (label if isinstance(label, list) else (label.get("entities") or label.get("annotations") or [])):
            if not isinstance(e, dict): continue
            name = e.get("text") or e.get("entity") or ""
            etype = e.get("label") or e.get("type") or ""
            if not name:
                continue
            entities.append({"file": rel, "entity": name, "type": etype})
            entity_id = "entity:" + etype + ":" + name
            if not any(n["id"] == entity_id for n in nodes):
                nodes.append({"id": entity_id, "type": "entity", "name": name, "category": etype})
            edges.append({"source": rel, "target": entity_id,
                          "type": "mentions", "entity_type": etype})
    terms = [{"term": t.term, "en_name": t.en_name, "category": t.category,
              "definition": t.definition,
              "synonyms": json.loads(t.synonyms or "[]"), "unit": t.unit}
             for t in db.query(SemanticTerm).filter(SemanticTerm.status == "现行").all()
             if cur is not None and cur.can_view(t.secret_level)]
    return {"scene": "graph", "dataset": {"id": ds.id, "name": ds.name,
                                          "data_type": ds.data_type,
                                          "secret_level": ds.secret_level},
            "version": _version_brief(v),
            "nodes": nodes, "edges": edges, "entities": entities,
            "snippets": snippets, "terms": terms,
            "lineage": vs.lineage_of(db, v)}


# ---------------- 对外接口 ----------------

@router.post("/supply")
def supply_data(body: SupplyBody, request: Request,
                cur: CurrentUser = Depends(require_login),
                db: Session = Depends(get_db)):
    """数据应用层场景化供数入口：train / analyze / quality / graph。"""
    if body.scene not in SCENES:
        raise HTTPException(status_code=400,
                            detail=f"scene 必须为 { '/'.join(SCENES) }")
    ds = vs.get_dataset(db, body.dataset_id, cur)
    v = _get_input_version(db, ds, body.version_id)
    vs.require_version_ready(db,v)
    cur.require_view(ds.secret_level)  # 所有场景至少要求可见密级
    if body.scene == "quality":
        cur.require_operate(ds.secret_level)  # 评估写库留痕，需可操作密级

    handler = {"train": lambda: _scene_train(db, ds, v, body),
               "analyze": lambda: _scene_analyze(db, ds, v, body),
               "quality": lambda: _scene_quality(db, cur, ds, v, body),
               "graph": lambda: _scene_graph(db, ds, v, body, cur)}[body.scene]
    result = handler()

    write_log(db, user_id=cur.id, username=cur.username, action="数据服务调用",
              module="data_service",
              detail=(f"scene={body.scene} 数据集 {ds.name}(id={ds.id}) "
                      f"V{v.version_no}（include={body.include or '全部'}）"),
              ip=client_ip(request))
    return result
