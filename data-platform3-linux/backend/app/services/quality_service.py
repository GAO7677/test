"""数据质量评估服务（8.4）：基于真实文件内容计算六维度评分，关联参考标准计算合规性。

六维度：完整性、准确性、一致性、时效性、规范性、可访问性。
版本级评估：写入 quality_results(version_id) 并回写 dataset_versions.quality_score / quality_evaluated_at。
"""
from __future__ import annotations
from typing import Optional

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..models import (DATA_TYPE_MODELS, Dataset, DatasetVersion, QualityResult,
                      RefStandard)
from . import file_service as fsvc

DIM_NAMES = {"completeness": "完整性", "accuracy": "内容有效性", "consistency": "一致性",
             "timeliness": "时效性", "standardization": "规范性", "accessibility": "可访问性"}


def _score_files(data_type: str, files: list[Path]) -> dict:
    """基于真实文件内容计算六维度得分与明细。

    版本目录中同时存在数据文件与标注文件（<_stem>_label.json / _manifest.json）。
    标注文件只参与「规范性（标注配套率）」统计，不参与内容解析与格式统计；
    否则会把 _label.json 当视频/音频解析，既拉低得分又产生大量解析错误日志。
    """
    data_files = [p for p in files
                  if not p.name.endswith("_label.json") and p.name != "_manifest.json"]
    details: dict = {"file_count": len(data_files),
                     "label_file_count": len(files) - len(data_files), "checks": []}
    if not data_files:
        # 空版本/无数据文件：六维记 0 分，但仍返回完整结构（含 metrics），
        # 否则 evaluate_quality 中 scored.pop("metrics") 会抛 KeyError 导致 500。
        return {**{d: 0.0 for d in DIM_NAMES},
                "details": details,
                "metrics": {"file_count": 0, "nonempty_rate": 0.0, "parseable_rate": 0.0,
                            "content_pass_rate": 0.0, "format_ext_count": 0,
                            "named_rate": 0.0, "avg_age_days": 0.0,
                            "label_coverage": 0.0, "readable_rate": 0.0}}

    n = len(data_files)
    # ---- 完整性：文件非空率 + 元数据可解析率 ----
    nonempty = sum(1 for p in data_files if p.stat().st_size > 0)
    parseable = 0
    for p in data_files[:20]:
        if fsvc.extract_metadata(data_type, p):
            parseable += 1
    completeness = (nonempty / n * 70
                    + (parseable / min(n, 20)) * 30)
    details["checks"].append(f"非空文件 {nonempty}/{n}，元数据可解析 {parseable}/{min(n, 20)}")

    # ---- 准确性：按类型的内容校验 ----
    accuracy = _accuracy(data_type, data_files, details)

    # ---- 一致性：格式统一率 + 命名规范率 ----
    exts = {p.suffix.lower() for p in data_files}
    format_consistency = 100 / max(len(exts), 1) if len(exts) <= 3 else max(100 - (len(exts) - 3) * 15, 40)
    import re
    named = sum(1 for p in data_files if re.match(r"^[\w\-\u4e00-\u9fff]+\.\w+$", p.name))
    consistency = format_consistency * 0.6 + (named / n * 100) * 0.4
    details["checks"].append(f"扩展名种类 {len(exts)} 种，命名规范 {named}/{n}")

    # ---- 时效性：按文件修改时间距今天数衰减 ----
    mtimes = [p.stat().st_mtime for p in data_files]
    days_old = (datetime.now().timestamp() - np.mean(mtimes)) / 86400
    timeliness = float(max(100 - days_old * 2, 20))
    details["checks"].append(f"平均数据龄期 {days_old:.1f} 天")

    # ---- 规范性：标注配套率（_label.json 覆盖，按数据文件名配对）----
    from .label_schema import valid_label
    labelled = 0
    for p in data_files:
        try:
            label = json.loads(p.with_name(p.stem + "_label.json").read_text(encoding="utf-8"))
            labelled += int(valid_label(label))
        except (OSError, ValueError):
            pass
    coverage = labelled / n
    standardization = coverage * 100
    details["checks"].append(f"有效标注配套率 {coverage * 100:.0f}%（{labelled}/{n}），不代表人工核验或准确率")

    # ---- 可访问性：文件可读取率 ----
    readable = 0
    for p in data_files[:20]:
        try:
            with open(p, "rb") as f:
                f.read(64)
            readable += 1
        except OSError:
            pass
    accessibility = readable / min(n, 20) * 100
    details["checks"].append(f"可读取文件 {readable}/{min(n, 20)}")

    scores = {"completeness": round(min(completeness, 100), 1),
              "accuracy": round(min(accuracy, 100), 1),
              "consistency": round(min(consistency, 100), 1),
              "timeliness": round(min(timeliness, 100), 1),
              "standardization": round(min(standardization, 100), 1),
              "accessibility": round(min(accessibility, 100), 1)}
    # 可量化指标（供自定义标准规则引用）
    metrics = {
        "file_count": n,
        "nonempty_rate": round(nonempty / n * 100, 1),
        "parseable_rate": round(parseable / min(n, 20) * 100, 1),
        "content_pass_rate": round(_accuracy(data_type, data_files, {"checks": []}), 1),
        "format_ext_count": len(exts),
        "named_rate": round(named / n * 100, 1),
        "avg_age_days": round(float(days_old), 1),
        "label_coverage": round(coverage * 100, 1),
        "readable_rate": round(accessibility, 1),
    }
    return {**scores, "details": details, "metrics": metrics}


def _accuracy(data_type: str, files: list[Path], details: dict) -> float:
    """准确性：内容级抽样校验。"""
    sample = files[:10]
    ok = 0
    try:
        if data_type == "voice":
            import soundfile as sf
            for p in sample:
                if p.suffix.lower() not in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
                    continue
                try:
                    info = sf.info(str(p))
                    if info.frames > 0 and info.samplerate > 0:
                        ok += 1
                except Exception:
                    pass
        elif data_type == "image":
            from PIL import Image
            for p in sample:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    ok += 1
                except Exception:
                    pass
        elif data_type in ("text", "radar"):
            for p in sample:
                try:
                    text = p.read_bytes().decode("utf-8", errors="strict") if p.suffix.lower() != ".ply" else p.read_bytes()[:200].decode("ascii", errors="ignore")
                    if text.strip():
                        ok += 1
                except Exception:
                    pass
        elif data_type == "video":
            import cv2
            for p in sample:
                if p.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
                    continue  # 跳过非视频文件，避免 OpenCV 尝试解析而刷错误日志
                cap = cv2.VideoCapture(str(p))
                if cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                    ok += 1
                cap.release()
        else:
            ok = len(sample)
    except Exception:
        ok = len(sample) // 2
    details["checks"].append(f"内容校验通过 {ok}/{len(sample)}")
    return ok / max(len(sample), 1) * 100


OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
       ">": lambda a, b: a > b, "<": lambda a, b: a < b,
       "==": lambda a, b: a == b}


def _eval_custom_rules(rules: list, metrics: dict) -> dict:
    """自定义标准：逐条规则比对实际指标，符合率 = 通过规则占比。"""
    results = []
    passed = 0
    for r in rules:
        metric, op = r.get("metric", ""), r.get("op", ">=")
        try:
            threshold = float(r.get("threshold", 0))
        except (TypeError, ValueError):
            threshold = 0.0
        actual = metrics.get(metric)
        ok = bool(actual is not None and OPS.get(op, OPS[">="])(float(actual), threshold))
        passed += 1 if ok else 0
        results.append({"metric": metric, "op": op, "threshold": threshold,
                        "actual": actual, "passed": ok,
                        "name": r.get("name") or f"{metric} {op} {threshold}"})
    rate = round(passed / max(len(rules), 1) * 100, 1) if rules else 0.0
    return {"rate": rate, "rule_results": results}


def evaluate_quality(db: Session, data_type: str, data_id: int,
                     version_id: Optional[int], user_id: int,
                     standard_ids: Optional[list] = None) -> dict:
    """执行评估并落库。data_type 为五类之一（评估数据获取层，version_id=None），
    或传 version_id 评估预处理数据集版本。"""
    if version_id:
        v = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
        if not v:
            raise HTTPException(status_code=404, detail="版本不存在")
        ds = db.query(Dataset).filter(Dataset.id == v.dataset_id).first()
        dtype = ds.data_type if ds else data_type
        root = Path(v.file_path)
        files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
        data_type, data_id = dtype, v.dataset_id
    else:
        if data_type not in DATA_TYPE_MODELS:
            raise HTTPException(status_code=400, detail="数据类型无效")
        model = DATA_TYPE_MODELS[data_type]
        obj = db.query(model).filter(model.id == data_id, model.deleted_at.is_(None)).first()
        if not obj:
            raise HTTPException(status_code=404, detail="数据不存在")
        root = fsvc.dataset_dir(data_type, data_id)
        files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []

    scored = _score_files(data_type if data_type in DATA_TYPE_MODELS else "text", files)
    details = scored.pop("details")
    metrics = scored.pop("metrics")
    details["metrics"] = metrics
    overall = round(sum(scored.values()) / len(scored), 1)

    # ---- 关联参考标准计算合规性：可勾选标准；每条标准单独出分 ----
    q = db.query(RefStandard).filter(RefStandard.status == "现行")
    if standard_ids:
        q = q.filter(RefStandard.id.in_(standard_ids))
    standards = q.all()
    compliance = []
    for s in standards:
        rules = json.loads(s.rules) if s.rules else []
        entry = {"code": s.code, "name": s.name, "standard_id": s.id,
                 "is_custom": bool(s.is_custom)}
        if rules:
            # 自定义标准：按规则逐条判定
            ev = _eval_custom_rules(rules, metrics)
            entry.update({"matched_dimensions": [], "compliance_rate": ev["rate"],
                          "rule_results": ev["rule_results"]})
        else:
            dims = [d for d in ("规范性", "完整性", "准确性", "一致性", "时效性", "可访问性")
                    if d in (s.dimensions or "")]
            if not dims:
                # 标准未声明可映射维度时按全部六维评估（与 compliance_rate 兜底口径一致）
                dims = list(DIM_NAMES.values())
            vals = [scored[k] for k, n in DIM_NAMES.items() if n in dims]
            rate = round(sum(vals) / len(vals), 1) if vals else overall
            entry.update({"matched_dimensions": dims, "compliance_rate": rate,
                          "dimension_scores": {n: scored[k] for k, n in DIM_NAMES.items() if n in dims}})
        entry["level"] = ("符合" if entry["compliance_rate"] >= 80
                          else "基本符合" if entry["compliance_rate"] >= 60 else "不符合")
        compliance.append(entry)

    rec = QualityResult(data_id=data_id, data_type=data_type,
                        completeness=scored["completeness"], accuracy=scored["accuracy"],
                        consistency=scored["consistency"], timeliness=scored["timeliness"],
                        standardization=scored["standardization"],
                        accessibility=scored["accessibility"],
                        overall_score=overall,
                        details=json.dumps(details, ensure_ascii=False),
                        standards_compliance=json.dumps(compliance, ensure_ascii=False),
                        version_id=version_id, evaluated_by=user_id)
    db.add(rec)

    # 版本级评估：回写 dataset_versions.quality_score / quality_evaluated_at
    if version_id:
        v.quality_score = overall
        v.quality_evaluated_at = datetime.now()
    db.commit()

    d = {k: scored[k] for k in DIM_NAMES}
    d.update({"overall_score": overall, "details": details,
              "standards_compliance": compliance,
              "data_type": data_type, "data_id": data_id, "version_id": version_id,
              "evaluated_at": rec.created_at.strftime("%Y-%m-%d %H:%M:%S")})
    return d
