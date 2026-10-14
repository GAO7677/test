"""预处理任务路由（6.7）：自动标注/手动标注/模板辅助标注/清洗/集成/转换/采样/工作流。

创建接口统一支持 input_version_id（缺省取数据集 current_version_id）；
响应含 output_version_id；进度经列表接口轮询。
"""
from __future__ import annotations
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import (AutoLabelTask, CleanTask, Dataset,
                      IntegrateTask, ManualLabelTask, SampleDataset,
                      TransformTask, Workflow)
from ..security import CurrentUser, require_login, require_task_user
from ..services import preprocess as pp
from ..services import version_service as vs
from ..services import data_access as access
from ..services.log_service import client_ip, write_log
from ..utils.common import SECRET_LEVELS, paginate, secret_index, to_dict

router = APIRouter(prefix="/api", tags=["preprocess"])


def _task_dict(t) -> dict:
    d = to_dict(t)
    for field in ("source_ids", "rules_config", "rules", "params", "nodes", "edges",
                  "stats_before", "stats_after", "source_types", "detail"):
        if field in d:
            d[field] = _json_or(d[field])
    return d


def _json_or(v):
    if isinstance(v, str) and v:
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _list_tasks(db: Session, model, page: int, page_size: int, cur):
    q = db.query(model).order_by(model.created_at.desc())
    if not cur.unrestricted:
        q = q.filter(model.created_by == cur.id)
    result = paginate(q, page, page_size)
    result["items"] = [_task_dict(t) for t in result["items"]]
    return result


def _validate_dataset_and_version(db: Session, dataset_id: int, input_version_id: Optional[int],
                                  cur: Optional[CurrentUser] = None):
    ds = vs.get_dataset(db, dataset_id, cur)
    v = vs.validate_input_version(db, ds, input_version_id)
    if cur is not None:
        cur.require_operate(ds.secret_level)  # 密级操作权限校验
    return ds, v


# ---------------- 内置规则查询 ----------------
@router.get("/auto-label/rules")
def auto_label_rules(data_type: str = "", cur: CurrentUser = Depends(require_login)):
    """返回内置规则列表，并为需要深度学习模型的规则附加 model_ready 状态。"""
    from ..services import vad_service, ner_service

    def _attach_model_status(rule_list: list) -> list:
        result = []
        for r in rule_list:
            d = dict(r)
            if d["key"] == "silero_vad":
                d["model_ready"] = vad_service.model_available()
                d["model_name"] = "Silero VAD"
            elif d["key"] == "spacy_ner":
                d["model_ready"] = ner_service.model_available()
                d["model_name"] = "Spacy zh_core_web_sm"
            elif d["key"] == "whisper_transcribe":
                from ..services import whisper_service
                d["model_ready"] = whisper_service.model_available()
                d["model_name"] = "faster-whisper"
            elif d["key"] == "gliner_ner":
                from ..services import gliner_service
                d["model_ready"] = gliner_service.model_available()
                d["model_name"] = "GliNER"
            elif d["key"] == "sam3_detect":
                from ..services import sam3_service
                d["model_ready"] = sam3_service.model_available()
                d["model_name"] = "SAM 3"
            elif d["key"] == "sam2_video":
                from ..services import sam2_service
                d["model_ready"] = sam2_service.model_available()
                d["model_name"] = "SAM 2"
            elif d["key"] == "coco_detect":
                from ..services import detector
                from importlib.util import find_spec
                d["model_ready"] = detector._default_model_path().is_file() and (find_spec("onnxruntime") is not None or find_spec("cv2") is not None)
            else:
                d["model_ready"] = True  # 规则算法无需模型
            result.append(d)
        return result

    if data_type:
        return {"items": _attach_model_status(pp.BUILTIN_RULES.get(data_type, []))}
    return {k: _attach_model_status(v) for k, v in pp.BUILTIN_RULES.items()}


def _validate_files(version, dtype):
    from pathlib import Path
    from ..services.file_service import allowed_file
    files = pp.data_files(Path(version.file_path))
    if not files: raise HTTPException(400, '输入版本没有数据文件')
    if any(not allowed_file(dtype, p.name) for p in files):
        raise HTTPException(400, '输入版本包含不适用的文件类型，请选择原始数据版本')


def _validate_algorithm(dtype, algorithm, params, cur):
    rules = auto_label_rules(dtype, cur)['items']
    rule = next((r for r in rules if r['key'] == algorithm), None)
    if rule is None: raise HTTPException(400, '算法不存在或尚未开放')
    if not rule['model_ready']: raise HTTPException(400, '该算法的模型或运行依赖未就绪')
    try:
        import math
        for key in ('frameInterval','maxFrames','sliceDuration','size','topK','minArea','k','maxIter'):
            if key in params and (not math.isfinite(float(params[key])) or float(params[key]) <= 0):
                raise ValueError(key + '须为正数')
        for key in ('confThreshold','nmsThreshold','overlap'):
            if key in params and not 0 <= float(params[key]) < (1 if key == 'overlap' else 1.000001):
                raise ValueError(key + '超出范围')
        if float(params.get('maxFrames', 200)) > 1000: raise ValueError('单次最多处理1000个采样帧')
    except (ValueError, TypeError) as e: raise HTTPException(400, str(e))


def _validate_transform(dtype, kind, target, params):
    formats = {'image': {'jpg','jpeg','png','bmp','tiff'}, 'voice': {'wav','mp3','flac','m4a','ogg'}, 'radar': {'csv','ply'}}
    if kind == 'format' and target.lower() in formats.get(dtype, set()): return
    if kind == 'standardize' and dtype in ('image','voice'):
        try:
            if not 1 <= int(params.get('size',640)) <= 8192: raise ValueError()
            if not 0 < float(params.get('target_peak',.89)) <= 1: raise ValueError()
        except (ValueError,TypeError): raise HTTPException(400, '尺寸或音量参数无效')
        return
    raise HTTPException(400, '暂不支持此数据类型与转换方式；特征提取需独立产物格式，暂未开放')


# ---------------- 规则自动标注 ----------------
class AutoLabelBody(BaseModel):
    name: str
    data_type: str
    dataset_id: int
    algorithm: str
    rules_config: Dict = {}
    input_version_id: Optional[int] = None


@router.get("/auto-label")
def list_auto_label(page: int = 1, page_size: int = 20,
                    cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    return _list_tasks(db, AutoLabelTask, page, page_size, cur)


@router.post("/auto-label")
def create_auto_label(body: AutoLabelBody, request: Request,
                      cur: CurrentUser = Depends(require_task_user), db: Session = Depends(get_db)):
    if body.data_type not in pp.BUILTIN_RULES:
        raise HTTPException(status_code=400, detail="数据类型无效")
    ds, v = _validate_dataset_and_version(db, body.dataset_id, body.input_version_id, cur)
    if body.data_type != ds.data_type: raise HTTPException(400, '数据类型与数据集不一致')
    _validate_algorithm(ds.data_type, body.algorithm, body.rules_config, cur)
    _validate_files(v, ds.data_type)
    task = AutoLabelTask(name=body.name, data_type=body.data_type,
                         source_ids=json.dumps([ds.id]), algorithm=body.algorithm,
                         rules_config=json.dumps(body.rules_config, ensure_ascii=False),
                         input_version_id=v.id, created_by=cur.id)
    db.add(task)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="创建自动标注任务",
              module="auto_label", detail=f"{body.name}（算法 {body.algorithm}，输入 V{v.version_no}）",
              ip=client_ip(request))
    pp.start_auto_label(task.id)
    return _task_dict(task)


# ---------------- 手动标注 ----------------
class ManualLabelBody(BaseModel):
    dataset_id: int
    label_type: str = "manual"
    labels: List[dict] = []          # [{file: 文件名, label: {...}}]
    input_version_id: Optional[int] = None


@router.get("/manual-label")
def list_manual_label(page: int = 1, page_size: int = 20,
                      cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    return _list_tasks(db, ManualLabelTask, page, page_size, cur)


@router.post("/manual-label")
def create_manual_label(body: ManualLabelBody, request: Request,
                        cur: CurrentUser = Depends(require_task_user), db: Session = Depends(get_db)):
    """手动标注工作台提交：前端标注结果直接写入新版本目录。"""
    from pathlib import Path
    ds, v = _validate_dataset_and_version(db, body.dataset_id, body.input_version_id, cur)
    root=Path(v.file_path).resolve()
    for item in body.labels:
        rel=Path(str(item.get('file','')).replace('\\','/'))
        target=(root/rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or root not in target.parents or not target.is_file() or target.name.endswith('_label.json') or target.name=='_manifest.json':
            raise HTTPException(400,'只能标注所选版本中实际存在的数据文件')
    task = ManualLabelTask(source_id=ds.id, source_type=ds.data_type,
                           label_type=body.label_type, input_version_id=v.id,
                           created_by=cur.id, status="running")
    db.add(task)
    db.commit()
    try:
        output_v = vs.create_output_version(db, ds, input_version=v,
                                            task_type="manual_label", task_id=task.id,
                                            user_id=cur.id)
        out_dir = Path(output_v.file_path)
        pp.copy_all(Path(v.file_path), out_dir)
        for item in body.labels:
            rel = Path(item.get("file", "").replace("\\", "/"))
            if rel.name:
                # 镜像相对目录结构（文件夹上传的数据集子目录内同名文件不冲突）
                pp.write_label(out_dir, rel, item.get("label", {}), src_dir=Path("."))
        task.output_version_id = output_v.id
        task.result_path = str(out_dir)
        vs.refresh_version_stats(db, output_v)
        task.status = "completed"
        vs.refresh_version_stats(db, output_v)
        from datetime import datetime as _dt
        task.completed_at = _dt.now()
        db.commit()
    except Exception as e:
        task.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"标注写入失败：{e}")
    write_log(db, user_id=cur.id, username=cur.username, action="提交手动标注",
              module="manual_label", detail=f"数据集 {ds.name}，{len(body.labels)} 条标注",
              ip=client_ip(request))
    return _task_dict(task)


# ---------------- 数据清洗 ----------------
class CleanBody(BaseModel):
    name: str
    data_type: str
    dataset_id: int
    rules: Dict = {}   # {missing, anomaly, duplicate, format}
    input_version_id: Optional[int] = None


@router.get("/clean")
def list_clean(page: int = 1, page_size: int = 20,
               cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    return _list_tasks(db, CleanTask, page, page_size, cur)


@router.post("/clean")
def create_clean(body: CleanBody, request: Request,
                 cur: CurrentUser = Depends(require_task_user), db: Session = Depends(get_db)):
    ds, v = _validate_dataset_and_version(db, body.dataset_id, body.input_version_id, cur)
    task = CleanTask(name=body.name, data_type=body.data_type,
                     source_ids=json.dumps([ds.id]),
                     rules=json.dumps(body.rules, ensure_ascii=False),
                     input_version_id=v.id, created_by=cur.id)
    db.add(task)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="创建清洗任务",
              module="clean", detail=f"{body.name}（输入 V{v.version_no}）", ip=client_ip(request))
    pp.start_clean(task.id)
    return _task_dict(task)


# ---------------- 数据集成 ----------------
class IntegrateBody(BaseModel):
    name: str
    dataset_id: int
    combo_type: str = "raw"       # raw/feature/decision/semantic
    strategy: str = "raw"
    main_input_version_id: int               # 主输入版本（血缘父）
    other_input_version_ids: List[int] = []  # 其余输入版本（记入任务 detail）


@router.get("/integrate")
def list_integrate(page: int = 1, page_size: int = 20,
                   cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    return _list_tasks(db, IntegrateTask, page, page_size, cur)


@router.post("/integrate")
def create_integrate(body: IntegrateBody, request: Request,
                     cur: CurrentUser = Depends(require_task_user), db: Session = Depends(get_db)):
    if body.strategy != "raw" or body.combo_type != "raw":
        raise HTTPException(400, "当前仅支持文件集合并")
    from ..models import DatasetVersion
    ds = vs.get_dataset(db, body.dataset_id, cur)
    cur.require_operate(ds.secret_level)
    # 跨模态集成：主输入版本允许来自其他数据集（集成目标为新建/独立数据集时，其自身可能尚无版本）
    if body.main_input_version_id:
        main_v = db.query(DatasetVersion).filter(
            DatasetVersion.id == body.main_input_version_id).first()
        if not main_v:
            raise HTTPException(status_code=404,
                                detail=f"主输入版本 #{body.main_input_version_id} 不存在")
        vs.require_version_ready(db,main_v)
        main_ds = vs.get_dataset(db, main_v.dataset_id, cur)
        cur.require_operate(main_ds.secret_level)
    else:
        main_v = vs.validate_input_version(db, ds, None)
        main_ds = ds
    others, other_types = [], []
    max_src_idx = max(secret_index(ds.secret_level), secret_index(main_ds.secret_level))
    for vid in body.other_input_version_ids:
        # 跨数据集/跨模态集成：其余输入版本可来自任意数据集
        ov = db.query(DatasetVersion).filter(DatasetVersion.id == vid).first()
        if not ov:
            raise HTTPException(status_code=404, detail=f"输入版本 #{vid} 不存在")
        if ov.id == main_v.id:
            continue
        vs.require_version_ready(db,ov)
        ods = vs.get_dataset(db, ov.dataset_id, cur)
        cur.require_operate(ods.secret_level)
        max_src_idx = max(max_src_idx, secret_index(ods.secret_level))
        others.append(ov.id)
        if ods.data_type not in other_types and ods.data_type != ds.data_type:
            other_types.append(ods.data_type)
    # 产出数据集密级不得低于任一来源（防止高密级数据被"洗"到低密级）
    if secret_index(ds.secret_level) < max_src_idx:
        raise HTTPException(
            status_code=400,
            detail=f"产出数据集密级（{ds.secret_level}）不得低于任一来源数据集的最高密级"
                   f"（{SECRET_LEVELS[max_src_idx]}），请先调整目标数据集密级")
    private=access.private_owners(db)[1]
    inputs=[main_ds.id]+[db.get(DatasetVersion,vid).dataset_id for vid in others]
    if not cur.unrestricted and any(did in private for did in inputs) and ds.created_by!=cur.id:
        raise HTTPException(403, "私有生成数据请集成到自己的数据集")
    task = IntegrateTask(name=body.name,
                         source_ids=json.dumps([ds.id]),
                         source_types=json.dumps([ds.data_type] + other_types),
                         combo_type=body.combo_type, strategy=body.strategy,
                         input_version_id=main_v.id,
                         detail=json.dumps({"other_input_version_ids": others}),
                         created_by=cur.id)
    db.add(task)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="创建集成任务",
              module="integrate",
              detail=f"{body.name}（主输入 V{main_v.version_no}，其余输入 {others}）",
              ip=client_ip(request))
    pp.start_integrate(task.id)
    return _task_dict(task)


# ---------------- 数据转换 ----------------
class TransformBody(BaseModel):
    name: str
    data_type: str
    dataset_id: int
    transform_type: str      # format/standardize/feature/pipeline
    target_format: str = ""
    params: Dict = {}
    input_version_id: Optional[int] = None


@router.get("/transform")
def list_transform(page: int = 1, page_size: int = 20,
                   cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    return _list_tasks(db, TransformTask, page, page_size, cur)


@router.post("/transform")
def create_transform(body: TransformBody, request: Request,
                     cur: CurrentUser = Depends(require_task_user), db: Session = Depends(get_db)):
    if body.transform_type not in ("format", "standardize"):
        raise HTTPException(status_code=400, detail="转换类型无效")
    ds, v = _validate_dataset_and_version(db, body.dataset_id, body.input_version_id, cur)
    if body.data_type != ds.data_type: raise HTTPException(400, "数据类型与数据集不一致")
    _validate_files(v, ds.data_type)
    _validate_transform(ds.data_type, body.transform_type, body.target_format, body.params)
    task = TransformTask(name=body.name, data_type=body.data_type, source_id=ds.id,
                         transform_type=body.transform_type, target_format=body.target_format,
                         params=json.dumps(body.params, ensure_ascii=False),
                         input_version_id=v.id, created_by=cur.id)
    db.add(task)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="创建转换任务",
              module="transform",
              detail=f"{body.name}（{body.transform_type}→{body.target_format}，输入 V{v.version_no}）",
              ip=client_ip(request))
    pp.start_transform(task.id)
    return _task_dict(task)


# ---------------- 数据采样（训练/验证/测试三划分） ----------------
class SampleBody(BaseModel):
    name: str
    data_type: str
    dataset_id: int
    train_ratio: int = 70
    val_ratio: int = 10
    test_ratio: int = 20
    method: str = "random"   # random/stratified/time/kfold
    input_version_id: Optional[int] = None


@router.get("/sample")
def list_sample(page: int = 1, page_size: int = 20,
                cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    return _list_tasks(db, SampleDataset, page, page_size, cur)


@router.post("/sample")
def create_sample(body: SampleBody, request: Request,
                  cur: CurrentUser = Depends(require_task_user), db: Session = Depends(get_db)):
    if body.method != "random": raise HTTPException(400, "当前仅支持随机划分；其他方式暂未开放")
    # 三比例之和=100%（后端校验，前端同步校验）
    if body.train_ratio + body.val_ratio + body.test_ratio != 100:
        raise HTTPException(status_code=400,
                            detail=f"训练/验证/测试比例之和须等于 100%（当前 {body.train_ratio + body.val_ratio + body.test_ratio}%）")
    for r in (body.train_ratio, body.val_ratio, body.test_ratio):
        if r < 0 or r > 100:
            raise HTTPException(status_code=400, detail="比例须在 0-100 之间")
    ds, v = _validate_dataset_and_version(db, body.dataset_id, body.input_version_id, cur)
    task = SampleDataset(name=body.name, data_type=body.data_type, source_id=ds.id,
                         train_ratio=body.train_ratio, val_ratio=body.val_ratio,
                         test_ratio=body.test_ratio,
                         input_version_id=v.id, created_by=cur.id)
    db.add(task)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="创建采样任务",
              module="sample",
              detail=f"{body.name}（{body.train_ratio}:{body.val_ratio}:{body.test_ratio}，输入 V{v.version_no}）",
              ip=client_ip(request))
    pp.start_sample(task.id)
    return _task_dict(task)


# ---------------- 工作流 ----------------
class WorkflowBody(BaseModel):
    name: str
    data_type: str = ""
    dataset_id: int
    nodes: List[dict] = []   # [{type: clean/auto_label/transform/sample, params/algorithm/...}]
    edges: List[dict] = []
    input_version_id: Optional[int] = None


@router.get("/workflows")
def list_workflows(page: int = 1, page_size: int = 20,
                   cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    return _list_tasks(db, Workflow, page, page_size, cur)


def _validate_nodes(dtype, nodes, cur):
    for node in nodes:
        kind, params = node.get('type'), node.get('params', {})
        if kind == 'auto_label': _validate_algorithm(dtype, node.get('algorithm'), params, cur)
        elif kind == 'transform': _validate_transform(dtype, node.get('transform_type','format'), node.get('target_format',''), params)
        elif kind == 'sample':
            ratios = [params.get(k, d) for k,d in [('train_ratio',70),('val_ratio',10),('test_ratio',20)]]
            if any(not isinstance(r,int) or not 0 <= r <= 100 for r in ratios) or sum(ratios) != 100:
                raise HTTPException(400, '采样比例须在0至100且总和为100')
            if params.get('method','random') != 'random': raise HTTPException(400, '当前仅支持随机划分')
        elif kind != 'clean': raise HTTPException(400, '节点类型无效')


@router.post("/workflows")
def create_workflow(body: WorkflowBody, request: Request,
                    cur: CurrentUser = Depends(require_task_user), db: Session = Depends(get_db)):
    ds, v = _validate_dataset_and_version(db, body.dataset_id, body.input_version_id, cur)
    if body.edges: raise HTTPException(400, "仅支持按节点顺序执行的线性流程")
    if not body.nodes: raise HTTPException(400, "工作流不能为空")
    _validate_nodes(ds.data_type, body.nodes, cur)
    valid_algos = {r["key"] for r in pp.BUILTIN_RULES.get(ds.data_type, [])}
    for node in body.nodes:
        if node.get("type") not in ("clean", "auto_label", "transform", "sample"):
            raise HTTPException(status_code=400,
                                detail=f"节点类型限定为 clean/auto_label/transform/sample，收到：{node.get('type')}")
        if node.get("type") == "auto_label":
            algo = node.get("algorithm", "")
            if algo not in valid_algos:
                raise HTTPException(
                    status_code=400,
                    detail=f"算法「{algo}」不适用于{ds.data_type}模态，可选：{'/'.join(sorted(valid_algos))}")
        if node.get("type") == "sample":
            p = node.get("params", {})
            total = int(p.get("train_ratio", 70)) + int(p.get("val_ratio", 10)) + int(p.get("test_ratio", 20))
            if total != 100:
                raise HTTPException(status_code=400, detail=f"采样节点三比例之和须为 100%（当前 {total}%）")
    wf = Workflow(name=body.name, data_type=body.data_type or ds.data_type,
                  source_id=ds.id, nodes=json.dumps(body.nodes, ensure_ascii=False),
                  edges=json.dumps(body.edges, ensure_ascii=False),
                  input_version_id=v.id, created_by=cur.id)
    db.add(wf)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="创建工作流",
              module="workflow", detail=f"{body.name}（{len(body.nodes)} 个节点）",
              ip=client_ip(request))
    return _task_dict(wf)


@router.post("/workflows/{wid}/run")
def run_workflow(wid: int, request: Request,
                 cur: CurrentUser = Depends(require_task_user), db: Session = Depends(get_db)):
    wf = db.query(Workflow).filter(Workflow.id == wid).with_for_update().first()
    if not wf:
        raise HTTPException(status_code=404, detail="工作流不存在")
    if wf.created_by != cur.id and not cur.unrestricted:
        raise HTTPException(403, "只能执行自己的工作流")
    if wf.status == "running":
        raise HTTPException(status_code=400, detail="工作流正在执行中")
    if wf.source_id:
        ds, v = _validate_dataset_and_version(db, wf.source_id, wf.input_version_id, cur)
        _validate_nodes(ds.data_type, json.loads(wf.nodes or "[]"), cur)
        _validate_files(v, ds.data_type)
    wf.status = "running"
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action="执行工作流",
              module="workflow", detail=f"启动工作流 {wf.name}", ip=client_ip(request))
    pp.start_workflow(wf.id)
    return _task_dict(wf)
