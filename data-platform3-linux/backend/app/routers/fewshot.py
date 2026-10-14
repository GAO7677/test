"""少样本图像生成；所有路径、GPU、解释器由管理员配置，不接受客户端命令。"""
from __future__ import annotations
import json
import shutil
import uuid
import io
from contextlib import nullcontext
from pathlib import Path
from typing import List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import FewshotJob, Dataset
from ..security import CurrentUser, require_login, require_task_user
from ..services import fewshot_service as svc
from ..services import local_materials as materials
from ..services import download_service as downloads, data_access as access
from ..utils.common import secret_index
from ..services.log_service import write_log, client_ip
from ..utils.common import SECRET_LEVELS

router = APIRouter(prefix='/api/fewshot', tags=['fewshot'])


class LocalImage(BaseModel):
    dataset: str = Field(max_length=40)
    name: str = Field(min_length=1, max_length=255)


class Group(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    caption: str = Field(min_length=1, max_length=2000)
    count: int = Field(default=10, ge=1, le=1000)
    data_file_ids: List[int] = Field(default_factory=list, max_length=100)
    local_images: List[LocalImage] = Field(default_factory=list, max_length=100)


class Options(BaseModel):
    method: Literal['beyondobjects', 'chimeralora'] = 'beyondobjects'
    model: Literal['flux2_klein_base_4b', 'sd21'] = 'flux2_klein_base_4b'
    steps: int = Field(default=400, ge=2, le=4000)
    rank: Literal[4, 8, 16] = 4
    seed: int = Field(default=0, ge=0, le=2147480000)
    resolution: Literal[512, 768, 1024] = 512
    sampling_steps: int = Field(default=30, ge=1, le=100)


class JobBody(BaseModel):
    request_key: uuid.UUID
    name: str = Field(min_length=1, max_length=80)
    project: str = Field(default='', max_length=100)
    scene: str = Field(default='', max_length=100)
    secret_level: str = '内部'
    groups: List[Group] = Field(min_length=1, max_length=20)
    options: Options = Field(default_factory=Options)
    download_request_id: Optional[int] = None


def validate_body(body, cur):
    try:
        svc.engine_id(body.options.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if body.secret_level not in SECRET_LEVELS:
        raise HTTPException(422, '密级无效')
    cur.require_operate(body.secret_level)
    if sum(g.count for g in body.groups) > 1000:
        raise HTTPException(422, '一次最多生成 1000 张图像')
    if any(not g.caption.strip() or not g.name.strip() for g in body.groups):
        raise HTTPException(422, '分组名称和描述不能为空白')


@router.get('/capabilities')
def capabilities(cur: CurrentUser = Depends(require_login)):
    engines = []
    for key, spec in svc.ENGINES.items():
        try:
            svc.engine_settings(key)
            error = ''
        except ValueError as exc:
            error = str(exc)
        engines.append(dict(spec, id=key, ready=not error, error=error))
    ready = any(e['ready'] for e in engines)
    return {'ready': ready, 'error': '' if ready else '训练方案均不可用，请检查环境与权重配置',
            'engines': engines, 'model': 'FLUX.2 Klein Base 4B / SD2.1',
            'labeler': 'YOLOv8n · COCO 80 类', 'max_files': 100, 'max_file_mb': 20,
            'max_total_mb': 500, 'max_generated': 1000}


@router.get('/jobs')
def list_jobs(page: int = 1, all_users: bool = False, cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    levels = [level for level in SECRET_LEVELS if cur.can_view(level)]
    q = select(FewshotJob).where(FewshotJob.secret_level.in_(levels)).order_by(FewshotJob.id.desc())
    if all_users and not cur.unrestricted:
        raise HTTPException(403, '只有管理员可以查看其他用户任务')
    if not all_users:q=q.where(FewshotJob.created_by==cur.id)
    jobs = db.scalars(q.offset((max(1, page) - 1) * 30).limit(30)).all()
    return {'items': [svc.public_job(j) for j in jobs]}


def require_materials(cur: CurrentUser = Depends(require_task_user)):
    cur.require_view('内部')
    cur.require_operate('内部')
    return cur


@router.get('/materials')
def material_catalog(cur: CurrentUser = Depends(require_materials)):
    return {'items': [{'id': key, 'name': name, 'available': root.is_dir()}
                      for key, (name, root) in materials.CATALOG.items()]}


@router.get('/materials/{dataset}')
def material_list(dataset: str, page: int = Query(1, ge=1), q: str = Query('', max_length=255),
                  category: str = Query('', max_length=255),
                  cur: CurrentUser = Depends(require_materials)):
    return materials.browse_images(dataset, q, category, page)


@router.get('/materials/{dataset}/thumbnail')
def material_thumbnail(dataset: str, name: str, cur: CurrentUser = Depends(require_materials)):
    path = materials.resolve_image(dataset, name)
    try:
        if path.stat().st_size > 20 * 1024**2:
            raise ValueError('图片超过 20 MB')
        with Image.open(path) as original:
            if original.width * original.height > 40_000_000:
                raise ValueError('图片尺寸过大')
            original.thumbnail((256, 256))
            thumb = ImageOps.exif_transpose(original).convert('RGB')
            out = io.BytesIO()
            thumb.save(out, format='JPEG', quality=80)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, '无法预览此图片') from exc
    return Response(out.getvalue(), media_type='image/jpeg', headers={'Cache-Control': 'private, max-age=300'})


@router.post('/jobs')
async def create_job(request: Request, cur: CurrentUser = Depends(require_task_user), db: Session = Depends(get_db)):
    async with request.form(max_files=100, max_fields=25) as form:
        try:
            body = JobBody.model_validate_json(str(form.get('manifest', '')))
        except ValidationError as exc:
            raise HTTPException(422, '任务参数不完整或超出范围') from exc
        validate_body(body, cur)
        try:
            svc.engine_settings(svc.engine_id(body.options.model_dump()))
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        if any(g.local_images for g in body.groups):
            require_materials(cur)
        saved=downloads.get_request(db,body.download_request_id,cur,own=True) if body.download_request_id else None
        if any(g.data_file_ids for g in body.groups) and saved is None:
            raise HTTPException(422,'候选素材必须关联下载请求')
        if sum(len(form.getlist(f'files_{i}')) + len(g.local_images) + len(g.data_file_ids) for i, g in enumerate(body.groups)) > 100:
            raise HTTPException(422, '一次最多选择 100 张素材')
        key = str(body.request_key)
        old = db.scalar(select(FewshotJob).where(FewshotJob.request_key == key))
        if old:
            return svc.public_job(svc.get_job(db, old.id, cur, write=True))
        try:
            folder = svc.new_job_dir(body.request_key.hex)
        except (ValueError, FileExistsError) as exc:
            raise HTTPException(409, str(exc)) from exc
        job = FewshotJob(request_key=key, name=body.name, project=body.project, scene=body.scene,
                         secret_level=body.secret_level, job_dir=str(folder),
                         options=body.options.model_dump_json(), created_by=cur.id, download_request_id=body.download_request_id)
        raw = None
        try:
            db.add(job)
            db.flush()
            raw = svc.new_raw(db, job, body.name + ' · 训练素材', '真实采集')
            total = count = 0
            pixels = set()
            groups = []
            for i, group in enumerate(body.groups):
                files = [(upload, None, None) for upload in form.getlist(f'files_{i}')]
                files += [(None, ref, None) for ref in group.local_images]
                files += [(None, None, fid) for fid in group.data_file_ids]
                if not files:
                    raise ValueError(f'分组 {group.name} 没有图片')
                group_dir = folder / 'support' / f'group_{i:03d}'
                group_dir.mkdir(parents=True)
                (group_dir / 'caption.txt').write_text(group.caption.strip() + '\n')
                sources = []
                for j, (upload, ref, fid) in enumerate(files):
                    source = materials.resolve_image(ref.dataset, ref.name) if ref else None
                    source_record = ref.model_dump() if ref else {'upload': getattr(upload,'filename',None)}
                    if fid is not None:
                        record, original, source = downloads.resolve_candidate(db,saved,fid,cur)
                        access.require_raw(db,cur,'image',original,operate=True)
                        if secret_index(body.secret_level)<secret_index(original.secret_level):
                            raise ValueError('任务密级不得低于候选素材密级')
                        source_record={'download_request_id':saved.id,'data_file_id':fid,'original_name':record.filename}
                    ext = (source if source else Path(getattr(upload, 'filename', '') or '')).suffix.lower()
                    if ext not in svc.IMAGE_EXTS or (source is None and not hasattr(upload, 'read')):
                        raise ValueError('仅支持 PNG/JPEG/WebP/BMP/TIFF 图像')
                    target = group_dir / f'{j:04d}{ext}'
                    size = 0
                    with target.open('wb') as out, (source.open('rb') if source else nullcontext()) as local:
                        while chunk := (local.read(1024 * 1024) if local else await upload.read(1024 * 1024)):
                            size += len(chunk)
                            total += len(chunk)
                            if size > 20 * 1024**2 or total > 500 * 1024**2:
                                raise ValueError('单图上限 20MB，一次上传上限 500MB')
                            out.write(chunk)
                    _, fingerprint = svc.image_info(target)
                    if fingerprint in pixels:
                        raise ValueError('素材包含重复图像，请移除后重新提交')
                    pixels.add(fingerprint)
                    svc.add_file(db, raw, target, f'group_{i:03d}/{target.name}')
                    sources.append({'file': target.name, 'sha256': svc.sha(target),
                                    'source': source_record})
                    count += 1
                groups.append({'name': group.name, 'caption': group.caption.strip(), 'count': group.count,
                               'folder': f'group_{i:03d}', 'images': len(files), 'sources': sources})
            if count > 100:
                raise ValueError('一次最多上传 100 张素材')
            job.groups = json.dumps(groups, ensure_ascii=False)
            job.source_data_id = raw.id
            db.commit()
        except Exception as exc:
            db.rollback()
            shutil.rmtree(folder)
            if raw is not None and raw.file_path:
                shutil.rmtree(raw.file_path, ignore_errors=True)
            if isinstance(exc, HTTPException):
                raise
            if isinstance(exc, (ValueError, OSError, IntegrityError)):
                raise HTTPException(422, str(exc)) from exc
            raise
    write_log(db, user_id=cur.id, username=cur.username, action='上传少样本素材', module='fewshot',
              detail=f'任务 #{job.id}，{count} 张图片，{len(groups)} 组', ip=client_ip(request))
    return svc.public_job(job)


@router.get('/jobs/{jid}')
def detail(jid: int, cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    job = svc.get_job(db, jid, cur)
    data = svc.public_job(job)
    log = Path(job.job_dir) / 'worker.log'
    if log.is_file():
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 12000))
            data['log'] = f.read().decode('utf-8', errors='replace')
    else:
        data['log'] = ''
    if job.dataset_id:
        ds=db.get(Dataset,job.dataset_id)
        data['current_version_id']=ds.current_version_id if ds and not ds.deleted_at else None
        data['images'] = [p.name for p in sorted((Path(job.batch_dir) / 'lora').glob('*.png'))[:24]]
    return data


@router.post('/jobs/{jid}/submit')
def submit(jid: int, request: Request, cur: CurrentUser = Depends(require_task_user), db: Session = Depends(get_db)):
    job = svc.get_job(db, jid, cur, write=True)
    db.refresh(job, with_for_update=True)
    if job.status not in ('draft', 'failed', 'queued'):
        raise HTTPException(409, '任务已开始或已完成')
    if job.smoke:
        raise HTTPException(409, '验收任务不允许通过页面重新训练')
    try:
        svc.job_engine(job)
        svc.ensure_worker()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(503, str(exc)) from exc
    job.status, job.error = 'queued', ''
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action='提交少样本任务', module='fewshot',
              detail=f'任务 #{job.id}', ip=client_ip(request))
    return svc.public_job(job)


@router.post('/jobs/{jid}/generate')
def regenerate(jid: int, request: Request, cur: CurrentUser = Depends(require_task_user), db: Session = Depends(get_db)):
    source = svc.get_job(db, jid, cur, write=True)
    if source.smoke or source.status != 'completed':
        raise HTTPException(409, '仅已完成的正式任务支持复用 LoRA')
    parent = svc.get_job(db, source.reuse_job_id, cur, write=True) if source.reuse_job_id else source
    if not (Path(parent.run_dir) / 'adapter/complete.json').is_file():
        raise HTTPException(409, 'LoRA 文件不存在')
    key = uuid.uuid4()
    folder = svc.new_job_dir(key.hex)
    shutil.copytree(Path(source.job_dir) / 'support', folder / 'support')
    options = json.loads(source.options)
    options['seed'] = key.int % 2147480000
    job = FewshotJob(request_key=str(key), name=source.name[:74] + ' · 再生成', project=source.project,
                     scene=source.scene, secret_level=source.secret_level, groups=source.groups,
                     options=json.dumps(options), source_data_id=source.source_data_id,
                     job_dir=str(folder), reuse_job_id=parent.id, created_by=cur.id,
                     download_request_id=source.download_request_id if source.created_by==cur.id else None)
    db.add(job)
    db.commit()
    write_log(db, user_id=cur.id, username=cur.username, action='复用 LoRA', module='fewshot',
              detail=f'任务 #{job.id}，LoRA 来源 #{parent.id}', ip=client_ip(request))
    return svc.public_job(job)


@router.get('/jobs/{jid}/images/{name}')
def image(jid: int, name: str, cur: CurrentUser = Depends(require_login), db: Session = Depends(get_db)):
    job = svc.get_job(db, jid, cur)
    if not job.dataset_id or Path(name).name != name or not name.endswith('.png'):
        raise HTTPException(404, '图片不存在')
    root = (Path(job.batch_dir) / 'lora').resolve()
    target = (root / name).resolve()
    if target.parent != root or not target.is_file():
        raise HTTPException(404, '图片不存在')
    return FileResponse(target, media_type='image/png')
