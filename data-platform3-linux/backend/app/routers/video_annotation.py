import json
from datetime import datetime
from fastapi import APIRouter,Depends,HTTPException,Request
from pydantic import BaseModel,Field
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import AutoLabelTask, AnnotationLabel
from ..security import CurrentUser,require_login,require_task_user
from ..services import video_annotation as va
from ..services.detector import COCO_CLASSES
from ..services.log_service import write_log,client_ip
from .tasks import _task_dict,_list_tasks
router=APIRouter(prefix='/api/video-annotation',tags=['video annotation'])

@router.get('/capabilities')
def capabilities(cur:CurrentUser=Depends(require_login)):
    return {**va.capability(),'classes':COCO_CLASSES}

@router.get('/labels')
def labels(cur:CurrentUser=Depends(require_login),db:Session=Depends(get_db)):
    return {'items': [x.name for x in db.query(AnnotationLabel).filter_by(user_id=cur.id).order_by(AnnotationLabel.id)]}

class LabelBody(BaseModel):
    name:str=Field(min_length=1,max_length=100)

@router.post('/labels')
def remember_label(body:LabelBody,cur:CurrentUser=Depends(require_task_user),db:Session=Depends(get_db)):
    name=body.name.strip()
    if not name:raise HTTPException(400,'标签不能为空')
    if not db.query(AnnotationLabel).filter_by(user_id=cur.id,name=name).first():
        from sqlalchemy.exc import IntegrityError
        db.add(AnnotationLabel(user_id=cur.id,name=name))
        try:db.commit()
        except IntegrityError:db.rollback()  # concurrent save of the same personal label
    return {'name':name}

@router.get('/exports')
def download_exports(dataset_id:int,input_version_id:int,file:str,cur:CurrentUser=Depends(require_login),db:Session=Depends(get_db)):
    from fastapi.responses import StreamingResponse
    from ..services.video_exports import exports
    from ..services.file_service import temp_dir
    import tempfile,zipfile
    _,v,p=va.resolve_video(db,dataset_id,input_version_id,file,cur,False)
    from pathlib import Path
    labelpath=p.with_name(p.stem+'_label.json').resolve()
    if Path(v.file_path).resolve() not in labelpath.parents:raise HTTPException(400,'标注文件路径无效')
    if not labelpath.is_file():raise HTTPException(404,'请先保存标注')
    contents=exports(json.loads(labelpath.read_text()),p.stem)
    if not contents:raise HTTPException(400,'此版本尚未选择导出文件，请在工作台保存')
    def stream():
        with tempfile.TemporaryFile(dir=temp_dir()) as buf:
            with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
                for name,data in contents.items():z.writestr(name,data)
            buf.seek(0)
            while chunk:=buf.read(1024*1024):yield chunk
    return StreamingResponse(stream(),media_type='application/zip',headers={'Content-Disposition':'attachment; filename=video_annotations.zip'})

@router.get('/metadata')
def metadata(dataset_id:int,input_version_id:int,file:str,cur:CurrentUser=Depends(require_login),db:Session=Depends(get_db)):
    _,_,p=va.resolve_video(db,dataset_id,input_version_id,file,cur,False)
    return va.metadata(p)

class Body(BaseModel):
    dataset_id:int
    input_version_id:int
    file:str
    start_frame:int=Field(ge=0)
    frame_count:int=Field(default=1,ge=1,le=300)
    output_mode:str='mask'
    output_modes:list[str]=Field(default_factory=list)
    workflow:str=''
    purpose:str='annotation'
    point_limit:int=Field(default=8,ge=1,le=32)
    prompts:list[dict]
    base_label:dict|list=Field(default_factory=list)

@router.post('')
def create(body:Body,request:Request,cur:CurrentUser=Depends(require_task_user),db:Session=Depends(get_db)):
    ds,v,p=va.resolve_video(db,body.dataset_id,body.input_version_id,body.file,cur)
    if body.workflow not in ('','manual','first_frame_auto') or body.purpose not in ('annotation','seed'):raise HTTPException(400,'标注流程无效')
    if body.workflow:
        if not body.output_modes or len(set(body.output_modes))!=len(body.output_modes) or any(k not in ('box','mask','trajectory') for k in body.output_modes):raise HTTPException(400,'请至少选择一种输出文件')
        if body.purpose=='seed' and body.frame_count!=1:raise HTTPException(400,'草稿mask只处理当前帧')
        if body.purpose=='annotation' and (body.workflow!='first_frame_auto' or body.start_frame!=0):raise HTTPException(400,'自动标注须从首帧开始')
        body.output_mode='mask'
    elif body.purpose=='seed':raise HTTPException(400,'草稿mask须指定工作流程')
    if body.output_mode not in ('trajectory','box','mask'):raise HTTPException(400,'请选择轨迹、检测框或mask')
    caps=va.capability();available=caps.get('types',{}).get(body.output_mode,{'ready':caps['ready']})
    if not available['ready']:raise HTTPException(503,'所选标注类型的模型未就绪')
    if body.workflow and body.purpose=='annotation' and 'trajectory' in body.output_modes and not caps.get('types',{}).get('trajectory',{'ready':caps['ready']})['ready']:raise HTTPException(503,'轨迹模型未就绪')
    meta=va.metadata(p)
    if body.start_frame+body.frame_count>meta['frame_count']:raise HTTPException(400,'处理区间超出视频帧数')
    va.check_prompts(body.prompts,meta,body.output_mode,body.point_limit)
    if body.workflow and 'trajectory' in body.output_modes:
        va.check_prompts(body.prompts,meta,'trajectory',body.point_limit,center_fallback=False,independent_box=True)
    try:va.flatten(body.base_label,meta['fps'])
    except (ValueError,TypeError,KeyError,AttributeError):raise HTTPException(400,'已有标注格式无法安全合并')
    params=body.model_dump();params.pop('dataset_id');params.pop('input_version_id')
    encoded=json.dumps(params,ensure_ascii=False)
    if len(encoded.encode())>32*1024*1024:raise HTTPException(413,'当前标注超过32MB，请缩短区间或减少目标')
    title=('草稿 mask' if body.purpose=='seed' else '、'.join({'trajectory':'点轨迹','box':'目标框','mask':'mask'}[k] for k in body.output_modes)) if body.workflow else {'trajectory':'点轨迹','box':'检测框','mask':'SAM mask'}[body.output_mode]
    task=AutoLabelTask(name='视频工作台 · '+title+' · '+p.name,
                       data_type='video',source_ids=json.dumps([ds.id]),algorithm=va.ALGORITHM,
                       rules_config=json.dumps({k:params[k] for k in ('file','start_frame','frame_count','output_mode','point_limit','output_modes','workflow','purpose')}),input_version_id=v.id,created_by=cur.id)
    db.add(task);db.flush()
    from ..config import settings
    work=settings.upload_dir/'video_annotation_jobs'/str(task.id)
    try:
        work.mkdir(parents=True,exist_ok=False)
        (work/'input.json').write_text(encoded)
        db.commit()
    except Exception:
        db.rollback()
        raise
    write_log(db,user_id=cur.id,username=cur.username,action='视频模型标注',module='auto_label',detail=f'{task.name}，{body.frame_count}帧',ip=client_ip(request))
    va.start(task.id)
    return compact(task)

def compact(t):
    d=_task_dict(t);params=d.pop('rules_config',{})
    d['start_frame']=params.get('start_frame',0);d['output_mode']=params.get('output_mode','mask');d['file']=params.get('file');d['frame_count']=params.get('frame_count');d['dataset_id']=d.get('source_ids',[None])[0]
    d.update({k:params.get(k) for k in ('output_modes','workflow','purpose')})
    return d

def get_task(db,tid,cur):
    t=db.get(AutoLabelTask,tid)
    if not t or t.algorithm!=va.ALGORITHM:raise HTTPException(404,'任务不存在')
    if not cur.unrestricted and t.created_by!=cur.id:raise HTTPException(403,'只能查看自己的视频任务')
    from ..services.version_service import get_dataset
    ds=get_dataset(db,json.loads(t.source_ids)[0],cur);cur.require_view(ds.secret_level)
    return t

@router.get('')
def listing(page:int=1,page_size:int=20,cur:CurrentUser=Depends(require_login),db:Session=Depends(get_db)):
    from ..utils.common import paginate
    q=db.query(AutoLabelTask).filter(AutoLabelTask.algorithm==va.ALGORITHM)
    q=q.filter(~AutoLabelTask.rules_config.contains('"purpose": "seed"'))
    if not cur.unrestricted:q=q.filter(AutoLabelTask.created_by==cur.id)
    result=paginate(q.order_by(AutoLabelTask.id.desc()),page,page_size);result['items']=[compact(t) for t in result['items']];return result

@router.get('/{tid}/input')
def task_input(tid:int,cur:CurrentUser=Depends(require_login),db:Session=Depends(get_db)):
    t=get_task(db,tid,cur)
    from ..config import settings
    params=json.loads((settings.upload_dir/'video_annotation_jobs'/str(t.id)/'input.json').read_text())
    if params.get('workflow')!='first_frame_auto' or params.get('purpose')!='annotation':raise HTTPException(400,'此任务没有可恢复的首帧准备结果')
    return {k:params[k] for k in ('base_label','output_modes','frame_count')}

@router.get('/{tid}/result')
def draft_result(tid:int,cur:CurrentUser=Depends(require_login),db:Session=Depends(get_db)):
    t=get_task(db,tid,cur)
    if t.status!='completed' or json.loads(t.rules_config).get('purpose')!='seed':raise HTTPException(409,'草稿mask尚未完成')
    from ..config import settings
    return json.loads((settings.upload_dir/'video_annotation_jobs'/str(t.id)/'result.json').read_text())

@router.get('/{tid}')
def get(tid:int,cur:CurrentUser=Depends(require_login),db:Session=Depends(get_db)):return compact(get_task(db,tid,cur))

@router.post('/{tid}/cancel')
def cancel(tid:int,cur:CurrentUser=Depends(require_task_user),db:Session=Depends(get_db)):
    t=get_task(db,tid,cur)
    from ..services.version_service import get_dataset
    cur.require_operate(get_dataset(db,json.loads(t.source_ids)[0],cur).secret_level)
    if t.status not in ('pending','running'):raise HTTPException(409,'任务已结束')
    t.status='cancelled';t.completed_at=datetime.now();db.commit();return compact(t)
