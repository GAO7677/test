"""Video assistance shares AutoLabelTask, permissions and DatasetVersion lifecycle."""
import json,os,signal,subprocess,threading,time,fcntl
from datetime import datetime
from pathlib import Path
from fastapi import HTTPException
from ..config import settings,base_dir
from ..database import SessionLocal
from ..models import AutoLabelTask,Dataset,DatasetVersion
from . import version_service as vs,preprocess as pp

ALGORITHM='sam2_video_workbench'
GPU7='GPU-439e8b22-75c0-55fc-283f-01f012c049ac'

def runtime():
    from .detector import _default_model_path
    return {**settings.data.get('video_annotation',{}),'detector_path':str(_default_model_path())}

def capability():
    c=runtime();ready=all(Path(c.get(k,'/nonexistent')).is_file() for k in ('python','checkpoint')) and Path(c.get('repository','/nonexistent'),'sam2','build_sam.py').is_file() and c.get('gpu_uuid')==GPU7
    points=Path(c.get('point_repository','/nonexistent'),'cotracker','predictor.py').is_file() and Path(c.get('point_checkpoint','/nonexistent')).is_file() and Path(c.get('python','/nonexistent')).is_file() and c.get('gpu_uuid')==GPU7
    detector=ready and Path(c['detector_path']).is_file()
    types={'trajectory':{'ready':points,'model':'CoTracker3'},'box':{'ready':detector,'model':'YOLOv8n + SAM 2.1 目标关联'},'mask':{'ready':ready,'model':'SAM 2.1 Hiera Tiny'}}
    return {'ready':ready,'types':types,'max_points_per_object':32,'model':'SAM 2.1 Hiera Tiny','max_frames':300,'max_objects':10,
            'reason':'' if ready else '视频跟踪模型或运行环境未就绪','gpu':'118 GPU 7'}

def resolve_video(db,did,vid,file,cur,operate=True):
    ds=vs.get_dataset(db,did,cur)
    (cur.require_operate if operate else cur.require_view)(ds.secret_level)
    if ds.data_type!='video':raise HTTPException(400,'仅视频数据可使用视频工作台模型')
    v=vs.validate_input_version(db,ds,vid);root=Path(v.file_path).resolve();rel=Path(file.replace('\\','/'));p=(root/rel).resolve()
    if rel.is_absolute() or '..' in rel.parts or root not in p.parents or not p.is_file() or p.suffix.lower() not in {'.mp4','.avi','.mov','.mkv'}:
        raise HTTPException(400,'请选择当前版本内的视频文件')
    return ds,v,p

def metadata(path):
    import cv2
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise HTTPException(400,'视频无法解码')
        return {'fps':cap.get(cv2.CAP_PROP_FPS),'frame_count':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:cap.release()

def check_prompts(prompts,meta,output_mode="mask",point_limit=8,center_fallback=True,independent_box=False):
    import math
    from .detector import COCO_CLASSES
    if not 1<=len(prompts)<=10:raise HTTPException(400,'请选择1至10个目标')
    ids=set()
    for p in prompts:
        tid=p.get('track_id');box=p.get('xyxy',[])
        if type(tid)!=int or tid<1 or tid in ids:raise HTTPException(400,'目标ID须为不重复的正整数')
        ids.add(tid)
        if output_mode=='box' and p.get('label') not in COCO_CLASSES:raise HTTPException(400,'请选择COCO80类别')
        if not isinstance(box,list) or len(box)!=4 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in box):raise HTTPException(400,'目标框无效')
        if not (0<=box[0]<box[2]<=meta['width'] and 0<=box[1]<box[3]<=meta['height']):raise HTTPException(400,'目标框超出视频尺寸')
        if not isinstance(p.get('label'),str) or not p['label'].strip() or len(p['label'])>100:raise HTTPException(400,'目标名称须为1至100字符')
        if output_mode=='trajectory':
            limit=p.get('point_limit',point_limit)
            if type(limit)!=int or not 1<=limit<=32:raise HTTPException(400,'每物体选点上限须为1至32')
            points=p.get('points',[])
            if not isinstance(points,list) or len(points)>limit:raise HTTPException(400,'物体选点数量超过该物体上限')
            pids=set()
            for point in points:
                if not isinstance(point,dict):raise HTTPException(400,'轨迹点格式无效')
                pid=point.get('point_id');xy=point.get('xy')
                if type(pid)!=int or pid<1 or pid in pids:raise HTTPException(400,'同一物体的点ID须为不重复正整数')
                pids.add(pid)
                if not isinstance(xy,list) or len(xy)!=2 or not all(type(x) in (int,float) and math.isfinite(x) for x in xy):raise HTTPException(400,'轨迹点坐标无效')
                region=[0,0,meta['width'],meta['height']] if independent_box else box
                if not (region[0]<=xy[0]<region[2] and region[1]<=xy[1]<region[3]):raise HTTPException(400,'起始点须位于视频画面内' if independent_box else '起始点须位于对应物体框内')
            if not points and center_fallback:p['points']=[{'point_id':1,'xy':[(box[0]+box[2])/2,(box[1]+box[3])/2],'origin':'box_center'}]
            p['point_limit']=limit
        if p.get('mask'):
            mask=p['mask']
            if not isinstance(mask,dict):raise HTTPException(400,'mask须为RLE对象')
            counts=mask.get('counts',[])
            if mask.get('size')!=[meta['height'],meta['width']] or not isinstance(counts,list) or len(counts)>meta['width']*meta['height']+1 or any(type(x)!=int or x<0 for x in counts) or sum(counts)!=meta['width']*meta['height']:
                raise HTTPException(400,'mask尺寸或RLE计数无效')

def flatten(label,fps):
    if isinstance(label,list):return label
    for key in ('annotations','items','boxes','detections'):
        if isinstance(label.get(key),list):
            rows=[]
            for a in label[key]:
                if isinstance(a.get('boxes'),list):rows.extend({**b,'frame':a['frame'],'time':a.get('time',a['frame']/fps)} for b in a['boxes'])
                else:rows.append(a)
            return rows
    if not label or any(k in label for k in ('caption','category','classification')):return []
    raise ValueError('旧标注格式无法安全合并，请先转换')

def merge_label(base,result,params):
    rows=flatten(base,result['fps']);ids={p['track_id'] for p in params['prompts']}
    start,end=result['start_frame'],result['end_frame']
    # Only replace these explicitly selected objects in the requested range.
    kind=result.get('annotation_type','mask')
    def same_kind(a):
        if params.get('workflow'):return a.get('type')!='video_point' or 'trajectory' in params.get('output_modes',[])
        if kind=='trajectory':return a.get('type')=='video_point'
        if kind=='mask' and a.get('source')=='yolov8n':return False
        return a.get('type')!='video_point'
    keep=[]
    for a in rows:
        affected=a.get('track_id') in ids and start<=a.get('frame',round(a.get('time',0)*result['fps']))<=end and same_kind(a)
        if not affected:keep.append(a)
        elif not params.get('workflow') and kind=='box' and a.get('mask'):
            # A detection result must not silently destroy a previously generated mask.
            b=dict(a);b['type']='video_mask';keep.append(b)
    out=dict(base) if isinstance(base,dict) else {}
    for k in ('annotations','items','boxes','detections'):out.pop(k,None)
    times={t['frame']:t for t in out.get('frame_times',[])};times.update({t['frame']:t for t in result['frame_times']})
    history=out.get('tracking_runs',[])+[{**{k:result[k] for k in ('start_frame','end_frame','model')},'annotation_type':kind,'track_ids':sorted(ids)}]
    out.update({k:v for k,v in result.items() if k not in ('annotations','frame_times')})
    generated=result['annotations']
    if params.get('workflow')=='first_frame_auto' and isinstance(base,dict):
        from copy import deepcopy
        manual=deepcopy(base.get('manual_annotations',[]))
        seeds={a['track_id']:a for a in manual if a.get('track_id') in ids and a.get('frame')==start and a.get('generation')}
        for a in seeds.values():
            # A persisted result has no pending editor point changes.
            a.pop('points_dirty',None)
            a['generation']={'status':'completed','output_modes':params.get('output_modes',[]),'end_frame':end}
        if seeds:
            # This is a new per-object generation. Remove obsolete generated rows
            # for these objects; preserve other objects and independent manual corrections.
            keep=[a for a in keep if a.get('track_id') not in seeds or (a.get('source')=='manual' and a.get('frame')!=start)]
            generated=[deepcopy(seeds[a['track_id']]) if a.get('track_id') in seeds and a.get('frame')==start and a.get('type')!='video_point' else a for a in generated]
            out['manual_annotations']=manual
    out.update(annotations=keep+generated,frame_times=sorted(times.values(),key=lambda x:x['frame']),tracking_runs=history,
               mask_format='COCO uncompressed RLE, column-major',review_status='pending')
    return out

def start(task_id):pp._run_in_thread(execute,task_id)

def execute(task_id):
    process=None;lock=None
    with SessionLocal() as db:
        task=db.get(AutoLabelTask,task_id)
        try:
            if task.status=='cancelled':return
            work=settings.upload_dir/'video_annotation_jobs'/str(task_id);assert (work/'input.json').is_file(), '任务输入文件缺失'
            lock=(settings.upload_dir/'video_annotation_jobs'/'gpu7.lock').open('a')
            deadline=time.monotonic()+1800
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    db.refresh(task)
                    if task.status=='cancelled':return
                    if time.monotonic()>deadline:raise RuntimeError('GPU7排队超时，请重试')
                    time.sleep(.5)
            db.refresh(task)
            if task.status=='cancelled':return
            params=json.loads((work/'input.json').read_text());input_v=db.get(DatasetVersion,task.input_version_id);ds=db.get(Dataset,input_v.dataset_id)
            root=Path(input_v.file_path).resolve();video=(root/params['file']).resolve()
            if root not in video.parents or not video.is_file():raise ValueError('输入视频不存在')
            task.status='running';db.commit()
            base=params.pop('base_label',[])
            from .detector import COCO_CLASSES
            manifest={**params,'classes':COCO_CLASSES,'runtime':runtime(),'video':str(video),'output':str(work/'result.json'),'progress':str(work/'progress.json')}
            (work/'request.json').write_text(json.dumps(manifest,ensure_ascii=False))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=GPU7,OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',DP_VIDEO_PARENT_PID=str(os.getpid()))
            with (work/'worker.log').open('w') as log:
                process=subprocess.Popen([runtime()['python'],'-u',str(base_dir()/'integrations/video_annotation/run.py'),str(work/'request.json')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+1800
                while process.poll() is None:
                    time.sleep(.5);db.refresh(task)
                    if task.status=='cancelled' or time.monotonic()>deadline:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                        if task.status=='cancelled':return
                        raise RuntimeError('视频推理超过30分钟限制，请缩短帧区间')
                    try:task.progress=json.loads((work/'progress.json').read_text())['progress'];db.commit()
                    except (FileNotFoundError,ValueError):pass
            if process.returncode:
                lines=(work/'worker.log').read_text(errors='replace').strip().splitlines()
                raise RuntimeError(('模型执行失败：'+(lines[-1] if lines else str(process.returncode)))[:500])
            db.refresh(task)
            if task.status=='cancelled':return
            result=json.loads((work/'result.json').read_text())
            if params.get('purpose')=='seed':
                pp._finish(db,task,True);return
            label=merge_label(base,result,params)
            output=vs.create_output_version(db,ds,input_version=input_v,task_type='auto_label',task_id=task.id,user_id=task.created_by,description='视频工作台标注结果，待人工核验')
            pp.copy_all(root,Path(output.file_path));pp.write_label(Path(output.file_path),Path(params['file']),label,src_dir=Path('.'))
            db.refresh(task)
            if task.status=='cancelled':return
            task.result_path=output.file_path;pp._finish(db,task,True)
        except Exception as e:
            db.rollback();task=db.get(AutoLabelTask,task_id)
            if task and task.status!='cancelled':pp._finish(db,task,False,str(e)[:500])
        finally:
            if process and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            if lock:lock.close()
