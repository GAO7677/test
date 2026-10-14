"""Audit reproduction only. Uses CPU, independent SQLite and tiny synthetic files."""
import json,os,secrets,sys,shutil
from pathlib import Path
from unittest.mock import patch
OUT=Path(os.environ['DP_AUDIT_DIR']).resolve();OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'fixtures.sqlite').exists(),'Use a new fixture path'
ROOT=Path(os.environ['DP_APP_ROOT']).resolve()
cfg=json.loads((ROOT/'config.json').read_text());cfg.update(upload_dir=str(OUT/'uploads'),log_dir=str(OUT/'logs'),jwt_secret=secrets.token_urlsafe(48))
cfg['database']={'host':'127.0.0.1','port':3307,'username':'audit','password':'unused','dbname':'audit'}
cp=OUT/'config.json';cp.write_text(json.dumps(cfg));cp.chmod(0o600)
os.environ.update(DP_CONFIG_PATH=str(cp),DATABASE_URL='sqlite:///'+str(OUT/'fixtures.sqlite'),CUDA_VISIBLE_DEVICES='')
sys.path.insert(0,str(ROOT/'backend'))
import numpy as np
import cv2
import soundfile as sf
from PIL import Image
from starlette.requests import Request
from app.database import SessionLocal,engine,init_schema
from app.models import User,Dataset,DatasetVersion,DataFile,TextData,ImageData,AutoLabelTask,CleanTask,TransformTask,SampleDataset
from app.security import CurrentUser
from app.services import preprocess as pp,version_service as vs,file_service as fs,quality_service as qs,whisper_service as whisper
from app.services.label_schema import label_names
from app.services.label_locator import find_label
from app.routers import tasks,datasets,service
assert engine.url.get_backend_name()=='sqlite'
init_schema();probes=[]
def record(id,title,observed,confirmed=True):
    assert confirmed,(id,title,observed)
    probes.append(dict(id=id,title=title,observed=observed,confirmed=True));print(id+' '+title,flush=True)
request=Request({'type':'http','client':('127.0.0.1',0),'headers':[]})
with SessionLocal() as db:
    u=User(username='audit_admin',password_hash='unused',display_name='audit',role='system_admin',status='active',must_change_password=False);db.add(u);db.commit();uid=u.id;cur=CurrentUser(u)

def dataset(dtype,files):
    with SessionLocal() as db:
        ds=Dataset(name='audit_'+dtype,data_type=dtype,secret_level='公开',created_by=uid);db.add(ds);db.flush();v=vs.create_initial_version(db,ds,uid)
        root=Path(v.file_path)
        for rel,value in files.items():
            p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
            if callable(value):value(p)
            elif isinstance(value,bytes):p.write_bytes(value)
            elif isinstance(value,(dict,list)):p.write_text(json.dumps(value,ensure_ascii=False))
            else:p.write_text(value)
        vs.refresh_version_stats(db,v);return ds.id,v.id,root

def execute(model,fn,did,vid,**kwargs):
    with SessionLocal() as db:
        fields=dict(name='audit',input_version_id=vid,created_by=uid,**kwargs)
        if model in (CleanTask,AutoLabelTask):fields['source_ids']=json.dumps([did])
        else:fields['source_id']=did
        t=model(**fields);db.add(t);db.commit();tid=t.id
    fn(tid)
    with SessionLocal() as db:
        t=db.get(model,tid);v=db.get(DatasetVersion,t.output_version_id) if t.output_version_id else db.query(DatasetVersion).filter(DatasetVersion.dataset_id==did,DatasetVersion.source_task_id==tid).order_by(DatasetVersion.id.desc()).first()
        return t.status,v.id,Path(v.file_path)

from fastapi import HTTPException
import io,zipfile

def rejected(fn):
    try: fn()
    except HTTPException as e: assert 400 <= e.status_code < 500; return
    raise AssertionError('request should fail before creating work')

# Data transformations preserve paired labels; unsafe changes cannot be published.
box={'caption':'plane','annotations':[{'type':'bbox','xyxy':[1000,200,1500,600],'label':'airplane'}]}
did,vid,root=dataset('image',{'large.png':lambda p:Image.new('RGB',(1600,800)).save(p),'large_label.json':box})
status,ov,out=execute(TransformTask,pp._exec_transform,did,vid,data_type='image',transform_type='standardize',params='{"size":640}')
assert status=='completed' and Image.open(out/'large.png').size==(640,320)
a=json.loads((out/'large_label.json').read_text());assert a['annotations'][0]['xyxy']==[400,80,600,240] and a['caption']=='plane'
with SessionLocal() as db:assert db.get(Dataset,did).current_version_id==ov
record('F01','缩放同步框和元数据且完成后切换版本',{})
span=[{'type':'span','start':2,'end':4,'text':'北京','label':'地名'}]
did,vid,root=dataset('text',{'doc.txt':'\n\n北京起飞。\n\n','doc_label.json':span})
status,ov,out=execute(CleanTask,pp._exec_clean,did,vid,data_type='text',rules='{"missing":true,"duplicate":false}')
a=json.loads((out/'doc_label.json').read_text())[0];assert status=='completed' and (out/'doc.txt').read_text()[a['start']:a['end']]=='北京'
record('F02','文本清洗同步实体偏移',{})
p=OUT/'records.json';p.write_text(json.dumps([{'id':1},{'id':1}],indent=2));d=OUT/'json_clean';pp.clean_file(p,{'duplicate':True},d);assert json.loads((d/p.name).read_text())==[{'id':1},{'id':1}]
record('F03','结构化文件不按物理行破坏内容',{})
did,vid,root=dataset('image',{'bad.png':b'bad','bad_label.json':box})
status,ov,out=execute(CleanTask,pp._exec_clean,did,vid,data_type='image',rules='{}');assert not (out/'bad_label.json').exists() and status=='failed'
record('F04','删除损坏数据不遗留孤儿标签且空结果失败',{})
status,ov,out=execute(AutoLabelTask,pp._exec_auto_label,did,vid,data_type='image',algorithm='edge_detect',rules_config='{}')
with SessionLocal() as db:assert db.get(Dataset,did).current_version_id==vid and not vs.version_ready(db,db.get(DatasetVersion,ov))
assert status=='failed' and not (out/'bad_label.json').exists()
record('F05','损坏图标注失败且保持原默认版本',{})
did,vid,root=dataset('image',{'ok.png':lambda p:Image.new('RGB',(100,50)).save(p)})
with SessionLocal() as db,patch.object(pp,'start_auto_label'):
    rejected(lambda:tasks.create_auto_label(tasks.AutoLabelBody(name='invalid',data_type='image',dataset_id=did,algorithm='made_up'),request,cur,db))
    rejected(lambda:tasks.create_auto_label(tasks.AutoLabelBody(name='invalid',data_type='text',dataset_id=did,algorithm='text_class'),request,cur,db))
    rejected(lambda:tasks.create_auto_label(tasks.AutoLabelBody(name='invalid',data_type='image',dataset_id=did,algorithm='template_match'),request,cur,db))
record('F06','假算法、错误模态和占位算法提交前拒绝',{})
with SessionLocal() as db,patch.object(pp,'start_transform'):
    rejected(lambda:tasks.create_transform(tasks.TransformBody(name='invalid',data_type='image',dataset_id=did,transform_type='pipeline'),request,cur,db))
    rejected(lambda:tasks.create_transform(tasks.TransformBody(name='invalid',data_type='image',dataset_id=did,transform_type='feature'),request,cur,db))
    rejected(lambda:tasks.create_transform(tasks.TransformBody(name='invalid',data_type='image',dataset_id=did,transform_type='format',target_format='mp4'),request,cur,db))
record('F07','未实现转换提交前拒绝',{})
status,ov,out=execute(TransformTask,pp._exec_transform,did,vid,data_type='image',transform_type='format',target_format='mp4',params='{}')
assert status=='failed' and not list(out.iterdir())
record('F08','转换失败不原样复制冒充成功',{})
for method in ['time','stratified','kfold']:
    with SessionLocal() as db,patch.object(pp,'start_sample'):
        rejected(lambda:tasks.create_sample(tasks.SampleBody(name='invalid',data_type='image',dataset_id=did,method=method),request,cur,db))
record('F09','未实现采样不再冒充随机执行',{})
did,vid,root=dataset('voice',{'tiny.wav':lambda p:sf.write(str(p),np.zeros(8000),8000)})
with patch.object(whisper,'transcribe',return_value=[{'start':0,'end':1,'text':'测试'}]) as stub:
    a=pp.label_voice(root/'tiny.wav','whisper_transcribe',{});assert a['segments'][0]['text']=='测试' and stub.call_args.args[0]==str(root/'tiny.wav')
record('F10','Whisper使用正确路径和分段文本',{})
with SessionLocal() as db:
    for params in [{'overlap':1},{'sliceDuration':0},{'overlap':-1}]:
        rejected(lambda:tasks.create_auto_label(tasks.AutoLabelBody(name='invalid',data_type='voice',dataset_id=did,algorithm='fixed_slice',rules_config=params),request,cur,db))
record('F11','危险切片参数提交前拒绝',{})
p=OUT/'tiny.mp4';w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),2,(64,48));assert w.isOpened()
for i in range(12):w.write(np.full((48,64,3),i*15,dtype=np.uint8))
w.release()
with patch('app.services.detector.detect_frame',return_value=[]):vl=pp.label_video(p,'coco_detect',{})
assert [i['time'] for i in vl['items']]==[0,2,4] and not vl['truncated'] and all(i['status']=='empty' for i in vl['items'])
with patch('app.services.detector.detect_frame',return_value=[]):vl=pp.label_video(p,'coco_detect',{'maxFrames':2})
assert vl['truncated'] and vl['processed_end_time']==2
record('F12','视频记录零检测帧并明确截断',{})
assert 'airplane' in label_names({'items':[{'label':'1个目标','boxes':[{'label':'airplane'}]}]})
assert '飞行日志' in label_names({'classification':{'category':'飞行日志'}})
assert not label_names({'error':'bad','boxes':[{'label':'airplane'}]})
record('F13','视频嵌套及文本分类检索完整并排除错误',{})
did,vid,root=dataset('image',{'a/same.png':lambda p:Image.new('RGB',(10,10)).save(p),'b/same.png':lambda p:Image.new('RGB',(10,10)).save(p),'a/same_label.json':{'error':'bad'}})
assert qs._score_files('image',pp.list_files(root))['metrics']['label_coverage']==0
(root/'a/same_label.json').write_text('{"boxes":[]}');assert qs._score_files('image',pp.list_files(root))['metrics']['label_coverage']==50
record('F14','标注配套按完整路径与合法性计数',{})
with SessionLocal() as db:
    ids=[]
    for content in ['one','two']:
        raw=TextData(name=content,format='TXT',secret_level='公开',created_by=uid);db.add(raw);db.flush();p=fs.dataset_dir('text',raw.id)/'same.txt';p.write_text(content)
        raw.file_path=str(p.parent);db.add(DataFile(dataset_id=raw.id,data_type='text',filename=p.name,stored_name=p.name,size_bytes=p.stat().st_size,stage='raw',status='complete'));ids.append(raw.id)
    db.commit();ds=datasets.create_dataset(datasets.DatasetBody(name='two sources',data_type='text',source_data_ids=ids,secret_level='公开'),request,cur,db)
    v=db.get(DatasetVersion,ds['current_version']['id']);assert sorted(p.read_text() for p in pp.data_files(Path(v.file_path)))==['one','two']
record('F15','同名来源文件分别保留且来源目录可追溯',{})
with SessionLocal() as db:
    raw=TextData(name='missing',format='TXT',secret_level='公开',created_by=uid);db.add(raw);db.flush();db.add(DataFile(dataset_id=raw.id,data_type='text',filename='missing.txt',stored_name='missing.txt',size_bytes=1,stage='raw',status='complete'));db.commit()
    count=db.query(Dataset).count();rejected(lambda:datasets.create_dataset(datasets.DatasetBody(name='rollback',data_type='text',source_data_ids=[raw.id]),request,cur,db));assert db.query(Dataset).count()==count
record('F16','复制失败不提交半成品数据集',{})
with SessionLocal() as db:
    raw=TextData(name='pair',format='TXT',secret_level='公开',created_by=uid);db.add(raw);db.flush();p=fs.dataset_dir('text',raw.id)/'pair.txt';p.write_text('\n北京');raw.file_path=str(p.parent)
    ds=Dataset(name='pair work',data_type='text',secret_level='公开',created_by=uid,source_data_ids=json.dumps([raw.id]));db.add(ds);db.flush();v=vs.create_initial_version(db,ds,uid)
    q=Path(v.file_path)/'pair.txt';q.write_text('北京');q.with_name('pair_label.json').write_text(json.dumps([{'start':0,'end':2,'text':'北京','type':'span'}]));vs.refresh_version_stats(db,v)
    assert find_label(db,'text',raw.id,p,'pair.txt',cur) is None
    q.write_bytes(p.read_bytes());assert find_label(db,'text',raw.id,p,'pair.txt',cur) is not None
    t=TransformTask(name='failed',source_id=ds.id,data_type='text',status='failed',transform_type='format',input_version_id=v.id);db.add(t);db.commit()
    bad=vs.create_output_version(db,ds,input_version=v,task_type='transform',task_id=t.id,user_id=uid)
    (Path(bad.file_path)/'pair.txt').write_bytes(p.read_bytes());(Path(bad.file_path)/'pair_label.json').write_text('{"boxes":[{"label":"bad"}]}')
    assert 'bad' not in label_names(find_label(db,'text',raw.id,p,'pair.txt',cur))
record('F17','原始下载仅复用内容一致且成功版本的标签',{})
did,vid,root=dataset('text',{'example.txt':'北京','example_label.json':[{'type':'span','start':0,'end':2,'text':'北京','label':'地名'}]})
with SessionLocal() as db:
    g=service._scene_graph(db,db.get(Dataset,did),db.get(DatasetVersion,vid),service.SupplyBody(scene='graph',dataset_id=did),cur)
    nodes={n['id'] for n in g['nodes']};assert g['edges'] and all(e['source'] in nodes and e['target'] in nodes for e in g['edges'])
record('F18','人工实体可供图且边端点存在',{})
from app.models import SemanticTerm
from app.routers.standards import list_terms,semantic_check,SemanticCheckBody
with SessionLocal() as db:
    u=User(username='public',password_hash='unused',role='visitor',view_level=0,operate_level=-1,status='active');db.add(u)
    db.add(SemanticTerm(term='密词',definition='test',synonyms='["秘密"]',secret_level='机密',status='现行'));db.commit();visitor=CurrentUser(u)
    assert not any(x['term']=='密词' for x in list_terms(1,100,'',visitor,db)['items'])
    assert semantic_check(SemanticCheckBody(text='秘密'),visitor,db)['hit_count']==0
record('F19','术语列表和检查执行密级隔离',{})
with SessionLocal() as db:
    for node in [{'type':'sample','params':{'train_ratio':-1,'val_ratio':1,'test_ratio':100}},{'type':'transform','transform_type':'feature'}]:
        rejected(lambda:tasks.create_workflow(tasks.WorkflowBody(name='invalid',dataset_id=did,nodes=[node]),request,cur,db))
record('F20','工作流复用能力和参数校验',{})
p=OUT/'points.csv';p.write_text('x,y,z,intensity\n1,2,3,10\n4,5,6,20\n');d=OUT/'points';d.mkdir();q=pp._convert_format(p,'.csv','ply',d);assert fs._load_radar_points(q).shape[1]==4
record('F21','点云转换保留强度列',{})
p=OUT/'crlf.txt';p.write_bytes('😀\r\n北京'.encode());assert fs._text_preview(p)['text']=='😀\r\n北京'
p=OUT/'gbk.txt';p.write_bytes('北京起飞'.encode('gbk'));assert fs._text_preview(p)['text']=='北京起飞'
record('F22','文本预览保留换行并正确解码GBK',{})
from app.routers.download import _zip_files
for iterator in [datasets._zip_version(root,'data,labels'),service._zip_supply(root,'data,labels',{},{}),_zip_files([('x.txt',root/'example.txt')],{})]:
    data=b''.join(iterator);assert zipfile.ZipFile(io.BytesIO(data)).testzip() is None
record('F23','三条下载路径的临时盘ZIP均可解压',{})
with SessionLocal() as db:
    ds=db.get(Dataset,did);v=db.get(DatasetVersion,vid);target=fs.settings.upload_dir/'processed'/str(did)/f'V{vs.next_version_no(db,did)}';target.mkdir();(target/'keep').write_text('do not delete')
    try:vs.copy_version(db,ds,v,uid)
    except FileExistsError:db.rollback()
    else:raise AssertionError('must reject destination conflict')
    assert (target/'keep').read_text()=='do not delete'
record('F24','版本目录冲突不删除其他产物',{})
(OUT/'results.json').write_text(json.dumps({'passed':True,'checks':probes,'count':len(probes),'gpu_used':False,'production_db_mutated':False},ensure_ascii=False,indent=2))
print(f'{len(probes)} checks passed')
