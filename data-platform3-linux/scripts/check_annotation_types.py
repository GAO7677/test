"""Isolated CPU contract/lifecycle checks. The model subprocess is a test double."""
import os,sys,json,secrets,shutil,time,subprocess,threading
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];O=Path(os.environ['DP_ACCEPTANCE_DIR']);O.mkdir(parents=True,exist_ok=True)
assert not (O/'fixtures.sqlite').exists(),'Use a fresh isolated directory'
cfg=json.loads((R/'config.json').read_text());cfg.update(upload_dir=str(O/'uploads'),log_dir=str(O/'logs'),jwt_secret=secrets.token_urlsafe(48))
cfg['fewshot']['output_root']=str(O/'fewshot');f=O/'config.json';f.write_text(json.dumps(cfg));f.chmod(0o600)
os.environ.update(DP_CONFIG_PATH=str(f),DATABASE_URL='sqlite:///'+str(O/'fixtures.sqlite'),CUDA_VISIBLE_DEVICES='')
sys.path.insert(0,str(R/'backend'));sys.path.insert(0,str(R/'integrations/video_annotation'))
from fastapi.testclient import TestClient
from app.main import create_app
from app.database import init_schema,seed_data,SessionLocal
from app.models import User,Dataset,DatasetVersion,AutoLabelTask
from app.security import create_token,hash_password
from app.services import version_service as vs,video_annotation as va
from mask_codec import encode,decode
import numpy as np
init_schema();seed_data();checks=[]
def check(name,condition=True):assert condition,name;checks.append(name);print('PASS '+name,flush=True)
def client(u):
 c=TestClient(create_app());c.cookies.set('dp_access_token',create_token(u));return c
with SessionLocal() as db:
 admin=db.query(User).filter_by(username='sysadmin').one();admin.must_change_password=False
 a=User(username='video_a',password_hash=hash_password('acceptance-only'),display_name='A',role='visitor',operate_level=5,view_level=5,status='active',must_change_password=False)
 b=User(username='video_b',password_hash=hash_password('acceptance-only'),display_name='B',role='visitor',operate_level=5,view_level=5,status='active',must_change_password=False)
 db.add_all([a,b]);db.commit();ca,cb,cadmin=client(a),client(b),client(admin)
 ds=Dataset(name='视频跟踪独立验收',data_type='video',secret_level='公开',created_by=a.id);db.add(ds);db.flush();v=vs.create_initial_version(db,ds,a.id)
 # Known short H.264 fixture from previous task, read-only source.
 prev=Path('/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_video_preview')
 source=prev/'uploads/video/1/data/video_a.mp4';shutil.copy2(source,Path(v.file_path)/'test.mp4')
 vs.refresh_version_stats(db,v);did,vid=ds.id,v.id;original=Path(v.file_path)
 (O/'browser_fixture.json').write_text(json.dumps({'token':create_token(admin),'dataset_id':did,'version_id':vid,'file':'test.mp4'}));(O/'browser_fixture.json').chmod(0o600)
mask=np.zeros((48,64),bool);mask[4:20,6:30]=1;mask[8:10,12:16]=0
check('RLE像素与孔洞无损往返',np.array_equal(mask,decode(encode(mask))))
body={'dataset_id':did,'input_version_id':vid,'file':'test.mp4','start_frame':0,'frame_count':3,'output_mode':'mask',
 'prompts':[{'track_id':1,'label':'person','xyxy':[6,4,30,20]}],
 'base_label':{'caption':'preserve','annotations':[{'type':'video_box','track_id':99,'frame':0,'time':0,'xyxy':[1,1,3,3],'label':'car'}]}}
check('匿名不能查看模型能力',TestClient(create_app()).get('/api/video-annotation/capabilities').status_code==401)
meta=ca.get('/api/video-annotation/metadata',params={k:body[k] for k in ('dataset_id','input_version_id','file')}).json()
check('工作台读取真实帧率与尺寸',meta['fps']==2 and meta['width']==64 and meta['frame_count']==12)
with patch.object(va,'capability',return_value={'ready':True}),patch.object(va,'start'):
 for name,changes in [('路径穿越',{'file':'../test.mp4'}),('错误版本',{'input_version_id':9999}),('越界区间',{'frame_count':13}),('非法类别',{'output_mode':'box','prompts':[{'track_id':1,'label':'bad','xyxy':[1,1,3,3]}]}),('重复目标ID',{'prompts':body['prompts']*2}),('非法mask',{'prompts':[{**body['prompts'][0],'mask':{'size':[48,64],'counts':[1]}}]})]:
  check(name+'被拒绝',ca.post('/api/video-annotation',json={**body,**changes}).status_code in (400,404,422))
 first=ca.post('/api/video-annotation',json=body);check('复用自动标注任务创建',first.status_code==200);tid=first.json()['id']
 check('其他用户不能读任务',cb.get(f'/api/video-annotation/{tid}').status_code==403)
 check('其他用户不能取消任务',cb.post(f'/api/video-annotation/{tid}/cancel').status_code==403)
 check('用户列表隔离',cb.get('/api/video-annotation').json()['total']==0)
 check('管理员可查看',cadmin.get(f'/api/video-annotation/{tid}').status_code==200)
 check('待执行任务可取消',ca.post(f'/api/video-annotation/{tid}/cancel').json()['status']=='cancelled')
 va.execute(tid)
 check('取消不生成版本',ca.get(f'/api/video-annotation/{tid}').json()['output_version_id'] is None)

testworker=O/'cpu_worker.py';testworker.write_text('''import sys,json,time,os
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())
assert os.environ['CUDA_VISIBLE_DEVICES']==m['runtime']['gpu_uuid']
if len(sys.argv)>2 and sys.argv[2]=='fail':raise RuntimeError('intentional CPU failure')
if len(sys.argv)>2 and sys.argv[2]=='slow':time.sleep(30)
out={'annotations':[{'type':'video_box','track_id':1,'frame':i,'time':i/2,'label':'person','xyxy':[6,4,30,20],'mask':json.loads(Path(sys.argv[1]).with_name('expected_mask.json').read_text())} for i in range(3)],'fps':2,'width':64,'height':48,'frame_times':[{'frame':i,'time':i/2} for i in range(3)],'start_frame':0,'end_frame':2,'model':'CPU test double','review_status':'pending'}
Path(m['output']).write_text(json.dumps(out))
''')
real_popen=subprocess.Popen
runtime={'python':sys.executable,'repository':'unused','checkpoint':'unused','gpu_uuid':va.GPU7}
def submit_and_run(mode='ok',async_run=False):
 with patch.object(va,'capability',return_value={'ready':True}),patch.object(va,'start'):
  res=ca.post('/api/video-annotation',json=body);assert res.status_code==200,res.text
 tid=res.json()['id'];work=O/'uploads/video_annotation_jobs'/str(tid);(work/'expected_mask.json').write_text(json.dumps(encode(mask)))
 def popen(cmd,**kw):return real_popen([sys.executable,str(testworker),cmd[-1],mode],**kw)
 def run():
  with patch.object(va,'runtime',return_value=runtime),patch.object(va.subprocess,'Popen',side_effect=popen):va.execute(tid)
 if async_run:
  thread=threading.Thread(target=run);thread.start();return tid,thread
 run();return tid

tid=submit_and_run();task=ca.get(f'/api/video-annotation/{tid}').json();check('进程完成后生成新版本',task['status']=='completed' and task['output_version_id']!=vid)
newvid=task['output_version_id'];label=ca.get(f'/api/datasets/{did}/versions/{newvid}/file',params={'path':'test_label.json'}).json()
check('保留旧类别、元数据和其他轨迹',label['caption']=='preserve' and any(a['track_id']==99 for a in label['annotations']))
check('mask和轨迹ID按帧回读',len([a for a in label['annotations'] if a['track_id']==1])==3 and np.array_equal(decode(label['annotations'][-1]['mask']),mask))
check('原版本不覆盖',not (original/'test_label.json').exists())
with SessionLocal() as db:check('成功版本提升当前指针',db.get(Dataset,did).current_version_id==newvid)
failed=submit_and_run('fail');check('推理失败如实记录',ca.get(f'/api/video-annotation/{failed}').json()['status']=='failed')
slow,thread=submit_and_run('slow',True)
for _ in range(50):
 if ca.get(f'/api/video-annotation/{slow}').json()['status']=='running':break
 time.sleep(.1)
check('运行中任务可取消',ca.post(f'/api/video-annotation/{slow}/cancel').json()['status']=='cancelled');thread.join(12)
check('取消终止本任务子进程',not thread.is_alive())
with SessionLocal() as db:check('失败和取消不提升版本指针',db.get(Dataset,did).current_version_id==newvid)
# A real editable mask fixture for the browser; CPU output is explicitly labelled above.
f=json.loads((O/'browser_fixture.json').read_text());f['result_version_id']=newvid;f['task_id']=tid;(O/'browser_fixture.json').write_text(json.dumps(f))
(O/'api_results.json').write_text(json.dumps({'passed':True,'checks':checks,'model':'CPU test double, not GPU model validation'},ensure_ascii=False,indent=2))

# New types and per-object limits: API contracts only, no model inference.
from copy import deepcopy
with patch.object(va,'capability',return_value={'ready':True}),patch.object(va,'start'):
 pointbody={**deepcopy(body),'output_mode':'trajectory','prompts':[{'track_id':1,'label':'target','xyxy':[1,1,30,30],'point_limit':8,'points':[]},{'track_id':2,'label':'target','xyxy':[31,1,63,30],'point_limit':8,'points':[{'point_id':i+1,'xy':[32+i,5]} for i in range(8)]}]}
 res=ca.post('/api/video-annotation',json=pointbody);check('每物体8点独立计数，共9点可提交',res.status_code==200)
 job=res.json();saved=json.loads((O/'uploads/video_annotation_jobs'/str(job['id'])/'input.json').read_text())
 check('无点物体补入框中心起始点',saved['prompts'][0]['points']==[{'point_id':1,'xy':[15.5,15.5],'origin':'box_center'}])
 for name,mutate in [('单物体超限',lambda b:b['prompts'][1]['points'].append({'point_id':9,'xy':[42,5]})),('框外点',lambda b:b['prompts'][1]['points'][0].update(xy=[0,0])),('重复点ID',lambda b:b['prompts'][1]['points'][1].update(point_id=1)),('非法上限',lambda b:b['prompts'][0].update(point_limit=33))]:
  bad=deepcopy(pointbody);mutate(bad);check(name+'被拒绝',ca.post('/api/video-annotation',json=bad).status_code==400)
 ca.post(f"/api/video-annotation/{job['id']}/cancel")
base={'annotations':[{'type':'video_box','frame':0,'track_id':1,'xyxy':[1,1,9,9],'mask':{'counts':[1]}},{'type':'video_point','frame':0,'track_id':1,'point_id':9,'xy':[3,3]},{'type':'video_point','frame':0,'track_id':2,'point_id':1,'xy':[4,4]}]}
result={'annotation_type':'trajectory','fps':2,'annotations':[{'type':'video_point','frame':0,'track_id':1,'point_id':1,'xy':[2,2]}],'frame_times':[{'frame':0,'time':0}],'start_frame':0,'end_frame':0,'model':'test'}
merged=va.merge_label(base,result,{'prompts':[{'track_id':1}]})
check('生成轨迹保留同物体框mask和其他物体轨迹',len(merged['annotations'])==3 and any(a.get('mask') for a in merged['annotations']) and any(a.get('track_id')==2 for a in merged['annotations']))
check('区间记录包含类型和物体ID',merged['tracking_runs'][-1]['annotation_type']=='trajectory' and merged['tracking_runs'][-1]['track_ids']==[1])
(O/'api_results.json').write_text(json.dumps({'passed':True,'checks':checks,'model':'CPU test double; not GPU validation'},ensure_ascii=False,indent=2))
