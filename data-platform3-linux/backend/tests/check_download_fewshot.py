"""Isolated CPU acceptance; simulated generation batch, no LoRA training.

Keep the resulting SQLite fixtures for browser checks. Run with a task-specific
DP_ACCEPTANCE_DIR; production MySQL and production tasks are never touched.
"""
import json
import os
import shutil
import secrets
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('DP_ACCEPTANCE_DIR','/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_download_fewshot_integration/acceptance'))
OUT.mkdir(parents=True,exist_ok=True)
config=json.loads((ROOT/'config.json').read_text())
config.update(upload_dir=str(OUT/'uploads'),log_dir=str(OUT/'logs'),jwt_secret=secrets.token_urlsafe(48))
config['database']={'host':'127.0.0.1','port':3307,'username':'acceptance','password':'unused','dbname':'download_acceptance_test'}
config['fewshot']['output_root']=str(OUT/'jobs')
cfg=OUT/'config.json';cfg.write_text(json.dumps(config));cfg.chmod(0o600)
os.environ.update(DP_CONFIG_PATH=str(cfg),DATABASE_URL='sqlite:///'+str(OUT/'fixtures.sqlite'),CUDA_VISIBLE_DEVICES='')
sys.path.insert(0,str(ROOT/'backend'))
from fastapi.testclient import TestClient
from PIL import Image
from app.main import create_app
from app.database import SessionLocal,engine,init_schema,seed_data
from app.models import Base,User,ImageData,DataFile,FewshotJob,Dataset,DatasetVersion,IntegrateTask,TransformTask
from app.security import hash_password,create_token
from app.services import file_service as fs, fewshot_service as few, version_service as vs
from app.services.data_access import excluded_dataset_ids

assert engine.url.get_backend_name()=='sqlite' and str(OUT) in str(engine.url)
Base.metadata.drop_all(engine);init_schema();seed_data()
checks=[]
def check(name,condition=True):
    assert condition,name
    checks.append(name);print('PASS '+name,flush=True)

def client_for(user):
    c=TestClient(create_app());c.cookies.set('dp_access_token',create_token(user));return c

def body(source,rid):
    return dict(request_key=str(uuid.uuid4()),name='下载关联验收',secret_level='公开',project='验收项目',scene='aircraft',download_request_id=rid,
                groups=[dict(name='aircraft',caption='a photograph of an aircraft',count=1,data_file_ids=[source])])

with SessionLocal() as db:
    admin=db.query(User).filter_by(username='sysadmin').one()
    users=[]
    for name in ['download_user_a','download_user_b']:
        user=User(username=name,password_hash=hash_password('acceptance-password'),display_name=name,role='visitor',status='active',view_level=5,operate_level=5,must_change_password=False)
        db.add(user);db.flush();users.append(user)
    db.commit();a,b=users
    raw=ImageData(name='共享候选飞机图像',format='PNG',secret_level='公开',project='验收项目',scene='aircraft',source_type='真实采集',created_by=admin.id,file_count=0,size_bytes=0)
    db.add(raw);db.flush();path=fs.dataset_dir('image',raw.id)/'aircraft.png'
    Image.new('RGB',(96,64),(20,100,170)).save(path)
    raw.file_path=str(path.parent);raw.file_count=1;raw.size_bytes=path.stat().st_size
    file=DataFile(dataset_id=raw.id,data_type='image',filename=path.name,stored_name=path.name,size_bytes=path.stat().st_size,file_hash=few.sha(path),stage='raw',status='complete',chunk_received=1)
    db.add(file);db.flush();fid=file.id;public_raw_id=raw.id
    path.with_name(path.stem+'_label.json').write_text(json.dumps({'annotations':[{'type':'bbox','xyxy':[5,5,70,45],'label':'airplane'}]}))
    db.commit()

ca,cb,cm=client_for(a),client_for(b),client_for(admin)
query=dict(data_type='image',scene='aircraft',project='验收项目',labels=['airplane'],secret_level='',required_count=3)
r=ca.post('/api/download/query',json=query);assert r.status_code==200,r.text
rid=r.json()['download_request_id'];check('query saved, matched source=1, candidate owned',r.json()['matched_count']==1)
rb=cb.post('/api/download/query',json=query);rb.raise_for_status();rid_b=rb.json()['download_request_id']
r=ca.post('/api/aigc/generate',json={'download_request_id':rid,'count':2});check('AIGC prepares context without submitting GPU task',r.status_code==200 and r.json()['status']=='awaiting_confirmation')
with SessionLocal() as db:check('no training task created by AIGC entry',db.query(FewshotJob).count()==0)
check('request and candidate thumbnail inaccessible to another user',cb.get(f'/api/download/requests/{rid}').status_code==403 and cb.get(f'/api/download/requests/{rid}/candidates/{fid}/thumbnail').status_code==403)
check('administrator reads another user request',cm.get(f'/api/download/requests/{rid}').status_code==200)
check('non-admin all-user request list rejected',cb.get('/api/download/requests?all_users=true').status_code==403)
check('request list isolated',all(x['created_by']==b.id for x in cb.get('/api/download/requests').json()['items']))
check('wrong candidate membership rejected',ca.get(f'/api/download/requests/{rid}/candidates/99999/thumbnail').status_code==422)
with patch.object(few,'engine_settings',return_value={}):
    r=ca.post('/api/fewshot/jobs',data={'manifest':json.dumps(body(fid,rid))});assert r.status_code==200,r.text
    job=r.json();jid=job['id']
    bad=body(fid,rid_b)
    check('foreign request cannot be linked on create',ca.post('/api/fewshot/jobs',data={'manifest':json.dumps(bad)}).status_code==403)
    bad=body(99999,rid)
    check('foreign candidate cannot be injected on create',ca.post('/api/fewshot/jobs',data={'manifest':json.dumps(bad)}).status_code==422)
check('candidate source provenance saved',job['download_request_id']==rid and job['groups'][0]['sources'][0]['source']['data_file_id']==fid and job['status']=='draft')
check('private source cannot be downloaded by other user',cb.get(f'/api/image/{job["source_data_id"]}/preview').status_code==403)
check('private task and list isolation',cb.get(f'/api/fewshot/jobs/{jid}').status_code==403 and cb.get('/api/fewshot/jobs').json()['items']==[])
check('administrator task read and all-user list',cm.get(f'/api/fewshot/jobs/{jid}').status_code==200 and cm.get('/api/fewshot/jobs?all_users=true').json()['items'][0]['id']==jid)
check('shared public dataset remains accessible',cb.get(f'/api/image/{public_raw_id}/preview').status_code==200)

# Simulate only the upstream generator's complete batch contract. Import and
# automatic annotation run the real platform code and CPU COCO detector.
with SessionLocal() as db:
    job=db.get(FewshotJob,jid);job.smoke=True
    batch=Path(job.job_dir)/'fixture_batch';(batch/'lora').mkdir(parents=True)
    dest=batch/'lora/generated.png';shutil.copy2(path,dest)
    (batch/'complete.json').write_text(json.dumps({'passed':True,'images':1}))
    (batch/'index.json').write_text(json.dumps([{'kind':'lora','image':'lora/generated.png','image_sha256':few.sha(dest)}]))
    job.batch_dir=str(batch);db.commit()
    few.import_batch(db,job);few.label_batch(db,job)
    job.status='completed';db.commit();did=job.dataset_id;vid=job.version_id;generated_raw=job.result_data_id
    genfid=db.query(DataFile).filter_by(data_type='image',dataset_id=generated_raw).one().id
check('automatic annotation completes on CPU, outputs linked to query',ca.get(f'/api/download/requests/{rid}').json()['jobs'][0]['status']=='completed')
for endpoint in [f'/api/image/{generated_raw}/preview',f'/api/image/{generated_raw}/viz',f'/api/datasets/{did}',f'/api/datasets/{did}/versions',f'/api/datasets/{did}/versions/{vid}/file?path=generated.png',f'/api/datasets/{did}/versions/{vid}/download',f'/api/datasets/{did}/lineage',f'/api/quality/image/{did}?version_id={vid}',f'/api/quality/image/{did}/export?version_id={vid}',f'/api/fewshot/jobs/{jid}/images/generated.png']:
    check('cross-user blocked '+endpoint,cb.get(endpoint).status_code==403)
check('pack cannot bypass result isolation',cb.post('/api/download/pack',json={'file_ids':[genfid]}).status_code==403)
check('quality write cannot bypass isolation',cb.post('/api/quality/evaluate',json={'data_type':'image','data_id':did,'version_id':vid}).status_code==403)
check('preprocessing cannot bypass isolation',cb.post('/api/clean',json={'name':'forbidden','data_type':'image','dataset_id':did}).status_code==403)
check('supply service cannot bypass isolation',cb.post('/api/service/supply',json={'scene':'analyze','dataset_id':did,'version_id':vid}).status_code==403)
check('private data excluded from raw list',generated_raw not in [x['id'] for x in cb.get('/api/image').json()['items']])
check('private data excluded from dataset list',did not in [x['id'] for x in cb.get('/api/datasets').json()['items']])
check('owner can download auto-labelled version immediately',ca.get(f'/api/datasets/{did}/versions/{vid}/download').status_code==200)
check('private file responses cannot survive account-switch cache',ca.get(f'/api/fewshot/jobs/{jid}/images/generated.png').headers.get('cache-control')=='private, no-store')
check('manual label cannot escape its selected version',ca.post('/api/manual-label',json={'dataset_id':did,'input_version_id':vid,'labels':[{'file':'../outside.png','label':{}}]}).status_code==400)
manual=ca.post('/api/manual-label',json={'dataset_id':did,'input_version_id':vid,'labels':[{'file':'generated.png','label':{'annotations':[{'type':'bbox','xyxy':[5,5,70,45],'label':'airplane'}]}}]})
assert manual.status_code==200,manual.text
latest=manual.json()['output_version_id']
detail=ca.get(f'/api/download/requests/{rid}').json()
check('manual result is an independently downloadable selected version',detail['jobs'][0]['current_version_id']==latest and len(detail['jobs'][0]['versions'])==3 and ca.get(f'/api/datasets/{did}/versions/{latest}/download').status_code==200)
with SessionLocal() as db:
    # Private status must propagate before an integration thread starts copying.
    target=Dataset(name='私有来源集成',data_type='mixed',secret_level='公开',created_by=a.id)
    public=Dataset(name='公开集成主输入',data_type='image',secret_level='公开',created_by=admin.id)
    db.add_all([target,public]);db.flush();pv=vs.create_initial_version(db,public,admin.id)
    task=IntegrateTask(name='pending private integration',source_ids=json.dumps([target.id]),source_types='["image"]',input_version_id=pv.id,detail=json.dumps({'other_input_version_ids':[latest]}),created_by=a.id)
    db.add(task);db.commit();targetid=target.id
    from app.security import CurrentUser
    check('pending integration inherits private visibility before file copy',targetid in excluded_dataset_ids(db,CurrentUser(b)))
check('administrator can download generated private version',cm.get(f'/api/datasets/{did}/versions/{latest}/download').status_code==200)
check('generation count above query shortage rejected',ca.post('/api/aigc/generate',json={'download_request_id':rid,'count':3}).status_code==422)
with SessionLocal() as db:
    target=db.get(Dataset,targetid);input_v=db.get(DatasetVersion,latest)
    unfinished=TransformTask(name='unfinished fixture',data_type='mixed',source_id=target.id,input_version_id=latest,transform_type='format',status='running',created_by=a.id)
    db.add(unfinished);db.commit()
    partial=vs.create_output_version(db,target,input_version=input_v,task_type='transform',task_id=unfinished.id,user_id=a.id)
    (Path(partial.file_path)/'partial.txt').write_text('incomplete output')
    partialid=partial.id
check('incomplete version ZIP download rejected',ca.get(f'/api/datasets/{targetid}/versions/{partialid}/download').status_code==409)
check('incomplete version cannot be used as a preprocessing input',ca.post('/api/clean',json={'name':'unfinished input','data_type':'mixed','dataset_id':targetid,'input_version_id':partialid}).status_code==409)
text_request=ca.post('/api/download/query',json={'data_type':'text','required_count':3}).json()['download_request_id']
check('unsupported supplementation modality rejected',ca.post('/api/aigc/generate',json={'download_request_id':text_request,'count':1}).status_code==400)
with SessionLocal() as db:
    original=db.get(ImageData,public_raw_id);original.source_type='AIGC';db.commit()
check('shared imported AIGC material stays public',cb.get(f'/api/image/{public_raw_id}/preview').status_code==200)
result=dict(passed=True,checks=checks,users={'a':a.id,'b':b.id,'admin':admin.id},request_id=rid,job_id=jid,dataset_id=did,auto_version_id=vid,current_version_id=latest,public_file_id=fid,config=str(cfg),sqlite=str(OUT/'fixtures.sqlite'),new_gpu_training_jobs=0,simulated_generation_batch=True)
(OUT/'api_acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'passed':True,'checks':len(checks),'new_gpu_training_jobs':0}),flush=True)
