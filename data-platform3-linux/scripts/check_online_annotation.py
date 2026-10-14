"""CPU-only annotation-entry acceptance using a task-specific SQLite database."""
import json
import os
import secrets
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ['DP_ACCEPTANCE_DIR']).resolve()
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT/'fixtures.sqlite').exists(), 'Use a fresh task-specific directory; do not erase fixtures still used by browsers'
config = json.loads((ROOT / 'config.json').read_text())
config.update(upload_dir=str(OUT/'uploads'), log_dir=str(OUT/'logs'), jwt_secret=secrets.token_urlsafe(48))
config['database'] = {'host':'127.0.0.1','port':3307,'username':'acceptance','password':'unused','dbname':'annotation_acceptance'}
config['fewshot']['output_root'] = str(OUT/'jobs')
cfg = OUT/'config.json'; cfg.write_text(json.dumps(config)); cfg.chmod(0o600)
os.environ.update(DP_CONFIG_PATH=str(cfg), DATABASE_URL='sqlite:///'+str(OUT/'fixtures.sqlite'), CUDA_VISIBLE_DEVICES='')
sys.path.insert(0, str(ROOT/'backend'))
from fastapi.testclient import TestClient
from app.main import create_app
from app.database import SessionLocal, engine, init_schema, seed_data
from app.models import Base, User, TextData, ImageData, DataFile, Dataset, DatasetVersion, TransformTask
from app.security import create_token, hash_password
from app.services import file_service as fs, version_service as vs

assert engine.url.get_backend_name() == 'sqlite' and str(OUT) in str(engine.url)
init_schema(); seed_data()
checks=[]
def check(name, condition):
    assert condition, name
    checks.append(name); print('PASS '+name, flush=True)
def client(user):
    c=TestClient(create_app(), raise_server_exceptions=False); c.cookies.set('dp_access_token', create_token(user)); return c

def source(db, name, content='北京机场的飞机正在起飞。', missing=False, empty=False):
    raw=TextData(name=name, format='TXT', secret_level='公开', created_by=admin.id, file_count=0, size_bytes=0)
    db.add(raw); db.flush()
    p=fs.dataset_dir('text',raw.id)/'example.txt'
    if not empty:
        if not missing:p.write_text(content)
        db.add(DataFile(dataset_id=raw.id,data_type='text',filename=p.name,stored_name=p.name,size_bytes=len(content.encode()),stage='raw',status='complete'))
        raw.file_count=1; raw.size_bytes=len(content.encode())
    db.commit();return raw.id,p

with SessionLocal() as db:
    admin=db.query(User).filter_by(username='sysadmin').one(); admin.must_change_password=False
    other=User(username='annotation_admin_b',password_hash=hash_password('test-only'),display_name='标注测试B',role='admin',status='active',must_change_password=False)
    visitor=User(username='annotation_visitor',password_hash=hash_password('test-only'),display_name='标注测试访客',role='visitor',status='active',view_level=5,operate_level=5,must_change_password=False)
    db.add_all([other,visitor]); db.commit()
    for i in range(101):db.add(Dataset(name=f'无关工作集 {i}',data_type='text',created_by=admin.id,secret_level='公开',created_at=datetime.now()+timedelta(days=1)))
    db.commit()
    rid,path=source(db,'在线标注接口验收')
    path.with_name('example_label.json').write_text(json.dumps({'annotations':[{'type':'span','start':0,'end':2,'text':'北京','label':'地名'}]}))
    browser_rid,browser_path=source(db,'在线标注浏览器验收')
    empty,_=source(db,'空数据验收',empty=True)
    missing,_=source(db,'缺失文件验收',missing=True)
    broken,_=source(db,'复制回滚验收')
    unsafe,_=source(db,'非法路径验收')
    file=db.query(DataFile).filter_by(dataset_id=unsafe,data_type='text').one();file.stored_name='../escape.txt';db.commit()
    baseline_count=db.query(Dataset).count()
    token=create_token(admin)
    fixture={'token':token,'raw_id':browser_rid,'raw_name':'在线标注浏览器验收','filename':browser_path.name}
    f=OUT/'browser_fixture.json';f.write_text(json.dumps(fixture));f.chmod(0o600)
    ca,cb,cv=client(admin),client(other),client(visitor)

url=f'/api/text/{rid}/annotation-context'
r=ca.get(url);check('只读预检返回候选数量而不创建副本',r.status_code==200 and r.json()['dataset_id'] is None and r.json()['file_count']==1)
with SessionLocal() as db:check('预检不更改数据集记录',db.query(Dataset).count()==baseline_count)
r=ca.post(url);check('确认后建立工作副本',r.status_code==200)
context=r.json();did,vid=context['dataset_id'],context['version_id']
check('原始ID与标注工作集ID正确区分',did!=rid)
check('目标位于前100数据集之外',did not in [d['id'] for d in ca.get('/api/datasets?page_size=100').json()['items']])
with SessionLocal() as db:
    v=db.get(DatasetVersion,vid);ds=db.get(Dataset,did)
    check('复制文件并继承标注及原密级',Path(v.file_path,'example.txt').read_text()==path.read_text() and Path(v.file_path,'example_label.json').is_file() and ds.secret_level=='公开' and ds.label_count==1)
    count=db.query(Dataset).count()
check('同一账号重复确认复用工作集',ca.post(url).json()==context and ca.get(url).json()==context)
with SessionLocal() as db:check('重复点击没有新建数据集',db.query(Dataset).count()==count)
manual=ca.post('/api/manual-label',json={'dataset_id':did,'input_version_id':vid,'label_type':'实体识别','labels':[{'file':'example.txt','label':[{'type':'span','start':0,'end':2,'text':'北京','label':'人工地名'}]}]})
check('手动标注实际生成后续版本',manual.status_code==200 and manual.json()['output_version_id']!=vid)
latest=manual.json()['output_version_id'];check('再次进入使用最新完成版本',ca.get(url).json()['version_id']==latest)
rb=cb.post(url);check('另一个管理员建立自己的工作集',rb.status_code==200 and rb.json()['dataset_id']!=did)
check('访客不能绕过现有数据集创建权限',cv.get(url).status_code==403 and cv.post(url).status_code==403)
check('不存在的原始数据返回404',ca.get('/api/text/999999/annotation-context').status_code==404 and ca.post('/api/text/999999/annotation-context').status_code==404)
check('空数据和缺失文件在创建前拒绝',ca.get(f'/api/text/{empty}/annotation-context').status_code==400 and ca.post(f'/api/text/{missing}/annotation-context').status_code==409)
check('无效相对路径拒绝',ca.post(f'/api/text/{unsafe}/annotation-context').status_code==409)
with SessionLocal() as db:before=db.query(Dataset).count()
before_dirs=set((OUT/'uploads'/'processed').glob('*/V1'))
with patch('app.routers.data.shutil.copy2', side_effect=OSError('simulated copy failure')):
    r=ca.post(f'/api/text/{broken}/annotation-context')
check('复制异常明确失败',r.status_code==500)
with SessionLocal() as db:
    check('失败回滚记录并移除本次副本',db.query(Dataset).count()==before and set((OUT/'uploads'/'processed').glob('*/V1'))==before_dirs)
    ds=db.get(Dataset,did)
    task=TransformTask(name='未完成输入',data_type='text',source_id=did,input_version_id=latest,transform_type='format',status='running',created_by=admin.id)
    db.add(task);db.commit()
    pending=vs.create_output_version(db,ds,input_version=db.get(DatasetVersion,latest),task_type='transform',task_id=task.id,user_id=admin.id)
    count=db.query(Dataset).count()
check('处理中仍进入上一完成版本并不另建副本',ca.get(url).json()['version_id']==latest and ca.post(url).json()['version_id']==latest)
with SessionLocal() as db:
    check('未完成版本不产生新工作集',db.query(Dataset).count()==count)
    db.get(Dataset,did).current_version_id=latest;db.commit()
check('原始文件及标注始终保持原样',path.read_text()=='北京机场的飞机正在起飞。' and json.loads(path.with_name('example_label.json').read_text())['annotations'][0]['label']=='地名')
check('浏览器源预检尚未创建副本',ca.get(f'/api/text/{browser_rid}/annotation-context').json()['dataset_id'] is None)
(OUT/'api_acceptance.json').write_text(json.dumps({'passed':True,'checks':checks,'count':len(checks)},ensure_ascii=False,indent=2))
print(f'{len(checks)} checks passed; fixtures retained for browser acceptance')
