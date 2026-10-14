"""Integration acceptance against the dedicated MySQL test database; never uses production files."""
import io
import json
import os
import sys
import uuid
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
if not os.environ.get('DP_CONFIG_PATH'):
    raise SystemExit('Set DP_CONFIG_PATH to the isolated test config')
from app.config import settings
assert settings.db['dbname'].endswith('_test'), 'Refusing to run against production DB'
from app.database import init_schema, seed_data, SessionLocal
from app.main import create_app
from app.models import FewshotJob, Dataset, DatasetVersion, ImageData, DataFile
from app.services import fewshot_service as svc
from app.services.label_locator import find_label
from app.security import CurrentUser
from app.models import User
from fastapi.testclient import TestClient
from sqlalchemy import select, func
init_schema(); seed_data()
client = TestClient(create_app())
assert client.get('/api/fewshot/jobs').status_code == 401
assert client.post('/api/auth/login', json={'username':'sysadmin','password':'admin'}).status_code == 200
assert client.get('/api/fewshot/capabilities').json()['ready']

def png(color):
    b = io.BytesIO(); Image.new('RGB',(64,64),color).save(b,format='PNG'); return b.getvalue()

body = {'request_key':str(uuid.uuid4()),'name':'API integration fixture','secret_level':'内部',
        'groups':[{'name':'A','caption':'a red plane','count':1},{'name':'B','caption':'a blue plane','count':1}],
        'options':{'steps':2,'sampling_steps':2}}
files = [('files_0',('a.png',png('red'),'image/png')),('files_1',('b.png',png('blue'),'image/png'))]
response = client.post('/api/fewshot/jobs', data={'manifest':json.dumps(body)}, files=files)
assert response.status_code == 200, response.text
job_id = response.json()['id']
assert client.post('/api/fewshot/jobs',data={'manifest':json.dumps(body)},files=files).json()['id']==job_id
assert len(response.json()['groups']) == 2
# Actual permission boundary: guest cannot see a restricted job or create tasks.
guest = TestClient(create_app())
assert guest.post('/api/auth/login',json={'username':'guest','password':'admin'}).status_code == 200
assert guest.get(f'/api/fewshot/jobs/{job_id}').status_code == 403
assert guest.post('/api/fewshot/jobs',data={'manifest':json.dumps(body)},files=files).status_code == 403
# Reject malformed and duplicate images, rolling back the new job.
bad = dict(body,request_key=str(uuid.uuid4()))
assert client.post('/api/fewshot/jobs',data={'manifest':json.dumps(bad)},files=[files[0],('files_1',('x.png',b'bad','image/png'))]).status_code == 422
bad['request_key']=str(uuid.uuid4())
assert client.post('/api/fewshot/jobs',data={'manifest':json.dumps(bad)},files=[files[0],('files_1',files[0][1])]).status_code == 422
with SessionLocal() as db:
    job = db.get(FewshotJob,job_id)
    config = svc.build_config(job,Path(job.job_dir)/'test_run')
    assert not config['generate']['compare_base'] and config['tracking']['enabled']
    assert config['runtime']['gpu'] != settings.data['fewshot']['forbidden_gpu']
    plan = svc.read_json(config['generate']['plan'])
    assert len(plan)==2 and plan[0]['prompt']=='a red plane' and plan[1]['prompt']=='a blue plane'
    batch = Path(job.job_dir)/'fixture_batch';(batch/'lora').mkdir(parents=True)
    records=[]
    for i,color in enumerate(('red','blue')):
        path=batch/'lora'/f'{i:06d}.png';path.write_bytes(png(color))
        records.append({'image':f'lora/{path.name}','kind':'lora','image_sha256':svc.sha(path),'prompt':plan[i]['prompt'],'seed':i})
    svc.dump(batch/'index.json',records);svc.dump(batch/'complete.json',{'passed':True,'images':2})
    job.batch_dir=str(batch);db.commit()
    # Hash mismatch must not create a dataset.
    original=(batch/'lora/000000.png').read_bytes();(batch/'lora/000000.png').write_bytes(b'corrupt')
    try: svc.import_batch(db,job)
    except ValueError: pass
    else: raise AssertionError('Accepted corrupt generation')
    (batch/'lora/000000.png').write_bytes(original)
    svc.import_batch(db,job)
    ids=(job.result_data_id,job.dataset_id,job.version_id)
    svc.import_batch(db,job)
    assert ids==(job.result_data_id,job.dataset_id,job.version_id)
    raw=db.get(ImageData,job.result_data_id)
    assert raw.source_type=='AIGC' and raw.file_count==2
    assert all(p.suffix=='.png' for p in Path(raw.file_path).iterdir())
    assert db.scalar(select(func.count()).select_from(DataFile).where(DataFile.dataset_id==raw.id))==2
    # Real CPU YOLO call on generated fixtures, no GPU allocation.
    svc.label_batch(db,job)
    assert job.status=='completed' and job.label_errors==0
    ds=db.get(Dataset,job.dataset_id);v=db.get(DatasetVersion,job.version_id)
    assert v.version_no==2 and len(list(Path(v.file_path).glob('*_label.json')))==2
    label=find_label(db,'image',raw.id,Path(raw.file_path)/'000000.png','000000.png')
    assert label is not None and label['algorithm']=='coco_detect'
    dataset_id,version_id=job.dataset_id,job.version_id
# Manual correction generates a new version and is discoverable from raw data.
response=client.post('/api/manual-label',json={'dataset_id':dataset_id,'input_version_id':version_id,
 'labels':[{'file':'000000.png','label':{'type':'image','boxes':[{'x':1,'y':2,'w':20,'h':15,'label':'airplane'}]}}]})
assert response.status_code==200,response.text
result=client.post('/api/download/query',json={'data_type':'image','labels':['airplane'],'required_count':1})
assert result.status_code==200 and any(x['dataset_id']==ids[0] for x in result.json()['files']),result.text
print(json.dumps({'passed':True,'job_id':job_id,'checks':['auth','permissions','multipart groups','idempotent upload','corrupt/duplicate rejection','generation plan','integrity before import','idempotent import','CPU YOLO','manual version','download label lookup']},ensure_ascii=False))
