"""Isolated API/config acceptance for all upstream methods; never launches training."""
import io,json,os,sys,uuid
from pathlib import Path
from unittest.mock import patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
assert os.environ.get('DP_CONFIG_PATH')
from app.config import settings
assert settings.db['dbname'].endswith('_test')
from app.main import create_app
from app.database import SessionLocal
from app.models import FewshotJob
from app.services import fewshot_service as svc
from fastapi.testclient import TestClient
client=TestClient(create_app())
assert client.post('/api/auth/login',json={'username':'sysadmin','password':'admin'}).status_code==200
engines=client.get('/api/fewshot/capabilities').json()['engines']
assert len(engines)==3 and all(e['ready'] for e in engines),engines
files=[]
for i in range(9):
 b=io.BytesIO();Image.new('RGB',(64,64),(i*20,50,99)).save(b,format='PNG')
 files.append((f'files_{0 if i<4 else 1}',(f'{i}.png',b.getvalue(),'image/png')))
records=[]
root=Path(settings.data['fewshot']['output_root']).parent/'engine_checks'
root.mkdir(exist_ok=True)
for engine in engines:
 body={'request_key':str(uuid.uuid4()),'name':'方法配置验收 '+engine['name'],
       'groups':[{'name':'A','caption':'a photograph of an aircraft','count':1},{'name':'B','caption':'a photograph of a car','count':1}],
       'options':{'method':engine['method'],'model':engine['model'],'steps':400}}
 with patch.object(svc,'ensure_worker',side_effect=AssertionError('Do not submit training')):
  response=client.post('/api/fewshot/jobs',data={'manifest':json.dumps(body)},files=files)
 assert response.status_code==200,response.text
 job_id=response.json()['id']
 with SessionLocal() as db,patch.object(svc,'gpu_memory',return_value=(10000,39140,49140)):
  job=db.get(FewshotJob,job_id)
  job.smoke=True;db.commit()
  config=svc.build_config(job,root/engine['id'])
  assert config['model']==engine['id']
  if engine['id']=='chimera_sd21':
   assert config['train']['gradient_accumulation']==1
   assert config['train']['checkpoint_every']>config['train']['steps']
  if engine['id']=='beyondobjects_sd21':assert config['train']['steps']==402
  assert config['runtime']['gpu']==settings.data['fewshot']['gpu']
  assert not config['tracking']['enabled']
  p=root/(engine['id']+'.json');svc.dump(p,config)
  records.append({'job':job_id,'engine':engine['id'],'config':str(p),'python':svc.job_engine(job)['python']})
body['request_key']=str(uuid.uuid4());body['options']={'method':'chimeralora','model':'flux2_klein_base_4b'}
assert client.post('/api/fewshot/jobs',data={'manifest':json.dumps(body)},files=files).status_code==422
assert svc.engine_id({'steps':400})=='flux2_klein_base_4b'
assert svc.training_steps({'steps':400,'method':'beyondobjects','model':'sd21'},[{'images':9}])==402
# Previously hidden OOM from the actual outer CalledProcessError pattern.
from types import SimpleNamespace
folder=root/'failure_fixture';folder.mkdir(exist_ok=True)
(folder/'worker.log').write_text('STAGE train\ntorch.OutOfMemoryError: CUDA out of memory\nCalledProcessError: exit 1\n')
fixture=SimpleNamespace(job_dir=str(folder),run_dir='',error='CalledProcessError(1)')
assert '显存不足' in svc.failure_message(fixture)
p=root/'configs.json';svc.dump(p,records)
print(json.dumps({'passed':True,'checks':['three ready engines','API method/model selection','unsupported pair rejection','legacy default','SD whole epochs','Chimera multi-head configuration','shared GPU pinning','OOM diagnosis'],'configs':records},ensure_ascii=False))
