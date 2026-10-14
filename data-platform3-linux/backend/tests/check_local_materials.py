"""Local material acceptance; isolated MySQL test DB only, no training."""
import io
import json
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
assert os.environ.get('DP_CONFIG_PATH'), 'Use isolated test configuration'
from app.config import settings
assert settings.db['dbname'].endswith('_test')
from app.main import create_app
from app.database import SessionLocal
from app.models import FewshotJob
from app.services import local_materials as materials
from app.services import fewshot_service as svc
from fastapi.testclient import TestClient

client = TestClient(create_app())
assert client.get('/api/fewshot/materials').status_code == 401
assert client.post('/api/auth/login', json={'username': 'sysadmin', 'password': 'admin'}).status_code == 200
catalog = client.get('/api/fewshot/materials').json()['items']
first_dataset = catalog[0]['id']
refs = []
for ds in catalog:
    result = client.get('/api/fewshot/materials/' + ds['id']).json()
    assert len(result['items']) == 24 and result['total'] >= 500
    item = result['items'][0]
    refs.append({'dataset': ds['id'], 'name': item['name']})
    thumb = client.get(f"/api/fewshot/materials/{ds['id']}/thumbnail", params={'name': item['name']})
    assert thumb.status_code == 200
    with Image.open(io.BytesIO(thumb.content)) as im:
        assert max(im.size) <= 256
    search = client.get('/api/fewshot/materials/' + ds['id'], params={'q': item['name']}).json()
    assert search['total'] == 1
assert client.get('/api/fewshot/materials/unknown').status_code == 404
assert client.get(f'/api/fewshot/materials/{first_dataset}', params={'page': 0}).status_code == 422
for name in ['../config.json', '/etc/passwd', '..\\private.jpg']:
    assert client.get(f'/api/fewshot/materials/{first_dataset}/thumbnail', params={'name': name}).status_code == 422
guest = TestClient(create_app())
assert guest.post('/api/auth/login', json={'username': 'guest', 'password': 'admin'}).status_code == 200
assert guest.get('/api/fewshot/materials').status_code == 403

body = {'request_key': str(uuid.uuid4()), 'name': '本地素材接入验收',
        'groups': [{'name': str(i), 'caption': 'a photograph', 'count': 1, 'local_images': [ref]}
                   for i, ref in enumerate(refs)]}
img = io.BytesIO()
Image.new('RGB', (48, 48), (31, 78, 142)).save(img, format='PNG')
files = [('files_0', ('uploaded.png', img.getvalue(), 'image/png'))]
with patch.object(svc, 'ensure_worker', side_effect=AssertionError('Must not train')):
    response = client.post('/api/fewshot/jobs', data={'manifest': json.dumps(body)}, files=files)
    assert response.status_code == 200, response.text
    jid = response.json()['id']
    assert response.json()['status'] == 'draft'
    assert client.post('/api/fewshot/jobs', data={'manifest': json.dumps(body)}, files=files).json()['id'] == jid
    with SessionLocal() as db:
        job = db.get(FewshotJob, jid)
        groups = json.loads(job.groups)
        assert [g['images'] for g in groups] == [2, 1, 1]
        for group in groups:
            for source in group['sources']:
                copied = Path(job.job_dir) / 'support' / group['folder'] / source['file']
                assert svc.sha(copied) == source['sha256']
                if 'dataset' in source['source']:
                    original = materials.resolve_image(**source['source'])
                    assert svc.sha(original) == source['sha256'] and not copied.is_symlink()
    # Local-only forms work without a file upload, duplicates still roll back.
    body['request_key'] = str(uuid.uuid4())
    result = client.post('/api/fewshot/jobs', data={'manifest': json.dumps(body)})
    assert result.status_code == 200, result.text
    body['request_key'] = str(uuid.uuid4())
    body['groups'][1]['local_images'] = [refs[0]]
    assert client.post('/api/fewshot/jobs', data={'manifest': json.dumps(body)}).status_code == 422
    body['request_key'] = str(uuid.uuid4())
    body['groups'][0]['local_images'] = [{'dataset': first_dataset, 'name': '../hidden.jpg'}]
    assert client.post('/api/fewshot/jobs', data={'manifest': json.dumps(body)}).status_code == 422
print(json.dumps({'passed': True, 'mixed_job': jid, 'local_only_job': result.json()['id'],
                  'checks': ['auth', 'catalog', 'pagination', 'search', 'thumbnails', 'path rejection',
                             'mixed/local-only creation', 'provenance hashes', 'duplicate rejection', 'no training']}))
