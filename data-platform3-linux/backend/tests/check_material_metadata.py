"""Read-only API acceptance against the isolated DB and actual local metadata index."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
assert os.environ.get('DP_CONFIG_PATH')
from app.config import settings
assert settings.db['dbname'].endswith('_test')
from app.main import create_app
from fastapi.testclient import TestClient

client = TestClient(create_app())
assert client.get('/api/fewshot/materials/aircraft').status_code == 401
assert client.post('/api/auth/login', json={'username': 'sysadmin', 'password': 'admin'}).status_code == 200
aircraft = client.get('/api/fewshot/materials/aircraft').json()
assert aircraft['total'] == 10000 and len(aircraft['categories']) == 100
category = aircraft['categories'][0]
filtered = client.get('/api/fewshot/materials/aircraft', params={'category': category['name']}).json()
assert filtered['total'] == category['count']
assert all(category['name'] in item['categories'] for item in filtered['items'])
assert client.get('/api/fewshot/materials/aircraft', params={'category': "invalid' OR 1=1--"}).json()['total'] == 0
assert client.get('/api/fewshot/materials/aircraft', params={'category': '__uncategorized__'}).json()['total'] == 0
rare = client.get('/api/fewshot/materials/rareplanes').json()
assert rare['total'] == 2853 and len(rare['categories']) == 3
assert all(item['categories'] for item in rare['items'])
assert sum(cat['count'] for cat in rare['categories']) > rare['total'], 'Expected multi-category images'
laion = client.get('/api/fewshot/materials/laionsg').json()
assert any(item['categories'] and item['caption'] for item in laion['items'])
laion_category = next(cat for cat in laion['categories'] if cat.get('value') != '__uncategorized__')
laion_filtered = client.get('/api/fewshot/materials/laionsg', params={'category': laion_category['name']}).json()
assert all(laion_category['name'] in item['categories'] for item in laion_filtered['items'])
hpd = client.get('/api/fewshot/materials/hpdv3').json()
assert hpd['total'] == 11781 and all(item['caption'] for item in hpd['items'])
assert not any(item['categories'] for item in hpd['items']), 'Do not fabricate HPD categories'
phrase = hpd['items'][0]['caption'][:35]
searched = client.get('/api/fewshot/materials/hpdv3', params={'q': phrase}).json()
assert searched['total'] and all(phrase.casefold() in (item['name']+' '+item['caption']).casefold()
                                for item in searched['items'])
name = aircraft['items'][0]['name']
assert client.get('/api/fewshot/materials/aircraft', params={'q': name}).json()['total'] == 1
print(json.dumps({'passed': True, 'categories': {'aircraft': 100, 'rareplanes': 3},
                  'checks': ['authentication', 'real category counts', 'multi-label membership', 'caption search',
                             'literal category filter', 'uncategorized images', 'no fabricated labels', 'filename search']}, ensure_ascii=False))
