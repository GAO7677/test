"""Live API acceptance of the imported subsets; CPU only, no LoRA submission.

Writes incremental evidence and leaves real task outputs visible in the platform.
Only runs once per report directory to prevent accidental duplicate submissions.
"""
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
import httpx
from app.database import SessionLocal
from app.models import User,DatasetVersion
from app.security import create_token
from app.services import version_service as vs

OUT=Path('/mnt/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_platform_dataset_demo')
REPORT=OUT/'api_acceptance.json'
records=[]


def record(check, **details):
    item=dict(check=check,**details);records.append(item)
    REPORT.write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print(json.dumps(item,ensure_ascii=False),flush=True)


def call(method,path,**kw):
    response=client.request(method,path,**kw)
    response.raise_for_status()
    return response


def task(path,body):
    item=call('POST',path,json=body).json()
    if path=='/workflows':call('POST',f'{path}/{item["id"]}/run')
    deadline=time.monotonic()+240
    while time.monotonic()<deadline:
        rows=call('GET',path,params={'page_size':100}).json()['items']
        state=next(x for x in rows if x['id']==item['id'])
        if state['status'] in ('completed','failed'):
            record(path,task_id=state['id'],status=state['status'],output_version_id=state.get('output_version_id'),error=state.get('error_msg'))
            assert state['status']=='completed',state
            return state
        time.sleep(1)
    raise TimeoutError(path)


def files(dsid,vid):
    return call('GET',f'/datasets/{dsid}/versions/{vid}/files',params={'page_size':500}).json()['items']


def seed(name,dtype,source, count, fixture=False):
    ds=call('POST','/datasets',json={'name':'验收 · '+name,'data_type':dtype,'secret_level':'公开','description':'现有样本的独立功能验收副本；不启动 LoRA。','source_data_ids':[source['raw_id']]}).json()
    vid=ds['current_version_id']
    with SessionLocal() as db:
        v=db.get(DatasetVersion,vid);root=Path(v.file_path)
        # API already copied raw examples. Keep only a small acceptance subset.
        chosen=sorted(p for p in root.rglob('*') if p.is_file())[:count]
        for p in sorted(root.rglob('*')):
            if p.is_file() and p not in chosen:p.unlink()
        if fixture:
            shutil.copy2(chosen[0],root/'duplicate.txt')
            text=chosen[0].read_text();(root/'lines.txt').write_text(text+'\n\n'+text)
        vs.refresh_version_stats(db,v)
    return ds['id'],vid


resume = '--after-image' in sys.argv
if REPORT.exists() and not resume:raise SystemExit('Acceptance report already exists; inspect it before a new run.')
if resume:records.extend(json.loads(REPORT.read_text()))
with SessionLocal() as db:
    token=create_token(db.query(User).filter_by(username='sysadmin').first())
client=httpx.Client(base_url='http://127.0.0.1:8000/api',cookies={'dp_access_token':token},timeout=120)
samples=json.loads((OUT/'import_report.json').read_text())
bykey={x['key']:x for x in samples}
if not resume:
    for sample in samples:
        dtype,rid,did,vid=sample['data_type'],sample['raw_id'],sample['dataset_id'],sample['version_id']
        preview=call('GET',f'/{dtype}/{rid}/preview').json()
        viz=call('GET',f'/{dtype}/{rid}/viz').json()
        ds=call('GET',f'/datasets/{did}').json()
        listed=files(did,vid)
        assert len(listed)==sample['data_count']==ds['data_count'],(sample,ds['data_count'])
        assert preview['data'] and viz['data']
        quality=call('POST','/quality/evaluate',json={'data_type':dtype,'data_id':rid}).json()
        version_quality=call('POST','/quality/evaluate',json={'data_type':dtype,'data_id':did,'version_id':vid}).json()
        record('import_preview_visualization_quality',key=sample['key'],raw_id=rid,dataset_id=did,files=len(listed),raw_quality=quality['overall_score'],version_quality=version_quality['overall_score'])

    aircraft=bykey['aircraft']
    stats=call('GET',f'/image/{aircraft["raw_id"]}/label-stats').json()
    assert len(stats['labels'])==100,stats
    query=call('POST','/download/query',json={'data_type':'image','scene':'aircraft','labels':['707-320']}).json()
    assert query['matched_count']==2 and all(f['file_id'] for f in query['files']),query
    unfiltered=call('POST','/download/query',json={'data_type':'image','scene':'aircraft'}).json()
    assert unfiltered['matched_count']==200 and all(f['file_id'] for f in unfiltered['files'])
    packed=call('POST','/download/pack',json={'file_ids':[x['file_id'] for x in query['files']]})
    with zipfile.ZipFile(io.BytesIO(packed.content)) as z:
        assert '_lineage.json' in z.namelist()
        assert len([n for n in z.namelist() if n.endswith('_label.json')])==2
    (OUT/'aircraft_label_download.zip').write_bytes(packed.content)
    record('100_classes_label_filter_download',matches=query['matched_count'],zip_bytes=len(packed.content))

    image_id,image_vid=seed('飞机 COCO 自动标注与修正','image',aircraft,4)
    auto=task('/auto-label',dict(name='验收 · COCO 80 类 CPU 检测',data_type='image',dataset_id=image_id,input_version_id=image_vid,algorithm='coco_detect',rules_config={'confThreshold':0.25}))
    with SessionLocal() as db:
        root=Path(db.get(DatasetVersion,auto['output_version_id']).file_path)
        labels=[json.loads(p.read_text()) for p in root.rglob('*_label.json')]
        assert labels and not any(p.get('error') for p in labels),labels
        total_boxes=sum(len(p.get('boxes',[])) for p in labels)
        assert total_boxes>0,labels
        path=next(p for p in root.rglob('*_label.json') if json.loads(p.read_text()).get('boxes'))
        label=json.loads(path.read_text());box=label['boxes'][0]
        filename=next(p.name for p in root.iterdir() if p.stem==path.stem[:-len('_label')] and not p.name.endswith('_label.json'))
        corrected=dict(annotations=[dict(type='bbox',xyxy=[box['x'],box['y'],box['x']+box['w'],box['y']+box['h']],label=box['label'])],source='manual_review',review_note='对已检测框进行人工确认；验收写入版本')
    manual=call('POST','/manual-label',json={'dataset_id':image_id,'input_version_id':auto['output_version_id'],'labels':[{'file':filename,'label':corrected}]}).json()
    assert manual['status']=='completed'
    record('coco_detection_manual_version',dataset_id=image_id,box_count=total_boxes,manual_task_id=manual['id'],manual_version_id=manual['output_version_id'])
    formatted=task('/transform',dict(name='验收 · JPG 转 PNG',data_type='image',dataset_id=image_id,input_version_id=manual['output_version_id'],transform_type='format',target_format='png'))
    assert all(x['name'].endswith('.png') for x in files(image_id,formatted['output_version_id']))
    sampled=task('/sample',dict(name='验收 · 图像划分及标注随行',data_type='image',dataset_id=image_id,input_version_id=formatted['output_version_id'],train_ratio=50,val_ratio=25,test_ratio=25))
    listed=files(image_id,sampled['output_version_id']);assert len(listed)==4 and all(x['has_label'] for x in listed)
    quality=call('POST','/quality/evaluate',json={'data_type':'image','data_id':image_id,'version_id':sampled['output_version_id']}).json()
    call('POST',f'/datasets/{image_id}/versions/{sampled["output_version_id"]}/publish',json={'description':'CPU 自动标注、修正、转换、采样验收产物'})
    lineage=call('GET',f'/datasets/{image_id}/lineage').json()
    assert len(lineage['items'])==5
    record('publish_lineage_quality',dataset_id=image_id,versions=len(lineage['items']),quality=quality['overall_score'])

else:
    aircraft=bykey['aircraft']
    auto=next(r for r in records if r['check']=='/auto-label')
    image_id=next(r for r in records if r['check']=='coco_detection_manual_version')['dataset_id']
    lineage=call('GET',f'/datasets/{image_id}/lineage').json()
    assert len(lineage['items'])==5
    record('publish_lineage_quality',dataset_id=image_id,versions=5)

clean_id,clean_vid=seed('caption 去重与空行清洗','text',bykey['hpdv3_captions'],2,fixture=True)
cleaned=task('/clean',dict(name='验收 · caption 去重与空行',data_type='text',dataset_id=clean_id,input_version_id=clean_vid,rules={'missing':True,'duplicate':True,'anomaly':True}))
assert len(files(clean_id,cleaned['output_version_id']))==3
assert cleaned['stats_after']['removed_duplicates']>=2 and cleaned['stats_after']['removed_missing']>=1,cleaned
record('clean_content_verified',dataset_id=clean_id,stats=cleaned['stats_after'])

text=bykey['laionsg_captions']
wf=task('/workflows',dict(name='验收 · 文本标注→清洗→采样',data_type='text',dataset_id=text['dataset_id'],input_version_id=text['version_id'],nodes=[{'type':'auto_label','algorithm':'keyword_extract','params':{'topK':5}}, {'type':'clean','params':{'duplicate':True,'missing':True}}, {'type':'sample','params':{'train_ratio':70,'val_ratio':10,'test_ratio':20}}]))
assert files(text['dataset_id'],wf['output_version_id'])
video=bykey['simulation_two']
task('/auto-label',dict(name='验收 · 仿真视频关键帧检测',data_type='video',dataset_id=video['dataset_id'],input_version_id=video['version_id'],algorithm='keyframe_extract',rules_config={'threshold':8,'minInterval':1}))
feature=task('/transform',dict(name='验收 · 图像梯度直方图特征',data_type='image',dataset_id=bykey['phyco_frames']['dataset_id'],input_version_id=bykey['phyco_frames']['version_id'],transform_type='feature'))
with SessionLocal() as db:
    root=Path(db.get(DatasetVersion,feature['output_version_id']).file_path)
    outputs=list(root.rglob('*_features.json'))
    assert len(outputs)==bykey['phyco_frames']['data_count']
    assert all(len(json.loads(p.read_text())['hog'])==9 for p in outputs)
integrate_id=call('POST','/datasets',json={'name':'验收 · 图文原始数据集成','data_type':'mixed','secret_level':'公开'}).json()['id']
integrated=task('/integrate',dict(name='验收 · 图像与 caption 文件集成',dataset_id=integrate_id,combo_type='raw',strategy='raw',main_input_version_id=bykey['hpdv3']['version_id'],other_input_version_ids=[bykey['hpdv3_captions']['version_id']]))
assert len(files(integrate_id,integrated['output_version_id']))==80
record('integration_files_verified',dataset_id=integrate_id,files=80)

with SessionLocal() as db:
    v=db.get(DatasetVersion,auto['output_version_id']);result=next(Path(v.file_path).glob('*_label.json'))
    payload=result.read_bytes()
writeback=call('POST','/writeback',data={'model_name':'YOLOv8n · CPU 功能验收','result_type':'COCO 目标框','original_data_type':'image','original_data_id':aircraft['raw_id'],'version':'demo-v1'},files={'file':('coco_demo_result.json',payload,'application/json')}).json()
assert call('GET',f'/writeback/{writeback["id"]}/download').content==payload
record('writeback_download',writeback_id=writeback['id'])
record('complete',imported_datasets=len(samples),imported_files=sum(x['data_count'] for x in samples),total_source_bytes=sum(x['bytes'] for x in samples))
