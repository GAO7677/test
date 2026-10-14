"""Import bounded, traceable examples from existing SSH 118 datasets.

Run on 118 with backend/.venv/bin/python. Sources are read-only; only copied
examples and generated caption exports are registered. Re-running skips names
already registered; inspect import_report.json before resuming a partial run.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.config import settings
from app.database import SessionLocal
from app.models import DATA_TYPE_MODELS, DataFile, Dataset, User, Project
from app.services import file_service as fs, version_service as vs

BASE = Path('/mnt/data/gaoya')
DATA = BASE / 'dataset'
AGENT = BASE / 'agent-data/datasets'
OUT = BASE / 'agent-data/outputs_v1/data_platform3_linux/20261013_platform_dataset_demo'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def balanced(paths, limit, per_group=2):
    groups = defaultdict(list)
    for path in paths:
        groups[path.parts[-3]].append(path)
    return list(itertools.islice((p for key in sorted(groups)
                                 for p in sorted(groups[key])[:per_group]), limit))


def collect():
    specs = []
    def add(key, name, dtype, rows, source_type='真实采集'):
        if rows:
            specs.append(dict(key=key, name='展示 · '+name, data_type=dtype,
                              rows=rows, source_type=source_type))
    index = sqlite3.connect(f'file:{settings.data["local_material_index"]}?mode=ro', uri=True)
    indexed = {}
    for key, count in [('aircraft', 200), ('rareplanes', 24), ('laionsg', 60), ('hpdv3', 40)]:
        spec = settings.data['local_materials'][key]
        if key == 'hpdv3':
            rows = index.execute('select name,caption,primary_category from images where dataset=? and caption!="" order by name limit ?', (key,count)).fetchall()
        else:
            rows = index.execute('select name,caption,primary_category from (select name,caption,primary_category,row_number() over(partition by primary_category order by name) as rn from images where dataset=? and primary_category!="") where rn<=? order by rn,primary_category limit ?', (key,2 if key=='aircraft' else 8,count)).fetchall()
        indexed[key] = rows
        samples = [dict(path=Path(spec['path'])/n, category=category, caption=caption)
                   for n,caption,category in rows]
        if key == 'aircraft':
            boxes = {}
            for line in (Path(spec['path']).parent/'images_box.txt').read_text().splitlines():
                fields=line.split(); boxes[fields[0]+'.jpg']=list(map(float,fields[1:]))
            for row in samples:
                row['label'] = {'annotations':[{'type':'bbox','xyxy':boxes[row['path'].name], 'label':row['category']}], 'source':'FGVC Aircraft original box + variant', 'category':row['category']}
        elif key == 'rareplanes':
            import ijson
            source = spec['metadata']['sources'][0]
            wanted = {s['path'].name:s for s in samples}
            with open(source,'rb') as f: cats={c['id']:c['name'] for c in ijson.items(f,'categories.item')}
            with open(source,'rb') as f: ids={i['id']:Path(i['file_name']).name for i in ijson.items(f,'images.item') if Path(i['file_name']).name in wanted}
            for row in samples:row['label']={'annotations':[],'source':source}
            with open(source,'rb') as f:
                for a in ijson.items(f,'annotations.item'):
                    name=ids.get(a['image_id'])
                    if name and not a.get('iscrowd'):
                        x,y,w,h=map(float,a['bbox'])
                        wanted[name]['label']['annotations'].append({'type':'bbox','xyxy':[x,y,x+w,y+h],'label':cats[a['category_id']]})
        else:
            for row in samples:row['label']={'caption':row['caption'],'category':row['category'],'source':'original indexed metadata; no inferred bounding boxes'}
        add(key,spec['name'],'image',samples, '仿真' if key=='rareplanes' else ('AIGC' if key=='hpdv3' else '真实采集'))
        if key in ('laionsg','hpdv3'):
            texts=[]
            for n,caption,category in rows:
                p=OUT/'caption_exports'/key/(Path(n).stem+'.txt')
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(caption,encoding='utf-8')
                texts.append(dict(path=p,original_source=str(Path(spec['path'])/n),caption=caption,category=category))
            add(key+'_captions',spec['name']+' · caption 文本','text',texts)
    crop=AGENT/'rareplanes_fewshot/20261012_role7_4shot_seed0'
    crops=[]
    for line in (crop/'train.jsonl').read_text().splitlines():
        row=json.loads(line)
        crops.append(dict(path=crop/row['image'],category=row['role'],label={'category':row['role'],'source':'RarePlanes real support crop original role'}))
    add('rareplanes_real','RarePlanes · 真实飞机裁剪','image',crops)
    for key,root,pattern,limit in [
        ('physics_iq',DATA/'physics-iq-benchmark','full-videos/take-1/30FPS/*.mp4',8),
        ('physics_iq_verified',DATA/'Anates-Labs-Research-Physics-IQ-Verified','full-videos/take-1/30FPS/*.mp4',8),
        ('physics_iq_verified2',DATA/'physics-IQ-benchmark-verified','full-videos/take-1/30FPS/*.mp4',8),
        ('simulation_single',DATA/'new_data0826','*/*/video.mp4',12),
        ('simulation_two',DATA/'new_data0826_twoobj','*/*/video.mp4',12),
        ('pybullet',DATA/'0717pybullet_5000_vbenchtop5','cases/*/*/videos/*.mp4',20)]:
        files=sorted(root.glob(pattern))
        if key.startswith('simulation'):
            groups=defaultdict(list)
            for p in files:groups[p.relative_to(root).parts[0]].append(p)
            files=[p for g in sorted(groups) for p in groups[g][:2]][:limit]
        elif key=='pybullet':
            groups=defaultdict(list)
            for p in files:groups[p.relative_to(root).parts[1]].append(p)
            files=[p for g in sorted(groups) for p in groups[g][:2]][:limit]
        else: files=files[:limit]
        rows=[dict(path=p,stored_name=str(p.relative_to(root)),category=p.relative_to(root).parts[0]) for p in files]
        add(key,root.name,'video',rows,'仿真' if key in ('simulation_single','simulation_two','pybullet') else '真实采集')
        metadata=[]
        for p in files:
            m=p.parent/'metadata.json'
            if key=='pybullet':m=p.parent.parent/'meta'/(p.stem+'.json')
            if m.exists():metadata.append(dict(path=m,stored_name=str(m.relative_to(root))))
        add(key+'_metadata',root.name+' · 场景元数据','text',metadata,'仿真')
    phyco=DATA/'nnsriram97-phyco_kubric'
    cases=[];seen=set()
    for line in (phyco/'test_500_balanced.txt').read_text().splitlines():
        rel=line.split('/nnsriram97-phyco_kubric/',1)[-1]
        folder=phyco/rel;category=Path(rel).parts[0]
        if category not in seen and (folder/'rgba.mp4').exists():
            seen.add(category);cases.append((folder,category))
        if len(cases)>=12:break
    add('phyco','PhyCo Kubric · RGB 视频','video',[dict(path=f/'rgba.mp4',stored_name=str((f/'rgba.mp4').relative_to(phyco)),category=c) for f,c in cases],'仿真')
    add('phyco_frames','PhyCo Kubric · 首帧','image',[dict(path=f/'rgba_firstframe.png',stored_name=str((f/'rgba_firstframe.png').relative_to(phyco)),category=c,label={'category':c,'caption':(phyco/c/'common_caption_cosmos.txt').read_text() if (phyco/c/'common_caption_cosmos.txt').exists() else '', 'source':'original scene category + caption'}) for f,c in cases if (f/'rgba_firstframe.png').exists()],'仿真')
    add('phyco_text','PhyCo Kubric · 场景 caption','text',[dict(path=phyco/c/'common_caption_cosmos.txt',stored_name=c+'/caption.txt',category=c) for _,c in cases if (phyco/c/'common_caption_cosmos.txt').exists()],'仿真')
    switches=DATA/'physics-iq-benchmark/switch-frames'
    add('physics_switches','Physics IQ · 切换帧','image',[dict(path=p) for p in sorted(switches.glob('*.jpg'))[:12]])
    for root in sorted(AGENT.glob('physv-*')):
        videos=sorted(root.glob('*.mp4'))[:8]
        add(root.name,root.name,'video',[dict(path=p) for p in videos],'仿真')
        add(root.name+'_metadata',root.name+' · 控制参数','text',
            [dict(path=p.with_suffix('.json')) for p in videos if p.with_suffix('.json').exists()], '仿真')
    for key,root,json_dir in [
        ('ava_analysis',DATA/'Iceclear-AVA/result_re_tmm','detections'),
        ('tad_analysis',DATA/'Shuai1995-TAD66K/tmm_result','jsons')]:
        derived=sorted((root/json_dir).glob('*.json'))[:20]
        derived+=sorted(p for p in (root/'csv').glob('*.csv') if p.stat().st_size<1024*1024)[:6]
        add(key,root.parent.name+' · 已有分析结果（非原图）','text',
            [dict(path=p,stored_name=str(p.relative_to(root))) for p in derived])
    return specs


def apply(specs):
    report=[]
    previous={r['key']:r for r in json.loads((OUT/'import_report.json').read_text())} if (OUT/'import_report.json').exists() else {}
    with SessionLocal() as db:
        user=db.query(User).filter(User.role=='system_admin',User.deleted_at.is_(None)).first()
        project='SSH118 现有数据集功能展示'
        if not db.query(Project).filter_by(name=project).first():
            db.add(Project(name=project,description='只复制有限样本，完整来源见展示清单。',created_by=user.id));db.commit()
        for spec in specs:
            ds=db.query(Dataset).filter(Dataset.name==spec['name'],Dataset.deleted_at.is_(None)).first()
            if ds:
                report.append(previous.get(spec['key'],dict(key=spec['key'],name=ds.name,dataset_id=ds.id,skipped=True)));continue
            dtype=spec['data_type']; rows=spec['rows']; model=DATA_TYPE_MODELS[dtype]
            raw=model(name=spec['name'],project=project,scene=spec['key'],source_type=spec['source_type'],secret_level='公开',created_by=user.id,remark='SSH118 有限展示样本；来源、类别与 caption 见清单，未修改原始数据。')
            db.add(raw);db.commit()
            directory=fs.dataset_dir(dtype,raw.id);manifest=[];labels=0
            for row in rows:
                src=row['path'];rel=Path(row.get('stored_name',src.name))
                if rel.is_absolute() or '..' in rel.parts:raise ValueError(rel)
                dst=directory/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
                digest=hashlib.sha256(dst.read_bytes()).hexdigest()
                db.add(DataFile(dataset_id=raw.id,data_type=dtype,filename=str(rel),stored_name=str(rel),size_bytes=dst.stat().st_size,file_hash=digest,stage='raw',status='complete',chunk_total=1,chunk_received=1))
                if row.get('label'):
                    save(dst.with_name(dst.stem+'_label.json'),row['label']);labels+=1
                manifest.append(dict(file=str(rel),source=str(src),original_source=row.get('original_source'),sha256=digest,category=row.get('category',''),caption=row.get('caption','')))
            save(directory/'_manifest.json',{'scope':'bounded demonstration subset','key':spec['key'],'items':manifest})
            raw.file_path=str(directory);raw.file_count=len(rows);raw.size_bytes=sum((directory/m['file']).stat().st_size for m in manifest)
            raw.format=','.join(sorted({r['path'].suffix[1:].upper() for r in rows}))[:20]
            raw.label_progress=round(labels/max(len(rows),1)*100);raw.status='已标注' if labels==len(rows) else '未标注';raw.label_type='原始类别/caption/目标框' if labels else ''
            metadata=fs.extract_metadata(dtype,directory/manifest[0]['file'])
            for key,value in metadata.items():
                if hasattr(raw,key):setattr(raw,key,value)
            ds=Dataset(name=spec['name'],data_type=dtype,description=f'来自 SSH118；{len(rows)} 个有限展示样本，保留来源清单。',source_data_ids=json.dumps([raw.id]),secret_level='公开',label_count=labels,created_by=user.id)
            db.add(ds);db.commit()
            v=vs.create_initial_version(db,ds,user.id,description='从现有数据复制展示子集及原始标注')
            shutil.copytree(directory,v.file_path,dirs_exist_ok=True);vs.refresh_version_stats(db,v)
            item=dict(key=spec['key'],name=spec['name'],data_type=dtype,raw_id=raw.id,dataset_id=ds.id,version_id=v.id,data_count=len(rows),original_label_files=labels,bytes=raw.size_bytes,categories=len({m['category'] for m in manifest if m['category']}))
            report.append(item);save(OUT/'import_report.json',report);print(json.dumps(item,ensure_ascii=False),flush=True)
    save(OUT/'import_report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    specs=collect()
    if args.apply:apply(specs)
    else:print(json.dumps([{k:v for k,v in s.items() if k!='rows'}|{'count':len(s['rows'])} for s in specs],ensure_ascii=False,indent=2))
