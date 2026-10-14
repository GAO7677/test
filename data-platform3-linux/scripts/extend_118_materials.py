"""Append three bounded demo image libraries to the existing material index."""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.config import settings
from app.services import file_service as fs

OUT=Path('/mnt/data/gaoya/agent-data/outputs_v1/data_platform3_linux/20261013_platform_dataset_demo')
samples={r['key']:r for r in json.loads((OUT/'import_report.json').read_text())}
index=Path(settings.data['local_material_index'])
fd,temporary=tempfile.mkstemp(prefix='materials-demo-',suffix='.sqlite',dir=index.parent)
os.close(fd)
shutil.copy2(index,temporary)
report=[]
with sqlite3.connect(temporary) as db:
    for key,title in [('rareplanes_real','RarePlanes · 真实飞机裁剪（展示子集）'),
                      ('phyco_frames','PhyCo Kubric · 仿真首帧（展示子集）'),
                      ('physics_switches','Physics IQ · 场景帧（展示子集）')]:
        source=fs.dataset_dir('image',samples[key]['raw_id'])
        manifest=json.loads((source/'_manifest.json').read_text())['items']
        destination=OUT/'fewshot_materials'/key
        destination.mkdir(parents=True,exist_ok=True)
        rows=[]
        for i,item in enumerate(manifest):
            original=source/item['file'];name=f'{i:03d}_{original.name}'
            shutil.copy2(original,destination/name)
            sidecar=original.with_name(original.stem+'_label.json')
            label=json.loads(sidecar.read_text()) if sidecar.exists() else {}
            category=label.get('category') or item.get('category','')
            rows.append(dict(name=name,caption=label.get('caption') or item.get('caption',''),categories=[category] if category else []))
        metadata=destination/'browse_metadata.json'
        metadata.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
        settings.data['local_materials'][key]=dict(name=title,path=str(destination),metadata=dict(kind='platform_demo_manifest',sources=[str(metadata)]))
        for table in ['roots','images','labels','category_counts']:db.execute(f'DELETE FROM {table} WHERE dataset=?',(key,))
        db.execute('INSERT INTO roots VALUES(?,?)',(key,str(destination)))
        for row in rows:
            primary=row['categories'][0] if row['categories'] else ''
            db.execute('INSERT INTO images(dataset,name,size,caption,primary_category) VALUES(?,?,?,?,?)',(key,row['name'],(destination/row['name']).stat().st_size,row['caption'],primary))
            db.executemany('INSERT INTO labels VALUES(?,?,?)',[(key,row['name'],c) for c in row['categories']])
        db.execute('INSERT INTO category_counts SELECT dataset,category,count(*) FROM labels WHERE dataset=? GROUP BY category',(key,))
        report.append(dict(key=key,images=len(rows),captioned=sum(bool(r['caption']) for r in rows),categories=len({c for r in rows for c in r['categories']})))
os.replace(temporary,index)
fd,temporary=tempfile.mkstemp(prefix='config-demo-',suffix='.json',dir=settings.path.parent)
with os.fdopen(fd,'w') as f:json.dump(settings.data,f,ensure_ascii=False,indent=2)
os.chmod(temporary,0o600);os.replace(temporary,settings.path)
(OUT/'fewshot_materials.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
