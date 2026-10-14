"""Portable static HTML report; open directly, no service needed."""
import html,json
from pathlib import Path
from .core import read

def build(run):
 run=Path(run);esc=lambda s:html.escape(str(s));parts=[]
 for batch in sorted((run/'generations').iterdir(),reverse=True):
  audit={(r['kind'],r['index']):r for r in read(batch/'index.json')} if (batch/'index.json').exists() else {}
  rows=[]
  for kind in ['base','lora']:
   for p in sorted((batch/kind).glob('*.json')):
    row=dict(audit.get((kind,int(p.stem))) or read(p))
    row.update(kind=kind,index=int(p.stem),image=kind+'/'+p.with_suffix('.png').name)
    rows.append(row)
  if not rows:continue
  parts.append('<h2>'+esc(batch.name)+'</h2>')
  for i in sorted({r['index'] for r in rows}):
   paired=[r for r in rows if r['index']==i];row=paired[0];cells=[]
   if row.get('caption_source_image'):
    url='../'+row['caption_source_image'];cells.append(f'<figure><a href="{esc(url)}"><img src="{esc(url)}" loading="lazy"></a><figcaption>Training source (not inference input)</figcaption></figure>')
   for r in paired:
    url='../generations/'+batch.name+'/'+r['image'];meta=url.rsplit('.',1)[0]+'.json'
    cells.append(f'<figure><a href="{esc(url)}"><img src="{esc(url)}" loading="lazy"></a><figcaption>{esc(r["kind"])} / seed {r["seed"]} · <a href="{esc(meta)}">JSON</a></figcaption></figure>')
   parts.append('<div class="pair">'+''.join(cells)+'</div><p>Prompt: '+esc(row['prompt'])+'</p>')
 status=read(run/'status.json');out=run/'report/index.html';out.parent.mkdir(exist_ok=True)
 out.write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FewShotLab</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:30px auto;padding:20px;background:#f5f7fa;color:#203044}.pair{display:flex;gap:12px}figure{flex:1;margin:0}img{width:100%}p{overflow-wrap:anywhere}</style><h1>'+esc(run.name)+'</h1><pre>'+esc(json.dumps(status,ensure_ascii=False,indent=2))+'</pre>'+''.join(parts));return out
