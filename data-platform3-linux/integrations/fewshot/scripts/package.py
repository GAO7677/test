#!/usr/bin/env python3
"""Package training/generation code and the example dataset."""
import argparse,hashlib,json,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[1]
if a.output.resolve().is_relative_to(root):raise ValueError('Write archives outside the source repository')
a.output.parent.mkdir(parents=True,exist_ok=True);manifest={}
with tarfile.open(a.output,'w:gz') as tar:
 for f in sorted(root.rglob('*')):
  if not f.is_file() or f.is_symlink() or any(s in ('.git','__pycache__','.pytest_cache') for s in f.relative_to(root).parts) or f.suffix in ('.pyc','.log'):continue
  rel='FewShotLab/'+str(f.relative_to(root));tar.add(f,arcname=rel);manifest[rel]=hashlib.sha256(f.read_bytes()).hexdigest()
 # Materialize the small example dataset when the checkout keeps it on a data disk.
 example=root/'asset'/'my_data'
 if example.is_symlink():
  for f in sorted(example.resolve().rglob('*')):
   if not f.is_file() or f.is_symlink():continue
   rel='FewShotLab/asset/my_data/'+str(f.relative_to(example.resolve()))
   tar.add(f,arcname=rel);manifest[rel]=hashlib.sha256(f.read_bytes()).hexdigest()
a.output.with_suffix(a.output.suffix+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
h=hashlib.sha256()
with a.output.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
a.output.with_suffix(a.output.suffix+'.sha256').write_text(h.hexdigest()+'  '+a.output.name+'\n')
print(a.output, len(manifest),'files',a.output.stat().st_size,'bytes')
