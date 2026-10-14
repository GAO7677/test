import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for row in json.loads((root/'vendor/sources.json').read_text())['files']:
 assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
print('All vendored source hashes match experiment snapshots.')
