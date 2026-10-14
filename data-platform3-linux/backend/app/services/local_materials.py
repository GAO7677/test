"""Read-only catalog of explicitly approved local image datasets (no recursive scans)."""
from pathlib import Path
import sqlite3
from fastapi import HTTPException
from ..config import settings

CATALOG = {
    'laion500': ('LAION 高分辨率 · 500', Path('/data/gaoya/dataset/laion-high-resolution-data_v1_500')),
    'laion2500': ('LAION 1024 · 2500', Path('/data/gaoya/dataset/laion1024_2500')),
    'aesthetic': ('Aesthetic Photos', Path('/data/gaoya/dataset/recoilme-aesthetic_photos_xs/images')),
}
# Deployment-specific allowlist: only administrators can edit config.json.
if 'local_materials' in settings.data:
    CATALOG = {key: (item['name'], Path(item['path']))
               for key, item in settings.data['local_materials'].items()}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def root_for(dataset: str) -> Path:
    if dataset not in CATALOG:
        raise HTTPException(404, '素材库不存在')
    root = CATALOG[dataset][1]
    if not root.is_dir():
        raise HTTPException(404, '素材库当前不可用')
    return root.resolve()


def resolve_image(dataset: str, name: str) -> Path:
    root = root_for(dataset)
    if not name or Path(name).name != name or '\\' in name or Path(name).suffix.lower() not in EXTENSIONS:
        raise HTTPException(422, '素材文件名无效')
    path = root / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
        raise HTTPException(404, '素材图片不存在')
    return path


def list_images(dataset: str, query: str = '') -> list[dict]:
    root = root_for(dataset)
    return [{'name': p.name, 'size': p.stat().st_size} for p in sorted(root.iterdir())
            if p.suffix.lower() in EXTENSIONS and not p.is_symlink() and p.is_file()
            and query.casefold() in p.name.casefold()]


def browse_images(dataset: str, query: str = '', category: str = '', page: int = 1) -> dict:
    root = root_for(dataset)
    index = settings.data.get('local_material_index')
    if index and Path(index).is_file():
        with sqlite3.connect(Path(index).resolve().as_uri() + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            stored_root = db.execute('SELECT path FROM roots WHERE dataset=?', (dataset,)).fetchone()
            if stored_root and stored_root['path'] == str(root):
                params = [dataset]
                conditions = ['i.dataset=?']
                if query:
                    conditions.append("instr(lower(i.name || ' ' || i.caption),lower(?))>0")
                    params.append(query)
                if category == '__uncategorized__':
                    conditions.append("i.primary_category=''")
                elif category:
                    conditions.append('EXISTS(SELECT 1 FROM labels l WHERE l.dataset=i.dataset '
                                      'AND l.name=i.name AND l.category=?)')
                    params.append(category)
                where = ' AND '.join(conditions)
                total = db.execute('SELECT COUNT(*) FROM images i WHERE ' + where, params).fetchone()[0]
                rows = db.execute('SELECT i.* FROM images i WHERE ' + where +
                                  ' ORDER BY i.primary_category,i.name LIMIT 24 OFFSET ?',
                                  params + [(page-1)*24]).fetchall()
                items = []
                for row in rows:
                    labels = [r[0] for r in db.execute('SELECT category FROM labels WHERE dataset=? AND name=? '
                                                      'ORDER BY category', (dataset, row['name']))]
                    items.append(dict(name=row['name'], size=row['size'], caption=row['caption'],
                                      categories=labels, primary_category=row['primary_category']))
                categories = [dict(r) for r in db.execute('SELECT name,count FROM category_counts '
                                                         'WHERE dataset=? ORDER BY name', (dataset,))]
                uncategorized = db.execute("SELECT COUNT(*) FROM images WHERE dataset=? AND primary_category=''",
                                           (dataset,)).fetchone()[0]
                if uncategorized:
                    categories.append(dict(name='未分类', value='__uncategorized__', count=uncategorized))
                return dict(items=items, total=total, categories=categories)
    items = list_images(dataset, query)
    if category and category != '__uncategorized__':
        items = []
    return dict(items=[dict(row, caption='', categories=[], primary_category='')
                       for row in items[(page-1)*24:page*24]], total=len(items), categories=[])
