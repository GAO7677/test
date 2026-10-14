"""Build a read-only SQLite browse index from explicitly configured metadata sources.

Requires ijson for streaming large JSON; never changes source images or annotations.
"""
import argparse
import json
import os
import sqlite3
from pathlib import Path
import ijson

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def records(path, prefix='item'):
    with Path(path).open('rb') as f:
        yield from ijson.items(f, prefix)


def build(config, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.building.sqlite')
    if temporary.exists():
        temporary.unlink()
    db = sqlite3.connect(temporary)
    db.executescript('''
        PRAGMA journal_mode=OFF;
        CREATE TABLE roots(dataset TEXT PRIMARY KEY, path TEXT NOT NULL);
        CREATE TABLE images(dataset TEXT, name TEXT, size INTEGER, caption TEXT DEFAULT '',
                            primary_category TEXT DEFAULT '', PRIMARY KEY(dataset,name));
        CREATE TABLE labels(dataset TEXT, name TEXT, category TEXT,
                            PRIMARY KEY(dataset,name,category));
        CREATE INDEX label_category ON labels(dataset,category,name);
        CREATE TABLE category_counts(dataset TEXT, name TEXT, count INTEGER,
                                     PRIMARY KEY(dataset,name));
    ''')
    report = {}
    try:
        for dataset, spec in config['local_materials'].items():
            root = Path(spec['path']).resolve()
            files = {p.name: p.stat().st_size for p in root.iterdir()
                     if p.is_file() and not p.is_symlink() and p.suffix.lower() in EXTENSIONS}
            db.execute('INSERT INTO roots VALUES(?,?)', (dataset, str(root)))
            db.executemany('INSERT INTO images(dataset,name,size) VALUES(?,?,?)',
                           ((dataset, name, size) for name, size in files.items()))
            print(f'{dataset}: {len(files)} images; reading metadata', flush=True)

            def apply(name, caption='', categories=()):
                name = Path(name).name
                if name not in files:
                    return
                if caption:
                    db.execute("UPDATE images SET caption=? WHERE dataset=? AND name=? AND caption=''",
                               (str(caption)[:10000], dataset, name))
                categories = {str(c).strip() for c in categories if str(c).strip()}
                db.executemany('INSERT OR IGNORE INTO labels VALUES(?,?,?)',
                               ((dataset, name, cat) for cat in categories))

            metadata = spec.get('metadata', {})
            kind = metadata.get('kind')
            sources = metadata.get('sources', [])
            if kind == 'platform_demo_manifest':
                for source in sources:
                    for row in records(source):
                        apply(row['name'], row.get('caption', ''), row.get('categories', []))
            elif kind == 'fgvc_variant':
                for source in sources:
                    for line in Path(source).read_text().splitlines():
                        identifier, variant = line.split(' ', 1)
                        apply(identifier + '.jpg', categories=[variant])
            elif kind == 'rareplanes_coco':
                for source in sources:
                    categories = {str(r['id']): r['name'] for r in records(source, 'categories.item')}
                    names = {str(r['id']): r['file_name'] for r in records(source, 'images.item')
                             if Path(r['file_name']).name in files}
                    for row in records(source, 'annotations.item'):
                        name = names.get(str(row['image_id']))
                        category = categories.get(str(row['category_id']))
                        if name and category:
                            apply(name, categories=[category])
            elif kind == 'laion_sg':
                for source in sources:
                    for row in records(source):
                        apply(row['name'], row.get('caption_ori', ''),
                              [item['label'] for item in row.get('items', []) if item.get('label')])
            elif kind == 'hpd_pairs':
                for source in sources:
                    for row in records(source):
                        for key in ('path1', 'path2'):
                            if row.get(key):
                                apply(row[key], row.get('prompt', ''))
            elif kind:
                raise ValueError('Unknown metadata kind: ' + kind)
            db.execute('''UPDATE images SET primary_category=COALESCE(
                (SELECT MIN(category) FROM labels WHERE labels.dataset=images.dataset
                 AND labels.name=images.name), '') WHERE dataset=?''', (dataset,))
            db.execute('INSERT INTO category_counts SELECT dataset,category,COUNT(*) FROM labels '
                       'WHERE dataset=? GROUP BY dataset,category', (dataset,))
            row = db.execute("SELECT COUNT(*),SUM(caption!=''),SUM(primary_category!='') FROM images WHERE dataset=?",
                             (dataset,)).fetchone()
            report[dataset] = dict(images=row[0], captions=row[1] or 0, categorized=row[2] or 0,
                                   categories=db.execute('SELECT COUNT(*) FROM category_counts WHERE dataset=?',
                                                         (dataset,)).fetchone()[0], sources=sources)
            print(dataset, json.dumps(report[dataset], ensure_ascii=False), flush=True)
        db.execute('CREATE INDEX image_category ON images(dataset,primary_category,name)')
        db.commit()
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    finally:
        db.close()
    os.replace(temporary, output)
    output.with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(json.loads(Path(args.config).read_text()), args.output)
