"""Selected video annotation exports, derived at download time (not source data)."""
import json
from pathlib import PurePosixPath

KINDS = ('box', 'mask', 'trajectory')

def exports(label, stem):
    if not isinstance(label, dict): return {}
    kinds = label.get('export_types', [])
    rows = label.get('annotations', [])
    result = {}
    common = {k: label[k] for k in ('fps', 'width', 'height', 'frame_times', 'review_status', 'workflow') if k in label}
    for kind in KINDS:
        if kind not in kinds: continue
        items = []
        for a in rows:
            point = a.get('type') == 'video_point'
            if (kind == 'trajectory') != point: continue
            keys = ['track_id', 'label', 'frame', 'time', 'outside', 'source', 'review_status']
            if kind == 'box': keys += ['xyxy']
            elif kind == 'mask': keys += ['mask']
            else: keys += ['point_id', 'xy', 'visible', 'seed_origin']
            if kind == 'mask' and not a.get('mask') and not a.get('outside'): continue
            if kind == 'box' and not a.get('xyxy') and not a.get('outside'): continue
            items.append({k:a[k] for k in keys if k in a})
        name = str(PurePosixPath('annotation_exports') / (stem + '_' + kind + '.json'))
        result[name] = json.dumps({**common, 'annotation_type':kind, 'annotations':items}, ensure_ascii=False)
    return result
