"""Keep paired annotations aligned with changed content; reject ambiguous geometry."""
import copy
import json
from difflib import SequenceMatcher
from pathlib import Path


def read_text(path):
    raw = Path(path).read_bytes()
    for encoding in ('utf-8-sig', 'gb18030'):
        try: return raw.decode(encoding)
        except UnicodeDecodeError: pass
    raise ValueError('无法识别文本编码，请转换为 UTF-8')


def carry_label(source, target, *, image_scale=None, text_change=False):
    lp = source.with_name(source.stem + '_label.json')
    if not lp.is_file() or not target.is_file(): return
    from .label_schema import valid_label
    label = json.loads(lp.read_text(encoding='utf-8'))
    if not valid_label(label): return
    label = copy.deepcopy(label)
    blocks = None
    if text_change:
        before, after = read_text(source), read_text(target)
        if before != after:
            if max(len(before), len(after)) > 200000:
                raise ValueError('带标注文本超过20万字符，暂不支持改变字符位置的清洗')
            blocks = SequenceMatcher(None, before, after, autojunk=False).get_matching_blocks()
    def walk(a):
        if isinstance(a, list):
            for item in a: walk(item)
        elif isinstance(a, dict):
            if image_scale:
                sx, sy = image_scale
                if any(k in a for k in ('segmentation', 'mask', 'polygon', 'bbox')):
                    raise ValueError('此标注含尚不支持缩放的几何格式，请先保留原尺寸或转换标注格式')
                if 'xyxy' in a:
                    a['xyxy'] = [v * (sx if i % 2 == 0 else sy) for i,v in enumerate(a['xyxy'])]
                for k in ('x','w','width'):
                    if isinstance(a.get(k), (int,float)): a[k] *= sx
                for k in ('y','h','height'):
                    if isinstance(a.get(k), (int,float)): a[k] *= sy
            if blocks is not None and isinstance(a.get('start'), int) and isinstance(a.get('end'), int):
                start, end = a['start'], a['end']
                unit = a.get('offset_unit', 'codepoint')
                if unit == 'utf16':
                    start = len(before.encode('utf-16-le')[:start*2].decode('utf-16-le'))
                    end = len(before.encode('utf-16-le')[:end*2].decode('utf-16-le'))
                match = next((b for b in blocks if b.a <= start <= end <= b.a+b.size), None)
                if match is None:
                    raise ValueError('清洗将删除或改写已标注片段，请先修正标注或关闭此清洗规则')
                ns, ne = start + match.b-match.a, end + match.b-match.a
                if unit == 'utf16':
                    ns, ne = len(after[:ns].encode('utf-16-le'))//2, len(after[:ne].encode('utf-16-le'))//2
                a['start'], a['end'] = ns, ne
            for item in a.values():
                if isinstance(item, (dict,list)): walk(item)
    walk(label)
    target.with_name(target.stem + '_label.json').write_text(json.dumps(label, ensure_ascii=False), encoding='utf-8')
