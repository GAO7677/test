"""Read categories without treating processing errors as annotations."""
def valid_label(label):
    if isinstance(label, list):
        return all(isinstance(a, (dict, list, str)) for a in label)
    return (isinstance(label, dict) and not label.get('error')
            and not label.get('requires_relabel')
            and any(k in label for k in ('annotations','detections','items','segments',
                    'entities','boxes','labels','keywords','classification','sentiment','category','caption')))


def label_names(label):
    if not valid_label(label):
        return []
    names = []
    def visit(item):
        if isinstance(item, list):
            for child in item: visit(child)
        elif isinstance(item, dict):
            nested = isinstance(item.get('boxes'), list)
            name = item.get('category') or (None if nested else item.get('label')) or item.get('word')
            if not name and item.get('entity'):
                name = item.get('type')
            if name: names.append(str(name))
            for key in ('annotations','detections','items','segments','entities','boxes',
                        'labels','keywords','classification','sentiment'):
                if key in item: visit(item[key])
        elif isinstance(item, str): names.append(item)
    visit(label)
    return list(dict.fromkeys(names))
