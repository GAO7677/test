"""Owner isolation for few-shot inputs, outputs, and derived versions.

Existing shared datasets keep their visibility. Administrator access still uses
the application's explicit unrestricted role. The cache lives in one DB session.
"""
import json
from fastapi import HTTPException
from ..models import Dataset, DatasetVersion, FewshotJob, IntegrateTask


def _ids(value):
    try:
        result=json.loads(value or '[]')
        return result if isinstance(result,list) else []
    except (ValueError,TypeError):
        return []


def private_owners(db):
    if 'private_owners' in db.info:
        return db.info['private_owners']
    raw, private = {}, set()
    for job in db.query(FewshotJob).order_by(FewshotJob.id).all():
        for rid in (job.source_data_id,job.result_data_id):
            if rid:raw.setdefault(rid,job.created_by)
        if job.dataset_id:private.add(job.dataset_id)
    datasets=db.query(Dataset).all()
    for ds in datasets:
        if ds.data_type=='image' and set(_ids(ds.source_data_ids)) & raw.keys():
            private.add(ds.id)
    # A copy/cleanup stays in the same dataset. Cross-dataset integration and
    # its later copies must inherit private status even with a public main input.
    versions=db.query(DatasetVersion).all()
    by_version={v.id:v.dataset_id for v in versions}
    parents={ds.id:set() for ds in datasets}
    for v in versions:
        parent=by_version.get(v.parent_version_id)
        if parent and parent!=v.dataset_id:parents[v.dataset_id].add(parent)
    for task in db.query(IntegrateTask).all():
        try:other=json.loads(task.detail or '{}').get('other_input_version_ids',[])
        except (ValueError,AttributeError):other=[]
        targets=set(_ids(task.source_ids))
        output=by_version.get(task.output_version_id)
        if output:targets.add(output)
        inputs=[task.input_version_id]+other
        for target in targets & parents.keys():
            parents[target].update(by_version[x] for x in inputs if x in by_version)
    changed=True
    while changed:
        inherited={dsid for dsid,sources in parents.items() if sources & private}
        changed=bool(inherited-private);private.update(inherited)
    result=(raw,{ds.id:ds.created_by for ds in datasets if ds.id in private})
    db.info['private_owners']=result
    return result


def excluded_raw_ids(db,cur,data_type='image'):
    if cur.unrestricted or data_type!='image':return []
    return [rid for rid,owner in private_owners(db)[0].items() if owner!=cur.id]


def excluded_dataset_ids(db,cur):
    if cur.unrestricted:return []
    return [did for did,owner in private_owners(db)[1].items() if owner!=cur.id]


def require_raw(db,cur,data_type,obj,operate=False):
    if obj is None or obj.deleted_at:raise HTTPException(404,'数据不存在')
    if obj.id in excluded_raw_ids(db,cur,data_type):
        raise HTTPException(403,'只能访问自己任务的素材和生成数据')
    (cur.require_operate if operate else cur.require_view)(obj.secret_level)
    return obj


def require_dataset(db,cur,ds,operate=False):
    if ds.id in excluded_dataset_ids(db,cur):
        raise HTTPException(403,'只能访问自己任务的数据集及预处理版本')
    (cur.require_operate if operate else cur.require_view)(ds.secret_level)
    return ds
