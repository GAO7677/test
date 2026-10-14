"""Saved query context, source candidates and generation-to-download lineage."""
import json
from pathlib import Path
from fastapi import HTTPException
from ..models import DownloadRequest,DataFile,ImageData,Dataset,DatasetVersion,FewshotJob,User
from . import data_access as access, file_service as fs
from .label_locator import find_label


def get_request(db,rid,cur,own=False):
    request=db.get(DownloadRequest,rid)
    if not request:raise HTTPException(404,'下载请求不存在')
    if request.created_by!=cur.id and (own or not cur.unrestricted):
        raise HTTPException(403,'只能操作自己的下载请求')
    return request


def resolve_candidate(db,request,fid,cur):
    if fid not in json.loads(request.candidate_file_ids):
        raise HTTPException(422,'素材不属于这次下载查询的候选图像')
    file=db.get(DataFile,fid)
    if not file or file.data_type!='image' or file.stage!='raw' or file.status!='complete' or file.deleted_at:
        raise HTTPException(404,'候选图像已不可用，请重新查询')
    raw=access.require_raw(db,cur,'image',db.get(ImageData,file.dataset_id))
    if raw.deleted_at:raise HTTPException(404,'来源数据已删除')
    root=fs.dataset_dir('image',raw.id).resolve()
    path=(root/file.stored_name).resolve()
    if root not in path.parents or not path.is_file():raise HTTPException(404,'候选文件不存在')
    return file,raw,path


def candidates(db,request,cur):
    items=[]
    for fid in json.loads(request.candidate_file_ids):
        try:file,raw,path=resolve_candidate(db,request,fid,cur)
        except HTTPException:continue
        label=find_label(db,'image',raw.id,path,file.stored_name,cur) or {}
        items.append(dict(file_id=file.id,name=file.filename,size=path.stat().st_size,
                          dataset_id=raw.id,secret_level=raw.secret_level,
                          caption=label.get('caption','') if isinstance(label,dict) else ''))
    return items


def create_request(db,cur,query,files):
    ids=[];size=0;seen=set()
    if query['data_type']=='image':
        for row in files:
            file=db.get(DataFile,row.get('file_id')) if row.get('file_id') else None
            if not file:continue
            if file.file_hash and file.file_hash in seen:continue
            length=file.size_bytes or 0
            if not 0<length<=20*1024**2 or size+length>500*1024**2:continue
            if Path(file.stored_name).suffix.lower() not in {'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}:continue
            ids.append(file.id);size+=length
            if file.file_hash:seen.add(file.file_hash)
            if len(ids)==100:break
    shortage=max(0,query['required_count']-len(files))
    request=DownloadRequest(created_by=cur.id,query_json=json.dumps(query,ensure_ascii=False),
                            candidate_file_ids=json.dumps(ids),matched_count=len(files),
                            generation_count=min(shortage,1000))
    db.add(request);db.commit()
    return request


def request_summary(db,request):
    owner=db.get(User,request.created_by)
    return dict(id=request.id,created_by=request.created_by,
                owner_name=owner.display_name or owner.username if owner else str(request.created_by),
                query=json.loads(request.query_json),matched_count=request.matched_count,
                generation_count=request.generation_count,created_at=request.created_at.isoformat())


def request_detail(db,request,cur):
    from .fewshot_service import public_job,get_job
    from .version_service import version_ready
    result=request_summary(db,request)
    result['candidates']=candidates(db,request,cur)
    jobs=[]
    for job in db.query(FewshotJob).filter_by(download_request_id=request.id).order_by(FewshotJob.id.desc()).all():
        get_job(db,job.id,cur)
        item=public_job(job)
        ds=db.get(Dataset,job.dataset_id) if job.dataset_id else None
        item['versions']=[]
        if ds and not ds.deleted_at:
            access.require_dataset(db,cur,ds)
            versions=db.query(DatasetVersion).filter_by(dataset_id=ds.id).order_by(DatasetVersion.version_no.desc()).all()
            item['versions']=[dict(id=v.id,version_no=v.version_no,stage=v.stage,status=v.status,data_count=v.data_count,ready=version_ready(db,v)) for v in versions]
            item['current_version_id']=ds.current_version_id
        jobs.append(item)
    result['jobs']=jobs
    return result
