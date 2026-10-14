"""Workflow contracts in a fresh isolated DB; worker is a CPU test double."""
import runpy,io,zipfile
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
ctx=runpy.run_path(str(Path(__file__).with_name('check_annotation_types.py')))
globals().update({k:v for k,v in ctx.items() if not k.startswith('__')})
from mask_codec import sample_points
from app.models import AnnotationLabel
from app.services.video_exports import exports
checks=[]
def check(name,condition=True):
 assert condition,name;checks.append(name);print('PASS '+name,flush=True)
for name in ('自定义车辆',' 自定义车辆 '):check('个人标签可保存',ca.post('/api/video-annotation/labels',json={'name':name}).status_code==200)
check('标签去重且去空格',ca.get('/api/video-annotation/labels').json()['items']==['自定义车辆'])
check('另一个用户没有共享标签',cb.get('/api/video-annotation/labels').json()['items']==[])
check('管理员标签库也不混入别人的标签',cadmin.get('/api/video-annotation/labels').json()['items']==[])
points=sample_points(mask,8)
check('自动8点互异且避开mask孔洞',len(points)==8 and len({tuple(p['xy']) for p in points})==8 and all(mask[p['xy'][1],p['xy'][0]] for p in points))
check('空mask不退化为框中心',sample_points(np.zeros_like(mask),8)==[])
workflow={**deepcopy(body),'workflow':'first_frame_auto','purpose':'annotation','output_modes':['box','mask','trajectory']}
workflow['prompts'][0]['label']='自定义车辆';workflow['prompts'][0]['points']=[]
with patch.object(va,'capability',return_value={'ready':True}),patch.object(va,'start'):
 for name,changes in [('空输出',{'output_modes':[]}),('未知输出',{'output_modes':['depth']}),('重复输出',{'output_modes':['mask','mask']}),('不是首帧',{'start_frame':1}),('手动模式后台传播',{'workflow':'manual'}),('多帧草稿',{'purpose':'seed','frame_count':2})]:
  check(name+'被拒绝',ca.post('/api/video-annotation',json={**workflow,**changes}).status_code==400)
 response=ca.post('/api/video-annotation',json=workflow);assert response.status_code==200,response.text
 job=response.json();saved=json.loads((O/'uploads/video_annotation_jobs'/str(job['id'])/'input.json').read_text())
 check('新流程自定义标签不受COCO80限制',saved['prompts'][0]['label']=='自定义车辆')
 check('新流程无点保持待mask采样，不补框中心',saved['prompts'][0]['points']==[])
 ca.post(f"/api/video-annotation/{job['id']}/cancel")
 seed={**workflow,'purpose':'seed','frame_count':1,'workflow':'manual'}
 response=ca.post('/api/video-annotation',json=seed);assert response.status_code==200,response.text
 tid=response.json()['id']
 check('草稿结果未完成不能读取',ca.get(f'/api/video-annotation/{tid}/result').status_code==409)
 check('草稿任务不混入最终任务列表',tid not in [a['id'] for a in ca.get('/api/video-annotation',params={'page_size':100}).json()['items']])
work=O/'uploads/video_annotation_jobs'/str(tid);(work/'expected_mask.json').write_text(json.dumps(encode(mask)))
with SessionLocal() as db:before=db.query(DatasetVersion).count();current=db.get(Dataset,did).current_version_id
real_popen=ctx['real_popen']
def popen(cmd,**kw):return real_popen([sys.executable,str(testworker),cmd[-1],'ok'],**kw)
with patch.object(va,'runtime',return_value=runtime),patch.object(va.subprocess,'Popen',side_effect=popen):va.execute(tid)
check('草稿mask可读取',ca.get(f'/api/video-annotation/{tid}/result').status_code==200)
check('他人不可读取草稿mask',cb.get(f'/api/video-annotation/{tid}/result').status_code==403)
with SessionLocal() as db:check('草稿无新版本且不改变当前版本',db.query(DatasetVersion).count()==before and db.get(Dataset,did).current_version_id==current)
label={'fps':2,'export_types':['box','mask','trajectory'],'workflow':'manual','annotations':[
 {'type':'video_box','frame':0,'time':0,'track_id':1,'label':'自定义车辆','xyxy':[6,4,30,20],'mask':encode(mask)},
 {'type':'video_point','frame':0,'time':0,'track_id':1,'point_id':1,'xy':[7,5],'label':'自定义车辆','visible':True}]}
response=ca.post('/api/manual-label',json={'dataset_id':did,'input_version_id':vid,'label_type':'video','labels':[{'file':'test.mp4','label':label}]})
assert response.status_code==200,response.text
newvid=response.json()['output_version_id']
params={'dataset_id':did,'input_version_id':newvid,'file':'test.mp4'}
z=ca.get('/api/video-annotation/exports',params=params);assert z.status_code==200,z.text
with zipfile.ZipFile(io.BytesIO(z.content)) as archive:
 check('一次下载三个独立标注文件',len(archive.namelist())==3)
 boxes=json.loads(archive.read('annotation_exports/test_box.json'))['annotations']
 masks=json.loads(archive.read('annotation_exports/test_mask.json'))['annotations']
 tracks=json.loads(archive.read('annotation_exports/test_trajectory.json'))['annotations']
 check('框mask轨迹各自无冗余字段',len(boxes)==len(masks)==len(tracks)==1 and 'mask' not in boxes[0] and 'xyxy' not in masks[0] and 'mask' not in tracks[0])
z=ca.get(f'/api/datasets/{did}/versions/{newvid}/download',params={'include':'labels'})
with zipfile.ZipFile(io.BytesIO(z.content)) as archive:check('数据集标签下载也包含所选导出文件',len([n for n in archive.namelist() if n.startswith('annotation_exports/')])==3 and 'test.mp4' not in archive.namelist())
listing=ca.get(f'/api/datasets/{did}/versions/{newvid}/files').json();check('导出文件不污染数据文件清单',listing['total']==1)
check('只选框只导出框文件',list(exports({**label,'export_types':['box']},'test'))==['annotation_exports/test_box.json'])
merged=va.merge_label({'annotations':label['annotations']}, {'fps':2,'start_frame':0,'end_frame':0,'frame_times':[],'annotations':label['annotations'],'model':'test','annotation_type':'mask','export_types':['box','mask','trajectory']},workflow)
check('组合生成替换同一物体区间，无重复轨迹',len(merged['annotations'])==2)
partial=va.merge_label({'annotations':label['annotations']}, {'fps':2,'start_frame':0,'end_frame':0,'frame_times':[],'annotations':[label['annotations'][0]],'model':'test','annotation_type':'mask','export_types':['mask']}, {**workflow,'output_modes':['mask']})
check('只重跑mask保留未请求的旧轨迹',any(a['type']=='video_point' for a in partial['annotations']))
# Background batches freeze all confirmed seeds; edited boxes do not clip point prompts.
prepared=[]
for obj in (1,2):
 prepared.append({'type':'video_box','track_id':obj,'frame':0,'time':0,'label':f'object{obj}',
  'xyxy':[18,12,24,18],'mask':encode(mask),'box_edited':True,'tracking_points':[{'point_id':1,'xy':[7,5],'origin':'manual'}],
  'source':'manual','generation':{'status':'prepared'}})
batch={**deepcopy(workflow),'base_label':{'annotations':deepcopy(prepared),'manual_annotations':deepcopy(prepared)},
 'prompts':[{**deepcopy(a),'points':a['tracking_points']} for a in prepared]}
with patch.object(va,'capability',return_value={'ready':True}),patch.object(va,'start'):
 r=ca.post('/api/video-annotation',json=batch);assert r.status_code==200,r.text
 batch_id=r.json()['id']
 check('多物体一次提交且允许点在修改后的结果框之外',r.status_code==200)
 restored=ca.get(f'/api/video-annotation/{batch_id}/input').json()
 check('关闭页面后可回读全部首帧框mask点',restored['base_label']==batch['base_label'])
 check('任务输入不暴露运行配置或路径',set(restored)=={'base_label','output_modes','frame_count'})
 check('其他用户不能读取首帧输入',cb.get(f'/api/video-annotation/{batch_id}/input').status_code==403)
 check('管理员可读取首帧输入',cadmin.get(f'/api/video-annotation/{batch_id}/input').status_code==200)
 invalid=deepcopy(batch);invalid['prompts'][0]['points'][0]['xy']=[64,5]
 check('独立选点仍拒绝画面越界',ca.post('/api/video-annotation',json=invalid).status_code==400)
 check('草稿任务不提供重试准备入口',ca.get(f'/api/video-annotation/{tid}/input').status_code==400)
 ca.post(f'/api/video-annotation/{batch_id}/cancel')
 check('取消后仍能恢复原始准备结果',ca.get(f'/api/video-annotation/{batch_id}/input').json()==restored)
result={'fps':2,'start_frame':0,'end_frame':1,'frame_times':[], 'model':'CPU test', 'annotations':[
 {**deepcopy(a),'frame':frame,'source':'sam2.1','xyxy':[6,4,30,20]} for a in prepared for frame in (0,1)]}
merged=va.merge_label(batch['base_label'],result,batch)
check('批量合并保留每个首帧的独立框和修正mask',all(a['xyxy']==[18,12,24,18] and a['mask']==encode(mask) for a in merged['annotations'] if a['frame']==0))
check('后续帧保留SAM外接框',all(a['xyxy']==[6,4,30,20] for a in merged['annotations'] if a['frame']==1))
check('全部物体完成状态同时持久化',len(merged['manual_annotations'])==2 and all(a['generation']['status']=='completed' for a in merged['manual_annotations']))
(O/'workflow_results.json').write_text(json.dumps({'passed':True,'checks':checks,'worker':'CPU test double'},ensure_ascii=False,indent=2))
print('Workflow CPU checks passed',flush=True)
