"""CPU-only matching/merge regressions. Run with DP_CONFIG_PATH set to test config."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'backend'),str(R/'integrations/video_annotation')]
from selected_detection import associate
from app.services.video_annotation import merge_label
from copy import deepcopy
rows=[{'track_id':1,'label':'person','xyxy':[0,0,10,10]},{'track_id':2,'label':'person','xyxy':[2,2,12,12]},{'track_id':3,'label':'car','xyxy':[0,0,10,10]}]
out=associate(deepcopy(rows),[{'label':'person','xyxy':[0,0,10,10],'confidence':.9}])
assert sum(not a['outside'] for a in out)==1
assert all('xyxy' not in a and a['status']=='not_detected' for a in out if a['outside'])
print('PASS one detection assigned to at most one selected object; wrong class/missing stay missing')
base={'annotations':[{'type':'video_box','frame':0,'track_id':1,'source':'yolov8n','xyxy':[1,1,9,9]},{'type':'video_mask','frame':0,'track_id':1,'mask':{'counts':[1]},'source':'sam2.1'},{'type':'video_point','frame':0,'track_id':1,'point_id':1,'xy':[3,3]}]}
res={'annotation_type':'mask','fps':2,'annotations':[{'type':'video_box','frame':0,'track_id':1,'mask':{'counts':[2]},'xyxy':[2,2,8,8],'source':'sam2.1'}],'frame_times':[{'frame':0,'time':0}],'start_frame':0,'end_frame':0,'model':'test'}
merged=merge_label(base,res,{'prompts':[{'track_id':1}]})
assert len(merged['annotations'])==3 and any(a.get('source')=='yolov8n' for a in merged['annotations'])
assert [a['mask']['counts'] for a in merged['annotations'] if a.get('mask')]==[[2]]
print('PASS regenerate mask replaces only mask; detection box and point trajectories preserved')
