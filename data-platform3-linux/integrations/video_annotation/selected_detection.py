"""YOLO boxes associated to selected SAM objects. Missing detections stay missing."""
import numpy as np

def iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)

class SelectedDetector:
    def __init__(self,path):
        import onnxruntime as ort
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(path,sess_options=opts,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0].name
        # Same checked class order as the platform detector, supplied by the worker.
    def predict(self,image,classes):
        import cv2
        h,w=image.shape[:2]
        blob=cv2.dnn.blobFromImage(image,1/255.,(640,640),swapRB=True,crop=False)
        pred=self.session.run(None,{self.input:blob})[0][0].T
        boxes=[];scores=[];ids=[]
        for row in pred:
            cid=int(np.argmax(row[4:]));score=float(row[4+cid])
            if score<.4:continue
            x,y,bw,bh=map(float,row[:4]);x1=max(0,(x-bw/2)*w/640);y1=max(0,(y-bh/2)*h/640);x2=min(w,(x+bw/2)*w/640);y2=min(h,(y+bh/2)*h/640)
            if x2<=x1 or y2<=y1:continue
            boxes.append([x1,y1,x2-x1,y2-y1]);scores.append(score);ids.append(cid)
        found=[]
        for cid in sorted(set(ids)):
            members=[i for i,c in enumerate(ids) if c==cid]
            for local in np.asarray(cv2.dnn.NMSBoxes([boxes[i] for i in members],[scores[i] for i in members],.4,.45)).flatten():
                i=members[int(local)];x,y,bw,bh=boxes[i];found.append({'xyxy':[x,y,x+bw,y+bh],'label':classes[cid],'confidence':round(scores[i],4)})
        return found

def associate(rows,detections):
    pairs=sorted([(iou(row['xyxy'],d['xyxy']),i,j) for i,row in enumerate(rows) if row.get('xyxy') for j,d in enumerate(detections) if row['label']==d['label']],reverse=True)
    matches={};used=set()
    for score,i,j in pairs:
        if score>=.15 and i not in matches and j not in used:matches[i]=j;used.add(j)
    for i,row in enumerate(rows):
        row.pop('xyxy',None);row.pop('mask',None);row['source']='yolov8n';row['association']='sam2.1';row['outside']=i not in matches
        if i in matches:row.update(detections[matches[i]])
        else:row['status']='not_detected'
    return rows
