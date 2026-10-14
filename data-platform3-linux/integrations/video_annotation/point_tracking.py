"""Point trajectories from explicit queries; never substitute per-frame box centres."""
import sys
from pathlib import Path

def track(m,frames,timestamps,width,height,progress):
    import torch,numpy as np
    from PIL import Image
    sys.path.insert(0,m['runtime']['point_repository'])
    from cotracker.predictor import CoTrackerOnlinePredictor
    model=CoTrackerOnlinePredictor(checkpoint=m['runtime']['point_checkpoint']).to('cuda').eval()
    # Match model resolution while decoding chunks; avoid retaining full-resolution clips on GPU.
    mh,mw=model.interp_shape
    queries=[];owners=[]
    for obj in m['prompts']:
        for p in obj['points']:
            queries.append([0,p['xy'][0]*(mw-1)/(width-1),p['xy'][1]*(mh-1)/(height-1)])
            owners.append((obj,p))
    query=torch.tensor([queries],dtype=torch.float32,device='cuda')
    def chunk(start,end):
        imgs=[np.asarray(Image.open(Path(frames)/f'{i:06d}.jpg').resize((mw,mh))).copy() for i in range(start,end)]
        return torch.from_numpy(np.stack(imgs)).permute(0,3,1,2)[None].float().to('cuda')
    with torch.inference_mode():
        first=chunk(0,min(len(timestamps),model.step*2))
        model(first,is_first_step=True,queries=query,grid_size=0,add_support_grid=True)
        del first
        tracks=visible=None
        for start in range(0,max(1,len(timestamps)-model.step),model.step):
            video=chunk(start,min(start+model.step*2,len(timestamps)))
            tracks,visible=model(video,add_support_grid=True)
            del video
            progress(min(95,round(min(start+model.step*2,len(timestamps))/len(timestamps)*95)))
        tracks=tracks[0,:len(timestamps)].cpu().numpy();visible=visible[0,:len(timestamps)].cpu().numpy()
    if len(tracks)!=len(timestamps):raise ValueError('点跟踪输出帧数与请求不一致')
    rows=[]
    for i,t in enumerate(timestamps):
        for j,(obj,p) in enumerate(owners):
            x=float(tracks[i,j,0])*(width-1)/(mw-1);y=float(tracks[i,j,1])*(height-1)/(mh-1)
            if i==0:x,y=p['xy']
            valid=bool(np.isfinite(x) and np.isfinite(y));shown=bool(visible[i,j] and valid and 0<=x<width and 0<=y<height) or i==0
            rows.append({'type':'video_point','track_id':obj['track_id'],'point_id':p['point_id'],'frame':m['start_frame']+i,'time':t,
                'xy':[round(x,3),round(y,3)] if valid else None,'visible':shown,'outside':not shown,'label':obj['label'],
                'trajectory_run':str(Path(m['output']).parent.name),'seed_frame':m['start_frame'],'seed_origin':p.get('origin','manual'),'source':'cotracker3','review_status':'pending'})
    return rows,'CoTracker3 online'
