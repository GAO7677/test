"""One GPU-7-only inference subprocess; no web server, no training."""
import argparse,ctypes,os,signal,json,tempfile,sys,time
from pathlib import Path

def atomic_json(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False));tmp.replace(p)

def main():
    # Die with the web parent (Linux); never leave inference after a web restart.
    parent=int(os.environ.get('DP_VIDEO_PARENT_PID',os.getppid()));ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    if os.getppid()!=parent:raise RuntimeError('Parent exited')
    from mask_codec import encode,decode,bounds
    a=argparse.ArgumentParser();a.add_argument('manifest');args=a.parse_args()
    m=json.loads(Path(args.manifest).read_text());cfg=m['runtime'];out=Path(m['output'])
    sys.path.insert(0,cfg['repository'])
    import torch,numpy as np,av
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor
    assert os.environ['CUDA_VISIBLE_DEVICES']==cfg['gpu_uuid'] and cfg['gpu_uuid'].startswith('GPU-')
    if not torch.cuda.is_available():raise RuntimeError('GPU7 CUDA不可用')
    free,_=torch.cuda.mem_get_info()
    if free < int(cfg.get('min_free_gib',6))*1024**3:raise RuntimeError('GPU7空闲显存不足，请稍后重试；不会切换其他GPU')
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='frames-',dir=out.parent) as td:
        timestamps=[]; width=height=None; fps=None
        with av.open(m['video']) as video:
            stream=video.streams.video[0];fps=float(stream.average_rate or 0)
            if fps<=0:raise ValueError('无法读取视频帧率')
            # Sequential decode gives exact frame ordinals and actual presentation timestamps.
            origin=None
            for idx,frame in enumerate(video.decode(stream)):
                if origin is None:origin=float(frame.time or 0)
                if idx<m['start_frame']:continue
                if len(timestamps)>=m['frame_count']:break
                if frame.time is None:raise ValueError('视频缺少时间戳，无法可靠定位帧')
                timestamps.append(float(frame.time)-origin)
                image=frame.to_image();width,height=image.size
                if width*height>16777216:raise ValueError('视频分辨率超过1600万像素限制')
                image.save(Path(td)/f'{len(timestamps)-1:06d}.jpg',quality=95)
        if len(timestamps)!=m['frame_count']:raise ValueError('选择的帧区间超出视频实际长度，请缩短区间')
        if any(abs(t-(m['start_frame']+i)/fps)>0.25/fps for i,t in enumerate(timestamps)):
            raise ValueError('当前视频为可变帧率或时间戳不规则，请转换为固定帧率视频后标注，避免框与画面错位')
        if m['output_mode']=='trajectory' and not m.get('workflow'):
            from point_tracking import track
            output,model_name=track(m,td,timestamps,width,height,lambda p:atomic_json(m['progress'],{'progress':p}))
            atomic_json(out,{'annotations':output,'fps':fps,'width':width,'height':height,
                'frame_times':[{'frame':m['start_frame']+i,'time':t} for i,t in enumerate(timestamps)],
                'start_frame':m['start_frame'],'end_frame':m['start_frame']+len(timestamps)-1,
                'annotation_type':'trajectory','model':model_name,'review_status':'pending'})
            print('Point annotation inference completed',flush=True)
            return
        detector=None
        if m['output_mode']=='box' and not m.get('workflow'):
            from selected_detection import SelectedDetector,associate
            import cv2
            detector=SelectedDetector(cfg['detector_path'])
        model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',cfg['checkpoint'],device='cuda',apply_postprocessing=False)
        output=[];meta={p['track_id']:p for p in m['prompts']}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            state=model.init_state(td,offload_video_to_cpu=True,offload_state_to_cpu=True)
            for p in m['prompts']:
                if p.get('mask'):
                    mask=decode(p['mask'])
                    if mask.shape!=(height,width):raise ValueError('提示mask尺寸与原视频不同')
                    model.add_new_mask(state,frame_idx=0,obj_id=p['track_id'],mask=mask)
                else:
                    box=np.asarray(p['xyxy'],dtype=np.float32)
                    if not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):raise ValueError('提示框超出视频尺寸')
                    model.add_new_points_or_box(state,frame_idx=0,obj_id=p['track_id'],box=box)
            for i,ids,logits in model.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=len(timestamps)-1):
                if i>=len(timestamps):break
                frame_rows=[]
                for j,obj in enumerate(ids):
                    mask=(logits[j,0]>0).cpu().numpy();box=bounds(mask)
                    row={'type':'video_box','track_id':int(obj),'frame':m['start_frame']+i,'time':timestamps[i],
                         'label':meta[obj]['label'],'source':'sam2.1','review_status':'pending','outside':box is None}
                    if box:
                        row['xyxy']=box
                        if m['output_mode']=='mask':row['mask']=encode(mask)
                        if m.get('workflow') and i==0:
                            from mask_codec import sample_points
                            prompt=meta[obj]
                            seed_mask=decode(prompt['mask']) if prompt.get('mask') else mask
                            points=prompt.get('points') or sample_points(seed_mask,prompt.get('point_limit',8))
                            if not points and 'trajectory' in m['output_modes']:raise ValueError('SAM未得到有效物体mask，请重画框')
                            prompt['points']=points
                            row.update(tracking_points=points,point_limit=prompt.get('point_limit',8))
                    frame_rows.append(row)
                if detector:frame_rows=associate(frame_rows,detector.predict(cv2.imread(str(Path(td)/f'{i:06d}.jpg')),m['classes']))
                output.extend(frame_rows)
                atomic_json(m['progress'],{'progress':min(95,round((i+1)/len(timestamps)*(50 if m.get('workflow') and 'trajectory' in m.get('output_modes',[]) and m.get('purpose')!='seed' else 95)))})
        if m.get('workflow') and m.get('purpose')!='seed' and 'trajectory' in m['output_modes']:
            for prompt in m['prompts']:
                if not prompt.get('points'):raise ValueError('物体mask为空，无法采样轨迹点，请重画框')
            del state,model,logits
            import gc
            gc.collect();torch.cuda.empty_cache()
            from point_tracking import track
            points,_=track(m,td,timestamps,width,height,lambda p:atomic_json(m['progress'],{'progress':50+round(p*.45)}))
            output.extend(points)
        atomic_json(out,{'annotations':output,'fps':fps,'width':width,'height':height,
                         **({'export_types':m['output_modes'],'workflow':m['workflow']} if m.get('workflow') else {}),
                         'frame_times':[{'frame':m['start_frame']+i,'time':t} for i,t in enumerate(timestamps)],
                         'start_frame':m['start_frame'],'end_frame':m['start_frame']+len(timestamps)-1,
                         'annotation_type':m['output_mode'],'model':('SAM 2.1 + CoTracker3' if m.get('workflow') and 'trajectory' in m['output_modes'] and m.get('purpose')!='seed' else 'SAM 2.1 Hiera Tiny') if not detector else 'YOLOv8n + SAM 2.1 association','review_status':'pending'})
    print('Video annotation inference completed',flush=True)

if __name__=='__main__': main()
