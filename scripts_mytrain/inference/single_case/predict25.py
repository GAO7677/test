"""Single-case native Predict2.5 inference with explicit input and sampling settings."""
from common import arguments,finish,save,HIST

def main():
    a,c,out=arguments('predict25')
    import os,sys,types,hashlib
    from pathlib import Path
    native=Path(c['model_path']);os.chdir(native)
    sys.path[:0]=[str(native),str(native/'packages/cosmos-oss')]
    import projects
    package=types.ModuleType('projects.generic_acn75');package.__path__=[str(HIST/'frozen/predict25')]
    sys.modules['projects.generic_acn75']=package;projects.generic_acn75=package
    import torch
    from projects.generic_acn75.inference_runtime import BenchmarkInference
    from projects.generic_acn75.inference_io import read_prefix,video_info
    from cosmos_predict2._src.predict2.inference.video2world import resize_input
    from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
    from sampling import install_predict
    torch.cuda.set_device(0);torch.set_grad_enabled(False)
    install_predict(c['mode'],'N75',c['context_seed'],str(out/'triplet_audit.json'),c.get('context_cache'),c['training'])
    pipe=BenchmarkInference(c['settings'],c['training'],Path(c['adapter']),c['fps'],c['seed'])
    save(out/'weight_load_audit.json',pipe.weight_audit)
    # Archive sampler output for decoder-only repeatability diagnosis.
    original_decode=pipe.model.decode
    def decode(sample,*args,**kwargs):
        torch.save(sample.detach().cpu(),out/'final_latent.pt')
        return original_decode(sample,*args,**kwargs)
    pipe.model.decode=decode
    rgb=read_prefix(c['video'],8)
    video=torch.from_numpy(rgb.copy()).permute(0,3,1,2)
    video=resize_input(video,[512,896]);pad=video[-1:].expand(49-len(video),-1,-1,-1)
    video=torch.cat([video,pad]).permute(1,0,2,3).unsqueeze(0).cuda()
    from cosmos_predict2.config import InferenceArguments
    # Keep the native input-frame default used by the historical official entry.
    sample=InferenceArguments(name='single',inference_type='video2world',input_path=Path(c['video']),prompt=c['prompt'],resolution='512,896',num_output_frames=49)
    generated=pipe.generate_vid2world(prompt=c['prompt'],input_path=video,guidance=c['guidance'],seed=c['seed'],negative_prompt=c['negative_prompt'],num_steps=c['steps'],num_video_frames=49,num_latent_conditional_frames=sample.num_input_frames,resolution='512,896')
    assert list(generated.shape)==[1,3,49,512,896] and torch.isfinite(generated).all()
    save_img_or_video((1+generated[0])/2,str(out/'video'),fps=pipe.fps)
    import json
    audit=dict(cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,decoded_tensor_sha256=hashlib.sha256(generated.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest(),input_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),context=pipe.model.last_context_audit,triplet=json.loads((out/'triplet_audit.json').read_text()))
    finish(a,c,out,audit)

if __name__=='__main__':main()
