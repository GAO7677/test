"""Condition input role/time ablation on the four established state-update protocols."""
import argparse,json,runpy,sys
from pathlib import Path
import common


def condition_input(backbone,kw,conditional_time=None):
    import torch
    updated=dict(kw)
    if backbone=='wan':
        x=kw['hidden_states'];ts=kw['timestep'].clone()
        assert ts.shape[1]%x.shape[2]==0
        stride=ts.shape[1]//x.shape[2];ts[:,:2*stride]=0
        assert torch.equal(ts[:,2*stride:],kw['timestep'][:,2*stride:])
        updated['timestep']=ts
        role=dict(context_time=0.,future_time=float(ts[0,-1]),role='wan_clean_time_no_independent_role')
    elif backbone=='nano':
        x=kw['vision_tokens'][0];t,h,w=kw['vision_token_shapes'][0];stride=h*w
        levels=kw['vision_timesteps']
        if levels.numel()==t*stride:levels=levels[2*stride:]
        assert levels.numel()==(t-2)*stride
        updated.update(vision_timesteps=levels,vision_noisy_frame_indexes=[torch.arange(2,t,device=x.device)])
        role=dict(context_time=None,future_time=float(levels[-1]),role='condition_no_context_time_embedding')
    else:
        x=kw['x_B_C_T_H_W'];mask=torch.zeros_like(kw['condition_video_input_mask_B_C_T_H_W']);mask[:,:,:2]=1
        ts=kw['timesteps_B_T']
        if ts.numel()==1:ts=ts.reshape(1,1).expand(x.shape[0],x.shape[2])
        ts=ts.clone();future=ts[:,2:].clone()
        if conditional_time>=0:ts[:,:2]=conditional_time
        else:ts[:,:2]=ts[:,-1:]
        assert torch.equal(ts[:,2:],future)
        updated.update(timesteps_B_T=ts,condition_video_input_mask_B_C_T_H_W=mask)
        role=dict(context_time=float(ts[0,0]),future_time=float(ts[0,-1]),role='condition_mask_1',mask_context=1,mask_future=0)
    # No latent values or scheduler/velocity policy are changed by this intervention.
    return updated,x,role


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('backbone',choices=['wan','nano','predict25'])
    p.add_argument('--protocol',choices=['initial_equal','initial_075','clean_rebuild','clean_free'],required=True)
    opt,rest=p.parse_known_args();original=common.arguments;finish_orig=common.finish;calls=[]
    def arguments(backbone):
        a,c,out=original(backbone)
        assert a.no_cache
        import sampling
        from sampling import digest
        from initial_only_hooks import WanInitialOnly,NanoInitialOnly,install_predict_initial
        from clean_first_hooks import WanCleanRebuild,WanCleanFree,NanoCleanRebuild,NanoCleanFree,install_predict_rebuild,install_predict_free
        kind=opt.protocol
        wb,nb,installer={'initial_equal':(WanInitialOnly,NanoInitialOnly,install_predict_initial),
                        'initial_075':(WanInitialOnly,NanoInitialOnly,install_predict_initial),
                        'clean_rebuild':(WanCleanRebuild,NanoCleanRebuild,install_predict_rebuild),
                        'clean_free':(WanCleanFree,NanoCleanFree,install_predict_free)}[kind]
        assert c['mode']==('noisy' if kind=='initial_075' else 'equal_sigma')
        def observer(conditional_time=None):
            def before(module,args,kw):
                assert not args
                changed,x,role=condition_input(backbone,kw,conditional_time)
                role.update(call=len(calls),context_sha256=digest(x[:,:,:2]),future_sha256=digest(x[:,:,2:]),latent_unchanged=True)
                calls.append(role)
                return args,changed
            return before
        def subclass(base):
            class Condition(base):
                output=out
                def __enter__(self):
                    super().__enter__()
                    self.role_handle=self.pipe.transformer.register_forward_pre_hook(observer(),with_kwargs=True)
                    return self
                def __exit__(self,*exc):
                    self.role_handle.remove();return super().__exit__(*exc)
            return Condition
        sampling.ContextNoise=subclass(wb);sampling.NanoContext=subclass(nb)
        def install(*args,**kwargs):
            installer(*args,**kwargs)
            from projects.generic_acn75.inference_runtime import BenchmarkModel
            init=BenchmarkModel.__init__
            def init_condition(self,*args,**kwargs):
                init(self,*args,**kwargs)
                self.net.register_forward_pre_hook(observer(float(self.config.conditional_frame_timestep)),with_kwargs=True)
            BenchmarkModel.__init__=init_condition
        sampling.install_predict=install
        c.update(experiment='condition_role_all_steps',base_protocol=kind,context_role='condition_all_steps',
                 role_time_policy='native_clean',state_update_policy='unchanged_from_base_protocol',
                 base_hook_trace_time='requested_before_role_override; actual model time in condition_role_audit.json')
        common.save(out/'config.json',c)
        return a,c,out
    def finish(a,c,out,audit):
        expected=len(audit['triplet']['trace']) if opt.backbone=='predict25' else c['steps']*(1 if opt.backbone=='wan' else 2)
        assert len(calls)==expected,(len(calls),expected)
        if opt.backbone!='wan':
            for left,right in zip(calls[::2],calls[1::2]):
                assert all(left[k]==right[k] for k in ['context_sha256','future_sha256','context_time','future_time','role'])
        common.save(out/'condition_role_audit.json',dict(status='PASS',backbone=opt.backbone,protocol=opt.protocol,calls=calls))
        finish_orig(a,c,out,audit)
    common.arguments=arguments;common.finish=finish
    script=Path(__file__).with_name(opt.backbone+'.py');sys.argv=[str(script),*rest]
    runpy.run_path(str(script),run_name='__main__')
if __name__=='__main__':main()
