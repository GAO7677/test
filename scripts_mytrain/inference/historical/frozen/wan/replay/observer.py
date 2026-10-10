"""Process-local, opt-in observer. No kernel substitution or model output edits.

Extraction placement inspired by PUSA wan_video_pusa.py, commit
8a8c4f523a0cadb94ab7c9f1e0620a4847b92991 (Apache-2.0).
Token frame-mass estimator below is an independent implementation.
"""
import contextlib
import numpy as np
import torch

def query_indices(grid,seed=1729):
 t,h,w=grid
 if h<8 or w<16:raise ValueError('128-query nested spatial strata require grid >=8x16')
 rng=np.random.default_rng(seed);fine=[]
 for ys in np.array_split(np.arange(h),8):
  for xs in np.array_split(np.arange(w),16):fine.append(int(rng.choice(ys)*w+rng.choice(xs)))
 coarse=[]
 for y in range(4):
  for x in range(8):
   dy,dx=divmod(int(rng.integers(4)),2);coarse.append((2*y+dy)*16+2*x+dx)
 fine=np.array(fine);small=fine[coarse]
 return {32:np.stack([small+i*h*w for i in range(t)]),128:np.stack([fine+i*h*w for i in range(t)])}

@torch.no_grad()
def frame_mass(q,k,queries,key_frames,frames,mask=None,scale=None,head_chunk=4,query_chunk=64):
 # q/k: [1,N,H,D]; queries: [frame,sampled spatial position].
 assert q.shape[0]==k.shape[0]==1
 heads=q.shape[2];ids=queries.flatten().to(q.device);kf=key_frames.to(q.device)
 scale=q.shape[-1]**-.5 if scale is None else scale
 result=torch.zeros(heads,len(ids),frames,device=q.device,dtype=torch.float32)
 allowed=kf>=0
 for hs in range(0,heads,head_chunk):
  kk=k[0,:,hs:hs+head_chunk].detach().permute(1,2,0).float()
  for begin in range(0,len(ids),query_chunk):
   qi=ids[begin:begin+query_chunk]
   qq=q[0,qi,hs:hs+head_chunk].detach().permute(1,0,2).float()
   logits=torch.bmm(qq,kk)*scale
   logits.masked_fill_(~allowed[None,None,:],float('-inf'))
   if mask is not None:
    full=mask.expand(1,heads,q.shape[1],k.shape[1])[0,hs:hs+head_chunk][:,qi,:]
    if full.dtype==torch.bool:logits.masked_fill_(~full,float('-inf'))
    else:logits=logits+full.float()
   prob=logits.softmax(-1)
   if not torch.isfinite(prob).all():raise ValueError('No valid key or invalid logits')
   mass=torch.zeros(*prob.shape[:-1],frames,device=q.device)
   mass.scatter_add_(-1,kf.clamp_min(0).expand_as(prob),prob)
   result[hs:hs+head_chunk,begin:begin+len(qi)]=mass
 return result.reshape(heads,frames,queries.shape[1],frames).mean(2)

@torch.no_grad()
def pooled_map(q,k,grid):
 t,h,w=grid;heads=q.shape[2];d=q.shape[3]
 qq=q[0].detach().float().reshape(t,h*w,heads,d).mean(1).permute(1,0,2)
 kk=k[0].detach().float().reshape(t,h*w,heads,d).mean(1).permute(1,2,0)
 return (qq@kk*d**-.5).softmax(-1)

class Observer:
 def __init__(self,model,grid,nquery=32,blocks=None,indices=None):
  self.model=model;self.grid=tuple(grid);self.nquery=nquery;self.blocks=blocks;self.active=None;self.arrays={};self.metadata={}
  self.indices=indices if indices is not None else query_indices(grid)[nquery]
 def __enter__(self):
  import fastvideo.models.wan_v2v.model_wan_v2v as module
  self.module=module;self.original=module.dispatch_attention_fn;self.processors=[]
  owner=self
  class Tagged:
   def __init__(self,original,index):self.original=original;self.index=index
   def __call__(self,*args,**kwargs):
    previous=owner.active;owner.active=self.index
    try:return self.original(*args,**kwargs)
    finally:owner.active=previous
  for i,block in enumerate(self.model.blocks):
   self.processors.append(block.attn1.processor);block.attn1.processor=Tagged(block.attn1.processor,i)
  def dispatch(q,k,v,*args,**kwargs):
   if owner.active is not None and (owner.blocks is None or owner.active in owner.blocks):
    if args or kwargs.get('is_causal',False) or kwargs.get('dropout_p',0):raise ValueError('Unsupported attention options')
    t,h,w=owner.grid;n=t*h*w
    if q.shape[1]!=n or k.shape[1]!=n:raise ValueError('Not a complete full sequence')
    if kwargs.get('attn_mask') is not None:raise ValueError('Real run has unexpected mask; pooled proxy not defined for it')
    old=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    try:
     with torch.autocast(device_type=q.device.type,enabled=False):
      mass=frame_mass(q,k,torch.as_tensor(owner.indices,device=q.device),torch.arange(n,device=q.device)//(h*w),t,scale=kwargs.get('scale'))
      proxy=pooled_map(q,k,owner.grid)
     assert float((mass.sum(-1)-1).abs().max())<2e-5
     owner.arrays[owner.active]=(mass.cpu().numpy(),proxy.cpu().numpy())
     owner.metadata[owner.active]=dict(q_shape=list(q.shape),k_shape=list(k.shape),valid_keys=n,context_keys=2*h*w,context_key_fraction=2/t,scale=kwargs.get('scale') or q.shape[-1]**-.5,mask=None,bias=None,causal=False,backend=str(kwargs.get('backend')),dtype=str(q.dtype))
    finally:torch.backends.cuda.matmul.allow_tf32=old
   return owner.original(q,k,v,*args,**kwargs)
  module.dispatch_attention_fn=dispatch
  return self
 def __exit__(self,*exc):
  self.module.dispatch_attention_fn=self.original
  for block,processor in zip(self.model.blocks,self.processors):block.attn1.processor=processor

def js_distance(p,q):
 p=np.asarray(p,dtype=np.float64);q=np.asarray(q,dtype=np.float64);m=(p+q)/2
 kl=lambda a:np.sum(np.where(a>0,a*(np.log(np.maximum(a,1e-300))-np.log(np.maximum(m,1e-300))),0),axis=-1)
 return np.sqrt(np.maximum((kl(p)+kl(q))/2,0))
