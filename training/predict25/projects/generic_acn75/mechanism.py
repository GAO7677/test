"""Context-only RF extension; the sampler/interpolation are NVIDIA's implementations."""
import torch

def pack_context(rf, clean, noise, sigma, context_frames, mode, scales, grid_config=None):
    assert mode in scales and 0 < context_frames < clean.shape[2]
    context_sigma = sigma * scales[mode]
    if mode == 'N75':
        g=grid_config or {'unshifted_max':1.,'unshifted_min':.001,'size':1000}
        grid = torch.linspace(g['unshifted_max'], g['unshifted_min'], g['size'], device=clean.device)
        grid = rf.shift * grid / (1 + (rf.shift - 1) * grid)
        context_sigma = grid[(grid[None] - context_sigma[:,None]).abs().argmin(1)]
    frame_sigma = sigma[:,None].expand(clean.shape[0],clean.shape[2]).clone()
    frame_sigma[:,:context_frames] = context_sigma[:,None]
    mixed, target = rf.get_interpolation(noise,clean,sigma)
    context_mixed,_ = rf.get_interpolation(noise[:,:,:context_frames],clean[:,:,:context_frames],context_sigma)
    mixed[:,:,:context_frames] = context_mixed
    mask = clean.new_zeros((clean.shape[0],1,clean.shape[2],clean.shape[3],clean.shape[4]))
    if mode == 'C':
        mask[:,:,:context_frames]=1
    return mixed,target,frame_sigma,mask
