import torch
from projects.generic_acn75.mechanism import pack_context
from cosmos_predict2._src.predict2.schedulers.rectified_flow import RectifiedFlow

def test_context_protocol():
    rf=RectifiedFlow(None,shift=5)
    clean=torch.arange(2*3*5*2*2,dtype=torch.float32).reshape(2,3,5,2,2)/100
    noise=torch.ones_like(clean)*-2
    sigmas=torch.tensor([.2,.8]);scales={'A':1.,'C':0.,'N75':.75}
    reference,target=rf.get_interpolation(noise,clean,sigmas)
    for mode in scales:
        x,v,t,mask=pack_context(rf,clean,noise,sigmas,2,mode,scales)
        assert torch.equal(x[:,:,2:],reference[:,:,2:])
        assert torch.equal(v,target) and torch.equal(v,noise-clean)
        assert torch.allclose(x,noise*t[:,None,:,None,None]+clean*(1-t[:,None,:,None,None]))
        if mode=='A':
            assert torch.equal(x,reference) and mask.count_nonzero()==0
        elif mode=='C':
            assert torch.equal(x[:,:,:2],clean[:,:,:2]) and t[:,:2].count_nonzero()==0
            assert torch.all(mask[:,:,:2]==1) and mask[:,:,2:].count_nonzero()==0
        else:
            assert torch.all(t[:,0]<sigmas) and torch.all(t[:,0]>0) and mask.count_nonzero()==0
            assert torch.max((t[:,0]-sigmas*.75).abs())<.005
    print('A/C/N75 official RF context and future-target invariants PASS')
if __name__=='__main__':test_context_protocol()
