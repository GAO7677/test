"""COCO-style uncompressed RLE, column-major, exact pixels (including holes)."""
import numpy as np

def encode(mask):
    a=np.asarray(mask,dtype=np.uint8)
    flat=a.ravel(order='F'); changes=np.flatnonzero(flat[1:] != flat[:-1])+1
    counts=np.diff(np.r_[0,changes,len(flat)]).tolist()
    if flat.size and flat[0]: counts.insert(0,0)
    return {'size':list(a.shape),'counts':counts}

def decode(rle):
    h,w=map(int,rle['size']);counts=rle['counts']
    if h<=0 or w<=0 or h*w>16777216 or not isinstance(counts,list) or any(type(n)!=int or n<0 for n in counts) or sum(counts)!=h*w:
        raise ValueError('Invalid mask RLE')
    return np.repeat(np.arange(len(counts))%2,counts).reshape((h,w),order='F').astype(bool)

def bounds(mask):
    y,x=np.nonzero(mask)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None


def sample_points(mask, limit=8):
    """Deterministic spread of actual foreground pixels, including nonconvex objects."""
    y,x=np.nonzero(mask)
    if not len(x):return []
    coords=np.column_stack((x,y))
    # Bound distance computation for large masks without sampling any background.
    coords=coords[np.linspace(0,len(coords)-1,min(len(coords),4096),dtype=int)]
    centre=coords.mean(axis=0);index=int(np.argmin(((coords-centre)**2).sum(axis=1)))
    chosen=[];dist=np.full(len(coords),np.inf)
    for _ in range(min(limit,len(coords))):
        chosen.append(coords[index].tolist())
        dist=np.minimum(dist,((coords-coords[index])**2).sum(axis=1));dist[index]=-1
        index=int(np.argmax(dist))
    return [{'point_id':i+1,'xy':xy,'origin':'mask_sample'} for i,xy in enumerate(chosen)]
