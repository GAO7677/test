// COCO uncompressed RLE: column-major counts, starts with background.
export function maskCanvas(mask) {
  const [h,w]=mask.size
  if (!(h>0 && w>0 && h*w<=16777216)) throw Error('Mask尺寸无效')
  const c=document.createElement('canvas');c.width=w;c.height=h
  const ctx=c.getContext('2d'),im=ctx.createImageData(w,h)
  let pos=0,on=false
  for (const n of mask.counts) {
    if (!Number.isInteger(n)||n<0||pos+n>w*h) throw Error('Mask计数无效')
    if(on) for(let k=pos;k<pos+n;k++) {const at=((k%h)*w+Math.floor(k/h))*4;im.data.set([52,211,153,100],at)}
    pos+=n;on=!on
  }
  if(pos!==w*h) throw Error('Mask长度无效')
  ctx.putImageData(im,0,0);return c
}
export function canvasMask(c) {
  const w=c.width,h=c.height,a=c.getContext('2d').getImageData(0,0,w,h).data
  const counts=[];let old=0,run=0,minX=w,minY=h,maxX=-1,maxY=-1
  for(let x=0;x<w;x++)for(let y=0;y<h;y++) {
    const bit=a[(y*w+x)*4+3]>0?1:0
    if(bit===old)run++;else{counts.push(run);run=1;old=bit}
    if(bit){minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x);maxY=Math.max(maxY,y)}
  }
  counts.push(run)
  return {mask:{size:[h,w],counts},xyxy:maxX>=0?[minX,minY,maxX+1,maxY+1]:null}
}

// Evenly spread samples on actual foreground pixels, never holes/background.
export function sampleMaskPoints(mask,limit=8,box=null){
  const c=maskCanvas(mask),data=c.getContext('2d').getImageData(0,0,c.width,c.height).data
  if(box)for(let y=0;y<c.height;y++)for(let x=0;x<c.width;x++)if(x<box[0]||x>=box[2]||y<box[1]||y>=box[3])data[(y*c.width+x)*4+3]=0
  let count=0
  for(let i=3;i<data.length;i+=4)if(data[i])count++
  if(!count)return []
  const stride=Math.max(1,Math.ceil(count/4096)),coords=[]
  let n=0
  for(let i=3;i<data.length;i+=4)if(data[i] && n++%stride===0){const pixel=(i-3)/4;coords.push([pixel%c.width,Math.floor(pixel/c.width)])}
  const center=coords.reduce((a,p)=>[a[0]+p[0]/coords.length,a[1]+p[1]/coords.length],[0,0])
  const distance=(a,b)=>(a[0]-b[0])**2+(a[1]-b[1])**2
  let index=coords.reduce((best,p,i)=>distance(p,center)<distance(coords[best],center)?i:best,0)
  const dist=coords.map(()=>Infinity),points=[]
  for(let i=0;i<Math.min(limit,coords.length);i++){
    const xy=coords[index];points.push({point_id:i+1,xy,origin:'mask_sample'})
    for(let j=0;j<coords.length;j++)dist[j]=Math.min(dist[j],distance(coords[j],xy))
    dist[index]=-1;index=dist.reduce((best,d,j)=>d>dist[best]?j:best,0)
  }
  return points
}


export const hasMaskPixels=mask=>!!mask?.counts?.some((n,i)=>i%2===1 && n>0)

export function paintMaskStroke(canvas,from,to,diameter,erase=false){
  const ctx=canvas.getContext('2d')
  ctx.globalCompositeOperation=erase?'destination-out':'source-over'
  ctx.strokeStyle='rgba(52,211,153,1)';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=diameter;ctx.lineCap='round'
  ctx.beginPath()
  if(from){ctx.moveTo(...from);ctx.lineTo(...to);ctx.stroke()}
  else{ctx.arc(...to,diameter/2,0,2*Math.PI);ctx.fill()}
}

export function pointsInMask(canvas,points){
  const {width,height}=canvas,data=canvas.getContext('2d').getImageData(0,0,width,height).data
  return points.filter(p=>{
    const [x,y]=p.xy.map(Math.floor)
    return x>=0 && y>=0 && x<width && y<height && data[(y*width+x)*4+3]>0
  })
}
