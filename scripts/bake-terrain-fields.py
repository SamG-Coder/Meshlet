"""Write the render heightfield and material maps from simulated inventories."""
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from PIL import Image

def bake(out, rock, soils, eroded, deposited, q, plane, hard, n, world, meta):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    h=rock+soils.sum(axis=0);dx=world/(n-1)
    h.astype('<f4').tofile(out/'height.bin')
    pn=min(n,1024);sample=lambda a:zoom(a.reshape(n,n),pn/n,order=1).astype(np.float32)
    fields=np.zeros((pn,pn,11),np.float32);fields[:,:,0]=sample(h)
    for g in range(3):fields[:,:,1+g]=sample(soils[g])
    fields[:,:,5]=(sample(soils[2])*.12).clip(0,1)
    fields[:,:,6]=sample(eroded);fields[:,:,7]=sample(deposited);fields[:,:,8]=sample(q)
    fields[:,:,9]=sample(rock);fields[:,:,10]=sample(plane)
    fields.tofile(out/'preview-fields.bin')
    color=np.empty((n,n,3),np.uint8);normal=np.empty_like(color);orm=np.empty_like(color);biome=np.empty_like(color)
    for z in range(0,n,128):
        end=min(n,z+128);sl=slice(z*n,end*n);shape=(end-z,n)
        a=h.reshape(n,n)[max(0,z-1):min(n,end+1)];gy,gx=np.gradient(a,dx);off=1 if z else 0
        gx=gx[off:off+end-z];gy=gy[off:off+end-z];length=np.sqrt(1+gx*gx+gy*gy)
        normal[z:end]=np.clip(np.stack([-gx/length,gy/length,1/length],-1)*127.5+127.5,0,255).astype(np.uint8)
        cover=soils[:,sl].sum(axis=0).reshape(shape);exposed=np.exp(-cover/.25);hr=hard[sl].reshape(shape)
        fines=1-np.exp(-soils[2,sl].reshape(shape)*2)
        tone=.60+.07*hr*exposed+.025*fines
        for c,tint in enumerate((1.025,1.,.96)):color[z:end,:,c]=np.clip(tone*tint*255,0,255)
        cavity=np.maximum(0,gaussian_filter(a,1)[off:off+end-z]-a[off:off+end-z])
        orm[z:end,:,0]=np.clip(np.exp(-cavity*.16)*255,145,255);orm[z:end,:,1]=232;orm[z:end,:,2]=0
        biome[z:end,:,0]=np.clip(hr*exposed*255,0,255)
        biome[z:end,:,1]=np.clip((1-np.exp(-soils[0,sl].reshape(shape)*3))*255,0,255)
        biome[z:end,:,2]=np.clip(fines*255,0,255)
    Image.fromarray(color).save(out/'albedo.jpg',quality=94,subsampling=0)
    Image.fromarray(normal).save(out/'terrain-normal.png')
    Image.fromarray(orm).save(out/'orm.png');Image.fromarray(biome).save(out/'geology.png')
    a=fields[:,:,0];gz,gx=np.gradient(a,world/(pn-1));contrast=a-gaussian_filter(a,10)
    score=contrast-np.hypot(gx,gz)*50;score[:pn//6]=score[-pn//6:]=-1e6;score[:,:pn//6]=score[:,-pn//6:]=-1e6
    iz,ix=np.unravel_index(score.argmax(),score.shape)
    meta.update({'size':n,'previewSize':pn,'worldSize':world,'heightMin':float(h.min()),'heightMax':float(h.max()),'drainageSize':n,'logicalRenderGrid':524288,'logicalTriangles':2*524288**2,'spawn':{'x':float((ix/(pn-1)-.5)*world),'z':float((iz/(pn-1)-.5)*world),'height':float(a[iz,ix])},'materialMask':'R=hard exposed bedrock, G=deposited gravel cover, B=deposited fines'})
    (out/'simulation.json').write_text(json.dumps(meta,indent=2))
    return meta
