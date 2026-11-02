"""Bake animated impact normals and coverage for terrain collision decals."""
from pathlib import Path
import json
import numpy as np

out=Path('public/assets/rain-impacts');out.mkdir(parents=True,exist_ok=True)
n=128;frames=32;diameter=.8
z,x=np.mgrid[:n,:n].astype(np.float32);x=(x+.5)/n*diameter-diameter/2;z=(z+.5)/n*diameter-diameter/2
r=np.hypot(x,z);theta=np.arctan2(z,x);images=[]
for i in range(frames):
    t=i/(frames-1);radius=.025+t*.31
    ripple=np.exp(-((r-radius)/(.013+t*.015))**2)*np.exp(-t*3.8)*.006
    crown=np.exp(-((r-(.03+t*.08))/.012)**2)*np.maximum(0,np.sin(theta*11+np.sin(theta*4)))**2*np.exp(-t*14)*.016
    h=ripple+crown;gz,gx=np.gradient(h,diameter/n);normal=np.stack([-gx,np.ones_like(gx),-gz],-1);normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
    opacity=np.clip((ripple+crown)/.008,0,1)*np.clip(1-r/(diameter*.5),0,1)
    image=np.stack([normal[:,:,0]*.5+.5,normal[:,:,2]*.5+.5,np.clip(crown/.01,0,1),opacity],-1)
    images.append(np.rint(image*255).astype(np.uint8))
np.stack(images).tofile(out/'impact-array.bin')
(out/'impacts.json').write_text(json.dumps({'size':n,'frames':frames,'durationSeconds':.75,'diameterMetres':diameter,'channels':'R/G tangent normal X/Z, B crown highlight, A ripple coverage','source':'Radial capillary-wave and decaying crown approximation; baked once for instanced terrain collisions.'},indent=2))
print('Baked',frames,'impact frames at',n,'squared')
