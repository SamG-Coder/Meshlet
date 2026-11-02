"""Tileable 3D Perlin-style/value and Worley density inputs, not sky images."""
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

out=Path('public/assets/cloud-volume');out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(71823);n=128
def smooth_noise(size,cells,dim):
    a=rng.random((cells,)*dim).astype(np.float32)
    return zoom(a,(size/cells,)*dim,order=3,mode='grid-wrap',grid_mode=True,prefilter=True).clip(0,1)

z,y,x=np.mgrid[:n,:n,:n].astype(np.float32)
points=np.stack([x.ravel(),y.ravel(),z.ravel()],1)
volume=np.empty((n,n,n,4),np.uint8)
fbm=sum(smooth_noise(n,c,3)*w for c,w in [(4,.58),(8,.27),(16,.11),(32,.04)])
volume[:,:,:,0]=np.rint(fbm*255).astype(np.uint8)
for channel,cells in enumerate((4,8,16),1):
    seeds=np.stack(np.meshgrid(np.arange(cells),np.arange(cells),np.arange(cells),indexing='ij'),-1).reshape(-1,3)
    seeds=(seeds+rng.random(seeds.shape))*n/cells
    distance=cKDTree(seeds,boxsize=n).query(points,workers=4)[0]
    volume[:,:,:,channel]=np.rint((1-distance.reshape(n,n,n)/(n/cells)).clip(0,1)*255).astype(np.uint8)
    print('Worley octave',cells,flush=True)
volume.tofile(out/'shape-rgba.bin')
weather=sum(smooth_noise(512,c,2)*w for c,w in [(4,.65),(12,.25),(32,.1)])
weather=np.rint(weather.clip(0,1)*255).astype(np.uint8);weather.tofile(out/'weather-r.bin')
(out/'noise.json').write_text(json.dumps({'volumeSize':n,'weatherSize':512,'seed':71823,'channels':'R=smooth fractal density, GBA=inverted Worley F1 at 4/8/16 cells','repeatMetres':14000},indent=2))
