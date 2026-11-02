"""Bake ten independently generated source images into aligned material arrays.
Normals and relief are image-derived estimates, not measured scan geometry.
No source image is reused as another variant.
"""
from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import gaussian_filter
SOURCE=Path('public/assets/generated-ground')
OUT=Path('public/assets/ground-generated-v3');OUT.mkdir(parents=True,exist_ok=True)
N=2048;COUNT=10;METRES=2.;HEIGHT_RANGE=.06
def periodic(a):
    """Periodic-plus-smooth decomposition removes the wrap discontinuity."""
    v=np.zeros_like(a)
    v[0]=a[-1]-a[0];v[-1]=a[0]-a[-1]
    v[:,0]+=a[:,-1]-a[:,0];v[:,-1]+=a[:,0]-a[:,-1]
    yy,xx=np.mgrid[:N,:N]
    denom=2*np.cos(2*np.pi*xx/N)+2*np.cos(2*np.pi*yy/N)-4
    denom[0,0]=1
    f=np.fft.fft2(v)/denom;f[0,0]=0
    return (a-np.fft.ifft2(f).real).astype(np.float32)
files=[SOURCE/f'source-{i+1:02d}.png' for i in range(COUNT)]
assert all(p.exists() for p in files),'All ten generated images must exist before baking.'
hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
assert len(set(hashes))==COUNT,'Every material must use a distinct generated source.'
sources=[]
contact=Image.new('RGB',(5*320,2*346),(30,32,31))
for variant,p in enumerate(files):
    im=Image.open(p).convert('RGB')
    sources.append({'file':str(p),'sha256':hashes[variant],'nativeSize':list(im.size)})
    photo=np.asarray(im.resize((N,N),Image.Resampling.LANCZOS),np.float32)/255
    color=np.stack([periodic(photo[:,:,c]) for c in range(3)],-1).clip(0,1)
    # Keep mineral variation while matching the same neutral pale rock family.
    color*=np.array([.64,.626,.597],np.float32)/color.mean(axis=(0,1))
    color=color.clip(.025,.97)
    lum=color@np.array([.2126,.7152,.0722],np.float32)
    smooth=gaussian_filter(lum,9.0,mode='wrap')
    # Local cavity/fragment estimate. Suppress illumination drift over large areas.
    local=smooth-gaussian_filter(smooth,96,mode='wrap')
    lo,hi=np.percentile(local,[2,98])
    relief=np.clip((local-lo)/(hi-lo),0,1)
    # Centimetre-scale fragment bodies, sub-millimetre grain. Treating every
    # colour edge as a tall rock wall produced embossed, wrinkled ground.
    height=(.008+.028*relief+.00022*(lum-smooth)/max(float(lum.std()),.02)).clip(0,HEIGHT_RANGE)
    height=gaussian_filter(height,1,mode='wrap')
    dx=(np.roll(height,-1,1)-np.roll(height,1,1))/(2*METRES/N)
    dy=(np.roll(height,-1,0)-np.roll(height,1,0))/(2*METRES/N)
    length=np.sqrt(1+dx*dx+dy*dy)
    normal=np.stack([-dx/length,-dy/length,1/length],-1)*.5+.5
    cavities=np.maximum(0,gaussian_filter(height,8,mode='wrap')-height)+.4*np.maximum(0,gaussian_filter(height,35,mode='wrap')-height)
    ao=np.exp(-cavities*24).clip(.45,1)
    rough=(.87+.07*(1-relief)).clip(0,1)
    rgba=lambda a:np.concatenate([(a.clip(0,1)*255).astype(np.uint8),np.full((N,N,1),255,np.uint8)],-1)
    c=rgba(color);nm=rgba(normal)
    m=(np.stack([ao,rough,height/HEIGHT_RANGE,np.ones_like(height)],-1).clip(0,1)*255).astype(np.uint8)
    folder=OUT/f'variant-{variant+1:02d}';folder.mkdir(exist_ok=True)
    Image.fromarray(c[:,:,:3]).save(folder/'color.jpg',quality=96,subsampling=0)
    Image.fromarray(nm[:,:,:3]).save(folder/'normal.png')
    Image.fromarray(m[:,:,:3]).save(folder/'ao-roughness-height.png')
    for name,array in [('color',c),('normal',nm),('material',m)]:
        with (OUT/f'{name}-array.bin').open('wb' if variant==0 else 'ab') as f:array.tofile(f)
    thumb=ImageOps.fit(im,(320,320))
    contact.paste(thumb,((variant%5)*320,(variant//5)*346))
    ImageDraw.Draw(contact).text(((variant%5)*320+10,(variant//5)*346+324),f'{variant+1:02d}  Independent generated source',fill='white')
    print(f'Baked generated image {variant+1}/{COUNT}: native {im.size}',flush=True)
contact.save(OUT/'ten-generated-sources.jpg',quality=92)
manifest={'count':COUNT,'size':N,'metres':METRES,'heightRangeMetres':HEIGHT_RANGE,'format':'RGBA8 texture arrays, layer-major','packedChannels':['ambientOcclusion','roughness','height','one'],'normalConvention':'XY = negative height derivatives in the two projection axes','source':'Ten separate built-in image_gen generations; prompts in generated-ground/prompts.json','heightProvenance':'Image-derived multi-scale relief estimate; not measured or reconstructed physical geometry','sources':sources}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))

