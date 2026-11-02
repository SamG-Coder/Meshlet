"""Numerical integration checks on flat ground and a raised occluder."""
from pathlib import Path
import json, subprocess, sys
import numpy as np
from PIL import Image

root=Path('results/lighting-fixtures');n=128;results={}
for label,wall,albedo in [('flat',False,180),('ridge-dark',True,70),('ridge-light',True,230)]:
    source=root/label;source.mkdir(parents=True,exist_ok=True)
    h=np.zeros((n,n),np.float32)
    if wall:h[:,62:66]=100
    h.tofile(source/'height.bin');Image.new('RGB',(n,n),(albedo,)*3).save(source/'albedo.jpg')
    (source/'simulation.json').write_text(json.dumps({'size':n,'worldSize':256}))
    output=source/'bake'
    subprocess.run([sys.executable,'scripts/bake-terrain-lighting.py','--source',str(source),'--out',str(output),'--size','128','--samples','256','--bounces','6','--shadow-size','128','--shadow-samples','4'],check=True,stdout=subprocess.DEVNULL)
    results[label]={'sky':np.fromfile(output/'sky-irradiance.bin',dtype='<f2').reshape(n,n,4)[:,:,:3].astype(float),'bounce':np.fromfile(output/'sun-bounce.bin',dtype='<f2').reshape(n,n,4)[:,:,:3].astype(float),'shadow':np.array(Image.open(output/'sun-visibility.png'))/255.}

flat=results['flat'];assert flat['shadow'].min()==1;assert flat['bounce'].max()==0
# Analytic quadrature of the same sky model over a flat cosine-weighted hemisphere.
rng=np.random.default_rng(317);u=rng.random(1000000);phi=rng.random(len(u))*2*np.pi
direction=np.stack([np.sqrt(u)*np.cos(phi),np.sqrt(1-u),np.sqrt(u)*np.sin(phi)],-1)
sun=np.array([-.55,.65,-.52]);sun/=np.linalg.norm(sun)
t=direction[:,1]**.45;glow=np.maximum(0,direction@sun)**12*.25
sky=(1-t[:,None])*np.array([.78,.94,1.16])+t[:,None]*np.array([.12,.30,.63])+glow[:,None]*np.array([1,.9,.72])
expected=sky.mean(0)*np.pi;actual=flat['sky'].mean((0,1));assert np.allclose(actual,expected,rtol=.01),(actual,expected)
dark=results['ridge-dark'];light=results['ridge-light'];region=np.s_[60:95,78:103]
assert dark['shadow'][region].mean()<.05
assert dark['shadow'][35:90,10:45].mean()>.99
assert dark['shadow'][60:95,112:120].mean()>.99  # beyond height * sun.x / sun.y
assert dark['sky'][region].mean()<flat['sky'][region].mean()*.9
assert light['bounce'][region].mean()>dark['bounce'][region].mean()*4
print('PASS: flat-surface sky integral, occluder shadow direction, sky occlusion and albedo-dependent indirect sunlight.')
