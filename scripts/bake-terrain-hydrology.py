"""Bake runoff routing and near-surface water retention from the retained map.

No terrain editing. Soil coefficients are graphics-model assumptions, not a
calibrated hydrological survey. Routing must always descend the actual surface.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from scipy.ndimage import map_coordinates
from numba import njit

@njit(cache=True)
def native_streams(height,spacing):
    n=height.shape[0];count=n*n;receiver=np.full(count,-1,np.int32);degree=np.zeros(count,np.uint8)
    area=np.full(count,spacing*spacing,np.float32);queue=np.empty(count,np.int32)
    for z in range(1,n-1):
        for x in range(1,n-1):
            i=z*n+x;best=0.;dest=-1
            for oz in range(-1,2):
                for ox in range(-1,2):
                    if ox==0 and oz==0:continue
                    slope=(height[z,x]-height[z+oz,x+ox])/np.sqrt(ox*ox+oz*oz)
                    if slope>best:best=slope;dest=(z+oz)*n+x+ox
            receiver[i]=dest
            if dest>=0:degree[dest]+=1
    tail=0
    for i in range(count):
        if degree[i]==0:queue[tail]=i;tail+=1
    head=0
    while head<tail:
        i=queue[head];head+=1;dest=receiver[i]
        if dest>=0:
            area[dest]+=area[i];degree[dest]-=1
            if degree[dest]==0:queue[tail]=dest;tail+=1
    return receiver,area,head

ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,default=1024)
ap.add_argument('--source',default='public/assets/badlands-v3')
ap.add_argument('--out',default='public/assets/badlands-hydrology');args=ap.parse_args()
root=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
meta=json.loads((root/'simulation.json').read_text());n=args.size;world=meta['worldSize'];cell=world/n
source=np.fromfile(root/'height.bin',dtype='<f4').reshape(meta['size'],meta['size'])
coords=(np.arange(n,dtype=np.float32)+.5)/n*(meta['size']-1)
zz,xx=np.meshgrid(coords,coords,indexing='ij');h=map_coordinates(source,[zz,xx],order=1)
field=np.fromfile(root/'preview-fields.bin',dtype='<f4').reshape(meta['previewSize'],meta['previewSize'],11)
coords=(np.arange(n,dtype=np.float32)+.5)/n*(meta['previewSize']-1)
zz,xx=np.meshgrid(coords,coords,indexing='ij')
soil=np.stack([map_coordinates(field[:,:,i],[zz,xx],order=1) for i in (1,2,3)],-1).clip(0)
soil/=soil.sum(-1,keepdims=True).clip(.001)
gravel,sand,fines=np.moveaxis(soil,-1,0)
gz,gx=np.gradient(h,cell);slope=np.hypot(gx,gz)
conductivity=.4+10*gravel+6*sand
capacity=.7+4*fines+1.5*sand
storage=.06+3*np.exp(-slope*7)*(.3+.7*fines)
params=np.stack([conductivity,capacity,storage,fines],-1).astype('<f4')

# Strict D8 descent. Closed basins remain closed; no artificial channels are cut.
dirs=[(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
receiver=np.full((n,n),-1,np.int32);best=np.zeros((n,n),np.float32)
dx=np.zeros_like(best);dz=np.zeros_like(best);iz,ix=np.indices((n,n))
for ox,oz in dirs:
    nx=ix+ox;nz=iz+oz;valid=(nx>=0)&(nx<n)&(nz>=0)&(nz<n)
    neighbour=h[nz.clip(0,n-1),nx.clip(0,n-1)]
    descent=(h-neighbour)/(cell*np.hypot(ox,oz))
    use=valid&(descent>best)
    best[use]=descent[use];receiver[use]=(nz*n+nx)[use];dx[use]=ox;dz[use]=oz
# Open boundary: discharge leaving the domain is counted as an explicit loss.
boundary=(ix==0)|(iz==0)|(ix==n-1)|(iz==n-1)
receiver[boundary]=-2;dx[boundary]=np.where(ix[boundary]==0,-1,np.where(ix[boundary]==n-1,1,0))
dz[boundary]=np.where(iz[boundary]==0,-1,np.where(iz[boundary]==n-1,1,0));best[boundary]=np.maximum(best[boundary],.02)
area=np.full(n*n,cell*cell,np.float64);flat=receiver.ravel()
for i in np.argsort(h.ravel())[::-1]:
    if flat[i]>=0:area[flat[i]]+=area[i]
area=area.reshape(n,n)
drainage=np.stack([dx,dz,best,area/1e6],-1).astype('<f4')

# The visible channel paths use the full 4 m source, not the 32 m transport
# buckets. Coarse straight segments otherwise cut across the actual gullies.
print('Tracing native-resolution channel paths...',flush=True)
native=meta['size'];native_cell=world/(native-1)
native_receiver,native_area,visited=native_streams(source,native_cell)
assert visited==native*native
indices=np.flatnonzero((native_area>=25000)&(native_receiver>=0))
z,x=indices//native,indices%native;end=native_receiver[indices];ez,ex=end//native,end%native
channel_area=native_area[indices]
width=np.clip(.8+np.sqrt(channel_area/1e6)*6,.8,22)
parents={}
for i,j in zip(indices,end):
    if j not in parents or native_area[i]>native_area[parents[j]]:parents[j]=i
tangents=np.empty((len(indices),4),np.float32)
for k,(i,j) in enumerate(zip(indices,end)):
    before=parents.get(i,i);after=native_receiver[j]
    if after<0:after=j
    direction=np.array([j%native-i%native,j//native-i//native],np.float32)
    a=np.array([j%native-before%native,j//native-before//native],np.float32)
    b=np.array([after%native-i%native,after//native-i//native],np.float32)
    length=np.linalg.norm(direction)*native_cell
    tangents[k,:2]=a/max(np.linalg.norm(a),.001)*length
    tangents[k,2:]=b/max(np.linalg.norm(b),.001)*length
links=np.stack([x*native_cell-world/2,z*native_cell-world/2,
                ex*native_cell-world/2,ez*native_cell-world/2,width,
                x/(native-1),z/(native-1)],-1).astype('<f4')
links=np.concatenate([links,tangents],axis=1)
params.tofile(out/'retention.bin');drainage.tofile(out/'drainage.bin');links.tofile(out/'river-links.bin')
valid=flat>=0
assert np.all(h.ravel()[flat[valid]]<h.ravel()[valid])
assert np.isfinite(params).all() and np.isfinite(drainage).all()
assert np.all(params[:,:,:3]>0)
result={'source':root.name,'heightSha256':hashlib.sha256((root/'height.bin').read_bytes()).hexdigest(),
        'size':n,'worldSize':world,'cellMetres':cell,'riverLinks':len(links),
        'retentionChannels':['conductivity_mm_per_hour','near_surface_capacity_mm','depression_storage_mm','fines_fraction'],
        'drainageChannels':['receiver_dx','receiver_dz','downhill_slope','contributing_area_km2'],
        'riverLinkStride':11,'riverPathCellMetres':native_cell,'closedBasins':int((receiver==-1).sum()),'maxCatchmentKm2':float(area.max()/1e6),
        'riverView':{'x':float(links[channel_area.argmax(),0]),'z':float(links[channel_area.argmax(),1]),'targetX':float(links[channel_area.argmax(),2]),'targetZ':float(links[channel_area.argmax(),3])},
        'scope':'Strictly descending D8 surface routing, open region boundary, closed basins retained. Near-surface soil parameters are assumed from source sediment fractions. No terrain changes and no groundwater or calibrated flood prediction.'}
(out/'hydrology.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
