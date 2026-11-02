"""Landscape-scale stream-power erosion, followed by high-resolution material baking.

The coarse drainage solve and the fine hydraulic source are recorded separately.
This is an artistic geomorphology model, not a calibrated hydrology prediction.
"""
import json, time, argparse
from pathlib import Path
import numpy as np
from numba import njit
from scipy.ndimage import zoom, gaussian_filter
from PIL import Image

parser=argparse.ArgumentParser();parser.add_argument('--source',default='public/assets/erosion-v3');parser.add_argument('--out',default='public/assets/badlands-v3');args=parser.parse_args()
OUT=Path(args.out); OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path(args.source)
N=8192; R=2048; WORLD=32768.; DX=WORLD/R
t0=time.time()

@njit(cache=True)
def receivers(h, n, dx):
    dest=np.full(h.size,-1,np.int32)
    slope=np.zeros(h.size,np.float32)
    length=np.full(h.size,dx,np.float32)
    for z in range(1,n-1):
        for x in range(1,n-1):
            i=z*n+x; best=0.
            for dz in range(-1,2):
                for dx0 in range(-1,2):
                    if dz==0 and dx0==0: continue
                    j=(z+dz)*n+x+dx0
                    d=dx*(1.41421356 if dz!=0 and dx0!=0 else 1.)
                    s=(h[i]-h[j])/d
                    if s>best:best=s;dest[i]=j;length[i]=d
            slope[i]=best
    return dest,slope,length

@njit(cache=True)
def accumulate(order,dest,dx):
    area=np.full(dest.size,dx*dx,np.float32)
    for ii in range(order.size-1,-1,-1):
        i=order[ii];j=dest[i]
        if j>=0: area[j]+=area[i]
    return area

@njit(cache=True)
def erode(h,order,dest,area,length,dx):
    """Implicit downstream-first stream-power step, n=1, m=.5.

    Eroded solid is routed down the same graph and settles where transport slows.
    Remaining solid at the open domain boundary is accounted as exported.
    """
    before=h.copy(); cut=np.zeros(h.size,np.float32); n=int(np.sqrt(h.size))
    for ii in range(order.size):
        i=order[ii];j=dest[i]
        if j<0:continue
        strength=.35+.65*(.5+.5*np.sin(before[i]*.095))**2
        k=.18*np.sqrt(area[i])/length[i]/strength
        new=(before[i]+k*h[j])/(1+k)
        # Limit per-step incision while retaining a stable downstream slope.
        delta=min(2.5,max(0.,before[i]-new));h[i]-=delta;cut[i]=delta
    sediment=cut.copy();deposit=np.zeros(h.size,np.float32);exported=0.
    for ii in range(order.size-1,-1,-1):
        i=order[ii];j=dest[i]
        if j<0:
            z=i//n;x=i%n
            if z==0 or x==0 or z==n-1 or x==n-1:exported+=sediment[i];continue
            amount=min(sediment[i],.3)
        else:
            slope=max(0.,(h[i]-h[j])/length[i])
            amount=min(.15,sediment[i]*.035*np.exp(-slope*12))
        deposit[i]=amount;h[i]+=amount
        if j>=0:sediment[j]+=sediment[i]-amount
        else:exported+=sediment[i]-amount  # unresolved sink storage, reported separately in docs
    return cut,deposit,exported

source=np.memmap(SOURCE/'height.bin',mode='r',dtype='<f4',shape=(N,N))
h=np.array(source[::4,::4],dtype=np.float32)*1.55
# Subwatersheds start with correlated topography rather than independent per-cell noise.
rng=np.random.default_rng(781)
for res,amp in [(45,35),(110,16),(240,5)]:
    h+=zoom(rng.random((res,res),dtype=np.float32)-.5,(R/res,R/res),order=3)[:R,:R]*amp
h=h.ravel();initial=h.copy();total_cut=np.zeros_like(h);total_deposit=np.zeros_like(h);exported=0.
for iteration in range(96):
    dest,slope,length=receivers(h,R,DX)
    order=np.argsort(h).astype(np.int32)
    area=accumulate(order,dest,DX)
    cut,deposit,loss=erode(h,order,dest,area,length,DX)
    total_cut+=cut;total_deposit+=deposit;exported+=loss
    if iteration%16==0:print(json.dumps({'iteration':iteration,'seconds':round(time.time()-t0,1),'meanIncision':float(total_cut.mean())}),flush=True)
balance=abs(float(h.astype(np.float64).sum())+exported-float(initial.astype(np.float64).sum()))/abs(float(initial.astype(np.float64).sum()))
assert balance<1e-5, balance
# Retain the finer source variation, with the large-scale drainage incision imposed over it.
delta=(h-initial).reshape(R,R)
coarse_balance=balance
coarse_exported=exported*DX*DX
fine_source=np.array(source[::2,::2],dtype=np.float32)*1.55
h=(fine_source+zoom(delta,2,order=1)).ravel()
fine_initial=h.copy()
total_cut=zoom(total_cut.reshape(R,R),2,order=1).ravel()
total_deposit=zoom(total_deposit.reshape(R,R),2,order=1).ravel()
R=4096;DX=WORLD/R;exported=0.
for iteration in range(32):
    dest,slope,length=receivers(h,R,DX)
    order=np.argsort(h).astype(np.int32)
    area=accumulate(order,dest,DX)
    cut,deposit,loss=erode(h,order,dest,area,length,DX)
    total_cut+=cut;total_deposit+=deposit;exported+=loss
    if iteration%8==0:print(json.dumps({'fineIteration':iteration,'seconds':round(time.time()-t0,1)}),flush=True)
balance=max(coarse_balance,abs(float(h.astype(np.float64).sum())+exported-float(fine_initial.astype(np.float64).sum()))/abs(float(fine_initial.astype(np.float64).sum())))
assert balance<1e-5,balance
fine_delta=zoom((h.reshape(R,R)-fine_source),2,order=1)
height=np.array(source,dtype=np.float32)*1.55+fine_delta
# Native-resolution drainage incision adds branching metre-scale gullies.
# This pass only exports material; its removed volume is recorded separately.
native_dx=WORLD/N
print(json.dumps({'nativeDrainage':N,'seconds':round(time.time()-t0,1)}),flush=True)
native_flat=height.ravel()
ndest,nslope,nlength=receivers(native_flat,N,native_dx)
norder=np.argsort(native_flat).astype(np.int32)
narea=accumulate(norder,ndest,native_dx)
micro_cut=np.minimum(14.,np.log1p(narea/(native_dx*native_dx))*np.power(np.maximum(nslope,0),.6)*3.2).reshape(N,N)
micro_cut=gaussian_filter(micro_cut,.5)
native_removed=float(micro_cut.astype(np.float64).sum())*native_dx*native_dx
height-=micro_cut
height.tofile(OUT/'height.bin')
del ndest,nslope,nlength,norder,narea
del fine_delta,source
native=np.memmap(SOURCE/'fields.bin',mode='r',dtype='<f4',shape=(N,N,11))
preview=np.array(native[::8,::8],copy=True)
preview[:,:,0]=height[::8,::8]
preview[:,:,6]+=zoom(total_cut.reshape(R,R),.25,order=1)+micro_cut[::8,::8]
preview[:,:,7]+=zoom(total_deposit.reshape(R,R),.25,order=1)
preview[:,:,8]=zoom(area.reshape(R,R),.25,order=1)
preview.tofile(OUT/'preview-fields.bin')

# Bake pale warm-grey parent rock; color comes from incision, deposition, and exposed strata.
eroded=zoom(total_cut.reshape(R,R),2,order=1)+micro_cut
deposited=zoom(total_deposit.reshape(R,R),2,order=1)
albedo=np.empty((N,N,3),np.uint8);normal=np.empty_like(albedo);rough=np.empty_like(albedo)
for z in range(0,N,256):
    end=min(N,z+256);hs=height[max(0,z-1):min(N,end+1)]
    gy,gx=np.gradient(hs,WORLD/(N-1));offset=1 if z else 0
    gx=gx[offset:offset+end-z];gy=gy[offset:offset+end-z]
    lengthn=np.sqrt(1+gx*gx+gy*gy)
    normal[z:end,:,0]=((-gx/lengthn*.5+.5)*255).astype(np.uint8)
    normal[z:end,:,1]=((gy/lengthn*.5+.5)*255).astype(np.uint8)
    normal[z:end,:,2]=((1/lengthn*.5+.5)*255).astype(np.uint8)
    exposed=1-np.exp(-eroded[z:end]*.12)
    silt=1-np.exp(-deposited[z:end]*.7)
    strata=np.sin(height[z:end]*.11)*.022
    for c,(rock,sand) in enumerate(zip([.655,.64,.605],[.63,.613,.57])):
        value=rock+strata+exposed*.035-silt*(rock-sand)
        albedo[z:end,:,c]=np.clip(value*255,0,255).astype(np.uint8)
    lap=np.gradient(gy,WORLD/(N-1),axis=0)+np.gradient(gx,WORLD/(N-1),axis=1)
    rough[z:end,:,0]=np.clip(np.exp(-np.maximum(lap,0)*2.2)*255,100,255).astype(np.uint8);rough[z:end,:,1]=np.clip((.88-silt*.07)*255,0,255).astype(np.uint8);rough[z:end,:,2]=0
Image.fromarray(albedo).save(OUT/'albedo.jpg',quality=94,subsampling=0)
Image.fromarray(normal).save(OUT/'terrain-normal.png')
Image.fromarray(rough).save(OUT/'orm.png')

# A physically scaled 8 m repeat tile for fractured rock normals, color and roughness.
from scipy.spatial import cKDTree
T=1024
seeds=rng.random((1800,2))*T
wrapped=np.concatenate([seeds+[x*T,y*T] for y in [-1,0,1] for x in [-1,0,1]])
yy,xx=np.mgrid[:T,:T];points=np.stack([xx.ravel(),yy.ravel()],1)
dist,ids=cKDTree(wrapped).query(points,k=2)
edge=(dist[:,1]-dist[:,0]).reshape(T,T)
grain=gaussian_filter(rng.random((T,T)),1,mode='wrap')
micro=gaussian_filter(rng.random((T,T)),.45,mode='wrap')
relief=(1-np.exp(-edge*.55))*.006+(grain-.5)*.014+(micro-.5)*.006+(gaussian_filter(rng.random((T,T)),5,mode="wrap")-.5)*.12
dy,dx=np.gradient(relief,8/T);ll=np.sqrt(dx*dx+dy*dy+1)
nm=np.stack([-.5*dx/ll+.5,.5*dy/ll+.5,.5/ll+.5],-1)
Image.fromarray(np.clip(nm*255,0,255).astype(np.uint8)).save(OUT/'fracture-normal.png')
tone=.79+(grain-.5)*.35+np.minimum(edge,3)*.015
col=np.stack([tone*1.03,tone,tone*.92],-1)
Image.fromarray(np.clip(col*255,0,255).astype(np.uint8)).save(OUT/'fracture-color.jpg',quality=95)
Image.fromarray(np.clip((.83+micro*.13)*255,0,255).astype(np.uint8)).save(OUT/'fracture-roughness.png')

# Pick a high shoulder with a modest local slope and a view down into drainage.
coarse=height[::8,::8];gz,gx=np.gradient(coarse,WORLD/1024)
score=coarse-gaussian_filter(coarse,25)-np.hypot(gx,gz)*30
score[:180]=score[-180:]=-1e6;score[:,:180]=score[:,-180:]=-1e6
iz,ix=np.unravel_index(np.argmax(score),score.shape)
spawn={'x':float((ix/1023-.5)*WORLD),'z':float((iz/1023-.5)*WORLD),'height':float(coarse[iz,ix])}
meta={'size':N,'previewSize':1024,'worldSize':WORLD,'heightMin':float(height.min()),'heightMax':float(height.max()),'steps':1600,'drainageSize':R,'drainageSteps':128,'drainagePasses':[{'size':2048,'steps':96},{'size':4096,'steps':32}],'drainageCellMetres':DX,'nativeDrainageSize':N,'nativeIncisionVolumeM3':native_removed,'solidBudgetRelativeError':balance,'exportedOrUnresolvedSinkVolumeM3':coarse_exported+exported*DX*DX,'logicalRenderGrid':524288,'logicalTriangles':2*524288**2,'spawn':spawn,'seconds':time.time()-t0,'hydraulicSource':json.loads((SOURCE/'simulation.json').read_text()),'source':'8192-square hydraulic bake rescaled horizontally to a 32.768 km domain; 2048- then 4096-square stream-power solves; 8192-square final height bake. Sub-cell relief is procedural.'}
(OUT/'simulation.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta),flush=True)
