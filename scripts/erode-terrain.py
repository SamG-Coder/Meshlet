"""Conservative four-neighbour water and multi-grain sediment transport on CUDA.
Graphics model inspired by Mei et al. 2007 and Stava et al. 2008; not calibrated hydrology.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=2048);p.add_argument('--steps',type=int,default=320);p.add_argument('--out',default='public/assets/erosion');a=p.parse_args()
torch.manual_seed(701);dev='cuda';n=a.size;side=8192.;dx=side/(n-1);dt=.18
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
t0=time.time()
y,x=torch.meshgrid(torch.linspace(-1,1,n,device=dev),torch.linspace(-1,1,n,device=dev),indexing='ij')
def noise(res):return F.interpolate(torch.rand(1,1,res,res,device=dev),size=(n,n),mode='bicubic',align_corners=True)[0,0]
ridge=1-torch.abs(2*noise(10)-1)
rock=80+600*torch.exp(-((x+.25)**2*1.3+y*y*.65))*ridge+230*noise(5)+85*noise(22)+28*noise(64)+8*noise(180)
rock+=160*(1-y);rock-=125*torch.exp(-((x-.18*torch.sin(y*7))/.15)**2)
initial_rock=rock.clone();initial_height=rock.clone()
# Initial unconsolidated inventory: gravel, sand, fine soil/silt, expressed as thickness.
bed=torch.stack([torch.full_like(rock,.18),.25+.55*noise(18),.25+.8*noise(12)])
suspended=torch.zeros_like(bed);water=torch.zeros_like(rock);moisture=torch.zeros_like(rock)
eroded=torch.zeros_like(rock);deposited=torch.zeros_like(rock);flow=torch.zeros_like(rock)
initial_height+=bed.sum(0)
directions=[(1,0),(-1,0),(1,1),(-1,1)]
def shift(t,d):return torch.roll(t,d[0],d[1]+t.ndim-2)
def bound(q,d):
    if d==(1,0):q[-1,:]=0
    elif d==(-1,0):q[0,:]=0
    elif d==(1,1):q[:,-1]=0
    else:q[:,0]=0
    return q
initial_mass=(rock.double().sum()+bed.double().sum()).item();rain_total=0.;evap_total=0.
for step in range(a.steps):
    rain=.012*(.55+.45*torch.sin(torch.tensor(step*.027,device=dev))**2)
    water+=rain;rain_total+=float(rain)*n*n
    surface=rock+bed.sum(0);head=surface+water
    # Downhill flow is limited by donor water; closed boundaries conserve material.
    q=[bound(torch.relu(head-shift(head,(-d[0],d[1])))*(.08/dx),d) for d in directions]
    total=sum(q);scale=torch.minimum(torch.ones_like(total),water/(total+1e-8))
    q=[v*scale for v in q];outflow=sum(q);velocity=outflow/(water+1e-5)
    slope=torch.sqrt(((shift(surface,(1,0))-shift(surface,(-1,0)))/(2*dx))**2+((shift(surface,(1,1))-shift(surface,(-1,1)))/(2*dx))**2)
    # Grain-specific carrying capacities: coarse grains require stronger flow.
    for c,(threshold,capacity,settle) in enumerate(((.35,.65,.28),(.12,1.4,.11),(.025,2.2,.035))):
        cap=torch.relu(velocity-threshold)*water*(slope+.015)*capacity
        entrain=torch.minimum(bed[c],torch.relu(cap-suspended[c])*.22)
        bed[c]-=entrain;suspended[c]+=entrain;eroded+=entrain
        deposit=torch.minimum(suspended[c],torch.relu(suspended[c]-cap)*settle)
        bed[c]+=deposit;suspended[c]-=deposit;deposited+=deposit
    # Exposed stratified bedrock weathers at a rate set by the exposed geological band.
    hardness=.25+.75*(.5+.5*torch.sin(rock*.07+x*3+y*2))**3
    exposure=torch.exp(-bed.sum(0)*2)
    incision=torch.minimum(torch.full_like(rock,.0015),exposure*velocity*water*slope*.004/(hardness+.1))
    next_rock=rock-incision
    incision=rock-next_rock  # Transport the actual FP32 decrement, not the unrepresentable nominal amount.
    rock=next_rock;eroded+=incision
    for c,fraction in enumerate((.35,.45,.2)):suspended[c]+=incision*fraction
    # Conservative granular relaxation: soil cannot support arbitrarily steep spikes.
    if step % 4 == 0:
        surface=rock+bed.sum(0)
        moves=[bound(torch.relu(surface-shift(surface,(-d[0],d[1]))-dx*.67)*.12,d) for d in directions]
        available=bed.sum(0)
        limiter=torch.minimum(torch.ones_like(available),available/(sum(moves)+1e-8))
        fractions=bed/(available.unsqueeze(0)+1e-8)
        for d,m in zip(directions,moves):
            transfer=fractions*(m*limiter).unsqueeze(0)
            bed-=transfer;bed+=shift(transfer,d)
        del moves,available,limiter,fractions,transfer,m
    old_water=water.clone();old_sed=suspended.clone()
    for d,v in zip(directions,q):
        transported=old_sed*(v/(old_water+1e-8)).unsqueeze(0)
        suspended-=transported;suspended+=shift(transported,d)
        water-=v;water+=shift(v,d)
    water.clamp_(min=0);suspended.clamp_(min=0)
    flow+=outflow
    del old_water,old_sed,transported,q,total,scale,outflow,velocity,surface,head,slope,cap,entrain,deposit,exposure,incision,hardness
    infiltration=torch.minimum(water,.0025*(1+bed[1])/(1+bed[2]*2))
    water-=infiltration;moisture+=infiltration
    evaporation=water*.006;water-=evaporation
    soil_evap=moisture*.003;moisture-=soil_evap
    evap_total+=float((evaporation+soil_evap).double().sum())

    if step%40==0:print(json.dumps(dict(step=step,seconds=round(time.time()-t0,1),water=float(water.mean()),erosion=float(eroded.mean()),deposition=float(deposited.mean()))),flush=True)
height=rock+bed.sum(0)
torch.cuda.empty_cache()
mass=(rock.double().sum()+bed.double().sum()+suspended.double().sum()).item()
water_mass=float((water+moisture).double().sum())
wet=1-torch.exp(-(water*2+moisture*.7))
cover=1-torch.exp(-bed.sum(0)*1.5);fractions=bed/(bed.sum(0)+1e-8)
band=.5+.5*torch.sin(rock*.07+x*3+y*2)
rock_color=torch.stack([.54+.22*band,.51+.20*band,.44+.18*band],-1)
palette=torch.tensor([[.53,.51,.45],[.74,.68,.54],[.43,.40,.31]],device=dev)
sed_color=torch.einsum('chw,cd->hwd',fractions,palette)
color=(rock_color*(1-cover[...,None])+sed_color*cover[...,None])*(1-.22*wet[...,None])
roughness=(.88-.46*wet).clamp(.22,.95)
fields=torch.stack([height,bed[0],bed[1],bed[2],water,moisture,eroded,deposited,flow,rock,initial_height],-1).cpu().numpy().astype('<f4')
fields.tofile(out/'fields.bin')
fields[:,:,0].copy().tofile(out/'height.bin')
client=fields[::max(1,n//1024),::max(1,n//1024)].copy();client.tofile(out/'preview-fields.bin')
Image.fromarray((color.cpu().numpy().clip(0,1)*255).astype(np.uint8)).save(out/'albedo.png')
orm=np.stack([np.ones((n,n)),roughness.cpu().numpy(),np.zeros((n,n))],-1)
Image.fromarray((orm*255).astype(np.uint8)).save(out/'orm.png')
metadata=dict(size=n,worldSize=side,steps=a.steps,seed=701,channels=['height','gravel','sand','silt','water','moisture','erosion','deposition','flow','bedrock','initialHeight'],heightMin=float(height.min()),heightMax=float(height.max()),solidMassRelativeError=abs(mass-initial_mass)/abs(initial_mass),waterBalanceRelativeError=abs(water_mass+evap_total-rain_total)/max(rain_total,1),seconds=time.time()-t0,logicalRenderGrid=65536,logicalTriangles=2*65536**2,disclaimer='Graphics-scale transport model, not a calibrated watershed forecast. Render microrelief is procedural and material-conditioned, not extra hydraulic simulation cells.')
metadata['previewSize']=client.shape[0]
(out/'simulation.json').write_text(json.dumps(metadata,indent=2));print(metadata,flush=True)
assert metadata['solidMassRelativeError']<1e-5, 'Solid mass conservation failed'
assert metadata['waterBalanceRelativeError']<.001, 'Water balance failed'



