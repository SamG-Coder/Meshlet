"""Offline CUDA diffuse path-traced light maps of the existing terrain.

Separate linear sky irradiance, unit-sun bounce irradiance and sun visibility
allow runtime cloud shadows without baking a fixed cloud photograph into rock.
"""
from pathlib import Path
import argparse, ctypes as c, hashlib, json, os, time
import numpy as np
from PIL import Image
import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,default=4096);ap.add_argument('--samples',type=int,default=256)
    ap.add_argument('--source',default='public/assets/badlands-v3');ap.add_argument('--bounces',type=int,default=6)
    ap.add_argument('--shadow-size',type=int,default=8192);ap.add_argument('--shadow-samples',type=int,default=8)
    ap.add_argument('--out',default='public/assets/badlands-lighting-hq');args=ap.parse_args()
    start=time.time();root=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'simulation.json').read_text());n=meta['size'];world=meta['worldSize'];dx=world/(n-1)
    torch.cuda.init();torch.zeros(1,device='cuda')
    lib=Path(torch.__file__).parent/'lib';dll_dir=os.add_dll_directory(str(lib))
    rtc=c.CDLL(str(lib/'nvrtc64_120_0.dll'));cuda=c.WinDLL('nvcuda.dll')
    def check(code):
        if code:raise RuntimeError(f'CUDA error {code}')
    code=Path('scripts/terrain-lighting.cu').read_bytes();program=c.c_void_p()
    rtc.nvrtcCreateProgram.argtypes=[c.POINTER(c.c_void_p),c.c_char_p,c.c_char_p,c.c_int,c.c_void_p,c.c_void_p]
    check(rtc.nvrtcCreateProgram(c.byref(program),code,b'terrain-lighting.cu',0,None,None))
    options=(c.c_char_p*2)(b'--gpu-architecture=compute_120',b'--use_fast_math')
    result=rtc.nvrtcCompileProgram(program,2,options)
    if result:
        length=c.c_size_t();rtc.nvrtcGetProgramLogSize(program,c.byref(length));log=c.create_string_buffer(length.value);rtc.nvrtcGetProgramLog(program,log);raise RuntimeError(log.value.decode())
    length=c.c_size_t();check(rtc.nvrtcGetPTXSize(program,c.byref(length)));ptx=c.create_string_buffer(length.value);check(rtc.nvrtcGetPTX(program,ptx))
    module=c.c_void_p();check(cuda.cuModuleLoadData(c.byref(module),ptx));kernels={}
    for name in ('bake','visibility'):
        f=c.c_void_p();check(cuda.cuModuleGetFunction(c.byref(f),module,name.encode()));kernels[name]=f
    def launch(name,count,values):
        actual=[c.c_void_p(a.data_ptr()) if isinstance(a,torch.Tensor) else c.c_int(a) if isinstance(a,int) else c.c_float(a) for a in values]
        pointers=(c.c_void_p*len(actual))(*[c.cast(c.byref(a),c.c_void_p) for a in actual])
        check(cuda.cuLaunchKernel(kernels[name],(count+127)//128,1,1,128,1,1,0,None,pointers,None));check(cuda.cuCtxSynchronize())
    h=np.fromfile(root/'height.bin',dtype='<f4').reshape(n,n);height=torch.from_numpy(h).cuda()
    cell=np.full((n,n),-1e10,dtype=np.float32);cell[:-1,:-1]=np.maximum.reduce([h[:-1,:-1],h[1:,:-1],h[:-1,1:],h[1:,1:]])
    levels=[cell];offsets=[0]
    while levels[-1].shape[0]>1:
        a=levels[-1];offsets.append(offsets[-1]+a.size);levels.append(a.reshape(a.shape[0]//2,2,a.shape[1]//2,2).max((1,3)))
    pyramid=torch.from_numpy(np.concatenate([a.ravel() for a in levels])).cuda();offset=torch.tensor(offsets,dtype=torch.int32,device='cuda');top=len(levels)-1
    albedo=np.asarray(Image.open(root/'albedo.jpg').resize((2048,2048)),np.float32)/255
    albedo=np.where(albedo<=.04045,albedo/12.92,((albedo+.055)/1.055)**2.4);color=torch.from_numpy(albedo).cuda()
    stats=torch.zeros(1,dtype=torch.int32,device='cuda');sun=np.array([-0.55,0.65,-0.52]);sun/=np.linalg.norm(sun)
    sky=np.empty((args.size,args.size,4),np.float32);bounce=np.empty_like(sky);rows=32
    print(json.dumps({'stage':'path tracing','source':n,'lightmap':args.size,'samples':args.samples,'bounces':args.bounces,'gpu':torch.cuda.get_device_name()}),flush=True)
    for row in range(0,args.size,rows):
        count=min(rows,args.size-row);a=torch.empty((count,args.size,4),device='cuda');b=torch.empty_like(a)
        launch('bake',count*args.size,[height,pyramid,offset,color,2048,n,top,dx,args.size,row,count,args.samples,args.bounces,*map(float,sun),a,b,stats])
        sky[row:row+count]=a.cpu().numpy();bounce[row:row+count]=b.cpu().numpy()
        if row%256==0:print(json.dumps({'stage':'path tracing','row':row,'seconds':round(time.time()-start,1)}),flush=True)
    # A narrow normal-guided reconstruction filter keeps canyon boundaries crisp.
    coords=np.minimum(n-1,((np.arange(args.size)+.5)*n/args.size).astype(int))
    guide=h[np.ix_(coords,coords)];gz,gx=np.gradient(guide,world/args.size)
    norm=np.stack([-gx,np.ones_like(gx),-gz],-1);norm/=np.linalg.norm(norm,axis=-1,keepdims=True)
    for value in (sky,bounce):
        accum=value.copy();weights=np.ones(guide.shape,np.float32)
        for zoff,xoff in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
            other=np.roll(norm,(zoff,xoff),(0,1));w=np.exp(-(1-(norm*other).sum(-1))*48)*(.16 if zoff*xoff else .3)
            if zoff>0:w[:zoff]=0
            if zoff<0:w[zoff:]=0
            if xoff>0:w[:,:xoff]=0
            if xoff<0:w[:,xoff:]=0
            accum+=np.roll(value,(zoff,xoff),(0,1))*w[:,:,None];weights+=w
        value[:]=accum/weights[:,:,None]
    sky.astype('<f2').tofile(out/'sky-irradiance.bin');bounce.astype('<f2').tofile(out/'sun-bounce.bin')
    visibility=np.empty((args.shadow_size,args.shadow_size),np.float32)
    for row in range(0,args.shadow_size,128):
        count=min(128,args.shadow_size-row);a=torch.empty((count,args.shadow_size),device='cuda')
        launch('visibility',count*args.shadow_size,[height,pyramid,offset,n,top,dx,args.shadow_size,row,count,args.shadow_samples,*map(float,sun),a,stats])
        visibility[row:row+count]=a.cpu().numpy()
        if row%1024==0:print(json.dumps({'stage':'sun visibility','row':row,'seconds':round(time.time()-start,1)}),flush=True)
    Image.fromarray(np.rint(visibility*255).astype(np.uint8)).save(out/'sun-visibility.png')
    assert np.isfinite(sky).all() and np.isfinite(bounce).all() and sky.min()>=0 and bounce.min()>=0
    truncated=int(stats.item());assert truncated==0,f'{truncated} traversal limits reached'
    result={'source':root.name,'heightSha256':hashlib.sha256((root/'height.bin').read_bytes()).hexdigest(),'worldSize':world,'sourceSize':n,'size':args.size,'samplesPerTexel':args.samples,'maxDiffuseBounces':args.bounces,'shadowSize':args.shadow_size,'shadowSamples':args.shadow_samples,'sunDirection':sun.tolist(),'sunAngularRadius':.00465,'encoding':'RGBA16F linear irradiance; sky RGB, unit white sun bounce RGB; alpha unused','denoising':'Narrow normal-guided reconstruction','seconds':time.time()-start,'truncatedRays':truncated,'skyRange':[float(sky[:,:,:3].min()),float(sky[:,:,:3].max())],'bounceRange':[float(bounce[:,:,:3].min()),float(bounce[:,:,:3].max())],'shadowFraction':float((visibility<.5).mean()),'scope':'Path tracing against the bilinear source heightfield used by the renderer. Sub-cell material relief is shaded at runtime and is not ray traced. Clouds modulate separated light components at runtime; moving-cloud indirect light is approximate.'}
    (out/'lighting.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
