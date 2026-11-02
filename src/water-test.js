import * as THREE from 'three/webgpu';
import { texture } from 'three/tsl';
import { createWaterGrid } from './water-solver.js';

export async function verifyWaterModel(){
 const n=64,renderer=new THREE.WebGPURenderer();await renderer.init();renderer.setSize(1,1);
 const make=data=>{const t=new THREE.DataTexture(data,n,n,THREE.RGBAFormat,THREE.FloatType);t.minFilter=t.magFilter=THREE.NearestFilter;t.needsUpdate=true;return texture(t);};
 const soil=new Float32Array(n*n*4),height=new Float32Array(soil.length),cloud=new Float32Array(soil.length);
 for(let z=0;z<n;z++)for(let x=0;x<n;x++){const i=(z*n+x)*4;soil.set([.4,1,.1,.7],i);height[i]=10-x*.2;cloud.set([.8,x<8?1:0,0,1],i);}
 const grid=createWaterGrid(renderer,{size:n,extent:256,world:256,origin:new THREE.Vector2(-128,-128),height:make(height),soil:make(soil),weather:make(cloud)});
 async function steps(count,seconds,rain){for(let i=0;i<count;i++){grid.step(seconds,rain);if(i%128===127){await renderer.backend.device.queue.onSubmittedWorkDone();await new Promise(r=>setTimeout(r,0));}}await renderer.backend.device.queue.onSubmittedWorkDone();}
 let budgetScale=1;
 async function sample(target=grid){const data=await renderer.readRenderTargetPixelsAsync(target.currentTarget(),0,0,n,n);let water=0,stored=0,flux=0,supply=0,downstream=0,min=Infinity;
  for(let i=0;i<data.length;i+=4){water+=data[i];stored+=data[i+1];flux+=data[i+2];supply+=data[i+3];min=Math.min(min,data[i],data[i+1]);if((i/4)%n>=10)downstream+=data[i];}
  const result={water,stored,flux,supply,downstream,min,balanceError:water+stored-supply};
  budgetScale=Math.max(budgetScale,Math.abs(supply));
  if(min<0||Math.abs(result.balanceError)>Math.max(.02,budgetScale*.0001))throw Error('Water accounting failed: '+JSON.stringify(result));return result;
 }
 const assert=(condition,message)=>{if(!condition)throw Error(message);};
 await steps(10,.1,0);const dry=await sample();assert(dry.water===0&&dry.stored===0,'Dry start creates no water');
 await steps(1024,.1,600);const wet=await sample();assert(wet.stored>1&&wet.flux>0,'Rain must wet soil and generate flow');assert(wet.downstream>1,'Flow must enter non-raining downstream cells');
 await steps(2,.1,0);const stopped=await sample();assert(stopped.water>wet.water*.8&&stopped.stored>0,'Water persists when rain stops');
 await steps(3600,1,0);const receding=await sample();assert(receding.water<wet.water*.1&&receding.water+receding.stored<wet.water+wet.stored,'Water recedes after rain');
 const lakeHeight=new Float32Array(soil.length),lakeInitial=new Float32Array(soil.length),lakeSoil=new Float32Array(soil.length),clear=new Float32Array(soil.length);
 for(let z=0;z<n;z++)for(let x=0;x<n;x++){const i=(z*n+x)*4;lakeHeight[i]=(x<4||z<4||x>59||z>59)?2:1+x*.005;lakeSoil.set([0,1,0,0],i);}
 for(let z=0;z<n;z++)for(let x=0;x<n;x++){
  const sx=(x+.5)/n*(n-1),sz=(z+.5)/n*(n-1),ix=Math.floor(sx),iz=Math.floor(sz),fx=sx-ix,fz=sz-iz;
  const h=(lakeHeight[(iz*n+ix)*4]*(1-fx)+lakeHeight[(iz*n+ix+1)*4]*fx)*(1-fz)+(lakeHeight[((iz+1)*n+ix)*4]*(1-fx)+lakeHeight[((iz+1)*n+ix+1)*4]*fx)*fz;
  const w=Math.max(0,1.6-h)*1000;lakeInitial.set([w,0,0,w],(z*n+x)*4);
 }
 const lake=createWaterGrid(renderer,{size:n,extent:256,world:256,origin:new THREE.Vector2(-128,-128),height:make(lakeHeight),soil:make(lakeSoil),weather:make(clear),initialState:lakeInitial});
 for(let i=0;i<32;i++)lake.step(.08,0);await renderer.backend.device.queue.onSubmittedWorkDone();
 const data=await renderer.readRenderTargetPixelsAsync(lake.currentTarget(),0,0,n,n);let maxFlux=0;
 for(let z=12;z<52;z++)for(let x=12;x<52;x++)maxFlux=Math.max(maxFlux,data[(z*n+x)*4+2]);
 assert(maxFlux<.03,'Level lake develops false currents: '+maxFlux);
 const damp=new Float32Array(soil.length),flat=new Float32Array(soil.length);
 for(let i=0;i<damp.length;i+=4)damp.set([.05,1,0,1.05],i);
 const drying=createWaterGrid(renderer,{size:n,extent:256,world:256,origin:new THREE.Vector2(-128,-128),height:make(flat),soil:make(soil),weather:make(clear),initialState:damp});
 // This retained 0.05 mm film is below depression storage: zero hydraulic
 // flux, so large hydrology-only steps do not bypass the wave timestep limit.
 for(let i=0;i<48;i++)drying.step(1800,0);const dried=await sample(drying);
 assert(dried.water<.01&&dried.stored<.01,'A dry day removes the thin film and soil moisture');
 const result={pass:true,fixture:'64² grid, 4 m cells, upstream cloudburst, downhill flow, recession, level lake and isolated soil dry-down.',dry,wet,stopped,receding,dried,lakeMaxFluxMmPerSecond:maxFlux};
 renderer.dispose();return result;
}
if(location.pathname==='/water-test.html')verifyWaterModel().then(result=>{document.body.dataset.result=JSON.stringify(result);document.getElementById('result').textContent=JSON.stringify(result,null,2);}).catch(error=>{document.body.dataset.error=error.stack;document.getElementById('result').textContent=error.stack;});
