import * as THREE from 'three/webgpu';
import { wgslFn, texture, uniform, screenUV } from 'three/tsl';

export const bilinearWater=wgslFn(`
fn bilinearWater(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
 let n=vec2<i32>(textureDimensions(tex));let p=uv*vec2<f32>(n)-.5;let i=vec2<i32>(floor(p));let f=fract(p);
 let a=textureLoad(tex,clamp(i,vec2<i32>(0),n-1),0);let b=textureLoad(tex,clamp(i+vec2<i32>(1,0),vec2<i32>(0),n-1),0);
 let c=textureLoad(tex,clamp(i+vec2<i32>(0,1),vec2<i32>(0),n-1),0);let d=textureLoad(tex,clamp(i+1,vec2<i32>(0),n-1),0);
 return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}`);

export const hydraulicFlux=wgslFn(`
fn hydraulicFlux(uv: vec2<f32>, state: texture_2d<f32>, bed: texture_2d<f32>, previous: texture_2d<f32>, dt: f32, cell: f32) -> vec4<f32> {
 let n=vec2<i32>(textureDimensions(state));let p=clamp(vec2<i32>(uv*vec2<f32>(n)),vec2<i32>(0),n-1);
 let s=textureLoad(state,p,0);let terrain=textureLoad(bed,p,0);let mobile=max(0.0,s.r-terrain.a);
 let level=terrain.r+s.r*.001;var output=vec4<f32>(0.0);let old=textureLoad(previous,p,0);
 let directions=array<vec2<i32>,4>(vec2<i32>(-1,0),vec2<i32>(1,0),vec2<i32>(0,-1),vec2<i32>(0,1));
 for(var k=0;k<4;k++){
  let q=p+directions[k];var otherBed=terrain.r-.05;var otherWater=0.0;
  if(all(q>=vec2<i32>(0))&&all(q<n)){otherBed=textureLoad(bed,q,0).r;otherWater=textureLoad(state,q,0).r;}
  let bedDifference=terrain.r-otherBed;let headDifference=bedDifference+(s.r-otherWater)*.001;
  let faceDepth=max(0.0,max(bedDifference+s.r*.001,otherWater*.001)-max(bedDifference,0.0));
  if(faceDepth>.00001){
   let accelerated=max(0.0,old[k]+9.81*faceDepth*dt*headDifference/cell);
   let friction=1.0+9.81*dt*.045*.045*old[k]/pow(max(faceDepth,.001),2.333333);
   output[k]=min(accelerated/friction,faceDepth*8.0);
  }
 }
 let total=dot(output,vec4<f32>(1.0));return output*min(1.0,mobile*.001*cell/max(total*dt,.000001));
}`);

export const hydraulicStep=wgslFn(`
fn hydraulicStep(uv: vec2<f32>, state: texture_2d<f32>, flux: texture_2d<f32>, bed: texture_2d<f32>,
 weather: texture_2d<f32>, origin: vec2<f32>, extent: f32, world: f32, dt: f32, rainRate: f32) -> vec4<f32> {
 let n=vec2<i32>(textureDimensions(state));let p=clamp(vec2<i32>(uv*vec2<f32>(n)),vec2<i32>(0),n-1);
 let s=textureLoad(state,p,0);let terrain=textureLoad(bed,p,0);let own=textureLoad(flux,p,0);
 let directions=array<vec2<i32>,4>(vec2<i32>(-1,0),vec2<i32>(1,0),vec2<i32>(0,-1),vec2<i32>(0,1));
 let opposite=array<i32,4>(1,0,3,2);var incoming=0.0;var exported=0.0;
 for(var k=0;k<4;k++){let q=p+directions[k];if(all(q>=vec2<i32>(0))&&all(q<n)){incoming+=textureLoad(flux,q,0)[opposite[k]];}else{exported+=own[k];}}
 let conversion=dt/(extent/f32(n.x))*1000.0;
 incoming*=conversion;exported*=conversion;let outgoing=dot(own,vec4<f32>(1.0))*conversion;let worldUV=(origin+uv*extent)/world+.5;
 let w=bilinearWater(weather,worldUV);let hours=dt/3600.0;let rain=w.g*rainRate*hours;
 var water=max(0.0,s.r-outgoing+incoming)+rain;var soil=s.g;
 let capacity=terrain.b;let conductivity=terrain.g;
 let intake=min(water,min(max(0.0,capacity-soil),conductivity*(1.0+.7*(1.0-soil/capacity))*hours));
 water-=intake;soil+=intake;
 let evap=(.16+.65*w.r)*hours;let surfaceLoss=min(water,evap);water-=surfaceLoss;
 let soilLoss=min(soil,(evap-surfaceLoss)*.6);soil-=soilLoss;
 let deepLoss=soil*(1.0-exp(-conductivity/capacity*.22*hours));soil-=deepLoss;
 return vec4<f32>(water,soil,outgoing/max(dt,.000001),s.a+rain-surfaceLoss-soilLoss-deepLoss-exported);
}`, [bilinearWater]);

const buildBed=wgslFn(`
fn buildWaterBed(uv: vec2<f32>, height: texture_2d<f32>, soil: texture_2d<f32>, origin: vec2<f32>, extent: f32, world: f32) -> vec4<f32> {
 let worldUV=(origin+uv*extent)/world+.5;let n=vec2<f32>(textureDimensions(height));
 let h=bilinearWater(height,(worldUV*(n-1.0)+.5)/n).r;let p=bilinearWater(soil,worldUV);
 return vec4<f32>(h,p.r,p.g,p.b);
}`, [bilinearWater]);
const shiftWater=wgslFn(`
fn shiftWater(uv: vec2<f32>, old: texture_2d<f32>, global: texture_2d<f32>, origin: vec2<f32>, oldOrigin: vec2<f32>, extent: f32, world: f32, ready: f32) -> vec4<f32> {
 let pos=origin+uv*extent;let previousUV=(pos-oldOrigin)/extent;
 var s=bilinearWater(global,pos/world+.5);
 if(ready>.5&&all(previousUV>=vec2<f32>(0.0))&&all(previousUV<=vec2<f32>(1.0))){s=bilinearWater(old,previousUV);}
 return vec4<f32>(s.rgb,s.r+s.g);
}`, [bilinearWater]);

export function createWaterGrid(renderer,{size,extent,world,origin,height,soil,weather,global,initialState}){
 const target=()=>new THREE.RenderTarget(size,size,{type:THREE.FloatType,minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter,depthBuffer:false});
 const zero=new THREE.DataTexture(initialState||new Float32Array(size*size*4),size,size,THREE.RGBAFormat,THREE.FloatType);zero.minFilter=zero.magFilter=THREE.NearestFilter;zero.needsUpdate=true;
 const noFlow=new THREE.DataTexture(new Float32Array(size*size*4),size,size,THREE.RGBAFormat,THREE.FloatType);noFlow.minFilter=noFlow.magFilter=THREE.NearestFilter;noFlow.needsUpdate=true;
 const states=[target(),target()],bed=target(),fluxes=[target(),target()],state=texture(zero),bedNode=texture(bed.texture),fluxNode=texture(noFlow);
 const originNode=uniform(origin),oldOrigin=uniform(origin.clone()),dt=uniform(1),rain=uniform(0),ready=uniform(0);
 const quad=fn=>{const m=new THREE.NodeMaterial({depthTest:false,depthWrite:false});m.fragmentNode=fn;return new THREE.QuadMesh(m);};
 const bedQuad=quad(buildBed(screenUV,height,soil,originNode,extent,world));
 const fluxQuad=quad(hydraulicFlux(screenUV,state,bedNode,fluxNode,dt,extent/size));
 const stepQuad=quad(hydraulicStep(screenUV,state,fluxNode,bedNode,weather,originNode,extent,world,dt,rain));
 const shiftQuad=global?quad(shiftWater(screenUV,state,global,originNode,oldOrigin,extent,world,ready)):null;
 let current=-1,flowIndex=-1,built=false;
 function rebuild(nextOrigin){
  oldOrigin.value.copy(originNode.value);originNode.value.copy(nextOrigin);
  renderer.setRenderTarget(bed);bedQuad.render(renderer);
  if(shiftQuad){current=(current+1)%2;renderer.setRenderTarget(states[current]);shiftQuad.render(renderer);state.value=states[current].texture;}
  fluxNode.value=noFlow;flowIndex=-1;
  built=true;ready.value=1;
 }
 return {state,bed:bedNode,flux:fluxNode,origin:originNode,extent,size,
  initialize(){if(!built)rebuild(originNode.value.clone());},
  recenter(x,z){const next=new THREE.Vector2(Math.floor((x-extent/2)/64)*64,Math.floor((z-extent/2)/64)*64);if(!built||next.distanceTo(originNode.value)>extent*.23)rebuild(next);},
  step(seconds,rainRate){if(!built)rebuild(originNode.value.clone());dt.value=seconds;rain.value=rainRate;flowIndex=(flowIndex+1)%2;renderer.setRenderTarget(fluxes[flowIndex]);fluxQuad.render(renderer);fluxNode.value=fluxes[flowIndex].texture;current=(current+1)%2;renderer.setRenderTarget(states[current]);stepQuad.render(renderer);state.value=states[current].texture;},
  reset(){state.value=zero;fluxNode.value=noFlow;current=-1;flowIndex=-1;built=false;ready.value=0;},
  currentTarget(){return current>=0?states[current]:null;}
 };
}
