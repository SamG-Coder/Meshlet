import * as THREE from 'three/webgpu';
import { wgslFn, texture, sampler, uniform, screenUV, positionWorld, positionLocal, attribute, uv, vec3, vec4, mix, smoothstep, cameraViewMatrix } from 'three/tsl';
import { createWaterGrid } from './water-solver.js';

const waterRead = wgslFn(`
fn waterRead(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
 let size=vec2<i32>(textureDimensions(tex));let p=uv*vec2<f32>(size)-0.5;let i=vec2<i32>(floor(p));let f=fract(p);
 let a=textureLoad(tex,clamp(i,vec2<i32>(0),size-1),0);let b=textureLoad(tex,clamp(i+vec2<i32>(1,0),vec2<i32>(0),size-1),0);
 let c=textureLoad(tex,clamp(i+vec2<i32>(0,1),vec2<i32>(0),size-1),0);let d=textureLoad(tex,clamp(i+1,vec2<i32>(0),size-1),0);
 return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}`);

const surfaceWetness = wgslFn(`
fn surfaceWetness(state: texture_2d<f32>, soil: texture_2d<f32>, uv: vec2<f32>, local: f32) -> vec2<f32> {
 let size=vec2<i32>(textureDimensions(state));let p=uv*vec2<f32>(size)-0.5;let cell=vec2<i32>(floor(p));let f=fract(p);var value=vec2<f32>(0.0);
 for(var z=0;z<2;z++){for(var x=0;x<2;x++){
  let at=clamp(cell+vec2<i32>(x,z),vec2<i32>(0),size-1);let s=textureLoad(state,at,0);let param=textureLoad(soil,at,0);
  let w=select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,z==1);
  let capacity=select(param.g,param.b,local>.5);
  value+=vec2<f32>(clamp(s.g/max(capacity,.001),0.0,1.0),1.0-exp(-s.r*0.08))*w;
 }}
 return value;
}`);

const heightRead = wgslFn(`
fn riverHeight(height: texture_2d<f32>, uv: vec2<f32>) -> f32 {
 let size=vec2<f32>(textureDimensions(height));
 return waterRead(height,(uv*(size-1.0)+0.5)/size).r;
}`, [waterRead]);

const reduceWater = wgslFn(`
fn terrainWaterReduce(uv: vec2<f32>, state: texture_2d<f32>) -> vec4<f32> {
 let size=vec2<i32>(textureDimensions(state));let base=vec2<i32>(floor(uv*64.0))*16;var result=vec4<f32>(0.0);
 for(var z=0;z<16;z++){for(var x=0;x<16;x++){result+=textureLoad(state,min(base+vec2<i32>(x,z),size-1),0);}}
 return result/256.0;
}`);

const dropPosition=wgslFn(`
fn rainDropPosition(local: vec3<f32>, seed: vec4<f32>, camera: vec3<f32>, time: f32, height: texture_2d<f32>, weather: texture_2d<f32>, world: f32, pixelAngle: f32, splash: f32) -> vec3<f32> {
 let middle=seed.x>=.65;let sx=select(seed.x/.65,(seed.x-.65)/.35,middle);
 let width=select(72.0,280.0,middle);
 let center=floor(camera.xz/12.0)*12.0+(vec2<f32>(sx,seed.y)-0.5)*width;
 let ground=riverHeight(height,center/world+0.5);
 let cloudBase=waterRead(weather,center/world+0.5).b;
 let diameter=0.8+seed.w*2.4;let speed=9.65-10.3*exp(-0.6*diameter);
 // Importance-sample the visible part of the cloud-to-ground column. The
 // particles still travel downward at terminal speed, in real seconds.
 let below=select(28.0,70.0,middle);let above=select(36.0,90.0,middle);
 let bottom=max(ground,camera.y-below);let top=max(bottom+.1,min(cloudBase,camera.y+above));
 let span=top-bottom;let flightSeconds=span/speed;let phase=fract(seed.z+time/flightSeconds);
 if(splash>0.5){
  let age=phase*flightSeconds;let radius=select(.10,.16+seed.w*.12,age<.75);
  let xz=center+local.xy*radius;
  return vec3<f32>(xz.x,riverHeight(height,xz/world+.5)+.012,xz.y);
 }
 let side=normalize(vec2<f32>(camera.z-center.y,center.x-camera.x));
 let drop=vec3<f32>(center.x,bottom+(1.0-phase)*span,center.y);
 // A subpixel drop needs a reconstruction footprint to survive rasterizing.
 // Coverage is reduced in the fragment shader instead of making white rods.
 let footprint=max(diameter*.001,length(camera-drop)*pixelAngle*.85);
 let xz=center+side*local.x*footprint;
 return vec3<f32>(xz.x,drop.y+local.y*speed/90.0,xz.y);
}`, [heightRead,waterRead]);

const dropColor=wgslFn(`
fn rainDropColor(p: vec3<f32>, texcoord: vec2<f32>, seed: vec4<f32>, camera: vec3<f32>, time: f32, rate: f32, world: f32, pixelAngle: f32,
 height: texture_2d<f32>, weather: texture_2d<f32>, weatherSampler: sampler,
 impacts: texture_2d_array<f32>, impactSampler: sampler, splash: f32) -> vec4<f32> {
 // Pixel-centre interpolation can extrapolate outside a subpixel triangle.
 // In particular, pow(negative, exponent) is undefined in WGSL.
 let uv=clamp(texcoord,vec2<f32>(0.0),vec2<f32>(1.0));
 let middle=seed.x>=.65;let below=select(28.0,70.0,middle);let above=select(36.0,90.0,middle);
 let column=textureSampleLevel(weather,weatherSampler,p.xz/world+0.5,0.0);
 let selection=fract(seed.x*213.73+seed.y*97.31);
 if(rate<=0.0 || selection>column.g*rate/100.0 || p.y>column.b){discard;}
 let ground=riverHeight(height,p.xz/world+0.5);if(p.y<ground){discard;}
 let horizontal=length(p.xz-camera.xz);
 let fade=select(1.0-smoothstep(32.0,48.0,horizontal),smoothstep(24.0,42.0,horizontal)*(1.0-smoothstep(105.0,145.0,horizontal)),middle);
 let distance=length(p-camera);let diameter=.8+seed.w*2.4;
 let coverage=clamp(diameter*.001/max(distance*pixelAngle*.85,.0001),select(.24,.12,middle),1.0);
 var opacity=max(0.0,sin(uv.y*3.14159))*(1.0-smoothstep(.12,.5,abs(uv.x-.5)))*coverage*.8;
 if(splash>0.5){
  let diameter=.8+seed.w*2.4;let speed=9.65-10.3*exp(-.6*diameter);
  if(middle){discard;}
  let bottom=max(ground,camera.y-below);let top=max(bottom+.1,min(column.b,camera.y+above));
  if(bottom>ground+.05){discard;}
  let flight=(top-bottom)/speed;let age=fract(seed.z+time/flight)*flight;
  if(age>2.5){discard;}
  if(age<.75){
   let frame=i32(clamp(age/.75*31.0,0.0,31.0));let baked=textureSampleLevel(impacts,impactSampler,uv,frame,0.0);
   let normal=normalize(vec3<f32>(baked.r*2.0-1.0,1.0,baked.g*2.0-1.0));
   let glint=pow(max(0.0,dot(normal,normalize(camera-p))),6.0);
   return vec4<f32>(vec3<f32>(.24,.31,.36)+baked.b*.7+glint*.25,baked.a*fade*.8);
  }
  let spot=1.0-smoothstep(.2,.5,length(uv-.5));
  return vec4<f32>(.055,.042,.03,spot*exp(-age)*fade*.22);
 }
 let glint=pow(max(0.0,1.0-abs(uv.x-.5)*2.0),8.0)*(.18+.18*column.r);
 let reflectedSky=vec3<f32>(.52,.60,.70)*(.7+.3*column.r);
 // Alpha blending supplies the transmitted background; keep the droplet's
 // reflected light separate so transparent samples preserve that background.
 let color=reflectedSky+glint;
 let heightFade=smoothstep(0.0,4.0,above-(p.y-camera.y))*smoothstep(0.0,4.0,(p.y-camera.y)+below);
 return vec4<f32>(color,opacity*fade*heightFade*smoothstep(.5,2.0,distance));
}`, [heightRead]);

const surfaceLevel=wgslFn(`
fn surfaceLevel(state: texture_2d<f32>, bed: texture_2d<f32>, uv: vec2<f32>) -> f32 {
 let size=vec2<i32>(textureDimensions(state));let p=uv*vec2<f32>(size)-.5;let i=vec2<i32>(floor(p));let f=fract(p);var level=0.0;var total=0.0;
 for(var z=0;z<2;z++){for(var x=0;x<2;x++){
  let at=clamp(i+vec2<i32>(x,z),vec2<i32>(0),size-1);let w=textureLoad(state,at,0).r;let b=textureLoad(bed,at,0).r;
  let weight=select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,z==1)*smoothstep(8.0,20.0,w);
  level+=(b+w*.001)*weight;total+=weight;
 }}
 if(total<.00001){return waterRead(bed,uv).r-.05;}return level/total;
}`, [waterRead]);
const waterPosition=wgslFn(`
fn waterPosition(local: vec3<f32>, state: texture_2d<f32>, bed: texture_2d<f32>, origin: vec2<f32>, extent: f32) -> vec3<f32> {
 let uv=local.xy+.5;let xz=origin+uv*extent;return vec3<f32>(xz.x,surfaceLevel(state,bed,uv),xz.y);
}`, [surfaceLevel]);

const waterColor=wgslFn(`
fn waterColor(p: vec3<f32>, screen: vec2<f32>, state: texture_2d<f32>, bed: texture_2d<f32>, flux: texture_2d<f32>, height: texture_2d<f32>, soil: texture_2d<f32>,
 origin: vec2<f32>, extent: f32, world: f32, eye: vec3<f32>, sun: vec3<f32>, sunEnergy: vec3<f32>, time: f32,
 background: texture_2d<f32>, backgroundSampler: sampler, weather: texture_2d<f32>, weatherSampler: sampler) -> vec4<f32> {
 let local=(p.xz-origin)/extent;let s=waterRead(state,local);let actualGround=riverHeight(height,p.xz/world+.5);
 let depth=p.y-actualGround;
 if(depth<.015||s.r<12.0){discard;}
 let cell=1.0/f32(textureDimensions(state).x);
 let left=surfaceLevel(state,bed,local-vec2<f32>(cell,0));let right=surfaceLevel(state,bed,local+vec2<f32>(cell,0));
 let back=surfaceLevel(state,bed,local-vec2<f32>(0,cell));let front=surfaceLevel(state,bed,local+vec2<f32>(0,cell));
 let f=waterRead(flux,local);let velocity=vec2<f32>(f.g-f.r,f.a-f.b);let direction=normalize(velocity+vec2<f32>(.00001));
 let speed=clamp(length(velocity)/max(s.r*.001,.001),.02,3.0);
 let along=dot(p.xz,direction);let ripple=sin(along*8.0-time*speed*8.0)*.017+sin(p.x*17.0+p.z*11.0-time*2.0)*.006;
 let n=normalize(vec3<f32>((left-right)/(2.0*cell*extent)+ripple,1.0,(back-front)/(2.0*cell*extent)+ripple*.6));
 let view=normalize(eye-p);let reflection=reflect(-view,n);let fresnel=.02+.98*pow(1.0-max(0.0,dot(n,view)),5.0);
 let scene=textureSampleLevel(background,backgroundSampler,clamp(screen+n.xz*.0015*min(depth,.5),vec2<f32>(.001),vec2<f32>(.999)),0.0).rgb;
 let sediment=waterRead(soil,p.xz/world+.5).a;
 let attenuation=exp(-depth*(vec3<f32>(.5,.25,.13)+sediment*vec3<f32>(1.5,1.7,2.0)));
 let under=scene*attenuation+vec3<f32>(.12,.086,.055)*(1.0-attenuation);
 let cloud=textureSampleLevel(weather,weatherSampler,p.xz/world+.5,0.0);
 let sky=mix(vec3<f32>(.36,.43,.52),vec3<f32>(.12,.30,.63),pow(max(0.0,reflection.y),.45));
 let glint=pow(max(0.0,dot(reflect(-sun,n),view)),220.0)*cloud.r*sunEnergy*.25;
 let edge=min(min(local.x,1.0-local.x),min(local.y,1.0-local.y));
 return vec4<f32>(mix(under,sky,fresnel)+glint,smoothstep(.015,.05,depth)*smoothstep(0.0,.08,edge));
}`, [waterRead,heightRead,surfaceLevel]);

async function load(url){const r=await fetch(url);if(!r.ok)throw Error('Hydrology asset failed: '+url);return r.arrayBuffer();}
function dataTexture(data,n){const t=new THREE.DataTexture(data,n,n,THREE.RGBAFormat,THREE.FloatType);t.minFilter=t.magFilter=THREE.NearestFilter;t.needsUpdate=true;return t;}

export async function createTerrainWater(renderer,material,context){
 const {shadowTarget,refractionTarget,heightBuffer,terrainMeta,camera,sunDirection,sunEnergy}=context;
 const root='/assets/badlands-hydrology/';const meta=await fetch(root+'hydrology.json').then(r=>r.json());
 const soilBytes=await load(root+'retention.bin');
 const impactMeta=await fetch('/assets/rain-impacts/impacts.json').then(r=>r.json());
 const impacts=new THREE.DataArrayTexture(new Uint8Array(await load('/assets/rain-impacts/impact-array.bin')),impactMeta.size,impactMeta.size,impactMeta.frames);
 impacts.minFilter=impacts.magFilter=THREE.LinearFilter;impacts.needsUpdate=true;const impactNode=texture(impacts);
 const soil=dataTexture(new Float32Array(soilBytes),meta.size),soilNode=texture(soil),weatherNode=texture(shadowTarget.texture);
 const height=new THREE.DataTexture(new Float32Array(heightBuffer),terrainMeta.size,terrainMeta.size,THREE.RedFormat,THREE.FloatType);
 height.minFilter=height.magFilter=THREE.NearestFilter;height.needsUpdate=true;const heightNode=texture(height);
 const rainRate=uniform(0),world=uniform(meta.worldSize),seconds=uniform(0),backgroundNode=texture(refractionTarget.texture);
 const pixelAngle=uniform(2*Math.tan(THREE.MathUtils.degToRad(camera.fov*.5))/innerHeight);
 const regional=createWaterGrid(renderer,{size:1024,extent:meta.worldSize,world:meta.worldSize,origin:new THREE.Vector2(-meta.worldSize/2,-meta.worldSize/2),height:heightNode,soil:soilNode,weather:weatherNode});
 const local=createWaterGrid(renderer,{size:256,extent:256,world:meta.worldSize,origin:new THREE.Vector2(-128,-128),height:heightNode,soil:soilNode,weather:weatherNode,global:regional.state});
 const stateNode=regional.state;
 const statsTarget=new THREE.RenderTarget(64,64,{type:THREE.FloatType,minFilter:THREE.NearestFilter,magFilter:THREE.NearestFilter,depthBuffer:false});
 const statsMat=new THREE.NodeMaterial({depthTest:false,depthWrite:false});statsMat.fragmentNode=reduceWater(screenUV,stateNode);const statsQuad=new THREE.QuadMesh(statsMat);
 const wetUV=positionWorld.xz.div(world).add(.5),localUV=positionWorld.xz.sub(local.origin).div(local.extent);
 const localEdge=localUV.x.min(localUV.y).min(localUV.x.oneMinus()).min(localUV.y.oneMinus());
 const wet=mix(surfaceWetness(stateNode,soilNode,wetUV,0),surfaceWetness(local.state,local.bed,localUV,1),smoothstep(0,.15,localEdge)),properties=waterRead(soilNode,wetUV);
 const saturation=wet.x,film=wet.y;
 const wetness=saturation.mul(.82).add(film.mul(.35)).clamp(0,1);
 const flat=smoothstep(.45,.9,material.terrainBaseNormalNode.y);
 const mud=wetness.mul(properties.a).mul(smoothstep(.3,.9,saturation)).mul(flat);
 const puddle=film.mul(flat).mul(smoothstep(.28,.7,material.terrainHeightDetailNode).oneMinus());
 const dryColor=material.colorNode,dryRoughness=material.roughnessNode,dryNormal=material.normalNode;
 const mudColor=vec3(.115,.081,.05).mul(material.terrainHeightDetailNode.mul(.45).add(.75));
 material.colorNode=mix(dryColor.mul(mix(vec3(1),vec3(.46,.43,.39),wetness)),mudColor,mud.mul(.85));
 material.roughnessNode=mix(mix(dryRoughness,.48,wetness),.12,puddle);
 material.clearcoatNode=wetness.mul(.22).add(film.mul(.42)).add(puddle.mul(.36)).clamp(0,1);
 material.clearcoatRoughnessNode=mix(.3,.07,puddle);
 const upright=cameraViewMatrix.mul(vec4(0,1,0,0)).xyz.normalize();
 const bedNormal=cameraViewMatrix.mul(vec4(material.terrainBaseNormalNode,0)).xyz.normalize();
 material.normalNode=mix(dryNormal,bedNormal,mud.mul(.72).add(puddle.mul(.28)).clamp(0,1)).normalize();
 material.clearcoatNormalNode=mix(material.normalNode,upright,puddle.mul(.8)).normalize();

 const geometry=new THREE.PlaneGeometry(1,1,255,255),riversMat=new THREE.NodeMaterial({transparent:true,depthWrite:false,side:THREE.DoubleSide});
 riversMat.positionNode=waterPosition(positionLocal,local.state,local.bed,local.origin,local.extent);
 riversMat.fragmentNode=waterColor(positionWorld,screenUV,local.state,local.bed,local.flux,heightNode,soilNode,local.origin,local.extent,world,uniform(camera.position),sunDirection,sunEnergy,seconds,backgroundNode,sampler(backgroundNode),weatherNode,sampler(weatherNode));
 const riverScene=new THREE.Scene(),rivers=new THREE.Mesh(geometry,riversMat);rivers.frustumCulled=false;riverScene.add(rivers);
 const dropGeometry=new THREE.InstancedBufferGeometry(),dropPlane=new THREE.PlaneGeometry(1,1),dropCount=60000;
 dropGeometry.index=dropPlane.index;dropGeometry.attributes=dropPlane.attributes;
 const seeds=new Float32Array(dropCount*4);let randomSeed=417;
 for(let i=0;i<seeds.length;i++){randomSeed=(Math.imul(randomSeed,1664525)+1013904223)>>>0;seeds[i]=randomSeed/4294967296;}
 dropGeometry.setAttribute('rainSeed',new THREE.InstancedBufferAttribute(seeds,4));dropGeometry.instanceCount=dropCount;
 const rainMeshes=[],rainScene=new THREE.Group();riverScene.add(rainScene);
 for(const splash of [0,1]){
  const m=new THREE.NodeMaterial({transparent:true,depthWrite:false,depthTest:true,side:THREE.DoubleSide,forceSinglePass:true}),seed=attribute('rainSeed','vec4'),origin=uniform(camera.position);
  const rainUV=uv().toVarying('rainUV').setInterpolation('perspective','centroid');
  m.positionNode=dropPosition(positionLocal,seed,origin,seconds,heightNode,weatherNode,world,pixelAngle,splash);
  m.fragmentNode=dropColor(positionWorld,rainUV,seed,origin,seconds,rainRate,world,pixelAngle,heightNode,weatherNode,sampler(weatherNode),impactNode,sampler(impactNode),splash);
  const mesh=new THREE.Mesh(dropGeometry,m);mesh.frustumCulled=false;rainScene.add(mesh);rainMeshes.push(mesh);
 }
 let lastUpdate=0,simSeconds=0,lastStats=0,reading=false,latest=null,cycle=null,regionalPending=0,lastRegional=0,waterActive=false;
 const rainControl=document.getElementById('rain-rate'),speedControl=document.getElementById('weather-speed'),status=document.getElementById('water-status');
 const testButton=document.getElementById('rain-test');
 document.getElementById('solver-check').onclick=async e=>{const button=e.target;button.disabled=true;button.textContent='Checking flow and water balance...';try{const {verifyWaterModel}=await import('./water-test.js');const result=await verifyWaterModel();document.body.dataset.solverTest=JSON.stringify(result);button.textContent='Solver checks passed';}catch(error){document.body.dataset.solverTest=JSON.stringify({pass:false,error:error.stack});button.textContent='Solver check failed';}finally{button.disabled=false;}};
 function endTest(){cycle=null;rainControl.disabled=speedControl.disabled=false;testButton.textContent='Test rain → runoff → dry / 75 s';}
 function reset(){regional.reset();local.reset();simSeconds=0;latest=null;regionalPending=0;waterActive=false;}
 document.getElementById('water-reset').onclick=()=>{endTest();reset();};
 document.getElementById('rain-stop').onclick=()=>{endTest();rainControl.value='0';};
 testButton.onclick=()=>{
  if(cycle){endTest();rainControl.value='0';return;}
  reset();rainControl.value='60';speedControl.value='30';context.setCoverage(.78);cycle={start:null,stage:'raining',records:[]};
  rainControl.disabled=speedControl.disabled=true;testButton.textContent='Stop rain / dry test';delete document.body.dataset.rainTest;
 };
 async function collect(time){
  reading=true;
  try{
   const pixels=await renderer.readRenderTargetPixelsAsync(statsTarget,0,0,64,64);const sums=[0,0,0,0];
   for(let i=0;i<pixels.length;i++)sums[i%4]+=pixels[i]/4096;
   const balance=sums[0]+sums[1]-sums[3];
   latest={simSeconds,meanSurfaceMm:sums[0],meanSoilMm:sums[1],meanFluxMmPerSecond:sums[2],netSupplyMm:sums[3],balanceErrorMm:balance,rainRate:rainRate.value,stage:cycle?.stage||'interactive'};
   document.body.dataset.hydrology=JSON.stringify(latest);
   status.textContent=`${rainRate.value>0?'Raining':'Drying'} / ground ${sums[1].toFixed(2)} mm / runoff ${sums[2].toFixed(4)} mm/s / ${(simSeconds/3600).toFixed(2)} h`;
   if(cycle)cycle.records.push({...latest,time});
  }finally{reading=false;}
 }
 return {meta,rainRate,render(){
  rainScene.visible=rainRate.value>0;
  renderer.autoClear=false;renderer.render(riverScene,camera);
  renderer.autoClear=true;
 },update(time){
  seconds.value=time/1000;
  pixelAngle.value=2*Math.tan(THREE.MathUtils.degToRad(camera.fov*.5))/innerHeight;
  if(cycle){
   if(cycle.start===null)cycle.start=time;
   const elapsed=(time-cycle.start)/1000;
   if(elapsed>=30&&cycle.stage==='raining'){cycle.stage='drying';rainControl.value='0';speedControl.value='60';context.setCoverage(.20);}
   if(elapsed>=75){
    document.body.dataset.rainTest=JSON.stringify({records:cycle.records,finished:true});endTest();speedControl.value='30';
   }
  }
  const elapsed=lastUpdate?Math.min(.1,(time-lastUpdate)/1000):0;lastUpdate=time;
  rainRate.value=Number(rainControl.value);const simDt=Math.min(.48,elapsed*Number(speedControl.value));waterActive||=rainRate.value>0;
  regional.initialize();local.recenter(camera.position.x,camera.position.z);
  if(simDt>0&&waterActive){regionalPending+=simDt;if(time-lastRegional>100){regional.step(regionalPending,rainRate.value);regionalPending=0;lastRegional=time;}const fineSteps=Math.max(1,Math.ceil(simDt/.08));for(let step=0;step<fineSteps;step++)local.step(simDt/fineSteps,rainRate.value);simSeconds+=simDt;}
  if(time-lastStats>1200&&!reading){renderer.setRenderTarget(statsTarget);statsQuad.render(renderer);lastStats=time;void collect(time);}
 }};
}
