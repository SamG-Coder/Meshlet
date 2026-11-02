import * as THREE from 'three/webgpu';
import { wgslFn, texture, texture3D, sampler, uniform, vec3, vec4, uv, screenUV, positionWorld, mix, positionViewDirection, normalView, clearcoatNormalView, roughness, clearcoatRoughness, cameraWorldMatrix } from 'three/tsl';
import { createTerrainWater } from './terrain-water.js';

// All positions are metres. These density functions are shared by the view
// ray marcher and the light-space ground transmittance pass.
const density = wgslFn(`
fn terrainCloudDensity(p: vec3<f32>, seconds: f32, coverage: f32,
 shape: texture_3d<f32>, shapeSampler: sampler, weather: texture_2d<f32>, weatherSampler: sampler) -> f32 {
 let h=(p.y-2600.0)/1800.0;
 if(h<=0.0 || h>=1.0 || coverage<=0.001){return 0.0;}
 let wind=vec3<f32>(seconds*11.0,0.0,seconds*4.0);
 let q=p+wind;
 let w=textureSampleLevel(weather,weatherSampler,q.xz/42000.0,0.0).r;
 let cover=clamp(coverage+(w-0.5)*0.85,0.0,1.0);
 let low=textureSampleLevel(shape,shapeSampler,q/vec3<f32>(8500.0,4200.0,8500.0),0.0);
 let middle=textureSampleLevel(shape,shapeSampler,q/2800.0,0.0).g;
 let base=low.r*0.3+low.g*0.55+middle*0.15;
 let profile=smoothstep(0.0,0.09,h)*(1.0-smoothstep(0.45,1.0,h));
 let body=clamp((base-(1.0-cover))/0.28,0.0,1.0)*profile;
 if(body<0.001){return 0.0;}
 let detail=textureSampleLevel(shape,shapeSampler,q/2200.0,0.0).b;
 return max(0.0,body-(1.0-detail)*0.25*(1.0-body))*1.5;
}`);

const groundShadow = wgslFn(`
fn terrainCloudShadow(p: vec3<f32>, sun: vec3<f32>, seconds: f32, coverage: f32,
 shape: texture_3d<f32>, shapeSampler: sampler, weather: texture_2d<f32>, weatherSampler: sampler) -> f32 {
 let start=max(0.0,(2600.0-p.y)/sun.y);let end=max(0.0,(4400.0-p.y)/sun.y);
 let step=(end-start)/24.0;var optical=0.0;
 for(var i=0;i<24;i++){let at=p+sun*(start+(f32(i)+0.5)*step);
  optical+=terrainCloudDensity(at,seconds,coverage,shape,shapeSampler,weather,weatherSampler)*step*0.008;
 }
 return exp(-optical);
}`, [density]);

const rainColumn = wgslFn(`
fn terrainRainColumn(p: vec3<f32>, seconds: f32, coverage: f32,
 shape: texture_3d<f32>, shapeSampler: sampler, weather: texture_2d<f32>, weatherSampler: sampler) -> vec2<f32> {
 var column=0.0;var base=4400.0;
 for(var i=0;i<16;i++){
  let y=2600.0+(f32(i)+0.5)*112.5;
  let d=terrainCloudDensity(vec3<f32>(p.x,y,p.z),seconds,coverage,shape,shapeSampler,weather,weatherSampler);
  column+=d*112.5;if(d>.015){base=min(base,y);}
 }
 return vec2<f32>(smoothstep(35.0,420.0,column),base);
}`, [density]);

const skyRadiance = wgslFn(`
fn terrainSky(d: vec3<f32>, sun: vec3<f32>) -> vec3<f32> {
 let h=pow(clamp(d.y,0.0,1.0),0.45);let mu=max(0.0,dot(d,sun));
 let sky=mix(vec3<f32>(0.78,0.94,1.16),vec3<f32>(0.12,0.30,0.63),h)+vec3<f32>(1.0,0.9,0.72)*pow(mu,12.0)*0.25;
 let disc=smoothstep(0.999985,0.999992,mu)*18.0;
 return sky+vec3<f32>(1.0,0.93,0.80)*disc;
}`);

const reconstruct = wgslFn(`
fn terrainViewPoint(uv: vec2<f32>, depth: f32, invProjection: mat4x4<f32>, invView: mat4x4<f32>) -> vec3<f32> {
 let v=invProjection*vec4<f32>(uv.x*2.0-1.0,1.0-uv.y*2.0,depth,1.0);
 return (invView*vec4<f32>(v.xyz/v.w,1.0)).xyz;
}`);

const wetEnvironment=wgslFn(`
fn wetEnvironment(p: vec3<f32>, d: vec3<f32>, rough: f32, sun: vec3<f32>, seconds: f32, coverage: f32,
 shape: texture_3d<f32>, shapeSampler: sampler, weather: texture_2d<f32>, weatherSampler: sampler) -> vec3<f32> {
 if(d.y<=0.02){return vec3<f32>(.16,.145,.12);}
 var sky=terrainSky(d,sun);let average=vec3<f32>(.46,.58,.77);
 if(rough<0.65){
  let start=max(0.0,(2600.0-p.y)/d.y);let end=max(start,(4400.0-p.y)/d.y);let step=(end-start)/6.0;var optical=0.0;
  for(var i=0;i<6;i++){optical+=terrainCloudDensity(p+d*(start+(f32(i)+0.5)*step),seconds,coverage,shape,shapeSampler,weather,weatherSampler)*step*.005;}
  let cloud=vec3<f32>(.53,.58,.66)*(0.8+0.3*max(0.0,dot(d,sun)));
  sky=mix(sky,cloud,1.0-exp(-optical));
 }
 return mix(sky,average,rough*rough*.85);
}`, [skyRadiance,density]);

const marchClouds = wgslFn(`
fn terrainCloudView(uv: vec2<f32>, depth: f32, invProjection: mat4x4<f32>, invView: mat4x4<f32>, origin: vec3<f32>,
 sun: vec3<f32>, sunEnergy: vec3<f32>, seconds: f32, coverage: f32,
 shape: texture_3d<f32>, shapeSampler: sampler, weather: texture_2d<f32>, weatherSampler: sampler) -> vec4<f32> {
 let point=terrainViewPoint(uv,min(depth,0.9999999),invProjection,invView);
 let rd=normalize(point-origin);let limit=select(length(point-origin),55000.0,depth>=1.0);
 if(abs(rd.y)<0.00001){return vec4<f32>(0.0);}
 let t0=(2600.0-origin.y)/rd.y;let t1=(4400.0-origin.y)/rd.y;
 let start=max(0.0,min(t0,t1));let end=min(limit,max(t0,t1));
 if(end<=start || coverage<=0.001){return vec4<f32>(0.0);}
 let step=min(500.0,(end-start)/80.0);var trans=1.0;var color=vec3<f32>(0.0);
 let jitter=fract(sin(dot(uv,vec2<f32>(127.1,311.7)))*43758.5453)*0.65+0.18;
 let mu=dot(rd,sun);let g=0.67;
 let forward=(1.0-g*g)/(12.5663706*pow(1.0+g*g-2.0*g*mu,1.5));
 let back=0.9375/(12.5663706*pow(1.0625+0.5*mu,1.5));
 let phase=forward*0.8+back*0.2;
 for(var i=0;i<96;i++){
  let distance=start+(f32(i)+jitter)*step;if(distance>=end || trans<0.008){break;}
  let p=origin+rd*distance;let den=terrainCloudDensity(p,seconds,coverage,shape,shapeSampler,weather,weatherSampler);
  if(den<0.001){continue;}
  let lightStep=max(30.0,(4400.0-p.y)/sun.y/6.0);var optical=0.0;
  for(var j=0;j<6;j++){
   optical+=terrainCloudDensity(p+sun*(f32(j)+0.5)*lightStep,seconds,coverage,shape,shapeSampler,weather,weatherSampler)*lightStep*0.008;
  }
  let sunT=exp(-optical);let height=clamp((p.y-2600.0)/1800.0,0.0,1.0);
  // Direct single scattering plus a low-frequency multi-scattering approximation.
  let illumination=vec3<f32>(0.38,0.46,0.59)*(0.55+height*0.45)+sunEnergy*(phase*sunT+0.065*pow(sunT,0.4)+0.035*pow(sunT,0.15));
  let opacity=1.0-exp(-den*0.008*step);let aerial=exp(-distance*0.000018);
  let lit=illumination*aerial+terrainSky(rd,sun)*(1.0-aerial);
  color+=trans*opacity*lit;trans*=1.0-opacity;
 }
 return vec4<f32>(color,1.0-trans);
}`, [density,skyRadiance,reconstruct]);

const rainShaftNoise=wgslFn(`
fn rainShaftNoise(p: vec2<f32>) -> f32 {
 let i=floor(p);let f=fract(p);let s=f*f*(3.0-2.0*f);
 let a=fract(sin(dot(i,vec2<f32>(127.1,311.7)))*43758.5453);
 let b=fract(sin(dot(i+vec2<f32>(1.0,0.0),vec2<f32>(127.1,311.7)))*43758.5453);
 let c=fract(sin(dot(i+vec2<f32>(0.0,1.0),vec2<f32>(127.1,311.7)))*43758.5453);
 let d=fract(sin(dot(i+vec2<f32>(1.0,1.0),vec2<f32>(127.1,311.7)))*43758.5453);
 return mix(mix(a,b,s.x),mix(c,d,s.x),s.y);
}`);
const composite = wgslFn(`
fn terrainAtmosphere(uv: vec2<f32>, depth: f32, base: vec3<f32>, clouds: vec4<f32>,
 invProjection: mat4x4<f32>, invView: mat4x4<f32>, origin: vec3<f32>, sun: vec3<f32>,
 shadows: texture_2d<f32>, shadowSampler: sampler, heights: texture_2d<f32>, heightSampler: sampler, rainRate: f32, world: f32, seconds: f32) -> vec4<f32> {
 let point=terrainViewPoint(uv,min(depth,0.9999999),invProjection,invView);let ray=point-origin;
 let distance=length(ray);let rd=normalize(ray);let sky=terrainSky(rd,sun);var result=sky;
 if(depth<1.0){
  let air=exp(-distance*0.000015*exp(-max(0.0,(point.y+origin.y)*0.5)/4600.0));
  var sunVisibility=0.0;
  for(var i=0;i<6;i++){
   let p=origin+ray*(f32(i)+0.5)/6.0;
   let projected=p.xz-sun.xz*max(0.0,p.y-900.0)/sun.y;
   sunVisibility+=textureSampleLevel(shadows,shadowSampler,projected/world+0.5,0.0).r/6.0;
  }
  let airColor=sky*(0.78+sunVisibility*0.22);
  result=mix(airColor,base,air);
 }
 if(rainRate>0.0){
  let end=min(select(distance,55000.0,depth>=1.0),18000.0);var optical=0.0;var light=0.0;
  for(var i=0;i<48;i++){
   let a=f32(i)/48.0;let b=f32(i+1)/48.0;let t=(a*a+b*b)*.5*end;
   let step=(b*b-a*a)*end;let p=origin+rd*t;let coord=p.xz/world+.5;
   if(any(coord<vec2<f32>(0.0))||any(coord>vec2<f32>(1.0))){continue;}
   let w=textureSampleLevel(shadows,shadowSampler,coord,0.0);
   let h=textureSampleLevel(heights,heightSampler,coord,0.0).r;
   if(p.y>h && p.y<w.b){
    // Elongated variations form distinct shafts beneath the raining cloud
    // footprint, rather than applying one uniform screen-wide fog layer.
    let q=p.xz+vec2<f32>(.7,.25)*(w.b-p.y)/7.0;
    let drift=vec2<f32>(seconds*11.0,seconds*4.0);
    let broad=rainShaftNoise((q+drift)/380.0);
    let streak=rainShaftNoise(vec2<f32>((q.x+q.y*.37)/32.0,(p.y+seconds*7.0)/42.0));
    let shaft=(.12+.88*smoothstep(.20,.76,broad))*(.3+.7*streak);
    let baseFade=smoothstep(0.0,90.0,w.b-p.y);
    let d=w.g*shaft*baseFade*step*.00024*rainRate/60.0;
    optical+=d;light+=d*w.r;
   }
  }
  let curtain=vec3<f32>(.25,.30,.36)*(0.7+0.6*light/max(optical,.001));
  result=mix(result,curtain,1.0-exp(-optical));
 }
 return vec4<f32>(result*(1.0-clouds.a)+clouds.rgb,1.0);
}`, [skyRadiance,reconstruct,rainShaftNoise]);

async function bytes(url){const r=await fetch(url);if(!r.ok)throw Error('Lighting asset failed: '+url);return r.arrayBuffer();}
function halfTexture(buffer,size){
 const t=new THREE.DataTexture(new Uint16Array(buffer),size,size,THREE.RGBAFormat,THREE.HalfFloatType);
 t.minFilter=THREE.LinearFilter;t.magFilter=THREE.LinearFilter;t.needsUpdate=true;return t;
}

export async function createTerrainLighting(renderer,scene,camera,material,heightBuffer,terrainMeta){
 const root='/assets/badlands-lighting-hq/',cloudRoot='/assets/cloud-volume/';
 const [meta,noiseMeta]=await Promise.all([fetch(root+'lighting.json').then(r=>r.json()),fetch(cloudRoot+'noise.json').then(r=>r.json())]);
 const [skyBytes,sunBytes,noiseBytes,weatherBytes,visibility]=await Promise.all([
  bytes(root+'sky-irradiance.bin'),bytes(root+'sun-bounce.bin'),bytes(cloudRoot+'shape-rgba.bin'),bytes(cloudRoot+'weather-r.bin'),new THREE.TextureLoader().loadAsync(root+'sun-visibility.png')
 ]);
 const skyMap=halfTexture(skyBytes,meta.size),sunMap=halfTexture(sunBytes,meta.size);
 visibility.minFilter=THREE.LinearMipmapLinearFilter;visibility.magFilter=THREE.LinearFilter;
 const shape=new THREE.Data3DTexture(new Uint8Array(noiseBytes),noiseMeta.volumeSize,noiseMeta.volumeSize,noiseMeta.volumeSize);
 shape.format=THREE.RGBAFormat;shape.minFilter=shape.magFilter=THREE.LinearFilter;shape.wrapS=shape.wrapT=shape.wrapR=THREE.RepeatWrapping;shape.needsUpdate=true;
 const weather=new THREE.DataTexture(new Uint8Array(weatherBytes),noiseMeta.weatherSize,noiseMeta.weatherSize,THREE.RedFormat);
 weather.minFilter=weather.magFilter=THREE.LinearFilter;weather.wrapS=weather.wrapT=THREE.RepeatWrapping;weather.needsUpdate=true;
 const sourceHeight=new Float32Array(heightBuffer),heightSize=1024,smallHeight=new Uint16Array(heightSize*heightSize);
 for(let z=0;z<heightSize;z++)for(let x=0;x<heightSize;x++)smallHeight[z*heightSize+x]=THREE.DataUtils.toHalfFloat(sourceHeight[Math.round(z/(heightSize-1)*(terrainMeta.size-1))*terrainMeta.size+Math.round(x/(heightSize-1)*(terrainMeta.size-1))]);
 const heightMap=new THREE.DataTexture(smallHeight,heightSize,heightSize,THREE.RedFormat,THREE.HalfFloatType);heightMap.minFilter=heightMap.magFilter=THREE.LinearFilter;heightMap.needsUpdate=true;
 const sunDirection=uniform(new THREE.Vector3(...meta.sunDirection)),sunEnergy=uniform(new THREE.Vector3(5.0,4.65,4.12));
 const cloudTime=uniform(360),coverage=uniform(.52),cameraPosition=uniform(camera.position),invProjection=uniform(camera.projectionMatrixInverse),invView=uniform(camera.matrixWorld);
 const shapeNode=texture3D(shape),weatherNode=texture(weather),cloudArgs=[shapeNode,sampler(shapeNode),weatherNode,sampler(weatherNode)];
 const shadowTarget=new THREE.RenderTarget(1024,1024,{type:THREE.HalfFloatType,depthBuffer:false});
 const shadowMat=new THREE.NodeMaterial({depthTest:false,depthWrite:false});
 const shadowWorld=vec3(screenUV.x.sub(.5).mul(meta.worldSize),texture(heightMap,screenUV).r,screenUV.y.sub(.5).mul(meta.worldSize));
 shadowMat.fragmentNode=vec4(groundShadow(shadowWorld,sunDirection,cloudTime,coverage,...cloudArgs),rainColumn(shadowWorld,cloudTime,coverage,...cloudArgs),1);
 const shadowQuad=new THREE.QuadMesh(shadowMat);
 const groundUV=positionWorld.xz.div(meta.worldSize).add(.5);
 const movingShadow=texture(shadowTarget.texture,groundUV).r;
 const sun=new THREE.DirectionalLight(0xffffff,1);sun.position.copy(sunDirection.value).multiplyScalar(10000);
 sun.colorNode=sunEnergy.mul(texture(visibility,uv()).r).mul(movingShadow);scene.add(sun);
 // Data textures have row zero at UV zero; image textures already flip on upload.
 const lightUV=vec3(groundUV.x,groundUV.y,0).xy;
 const irradiance=texture(skyMap,lightUV).rgb.mul(mix(.78,1,movingShadow)).add(texture(sunMap,lightUV).rgb.mul(sunEnergy).mul(mix(.15,1,movingShadow)));
 material.setupLightMap=()=>new THREE.IrradianceNode(irradiance);
 scene.fog=null;scene.background=new THREE.Color(0x000000);
 renderer.toneMapping=THREE.AgXToneMapping;renderer.toneMappingExposure=1.0;
 const sceneTarget=new THREE.RenderTarget(1,1,{type:THREE.HalfFloatType,samples:4,depthTexture:new THREE.DepthTexture(1,1,THREE.FloatType)});
 const refractionTarget=new THREE.RenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false});
 const copyMaterial=new THREE.NodeMaterial({depthTest:false,depthWrite:false});copyMaterial.fragmentNode=texture(sceneTarget.texture,screenUV);const copyQuad=new THREE.QuadMesh(copyMaterial);
 const cloudTarget=new THREE.RenderTarget(1,1,{type:THREE.HalfFloatType,depthBuffer:false});
 const depth=texture(sceneTarget.depthTexture,screenUV).r;
 const cloudMat=new THREE.NodeMaterial({depthTest:false,depthWrite:false});
 cloudMat.fragmentNode=marchClouds(screenUV,depth,invProjection,invView,cameraPosition,sunDirection,sunEnergy,cloudTime,coverage,...cloudArgs);
 const cloudQuad=new THREE.QuadMesh(cloudMat);
 const compositeMat=new THREE.NodeMaterial({depthTest:false,depthWrite:false});
 const shadowTexture=texture(shadowTarget.texture);
 const compositeQuad=new THREE.QuadMesh(compositeMat);
 const fullscreenPipeline=new THREE.StandardDrawingPipeline();
 let previous=0,shadowUpdated=-Infinity,paused=false;
 const coverageControl=document.getElementById('cloud-cover');
 coverageControl.addEventListener('input',()=>{coverage.value=Number(coverageControl.value);shadowUpdated=-Infinity;});
 document.getElementById('cloud-pause').addEventListener('change',e=>paused=e.target.checked);
 document.getElementById('exposure').addEventListener('input',e=>renderer.toneMappingExposure=Number(e.target.value));
 document.getElementById('lighting-status').textContent=`Baked path lighting / ${meta.maxDiffuseBounces} bounces / volumetric clouds`;
 const water=await createTerrainWater(renderer,material,{shadowTarget,refractionTarget,heightBuffer,terrainMeta,camera,sunDirection,sunEnergy,
  setCoverage(value){coverage.value=value;coverageControl.value=String(value);shadowUpdated=-Infinity;}});
 const heightTextureNode=texture(heightMap);
 compositeMat.fragmentNode=composite(screenUV,depth,texture(sceneTarget.texture,screenUV).rgb,texture(cloudTarget.texture,screenUV),invProjection,invView,cameraPosition,sunDirection,shadowTexture,sampler(shadowTexture),heightTextureNode,sampler(heightTextureNode),water.rainRate,meta.worldSize,cloudTime);
 class SkySpecular extends THREE.LightingNode {
  static get type(){return 'TerrainSkySpecular';}
  setup(builder){
   const reflected=positionViewDirection.negate().reflect(normalView).transformDirection(cameraWorldMatrix);
   builder.context.radiance.addAssign(wetEnvironment(positionWorld,reflected,roughness,sunDirection,cloudTime,coverage,...cloudArgs));
   if(builder.context.lightingModel.clearcoatRadiance){
    const coatDirection=positionViewDirection.negate().reflect(clearcoatNormalView).transformDirection(cameraWorldMatrix);
    builder.context.lightingModel.clearcoatRadiance.addAssign(wetEnvironment(positionWorld,coatDirection,clearcoatRoughness,sunDirection,cloudTime,coverage,...cloudArgs));
   }
  }
 }
 // Only specular environment light is added here; diffuse sky light is baked.
 material.setupEnvironment=()=>new SkySpecular();
 let vehicleScene=null;
 function prepareVehicleMaterial(mat){
  mat.setupEnvironment=()=>new SkySpecular();
  // Ground irradiance supplies local canyon occlusion/bounce; orientation-aware
  // hemisphere lighting is added by the vehicle scene for vertical body panels.
  mat.setupLightMap=()=>new THREE.IrradianceNode(irradiance.mul(.35));
 }
 function configureVehicleScene(s){
  const light=new THREE.DirectionalLight(0xffffff,1);
  light.colorNode=sunEnergy.mul(texture(visibility,vec3(groundUV.x,groundUV.y.oneMinus(),0).xy).r).mul(movingShadow);
  light.castShadow=true;light.shadow.mapSize.set(1024,1024);light.shadow.camera.left=-12;light.shadow.camera.right=12;light.shadow.camera.top=12;light.shadow.camera.bottom=-12;light.shadow.camera.near=.5;light.shadow.camera.far=80;light.shadow.bias=-.0002;light.shadow.normalBias=.025;
  s.add(light,light.target,new THREE.HemisphereLight(0xb8ccdf,0x6c6555,.9));return light;
 }
 function resize(w,h){sceneTarget.setSize(w,h);refractionTarget.setSize(w,h);cloudTarget.setSize(Math.ceil(w*.5),Math.ceil(h*.5));}
 resize(innerWidth,innerHeight);
 return {meta,waterMeta:water.meta,resize,prepareVehicleMaterial,configureVehicleScene,sunDirection:sunDirection.value,setVehicleScene(s){vehicleScene=s;},render(time){
  if(previous&&!paused)cloudTime.value+=Math.min(.1,(time-previous)/1000);previous=time;
  const pipeline=renderer.drawingPipeline;
  renderer.drawingPipeline=fullscreenPipeline;
  if(time-shadowUpdated>80){renderer.setRenderTarget(shadowTarget);shadowQuad.render(renderer);shadowUpdated=time;}
  water.update(time);
  renderer.drawingPipeline=pipeline;
  renderer.setRenderTarget(sceneTarget);renderer.render(scene,camera);
  renderer.drawingPipeline=fullscreenPipeline;
  if(vehicleScene){const clear=renderer.autoClear;renderer.autoClear=false;renderer.render(vehicleScene,camera);renderer.autoClear=clear;}
  renderer.setRenderTarget(refractionTarget);copyQuad.render(renderer);
  renderer.setRenderTarget(sceneTarget);
  water.render();
  renderer.setRenderTarget(cloudTarget);cloudQuad.render(renderer);
  renderer.setRenderTarget(null);compositeQuad.render(renderer);
  renderer.drawingPipeline=pipeline;
  document.body.dataset.lighting=JSON.stringify({source:meta.source,bounces:meta.maxDiffuseBounces,samples:meta.samplesPerTexel,shadowSize:meta.shadowSize,coverage:coverage.value,cloudSeconds:cloudTime.value,paused,exposure:renderer.toneMappingExposure});
 }};
}
