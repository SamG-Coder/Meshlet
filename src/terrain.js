import * as THREE from 'three/webgpu';
import { attribute, positionLocal, cameraPosition, uniform, vec3, uv, texture, mix, max, smoothstep, uniformArray, int, uint, vec2, vec4, select } from 'three/tsl';
import { MeshletDrawingPipeline } from 'three/addons/pipelines/MeshletDrawingPipeline.js';
import { ready } from 'three/addons/utils/MeshletGeometryUtils.js';
import { loadGroundVariants, groundVariantNodes } from './ground-material.js';
import { unpackMeshlets } from './meshlet-transfer.js';
import { balanceCover, coverEdgeFlags } from './terrain-hlod.js';
import { makeSurface, hash } from './terrain-surface.js';
import { createTerrainLighting } from './terrain-lighting.js';

const $=id=>document.getElementById(id),fmt=n=>Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:2}).format(n);
const state={errors:[],ready:false,records:[],transitions:0,workerMs:[],installMs:[]};
window.terrainLab=state;
const status=t=>$('status').textContent=t;
function fail(e){const message=String(e?.message||e);if(!state.errors.includes(message))state.errors.push(message);document.body.dataset.firstGpuError=state.errors[0];document.body.dataset.errorStack=e?.stack||'';status(message);}
addEventListener('error',e=>fail(e.message));addEventListener('unhandledrejection',e=>fail(e.reason));
const key=t=>`${t.level}/${t.tx}/${t.tz}`;
const children=t=>[0,1,2,3].map(i=>({level:t.level+1,tx:t.tx*2+(i%2),tz:t.tz*2+Math.floor(i/2)}));
const percentile=(a,p)=>a.length?[...a].sort((a,b)=>a-b)[Math.min(a.length-1,Math.floor(a.length*p))]:0;

async function start(){
 const loadStarted=performance.now();
 const root='/assets/badlands-v3/';
 document.body.dataset.terrainSource=root;
 const meta=await fetch(root+'simulation.json').then(r=>{if(!r.ok)throw Error('The new erosion bake is still being prepared.');return r.json();});
 const world=meta.worldSize,segments=256,maxLevel=Math.round(Math.log2(meta.logicalRenderGrid/segments));
 state.simulation=meta;state.modelYears=meta.simulatedYears||0;state.logicalTriangles=meta.logicalTriangles;
 $('logical').textContent=fmt(meta.logicalTriangles);$('area').textContent=`${(world/1000).toFixed(3)} km across / ${(world*world/1e6).toFixed(0)} km²`;
 $('provenance').textContent=`${meta.size.toLocaleString()}² source heightfield / ${(world/(meta.size-1)).toFixed(2)} m per height sample. Finer mesh spacing adds interpolation and shallow material relief. Ten independently generated 1,254² sources baked into 2K material arrays.${root.includes('badlands-rain')?' Experimental rain refinement; no calibrated geological age.':''}`;
 status('Loading eroded heightfield and surface maps...');
 const [heightBuffer,fieldBuffer,groundBakes]=await Promise.all([fetch(root+'height.bin').then(r=>r.arrayBuffer()),fetch(root+'preview-fields.bin').then(r=>r.arrayBuffer()),loadGroundVariants()]);
 const surface=makeSurface(new Float32Array(heightBuffer),new Float32Array(fieldBuffer),meta.size,meta.previewSize,world,groundBakes.height);
 const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('Hardware WebGPU is required.');
 const storageLimit=adapter.limits.maxStorageBufferBindingSize;
 const vertexCount=(segments+1)**2+(segments+1)*4;
 const poolCount=storageLimit>=1024**3?2:1;
 const tilesPerPool=Math.max(48,Math.floor(storageLimit*.82/(vertexCount*48)));
 const maxCacheTiles=Math.min(768,tilesPerPool*poolCount);
 const maxLeaves=Math.floor(maxCacheTiles*.72);
 const maximumOption=$('budget').options[$('budget').options.length-1];maximumOption.value=String(maxLeaves);maximumOption.textContent='Maximum / '+maxLeaves+' dense tiles';maximumOption.selected=true;
 const renderer=new THREE.WebGPURenderer({antialias:true,requiredLimits:{maxBufferSize:adapter.limits.maxBufferSize,maxStorageBufferBindingSize:storageLimit,maxSampledTexturesPerShaderStage:Math.min(24,adapter.limits.maxSampledTexturesPerShaderStage)}});
 document.body.dataset.gpuLimits=JSON.stringify({storageLimit,maxBuffer:adapter.limits.maxBufferSize,poolCount,maxCacheTiles,maxLeaves,segments});
 renderer.setPixelRatio(1);renderer.setSize(innerWidth,innerHeight);await renderer.init();await ready;
 if(!renderer.backend.isWebGPUBackend)throw Error('Hardware WebGPU required.');
 renderer.backend.device.addEventListener('uncapturederror',e=>fail(e.error));
 renderer.backend.device.lost.then(e=>{if(e.reason!=='destroyed')fail('GPU device lost: '+e.message);});
 renderer.info.autoReset=false;$('viewport').append(renderer.domElement);
 const pipeline=new MeshletDrawingPipeline({lodThreshold:0});renderer.drawingPipeline=pipeline;
 for(let p=0;p<poolCount;p++)pipeline.reserveGeometry(vertexCount*3*Math.ceil(maxCacheTiles/poolCount),Math.ceil((segments*segments*6+segments*24)*1.08*Math.ceil(maxCacheTiles/poolCount)),'terrain-'+p);
 const scene=new THREE.Scene();
 const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,.1,90000);
 const viewTarget=new THREE.Vector3(),lookEuler=new THREE.Euler(0,0,0,'YXZ');
 let fly=false,verticalVelocity=0,dragLook=null,canLock=true;
 const canvas=renderer.domElement;
 function setFly(value){fly=value;verticalVelocity=0;$('camera-mode').textContent=fly?'Fly mode / F to walk':'Walk mode / F to fly';document.body.dataset.cameraMode=fly?'fly':'walk';}
 $('camera-mode').onclick=()=>setFly(!fly);
 const locked=()=>document.pointerLockElement===canvas;
 function look(dx,dy){lookEuler.setFromQuaternion(camera.quaternion);lookEuler.y-=dx*.0016;lookEuler.x=Math.max(-Math.PI/2+.02,Math.min(Math.PI/2-.02,lookEuler.x-dy*.0016));camera.quaternion.setFromEuler(lookEuler);}
 canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;dragLook={x:e.clientX,y:e.clientY,distance:0};canvas.setPointerCapture(e.pointerId);});
 canvas.addEventListener('pointermove',e=>{if(locked()||!dragLook)return;const dx=e.clientX-dragLook.x,dy=e.clientY-dragLook.y;look(dx,dy);dragLook.distance+=Math.abs(dx)+Math.abs(dy);dragLook.x=e.clientX;dragLook.y=e.clientY;});
 canvas.addEventListener('pointerup',async e=>{
  const click=dragLook&&dragLook.distance<3;dragLook=null;
  if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);
  if(click&&canLock&&!locked()&&canvas.requestPointerLock){
   try{await canvas.requestPointerLock();}catch{canLock=false;document.body.dataset.lookInput='drag';}
  }
 });
 document.addEventListener('mousemove',e=>{if(locked())look(e.movementX,e.movementY);});
 document.addEventListener('pointerlockchange',()=>document.body.classList.toggle('looking',locked()));
 document.addEventListener('pointerlockerror',()=>{canLock=false;document.body.dataset.lookInput='drag';});
 canvas.addEventListener('pointercancel',()=>dragLook=null);
 setFly(false);
 const loader=new THREE.TextureLoader();loader.manager.onError=url=>fail("Texture failed: "+url);
 const [albedo,orm,terrainNormal,geologyMap]=await Promise.all(['albedo.jpg','orm.png','terrain-normal.png',...(meta.materialMask?['geology.png']:[])].map(p=>loader.loadAsync(root+p)));
 albedo.colorSpace=THREE.SRGBColorSpace;
 for(const t of [albedo,terrainNormal,orm])t.anisotropy=8;
 const clock=uniform(0),lod=attribute('lodData','vec4');
 const packed=uint(lod.z),tileUV=vec2(packed.bitAnd(511),packed.shiftRight(9)).div(segments);
 const birthValues=Array.from({length:2048},()=>new THREE.Vector4(-100,-100,0,0)),birthUniform=uniformArray(birthValues,'vec4');
 const edgeValues=Array.from({length:2048},()=>new THREE.Vector4()),edgeUniform=uniformArray(edgeValues,'vec4'),freeSlots=Array.from({length:2048},(_,i)=>2047-i);
 const edgeBirthValues=Array.from({length:2048},()=>new THREE.Vector4(-100,-100,-100,-100));
 const edgeBirthUniform=uniformArray(edgeBirthValues,'vec4');
 const edge=edgeUniform.element(int(lod.w));
 const borderArrival=smoothstep(0,.9,clock.sub(edgeBirthUniform.element(int(lod.w)))).oneMinus();
 const proximity=vec4(tileUV.x,tileUV.x.oneMinus(),tileUV.y,tileUV.y.oneMinus());
 const fade=smoothstep(0,6/segments,proximity).oneMinus();
 const toCoarse=fade.mul(max(edge.negate().max(0),borderArrival));
 const toFine=fade.mul(edge.max(0));
 const edgeCoarse=max(max(toCoarse.x,toCoarse.y),max(toCoarse.z,toCoarse.w));
 const edgeFine=max(max(toFine.x,toFine.y),max(toFine.z,toFine.w));
 const distance=positionLocal.xz.distance(cameraPosition.xz);
 const distanceMorph=smoothstep(lod.y.mul(4.8),lod.y.mul(6.4),distance);
 const arrivalMorph=smoothstep(0,.9,clock.sub(birthUniform.element(int(lod.w)).x)).oneMinus();
 const collapseStart=birthUniform.element(int(lod.w)).y;
 const collapseMorph=smoothstep(0,.9,clock.sub(collapseStart)).mul(select(collapseStart.greaterThanEqual(0),1,0));
 const morph=max(max(max(distanceMorph,arrivalMorph),collapseMorph),edgeCoarse).mul(edgeFine.oneMinus()).mul(select(lod.y.lessThan(world),1,0));
 const material=new THREE.MeshPhysicalNodeMaterial({map:albedo,roughnessMap:orm,aoMap:orm,aoMapIntensity:.7,roughness:1});
 material.positionNode=vec3(positionLocal.x,mix(positionLocal.y,lod.x,morph),positionLocal.z);
 material.userData.meshletConservativeBounds=true; // Worker bounds contain both morph endpoints.
 material.normalScale.set(.7,.7);
 const naturalNodes=groundVariantNodes(groundBakes,albedo,terrainNormal,orm,geologyMap);Object.assign(material,naturalNodes);material.normalMap=null;
 status('Loading path-traced lighting and cloud volume...');
 const lighting=await createTerrainLighting(renderer,scene,camera,material,heightBuffer,meta);
 state.lighting=lighting.meta;
 Object.assign(naturalNodes,{colorNode:material.colorNode,normalNode:material.normalNode,roughnessNode:material.roughnessNode});
 const fields=new Float32Array(fieldBuffer),pn=meta.previewSize,inspection=new Map();
 function inspectionMap(mode){
  if(inspection.has(mode))return inspection.get(mode);
  const bytes=new Uint8Array(pn*pn*4);
  for(let i=0;i<pn*pn;i++){const o=i*11;let rgb;
   if(mode==='sediment'){const total=fields[o+1]+fields[o+2]+fields[o+3]+.001;rgb=[(fields[o+1]*160+fields[o+2]*205+fields[o+3]*115)/total,(fields[o+1]*155+fields[o+2]*190+fields[o+3]*100)/total,(fields[o+1]*145+fields[o+2]*155+fields[o+3]*75)/total];}
   else{const v=mode==='wetness'?Math.min(1,(fields[o+4]+fields[o+5])*.4):Math.min(1,Math.log1p(fields[o+(mode==='erosion'?6:7)])/5);rgb=mode==='wetness'?[30+20*v,45+130*v,45+195*v]:[30+225*v,40+95*v,55-35*v];}
   bytes.set([...rgb,255],i*4);
  }
  const t=new THREE.DataTexture(bytes,pn,pn);t.flipY=true;t.colorSpace=THREE.SRGBColorSpace;t.magFilter=THREE.LinearFilter;t.minFilter=THREE.LinearFilter;t.needsUpdate=true;inspection.set(mode,t);return t;
 }
 $('surface').onchange=()=>{const natural=$('surface').value==='natural';material.map=natural?albedo:inspectionMap($('surface').value);material.colorNode=natural?naturalNodes.colorNode:null;material.needsUpdate=true;};
 $('lod').onchange=()=>{pipeline.lodThreshold=Number($('lod').value);pump();};$('clusters').onchange=()=>pipeline.debug=$('clusters').checked;

 const cache=new Map();
 const workers=Array.from({length:Math.min(4,Math.max(2,Math.floor((navigator.hardwareConcurrency||4)/2)))},()=>({worker:new Worker(new URL('./terrain-worker.js',import.meta.url),{type:'module'}),ready:false,busy:false}));
 state.workerCount=workers.length;
 let groundClearance=2;
 let workerReady=false,desired=[],wanted=new Set(),split=new Set(),queue=[],pending=new Set(),incoming=[],epoch=0,lastPlan=0,plannedPosition=null,plannedBudget=0;
 let samples=null,activeSplits=new Set(),lastCover='',lastSwitch=0,hasCollapses=false;
 let workingGroup=null;
 function publish(){
  const meshes=[...cache.values()].filter(e=>e.mesh.visible);
  state.visibleTriangles=meshes.reduce((a,e)=>a+e.sourceTriangles,0);
  state.residentTriangles=[...cache.values()].reduce((a,e)=>a+e.sourceTriangles,0);
  state.tileCount=meshes.length;state.cachedTiles=cache.size;state.maxTileLevel=Math.max(0,...meshes.map(e=>e.level));
  state.coverage=meshes.reduce((a,e)=>a+(world/2**e.level)**2,0)/(world*world);
  state.ready=desired.length>0&&desired.every(t=>cache.has(key(t)))&&!incoming.length;
  state.materialVariants=groundBakes.meta.count;
  if(state.ready&&!state.initialReadyMs)state.initialReadyMs=performance.now()-loadStarted;
  document.body.dataset.terrainState=JSON.stringify({firstGeometryMs:state.firstGeometryMs,initialReadyMs:state.initialReadyMs,segments,workers:workers.length,ready:state.ready,tiles:state.tileCount,cached:cache.size,spacing:world/(segments*2**state.maxTileLevel),pending:queue.length,materialVariants:state.materialVariants,modelYears:state.modelYears,coverage:state.coverage,transitions:state.transitions,errors:state.errors.length,workerP95:percentile(state.workerMs,.95),installP95:percentile(state.installMs,.95),cameraMode:fly?'fly':'walk',orientation:camera.quaternion.toArray(),camera:camera.position.toArray(),sharedWorkerData:state.sharedWorkerData});
  if(!samples&&!state.errors.length)status(`${state.ready?'Ready':'Streaming'} / ${state.tileCount} HLOD tiles / ${(world/(segments*2**state.maxTileLevel)).toFixed(3)} m / 10 baked material variants`);
 }
 function distanceTo(t){const span=world/2**t.level,x=-world/2+(t.tx+.5)*span,z=-world/2+(t.tz+.5)*span;return Math.hypot(x-camera.position.x,z-camera.position.z,groundClearance);}
 function plan(force=false){
  if(!workerReady)return;const budget=Math.min(maxLeaves,Number($('budget').value));
  if(!force&&plannedPosition&&plannedPosition.distanceTo(camera.position)<8&&plannedBudget===budget)return;
  plannedPosition=camera.position.clone();plannedBudget=budget;epoch++;groundClearance=Math.max(1.8,camera.position.y-surface.sample(camera.position.x,camera.position.z));
  const leaves=[{level:0,tx:0,tz:0}];
  while(leaves.length+3<=budget){let best=-1,score=0;
   for(let i=0;i<leaves.length;i++){const t=leaves[i];if(t.level>=maxLevel)continue;const span=world/2**t.level,d=distanceTo(t);const threshold=split.has(key(t))?4.0:3.2;const scoreHere=span/(d+1);if(d<span*threshold&&scoreHere>score){best=i;score=scoreHere;}}
   if(best<0)break;leaves.push(...children(leaves.splice(best,1)[0]));
  }
  desired=leaves;wanted=new Set();split=new Set();const nodes=new Map();
  for(const leaf of leaves){let t=leaf;while(true){const id=key(t);wanted.add(id);nodes.set(id,t);if(!t.level)break;t={level:t.level-1,tx:Math.floor(t.tx/2),tz:Math.floor(t.tz/2)};split.add(key(t));}}
  const inflight=new Set(incoming.map(d=>d.id));
  queue=[...nodes.values()].filter(t=>!cache.has(key(t))&&!pending.has(key(t))&&!inflight.has(key(t))).sort((a,b)=>a.level-b.level||distanceTo(a)-distanceTo(b));
  updateCover();pump();
 }
 function pump(){
  if(!workerReady)return;
  const parent=t=>t.level?(t.level-1)+'/'+Math.floor(t.tx/2)+'/'+Math.floor(t.tz/2):null;
  for(const channel of workers){
   if(!channel.ready||channel.busy)continue;
   if(cache.size+pending.size+incoming.length>=maxCacheTiles)break;
   const available=queue.filter(t=>!t.level||cache.has(parent(t)));
   let t,upgrade=false;
   if(available.length){
    let candidates=available.filter(t=>parent(t)===workingGroup);
    if(!candidates.length){
     available.sort((a,b)=>{const priority=t=>{const p={level:Math.max(0,t.level-1),tx:Math.floor(t.tx/2),tz:Math.floor(t.tz/2)};return (world/2**p.level)/(distanceTo(p)+2);};return priority(b)-priority(a);});
     workingGroup=parent(available[0]);candidates=available.filter(t=>parent(t)===workingGroup);
    }
    t=candidates[0];queue.splice(queue.indexOf(t),1);
   }else if(!queue.length&&Number($('lod').value)>0){
    t=[...cache.values()].filter(e=>e.data.lodCount===1&&!e.upgrading).sort((a,b)=>Number(b.mesh.visible)-Number(a.mesh.visible)||distanceTo(a)-distanceTo(b))[0];
    if(t){t.upgrading=true;upgrade=true;}
   }
   if(!t)continue;
   const id=key(t);pending.add(id);channel.busy=true;
   channel.worker.postMessage({level:t.level,tx:t.tx,tz:t.tz,id,segments,fullDetail:Number($('lod').value)===0,upgrade});
  }
 }
 function updateCover(){
  const visible=new Set(),newSplits=new Set(),previouslyVisible=new Set([...cache].filter(([,e])=>e.mesh.visible).map(([id])=>id));
  hasCollapses=false;
  function visit(t){const id=key(t),entry=cache.get(id);if(!entry)return;const c=children(t),span=world/2**t.level;
   if(split.has(id)&&c.every(t=>cache.has(key(t)))){
    delete entry.collapseStarted;
    for(const child of c){const e=cache.get(key(child)),u=birthValues[e.slot];if(u.y>=0){const t=Math.min(1,(clock.value-u.y)/.9);u.x=clock.value-.9*(1-t);u.y=-100;}}
    newSplits.add(id);c.forEach(visit);
   }else if(activeSplits.has(id)&&c.every(t=>cache.has(key(t)))){
    // Collapse one level at a time. Children reach their actual parent surface
    // before it replaces them, including when the old detail was still nearby.
    if(c.some(child=>activeSplits.has(key(child)))){newSplits.add(id);c.forEach(visit);hasCollapses=true;}
    else{
     if(entry.collapseStarted===undefined)entry.collapseStarted=clock.value;
     if(clock.value-entry.collapseStarted<.9){
      for(const child of c){const e=cache.get(key(child));birthValues[e.slot].y=entry.collapseStarted;visible.add(key(child));}
      newSplits.add(id);hasCollapses=true;
     }else{for(const child of c)birthValues[cache.get(key(child)).slot].y=-100;delete entry.collapseStarted;visible.add(id);}
    }
   }else visible.add(id);
  }
  visit({level:0,tx:0,tz:0});
  const balanced=balanceCover([...visible].map(id=>cache.get(id)));
  visible.clear();for(const t of balanced)visible.add(key(t));
  const flags=coverEdgeFlags(balanced);
  for(const [id,f]of flags){
   const slot=cache.get(id).slot,old=edgeValues[slot].toArray();
   for(let i=0;i<4;i++)if(old[i]===-1&&f[i]===0)edgeBirthValues[slot].setComponent(i,clock.value);
   edgeValues[slot].fromArray(f);
  }
  activeSplits=newSplits;
  for(const [id,e]of cache){
   e.mesh.visible=visible.has(id);
   if(e.mesh.visible){
    if(!previouslyVisible.has(id)){
     let parent=e,refining=false;
     while(parent.level){parent={level:parent.level-1,tx:Math.floor(parent.tx/2),tz:Math.floor(parent.tz/2)};if(previouslyVisible.has(key(parent))){refining=true;break;}}
     birthValues[e.slot].x=refining?clock.value:-100;e.everVisible=true;
    }
    e.used=epoch;
   }
  }
  const signature=[...visible].sort().join('|');if(signature!==lastCover){state.transitions++;lastSwitch=performance.now();lastCover=signature;}
  // Warm cache budget counts parents as well as leaves. Never evict an active cover or its parents.
  const limit=Math.min(maxCacheTiles-(queue.length?workers.length*2:0),Number($('budget').value)*2+48);
  for(const [id,e]of [...cache].sort((a,b)=>a[1].used-b[1].used)){if(cache.size<=limit)break;if(wanted.has(id)||visible.has(id)||activeSplits.has(id))continue;scene.remove(e.mesh);e.mesh.geometry.dispose();freeSlots.push(e.slot);edgeValues[e.slot].set(0,0,0,0);edgeBirthValues[e.slot].set(-100,-100,-100,-100);birthValues[e.slot].x=-100;cache.delete(id);}
  publish();
 }
 function install(d){
  const old=cache.get(d.id);
  if((old&&!d.upgrade)||(!old&&!wanted.has(d.id)))return false;
  const begin=performance.now(),g=new THREE.BufferGeometry(),slot=old?old.slot:freeSlots.pop();
  if(slot===undefined)throw Error('HLOD uniform slot pool exhausted');
  const poolLoads=Array.from({length:poolCount},()=>0);
  for(const e of cache.values())poolLoads[e.poolId]++;
  const poolId=old?old.poolId:poolLoads.indexOf(Math.min(...poolLoads));
  d.prepared.poolKey='terrain-'+poolId;
  for(let i=3;i<d.lodData.length;i+=4)d.lodData[i]=slot;
  const offset=new Map(d.prepared.attributeLayout).get('lodData').offset*4;
  for(let i=offset+3;i<d.prepared.vertexArray.length;i+=d.prepared.stride*4)d.prepared.vertexArray[i]=slot;
  // Share the worker-prepared vertex storage with BufferGeometry. Keeping a
  // second position/normal/LOD/index copy for every tile doubled CPU memory.
  const packedBuffer=new THREE.InterleavedBuffer(d.prepared.vertexArray,d.prepared.stride*4),uvs=d.prepared.lodCount===1?d.uv:new Float32Array(packedBuffer.count*2);
  if(d.prepared.lodCount!==1)for(let i=0;i<packedBuffer.count;i++){uvs[i*2]=d.prepared.vertexArray[i*packedBuffer.stride+3];uvs[i*2+1]=d.prepared.vertexArray[i*packedBuffer.stride+7];}
  g.setAttribute('position',new THREE.InterleavedBufferAttribute(packedBuffer,3,0));g.setAttribute('normal',new THREE.InterleavedBufferAttribute(packedBuffer,3,4));
  g.setAttribute('uv',new THREE.BufferAttribute(uvs,2));g.setAttribute('lodData',new THREE.InterleavedBufferAttribute(packedBuffer,4,offset));g.setIndex(new THREE.BufferAttribute(d.prepared.indexArray,1));
  g.boundingBox=new THREE.Box3(new THREE.Vector3(...d.bounds.slice(0,3)),new THREE.Vector3(...d.bounds.slice(3)));
  g.boundingSphere=g.boundingBox.getBoundingSphere(new THREE.Sphere());
  pipeline.registerPreparedGeometry(g,unpackMeshlets(d.prepared,g));
  if(old){old.mesh.geometry.dispose();old.mesh.geometry=g;old.data=d.prepared;old.upgrading=false;}
  else{const mesh=new THREE.Mesh(g,material);mesh.visible=false;scene.add(mesh);cache.set(d.id,{mesh,slot,poolId,sourceTriangles:d.index.length/3,data:d.prepared,level:d.level,tx:d.tx,tz:d.tz,used:epoch,everVisible:false});}
  if(!state.firstGeometryMs)state.firstGeometryMs=performance.now()-loadStarted;
  state.workerMs.push(d.buildMs);state.installMs.push(performance.now()-begin);return true;
 }
 for(const channel of workers){
  channel.worker.onmessage=e=>{
   const d=e.data;
   if(d.type==='ready'){channel.ready=true;workerReady=true;plan(true);return;}
   channel.busy=false;pending.delete(d.id);
   if(wanted.has(d.id)||cache.has(d.id))incoming.push(d);
   pump();
  };
  channel.worker.onerror=e=>fail(e.message);
 }
 function ridge(){const {x,z}=meta.spawn,h=surface.height(x,z);setFly(false);camera.position.set(x,h+1.8,z);viewTarget.set(x+600,h-75,z-1200);camera.lookAt(viewTarget);plannedPosition=null;plan(true);}
 function ground(){let {x,z}=meta.spawn,best=Infinity;const ox=x,oz=z;for(let dz=-40;dz<=40;dz+=4)for(let dx=-40;dx<=40;dx+=4){const px=ox+dx,pz=oz+dz,slope=Math.hypot(surface.height(px+1,pz)-surface.height(px-1,pz),surface.height(px,pz+1)-surface.height(px,pz-1))*.5;const score=slope+Math.hypot(dx,dz)*.0015;if(score<best){best=score;x=px;z=pz;}}const h=surface.height(x,z);setFly(false);camera.position.set(x,h+1.8,z);viewTarget.set(x+5,h-2,z-8);camera.lookAt(viewTarget);plannedPosition=null;plan(true);}
 function overview(){setFly(true);camera.position.set(11000,7200,13000);viewTarget.set(-1000,700,0);camera.lookAt(viewTarget);plannedPosition=null;plan(true);}
 $('overview').onclick=overview;$('ground').onclick=ground;$('ridge').onclick=ridge;$('budget').onchange=()=>plan(true);
 $('river').onclick=()=>{const p=lighting.waterMeta.riverView;setFly(true);const y=Math.max(surface.height(p.x,p.z),surface.height(p.x+45,p.z+45));camera.position.set(p.x+45,y+65,p.z+45);camera.lookAt(p.x,surface.height(p.x,p.z),p.z);plannedPosition=null;plan(true);};
 ridge();
 const shared=buffer=>{if(!crossOriginIsolated||typeof SharedArrayBuffer==='undefined')return buffer;const b=new SharedArrayBuffer(buffer.byteLength);new Uint8Array(b).set(new Uint8Array(buffer));return b;};
 const init={type:'init',height:shared(heightBuffer),fields:shared(fieldBuffer),detail:shared(groundBakes.height.data.buffer),detailConfig:groundBakes.meta,size:meta.size,previewSize:meta.previewSize,worldSize:world};
 state.sharedWorkerData=crossOriginIsolated;for(const channel of workers)channel.worker.postMessage(init);
 const keys=new Set();addEventListener('keydown',e=>{if(e.target.matches('input,select'))return;if(e.code==='KeyF'&&!e.repeat){setFly(!fly);return;}keys.add(e.code);if(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE'].includes(e.code))e.preventDefault();});addEventListener('keyup',e=>keys.delete(e.code));addEventListener('blur',()=>keys.clear());
 const forward=new THREE.Vector3(),right=new THREE.Vector3(),move=new THREE.Vector3();let previous=0,lastHud=0,motion=null;
 renderer.setAnimationLoop(time=>{
  const dt=previous?time-previous:16.7;previous=time;clock.value=time/1000;
  if(keys.size){
   camera.getWorldDirection(forward);if(!fly)forward.y=0;forward.normalize();right.crossVectors(forward,camera.up).normalize();
   move.set(0,0,0);if(keys.has('KeyW'))move.add(forward);if(keys.has('KeyS'))move.sub(forward);if(keys.has('KeyD'))move.add(right);if(keys.has('KeyA'))move.sub(right);
   if(fly){if(keys.has('KeyE'))move.y++;if(keys.has('KeyQ'))move.y--;}
   const boost=keys.has('ShiftLeft')||keys.has('ShiftRight');
   move.normalize().multiplyScalar(Math.min(dt,50)*.001*(fly?(boost?250:30):(boost?9:4.5)));camera.position.add(move);
  }
  if(motion){const t=Math.min(1,(time-motion.start)/12000);camera.position.copy(motion.camera).addScaledVector(motion.direction,t*220);}
  const floor=surface.height(camera.position.x,camera.position.z)+1.8;
  if(!fly){verticalVelocity-=18*Math.min(dt,50)*.001;camera.position.y+=verticalVelocity*Math.min(dt,50)*.001;}
  if(camera.position.y<floor){camera.position.y=floor;verticalVelocity=0;}
  const uploadStart=performance.now();let installed=false,uploads=0;
  while(incoming.length&&uploads<4&&performance.now()-uploadStart<3){installed=install(incoming.shift())||installed;uploads++;}
  if(installed){updateCover();pump();}
  renderer.info.reset();lighting.render(time);
  if(samples&&dt>0){samples.frames.push(dt);samples.drawn.push(pipeline.info.triangles);if(!state.ready)samples.streamingFrames++;if(pipeline.info.overflow)samples.overflow++;if(state.coverage!==1)samples.holes++;}
  if(time-lastPlan>150){lastPlan=time;plan();if(hasCollapses){updateCover();pump();}}
  if(time-lastHud>400){state.meshlets={...pipeline.info};state.gpuMiB=renderer.info.memory.total/1048576;state.fps=1000/dt;$('triangles').textContent=`${fmt(state.residentTriangles||0)} / ${fmt(pipeline.info.triangles)}`;$('fps').textContent=state.fps.toFixed(0);$('memory').textContent=state.gpuMiB.toFixed(0)+' MiB';publish();lastHud=time;}
 });
 async function measure(moving=false){
  if(samples)return;if(!state.ready){status('Wait for the initial HLOD cover to finish streaming.');return;}
  $('measure').disabled=$('movement').disabled=true;const before=state.transitions;
  samples={frames:[],drawn:[],streamingFrames:0,overflow:0,holes:0,hidden:document.visibilityState!=='visible'};
  const visibility=()=>{if(document.visibilityState!=='visible')samples&&(samples.hidden=true);};document.addEventListener('visibilitychange',visibility);
  if(moving){camera.getWorldDirection(forward);forward.y=0;forward.normalize();motion={start:performance.now(),camera:camera.position.clone(),direction:forward.clone()};}
  status(moving?'Measuring 220 m HLOD traversal...':'Measuring settled frame times...');
  await new Promise(r=>setTimeout(r,moving?12000:8000));motion=null;const s=samples;samples=null;document.removeEventListener('visibilitychange',visibility);
  const elapsed=s.frames.reduce((a,b)=>a+b,0);
  const result={mode:moving?'movement':'settled',fps:1000*s.frames.length/elapsed,p95:percentile(s.frames,.95),p99:percentile(s.frames,.99),maxFrameMs:Math.max(...s.frames),frames:s.frames.length,streamingFrames:s.streamingFrames,overflowFrames:s.overflow,uncoveredFrames:s.holes,transitions:state.transitions-before,valid:!s.hidden&&!s.overflow&&!s.holes&&state.errors.length===0,drawnMean:s.drawn.reduce((a,b)=>a+b,0)/s.drawn.length,residentTriangles:state.residentTriangles,materialVariants:state.materialVariants,meshlets:{...pipeline.info},workerP95:percentile(state.workerMs,.95),installP95:percentile(state.installMs,.95),gpuMiB:state.gpuMiB};
  state.records.push(result);document.body.dataset.benchmark=JSON.stringify(result);$('result').textContent=`${result.valid?'':'INVALID / '}${result.fps.toFixed(1)} FPS / p95 ${result.p95.toFixed(1)} ms / ${s.holes} uncovered frames / ${s.overflow} overflows`;
  $('measure').disabled=$('movement').disabled=false;
 }
 $('measure').onclick=()=>measure();$('movement').onclick=()=>measure(true);
 addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);lighting.resize(innerWidth,innerHeight);});
}
start().catch(fail);


