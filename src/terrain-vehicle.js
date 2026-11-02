import * as THREE from 'three/webgpu';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { createVehiclePhysics,vehicleSpec } from './vehicle-physics.js';

const ASSET='/assets/vehicles/raptor/source/2024_ford_f-150_raptor_r.glb';
export async function createTerrainVehicle(renderer,surface,spawn,lighting){
 const gltf=await new GLTFLoader().loadAsync(ASSET);
 const scene=new THREE.Scene(),chassis=new THREE.Group();scene.add(chassis);
 // The FBX export's 0.01 root transform makes this metre-authored model 100x too small.
 gltf.scene.scale.multiplyScalar(100*vehicleSpec.wheelbase/3.686);gltf.scene.updateMatrixWorld(true);
 const offset=new THREE.Vector3(0,-.70,-.1603*vehicleSpec.wheelbase/3.686);
 const wheels=Array.from({length:4},()=>{const steer=new THREE.Group(),spin=new THREE.Group();steer.add(spin);chassis.add(steer);return {steer,spin};});
 const wheelIndex=name=>/Ani_Wheel_FL/.test(name)?0:/Ani_Disc_Scale_FR/.test(name)?1:/Ani_Wheel_BR/.test(name)?3:/Ani_Wheel_BL/.test(name)?2:-1;
 const caliperIndex=name=>/Caliper_FL/.test(name)?0:/Caliper_FR/.test(name)?1:/Caliper_BR/.test(name)?3:/Caliper_BL/.test(name)?2:-1;
 const materials=new Map();let triangles=0;
 gltf.scene.traverse(o=>{
  if(!o.isMesh)return;
  const parentName=o.parent.name,wi=wheelIndex(parentName),ci=caliperIndex(parentName);
  const geometry=o.geometry.clone().applyMatrix4(o.matrixWorld).translate(offset.x,offset.y,offset.z);
  let mat=materials.get(o.material);
  if(!mat){mat=renderer.library.fromMaterial(o.material);mat.name=o.material.name;mat.transmission=0;
   if(/Window/.test(mat.name)){mat.transparent=true;mat.opacity=.32;mat.depthWrite=false;mat.color.setRGB(.13,.18,.20);mat.roughness=.08;}
   for(const k of ['map','normalMap','roughnessMap','metalnessMap'])if(mat[k])mat[k].anisotropy=8;
   lighting.prepareVehicleMaterial(mat);materials.set(o.material,mat);
  }
  const mesh=new THREE.Mesh(geometry,mat);mesh.name=o.name;mesh.castShadow=!mat.transparent;mesh.receiveShadow=true;
  triangles+=(geometry.index?.count||geometry.attributes.position.count)/3;
  const i=wi>=0?wi:ci;
  if(i>=0){const pivot=new THREE.Vector3().setFromMatrixPosition(o.parent.matrixWorld).add(offset);geometry.translate(-pivot.x,-pivot.y,-pivot.z);(wi>=0?wheels[i].spin:wheels[i].steer).add(mesh);}
  else chassis.add(mesh);
 });
 const physics=await createVehiclePhysics(surface.height,spawn);
 const sun=lighting.configureVehicleScene(scene);renderer.shadowMap.enabled=true;
 const shadowGeometry=new THREE.PlaneGeometry(24,24,48,48);shadowGeometry.rotateX(-Math.PI/2);
 const shadowMesh=new THREE.Mesh(shadowGeometry,new THREE.ShadowNodeMaterial({opacity:.65,depthWrite:false}));shadowMesh.receiveShadow=true;shadowMesh.frustumCulled=false;scene.add(shadowMesh);
 let shadowX=Infinity,shadowZ=Infinity,active=false,camReady=false,lastTelemetry=0,orbitYaw=0,orbitPitch=.34;
 const target=new THREE.Vector3(),desired=new THREE.Vector3(),heading=new THREE.Vector3(),cameraTarget=new THREE.Vector3();
 let snapshot=physics.state();
 function update(dt,keys,camera,time){
  physics.advance(dt,{throttle:active?Number(keys.has('KeyW'))-Number(keys.has('KeyS')):0,steer:active?Number(keys.has('KeyA'))-Number(keys.has('KeyD')):0,handbrake:!active||keys.has('Space')});
  snapshot=physics.state();chassis.position.copy(snapshot.position);chassis.quaternion.copy(snapshot.rotation);
  snapshot.wheels.forEach((w,i)=>{wheels[i].steer.position.set(w.x,w.y,w.z);wheels[i].steer.rotation.y=w.steer;wheels[i].spin.rotation.x=w.spin;});
  sun.target.position.copy(chassis.position);sun.position.copy(chassis.position).addScaledVector(lighting.sunDirection,35);sun.target.updateMatrixWorld();
  if(Math.hypot(chassis.position.x-shadowX,chassis.position.z-shadowZ)>.5){
   shadowX=chassis.position.x;shadowZ=chassis.position.z;const a=shadowGeometry.attributes.position;
   for(let i=0;i<a.count;i++){const x=shadowX+(i%49)/48*24-12,z=shadowZ+Math.floor(i/49)/48*24-12;a.setXYZ(i,x,surface.height(x,z)+.035,z);}a.needsUpdate=true;shadowGeometry.computeVertexNormals();
  }
  if(active){
   heading.set(0,0,1).applyQuaternion(chassis.quaternion);heading.y=0;heading.normalize();
   target.copy(chassis.position).add(new THREE.Vector3(0,.75,0));
   const distance=10+Math.min(3,Math.abs(snapshot.speed)*.08),angle=Math.atan2(heading.x,heading.z)+orbitYaw;
   desired.set(target.x-Math.sin(angle)*distance*Math.cos(orbitPitch),target.y+distance*Math.sin(orbitPitch),target.z-Math.cos(angle)*distance*Math.cos(orbitPitch));
   // Lift the boom above intermediate ridges as well as its endpoint.
   for(let i=1;i<=12;i++){const f=i/12,x=target.x+(desired.x-target.x)*f,z=target.z+(desired.z-target.z)*f;desired.y=Math.max(desired.y,target.y+(surface.height(x,z)+.7-target.y)/f);}
   if(!camReady){camera.position.copy(desired);cameraTarget.copy(target);camReady=true;}
   else{camera.position.lerp(desired,1-Math.exp(-dt*5));cameraTarget.lerp(target,1-Math.exp(-dt*9));}
   camera.position.y=Math.max(camera.position.y,surface.height(camera.position.x,camera.position.z)+.5);camera.lookAt(cameraTarget);
  }
  if(time-lastTelemetry>200){lastTelemetry=time;
   const contacts=snapshot.wheels.filter(w=>w.contact).length;
   document.getElementById('vehicle-status').textContent=`${Math.abs(snapshot.speed*3.6).toFixed(0)} km/h · ${snapshot.speed<-.3?'R':'D'} · ${contacts}/4 tyres grounded`;
   document.body.dataset.vehicle=JSON.stringify({...snapshot,active,triangles,riggedWheels:wheels.length,collisionShapes:physics.colliderCount});
  }
 }
 lighting.setVehicleScene(scene);
 function flip(){const p=physics.state();heading.set(0,0,1).applyQuaternion(new THREE.Quaternion().copy(p.rotation));physics.reset(p.position.x,p.position.z,Math.atan2(heading.x,heading.z));camReady=false;}
 function recover(){const p=physics.state().position;let best=Infinity,spot={x:p.x,z:p.z};for(let dz=-24;dz<=24;dz+=3)for(let dx=-24;dx<=24;dx+=3){const x=p.x+dx,z=p.z+dz,h=surface.height(x,z);const score=Math.abs(surface.height(x+3,z)-h)+Math.abs(surface.height(x-3,z)-h)+Math.abs(surface.height(x,z+4)-h)+Math.abs(surface.height(x,z-4)-h)+Math.hypot(dx,dz)*.008;if(score<best){best=score;spot={x,z};}}physics.reset(spot.x,spot.z);camReady=false;}
 return {update,get active(){return active;},orbit(dx,dy){orbitYaw-=dx*.005;orbitPitch=THREE.MathUtils.clamp(orbitPitch+dy*.004,-.08,1.25);},setActive(v){active=v;camReady=false;document.getElementById('drive').textContent=v?'Exit Raptor / V':'Drive Raptor / V';document.body.classList.toggle('driving',v);},reset:recover,flip,spawn(){physics.reset();camReady=false;},position:chassis.position};
}
