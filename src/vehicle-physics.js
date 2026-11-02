import RAPIER from '@dimforge/rapier3d-compat';

// Metres, seconds, kilograms. A game-tuned raycast suspension, not a Ford simulator.
export const vehicleSpec={mass:2700,wheelbase:3.693,track:1.876,radius:.4375,rest:.42,step:1/120};
export async function createVehiclePhysics(height,spawn){
 await RAPIER.init();
 const s=vehicleSpec,origin={x:spawn.x,y:height(spawn.x,spawn.z),z:spawn.z};
 const world=new RAPIER.World({x:0,y:-9.81,z:0});world.timestep=s.step;
 const body=world.createRigidBody(RAPIER.RigidBodyDesc.dynamic().setTranslation(0,1.2,0).setLinearDamping(.035).setAngularDamping(.55).setCcdEnabled(true).setCanSleep(false));
 // Convex compound: frame, engine/hood, cab/roof, bed and two bumpers.
 // Separate pieces preserve the truck silhouette when it lands on its roof/side.
 const hulls=[
  {half:[.83,.22,2.57],at:[0,-.08,0],mass:1650},
  {half:[.91,.31,.70],at:[0,.22,2.02],mass:400},
  {half:[.91,.60,1.12],at:[0,.66,.26],mass:450},
  {half:[.94,.36,.98],at:[0,.22,-1.88],mass:160},
  {half:[1.02,.16,.14],at:[0,-.05,2.75],mass:20},
  {half:[1.02,.16,.14],at:[0,-.05,-2.91],mass:20},
 ];
 for(const h of hulls)world.createCollider(RAPIER.ColliderDesc.cuboid(...h.half).setTranslation(...h.at).setMass(h.mass).setFriction(.65),body);
 const car=world.createVehicleController(body);car.indexUpAxis=1;car.setIndexForwardAxis=2;
 const mounts=[],tyreColliders=[];
 for(const z of [s.wheelbase/2,-s.wheelbase/2])for(const x of [s.track/2,-s.track/2]){
  const i=mounts.length,m={x,y:.055,z};mounts.push(m);
  car.addWheel(m,{x:0,y:-1,z:0},{x:-1,y:0,z:0},s.rest,s.radius);
  car.setWheelSuspensionStiffness(i,40);car.setWheelSuspensionCompression(i,4.2);car.setWheelSuspensionRelaxation(i,5.2);
  car.setWheelMaxSuspensionTravel(i,.22);car.setWheelMaxSuspensionForce(i,42000);
  car.setWheelFrictionSlip(i,1.65);car.setWheelSideFrictionStiffness(i,1);
  // Smaller than the tread radius: ray suspension carries vertical load while
  // these rounded solids catch sideways impacts and rolled-over wheels.
  tyreColliders.push(world.createCollider(RAPIER.ColliderDesc.cylinder(.145,s.radius-.06).setRotation({x:0,y:0,z:Math.SQRT1_2,w:Math.SQRT1_2}).setTranslation(x,-.31,z).setMass(0).setFriction(.8),body));
 }
 let accumulator=0,steering=0,throttle=0,steps=0,patchMs=0,tilesBuilt=0;
 // This immutable collision grid is independent of camera, HLOD and render
 // residency. The authoritative height callback includes the source bake and
 // material relief. Never replace a collider that currently supports a tyre.
 const terrainTiles=new Map(),tileSize=32,spacing=.5,n=tileSize/spacing;
 const indices=new Uint32Array(n*n*6);
 for(let z=0;z<n;z++)for(let x=0;x<n;x++){const a=z*(n+1)+x;indices.set([a,a+n+1,a+1,a+1,a+n+1,a+n+2],(z*n+x)*6);}
 function prepareTerrain(limit=1){
  const p=body.translation(),tx=Math.floor((p.x+origin.x)/tileSize),tz=Math.floor((p.z+origin.z)/tileSize),missing=[];
  for(let z=tz-1;z<=tz+1;z++)for(let x=tx-1;x<=tx+1;x++)if(!terrainTiles.has(`${x}/${z}`))missing.push({x,z,d:Math.hypot(x-tx,z-tz)});
  missing.sort((a,b)=>a.d-b.d);const start=performance.now();
  for(const t of missing.slice(0,limit)){
   const positions=new Float32Array((n+1)**2*3);
   for(let z=0;z<=n;z++)for(let x=0;x<=n;x++){const wx=t.x*tileSize+x*spacing,wz=t.z*tileSize+z*spacing;positions.set([wx-origin.x,height(wx,wz)-origin.y,wz-origin.z],(z*(n+1)+x)*3);}
   const collider=world.createCollider(RAPIER.ColliderDesc.trimesh(positions,indices,RAPIER.TriMeshFlags.FIX_INTERNAL_EDGES).setFriction(.8));
   terrainTiles.set(`${t.x}/${t.z}`,{...t,collider});tilesBuilt++;
  }
  for(const [key,t]of terrainTiles)if(Math.abs(t.x-tx)>2||Math.abs(t.z-tz)>2){world.removeCollider(t.collider,true);terrainTiles.delete(key);}
  if(missing.length)patchMs=performance.now()-start;
 }
 prepareTerrain(9);world.step();
 function reset(x=spawn.x,z=spawn.z,yaw=0){
  body.setTranslation({x:x-origin.x,y:height(x,z)-origin.y+1.15,z:z-origin.z},true);
  body.setRotation({x:0,y:Math.sin(yaw/2),z:0,w:Math.cos(yaw/2)},true);
  body.setLinvel({x:0,y:0,z:0},true);body.setAngvel({x:0,y:0,z:0},true);steering=throttle=accumulator=0;prepareTerrain(9);world.step();
 }
 function advance(dt,input={}){
  accumulator=Math.min(accumulator+Math.min(dt,.1),.1);
  prepareTerrain();
  while(accumulator>=s.step){
   const speed=car.currentVehicleSpeed(),abs=Math.abs(speed);
   const targetSteer=(input.steer||0)*(.52/(1+abs*.06));
   steering+=(targetSteer-steering)*(1-Math.exp(-s.step*7));
   const request=input.throttle||0,braking=request*speed<-.5;
   const targetThrottle=braking?0:request;
   throttle+=(targetThrottle-throttle)*(1-Math.exp(-s.step*5));
   // Torque-limited launch, power-limited cruise; reverse limited to 24 km/h.
   let force=braking||input.handbrake||Math.abs(throttle)<.001?0:throttle*Math.min(15500,390000/Math.max(8,abs));
   if((speed>48&&force>0)||(speed< -6.7&&force<0))force=0;
   for(let i=0;i<4;i++){
    // Ackermann: the inner front tyre turns further than the outer one.
    const turnRadius=s.wheelbase/Math.max(.0001,Math.tan(Math.abs(steering)));
    const inner=(steering>0)===(i===0);
    car.setWheelSteering(i,i<2&&Math.abs(steering)>.0001?Math.sign(steering)*Math.atan(s.wheelbase/(turnRadius+(inner?-1:1)*s.track/2)):0);
    car.setWheelEngineForce(i,force*.25);
    car.setWheelBrake(i,braking?110:input.handbrake&&i>=2?180:!request?2:0);
    car.setWheelFrictionSlip(i,input.handbrake&&i>=2?.65:1.65);
   }
   car.updateVehicle(s.step,RAPIER.QueryFilterFlags.EXCLUDE_DYNAMIC);
   for(let i=0;i<4;i++){
    tyreColliders[i].setTranslationWrtParent({x:mounts[i].x,y:mounts[i].y-car.wheelSuspensionLength(i),z:mounts[i].z});
    const a=car.wheelSteering(i)/2;tyreColliders[i].setRotationWrtParent({x:Math.sin(a)*Math.SQRT1_2,y:Math.sin(a)*Math.SQRT1_2,z:Math.cos(a)*Math.SQRT1_2,w:Math.cos(a)*Math.SQRT1_2});
   }
   world.step();accumulator-=s.step;steps++;
  }
 }
 function state(){const p=body.translation();return {position:{x:p.x+origin.x,y:p.y+origin.y,z:p.z+origin.z},rotation:body.rotation(),speed:car.currentVehicleSpeed(),steering,steps,patchMs,collisionSpacing:spacing,collisionTiles:terrainTiles.size,tilesBuilt,wheels:mounts.map((m,i)=>({x:m.x,y:m.y-car.wheelSuspensionLength(i),z:m.z,steer:car.wheelSteering(i),spin:car.wheelRotation(i),contact:car.wheelIsInContact(i),length:car.wheelSuspensionLength(i),force:car.wheelSuspensionForce(i)}))};}
 return {advance,state,reset,world,body,colliderCount:hulls.length+tyreColliders.length,dispose(){world.free();}};
}
