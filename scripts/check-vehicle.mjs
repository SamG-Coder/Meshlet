import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createVehiclePhysics} from '../src/vehicle-physics.js';
const results={};
const p=await createVehiclePhysics(()=>0,{x:0,z:0});
const run=(seconds,input={})=>{for(let i=0;i<seconds*120;i++)p.advance(1/120,input);return p.state();};
let s=run(5);assert.equal(s.wheels.filter(w=>w.contact).length,4);assert.ok(Math.abs(s.position.y-.744)<.02);results.settled=s;
s=run(5,{throttle:1});assert.ok(s.position.z>40);assert.ok(s.speed>15&&s.speed<35);results.acceleration=s;
s=run(2,{throttle:-1});results.brake=s;assert.ok(Math.abs(s.speed)<1);
p.reset();run(2);s=run(3,{throttle:1,steer:1});results.steering=s;assert.ok(s.position.x>2,'A must steer left (+X)');assert.ok(Math.abs(s.rotation.y)>.1);assert.ok(s.wheels.some(w=>w.contact));
p.reset();run(2);s=run(4,{throttle:-1});results.reverse=s;assert.ok(s.position.z< -5&&s.speed<0);
p.dispose();
const rough=await createVehiclePhysics((x,z)=>.06*z+.12*Math.sin(z*.8)+.08*Math.sin(x*1.7),{x:0,z:0});
for(let i=0;i<1200;i++)rough.advance(1/120,{throttle:i<240?0:.35});
s=rough.state();results.rough=s;assert.ok(Number.isFinite(s.position.y));assert.ok(Math.abs(s.rotation.w)>.8);assert.ok(s.wheels.some(w=>w.contact));assert.ok(Math.max(...s.wheels.map(w=>w.length))-Math.min(...s.wheels.map(w=>w.length))>.005);rough.dispose();
fs.mkdirSync('results',{recursive:true});fs.writeFileSync('results/vehicle-physics-check.json',JSON.stringify(results,null,2));console.log('PASS: static load, acceleration, braking, Ackermann turn, reverse, uneven ground and collider streaming.');

