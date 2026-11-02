import assert from 'node:assert/strict';
import { PlaneGeometry } from 'three/webgpu';
import { ready, createMeshlets } from 'three/addons/utils/MeshletGeometryUtils.js';
import { MeshletDrawingPipeline } from 'three/addons/pipelines/MeshletDrawingPipeline.js';
import { makeSurface } from '../src/terrain-surface.js';
import { packMeshlets, unpackMeshlets } from '../src/meshlet-transfer.js';

// Child vertices must begin exactly on the two triangles of their parent cell,
// including saddle cells where bilinear interpolation would visibly pop.
const height=new Float32Array([0,7,1,3,11,2,9,4,12]);
const fields=new Float32Array(3*3*11);
for(let i=0;i<9;i++)fields[i*11+7]=1000; // Suppress procedural relief for this exact fixture.
const s=makeSurface(height,fields,3,3,4);
for(let z=-2;z<=2;z+=.25)for(let x=-2;x<=2;x+=.25){
 const ix=Math.min(1,Math.floor((x+2)/2)),iz=Math.min(1,Math.floor((z+2)/2));
 const u=(x-(ix*2-2))/2,v=(z-(iz*2-2))/2;
 const a=height[iz*3+ix],b=height[iz*3+ix+1],c=height[(iz+1)*3+ix],d=height[(iz+1)*3+ix+1];
 const expected=u+v<=1?a+(b-a)*u+(c-a)*v:d+(c-d)*(1-u)+(b-d)*(1-v);
 assert.ok(Math.abs(s.parentHeight(x,z,2)-expected)<1e-5,`${x},${z}: parent interpolation mismatch`);
}
await ready;
const geometry=new PlaneGeometry(10,10,12,12);
const data=new Float32Array(geometry.attributes.position.count*4);
for(let i=0;i<data.length;i+=4)data.set([7,10,2,1],i);
const { Float32BufferAttribute }=await import('three/webgpu');
geometry.setAttribute('lodData',new Float32BufferAttribute(data,4));
const original=createMeshlets(geometry),packed=structuredClone(packMeshlets(original)),restored=unpackMeshlets(packed,geometry);
assert.deepEqual(restored.indexArray,original.indexArray);
assert.deepEqual(restored.vertexArray,original.vertexArray);
assert.deepEqual(restored.chunkArray,original.chunkArray);
assert.deepEqual(restored.attributeLayout,original.attributeLayout);
assert.equal(restored.hasAttribute('lodData'),true);
const pipeline=new MeshletDrawingPipeline();
pipeline.registerPreparedGeometry(geometry,restored);
assert.equal(pipeline._getClusters(geometry),restored,'Render thread must reuse worker result');
geometry.dispose();
assert.equal(pipeline._clusters.has(geometry),false,'Worker-prepared clusters must obey normal disposal');
console.log('PASS: parent-triangle continuity, transferable meshlet data, custom attributes, cache reuse and disposal.');

const {balanceCover,coverEdgeFlags,tileKey}=await import('../src/terrain-hlod.js');
let leaves=[{level:0,tx:0,tz:0}];
for(let depth=0;depth<6;depth++){
 const at=leaves.findIndex(t=>t.tx===0&&t.tz===0&&t.level===depth);
 const p=leaves.splice(at,1)[0];
 for(let i=0;i<4;i++)leaves.push({level:p.level+1,tx:p.tx*2+i%2,tz:p.tz*2+Math.floor(i/2)});
}
const balanced=balanceCover(leaves);
assert.equal(balanced.reduce((a,t)=>a+4**-t.level,0),1,'Balancing must preserve exact world coverage.');
const flags=coverEdgeFlags(balanced);
assert.ok([...flags.values()].some(v=>v.includes(-1)),'Mixed detail must stitch its fine boundary.');
for(const a of balanced)for(const b of balanced){
 if(a===b)continue;
 const sa=2**-a.level,sb=2**-b.level;
 const xOverlap=Math.min((a.tx+1)*sa,(b.tx+1)*sb)-Math.max(a.tx*sa,b.tx*sb);
 const zOverlap=Math.min((a.tz+1)*sa,(b.tz+1)*sb)-Math.max(a.tz*sa,b.tz*sb);
 if(xOverlap>=0&&zOverlap>=0)assert.ok(Math.abs(a.level-b.level)<=1,'Touching HLOD tiles must differ by at most one level.');
}
const {sampleVariantHeight}=await import('../src/material-variation.js');
const vd={count:10,size:8,metres:2,heightRangeMetres:.12,data:new Uint8Array(10*8*8*4)};
for(let i=2;i<vd.data.length;i+=4)vd.data[i]=(i*13)%256;
for(const [x,z] of [[0,.71],[-2,.45],[2,1.23],[.7,1.3],[-.7,-1.3]]){
 const a=sampleVariantHeight(vd,x-1e-7,z),b=sampleVariantHeight(vd,x+1e-7,z);
 assert.ok(Math.abs(a-b)<1e-5,'Stochastic variant height must remain continuous at cell and triangle boundaries.');
}
console.log('PASS: balanced HLOD cover, mixed-resolution edge flags, continuous stochastic material height.');


const locked=new PlaneGeometry(10,10,16,16);
locked.userData.meshletVertexLocks=new Uint8Array(locked.attributes.position.count);
for(let i=0;i<locked.attributes.position.count;i++){
 const u=locked.attributes.uv.getX(i),v=locked.attributes.uv.getY(i);
 if(u===0||u===1||v===0||v===1)locked.userData.meshletVertexLocks[i]=1;
}
const lc=createMeshlets(locked);
const boundary=[];
for(let i=0;i<lc.vertexArray.length/(lc.stride*4);i++){
 const x=lc.vertexArray[i*lc.stride*4],y=lc.vertexArray[i*lc.stride*4+1];
 if(Math.abs(x)===5||Math.abs(y)===5)boundary.push(i);
}
for(const l of lc.lodOffsetValues){
 const vertices=new Set(lc.indexArray.subarray(l.x*3,(l.x+l.y)*3));
 for(const v of boundary)assert.ok(vertices.has(v),'Every adaptive meshlet LOD must retain each locked tile-border vertex.');
}
console.log('PASS: adaptive meshlet simplification preserves every stitched tile boundary.');


locked.userData.meshletFullDetailOnly=true;
const fullOnly=createMeshlets(locked);
assert.equal(fullOnly.lodCount,1,'Full resident streaming must not generate unused simplified levels.');
assert.equal(fullOnly.lodOffsetValues[0].y,lc.lodOffsetValues[0].y,'Skipping simplification must preserve the complete source meshlet level.');
console.log('PASS: full-resident preparation skips unused simplification without losing source triangles.');

// Direct grid packing must preserve each input triangle, including all skirts,
// and bound both fine and parent positions throughout the HLOD morph.
const {packTerrainGrid}=await import('../src/terrain-meshlets.js');
const gridSize=16,rows=gridSize+1,gridCount=rows*rows;
const gp=new Float32Array(gridCount*3),gn=new Float32Array(gridCount*3),guv=new Float32Array(gridCount*2),gl=new Float32Array(gridCount*4),gi=new Uint32Array(gridSize*gridSize*6);
for(let z=0;z<rows;z++)for(let x=0;x<rows;x++){
 const i=z*rows+x,y=Math.sin(x*.7)*Math.cos(z*.4)*8;
 gp.set([x,y,z],i*3);gn.set([0,1,0],i*3);guv.set([x/gridSize,z/gridSize],i*2);gl.set([y+Math.sin(i)*4,16,x+z*512,0],i*4);
 if(x<gridSize&&z<gridSize)gi.set([i,i+rows,i+1,i+1,i+rows,i+rows+1],(z*gridSize+x)*6);
}
const direct=packTerrainGrid(gp,gn,guv,gl,gi,gridSize);
const triangleKeys=a=>Array.from({length:a.length/3},(_,i)=>Array.from(a.subarray(i*3,i*3+3)).join(',')).sort();
assert.deepEqual(triangleKeys(direct.indexArray),triangleKeys(gi),'Direct meshlets must preserve every triangle and its winding');
for(let c=0;c<direct.chunkCount;c++){
 const [cx,cy,cz,r]=direct.chunkArray.subarray(c*8,c*8+4);
 for(const i of direct.indexArray.subarray(c*192,(c+1)*192))for(const y of [gp[i*3+1],gl[i*4]])assert.ok(Math.hypot(gp[i*3]-cx,y-cy,gp[i*3+2]-cz)<=r+1e-5,'Cluster bounds must contain the whole morph');
}
const p2=new MeshletDrawingPipeline();
const aPool=p2._poolFor({poolKey:'terrain-a'}),bPool=p2._poolFor({poolKey:'terrain-b'});
assert.notEqual(aPool,bPool);assert.equal(p2._poolFor({poolKey:'terrain-a'}),aPool);
assert.notEqual(p2._getBatchKey({poolKey:'terrain-a',layoutKey:'lodData:4|'}),p2._getBatchKey({poolKey:'terrain-b',layoutKey:'lodData:4|'}),'Batches must never cross geometry buffer pools');
const block=()=>({vertexArray:new Float32Array(48),indexArray:new Uint32Array([0,1,2,0,2,3]),stride:3});
const blocks=Array.from({length:8},block);blocks.forEach(b=>aPool.add(b));
const vEnd=aPool.vertexEnd,iEnd=aPool.indexEnd;
for(let i=0;i<100;i++){
 const at=i%blocks.length;aPool.remove(blocks[at]);blocks[at]=block();aPool.add(blocks[at]);
 assert.equal(aPool.vertexEnd,vEnd,'Streaming replacements must reuse freed vertex addresses');
 assert.equal(aPool.indexEnd,iEnd,'Streaming replacements must reuse freed index addresses');
}
assert.equal(aPool.wasted,0);
p2.dispose();
console.log('PASS: direct grid triangles, conservative morph bounds, independent geometry pools and batch separation.');

