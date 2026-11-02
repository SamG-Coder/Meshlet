import { BufferGeometry, BufferAttribute } from 'three/webgpu';
import { createMeshlets, ready } from 'three/addons/utils/MeshletGeometryUtils.js';
import { packMeshlets } from './meshlet-transfer.js';
import { makeSurface } from './terrain-surface.js';
import { packTerrainGrid } from './terrain-meshlets.js';
let surface,world;
onmessage=async e=>{
 const d=e.data;
 if(d.type==='init'){
  world=d.worldSize;surface=makeSurface(new Float32Array(d.height),new Float32Array(d.fields),d.size,d.previewSize,world,{...d.detailConfig,data:new Uint8Array(d.detail)});
  await ready;postMessage({type:'ready'});return;
 }
 const started=performance.now(),{id,level,tx,tz,segments}=d;
 const span=world/2**level,step=span/segments,ox=-world/2+tx*span,oz=-world/2+tz*span;
 const row=segments+1,count=row*row+row*4;
 const pos=new Float32Array(count*3),normal=new Float32Array(count*3),uv=new Float32Array(count*2),lodData=new Float32Array(count*4);
 const grid=new Float32Array(row*row),locks=new Uint8Array(count);
 for(let z=0;z<row;z++)for(let x=0;x<row;x++)grid[z*row+x]=surface.height(ox+x*step,oz+z*step);
 let maxDelta=0,minHeight=Infinity,maxHeight=-Infinity;
 for(let z=0;z<row;z++)for(let x=0;x<row;x++){
  const i=z*row+x,wx=ox+x*step,wz=oz+z*step,h=grid[i];
  const x0=Math.min(segments-2,x&~1),z0=Math.min(segments-2,z&~1),fx=(x-x0)*.5,fz=(z-z0)*.5;
  const a=grid[z0*row+x0],b=grid[z0*row+x0+2],c=grid[(z0+2)*row+x0],dd=grid[(z0+2)*row+x0+2];
  const ph=level===0?h:fx+fz<=1?a+(b-a)*fx+(c-a)*fz:dd+(c-dd)*(1-fx)+(b-dd)*(1-fz);
  const lx=Math.max(0,x-1),rx=Math.min(segments,x+1),tz0=Math.max(0,z-1),bz=Math.min(segments,z+1);
  const nx=(grid[z*row+lx]-grid[z*row+rx])/((rx-lx)*step),nz=(grid[tz0*row+x]-grid[bz*row+x])/((bz-tz0)*step),len=Math.hypot(nx,1,nz);
  pos.set([wx,h,wz],i*3);normal.set([nx/len,1/len,nz/len],i*3);uv.set([wx/world+.5,1-(wz/world+.5)],i*2);
  // Local grid coordinates fit exactly in one float; w is a render-thread slot.
  lodData.set([ph,span,x+z*512,0],i*4);maxDelta=Math.max(maxDelta,Math.abs(h-ph));
  minHeight=Math.min(minHeight,h,ph);maxHeight=Math.max(maxHeight,h,ph);
  if(x===0||z===0||x===segments||z===segments)locks[i]=1;
 }
 const indices=new Uint32Array(segments*segments*6+segments*24);let at=0;
 for(let z=0;z<segments;z++)for(let x=0;x<segments;x++){const a=z*row+x,b=a+1,c=a+row;indices.set([a,c,b,b,c,c+1],at);at+=6;}
 const edges=[Array.from({length:row},(_,i)=>i),Array.from({length:row},(_,i)=>i*row+segments),Array.from({length:row},(_,i)=>segments*row+segments-i),Array.from({length:row},(_,i)=>(segments-i)*row)];
 let next=row*row;
 for(const edge of edges){
  const first=next;
  for(const i of edge){
   const skirt=Math.max(.3,Math.min(30,maxDelta+step*.15));
   pos.set([pos[i*3],pos[i*3+1]-skirt,pos[i*3+2]],next*3);
   normal.set(normal.subarray(i*3,i*3+3),next*3);uv.set(uv.subarray(i*2,i*2+2),next*2);
   lodData.set([lodData[i*4]-skirt,span,lodData[i*4+2],0],next*4);locks[next]=1;next++;
  }
  for(let k=0;k<segments;k++){indices.set([edge[k],first+k,edge[k+1],edge[k+1],first+k,first+k+1],at);at+=6;}
 }
 const geo=new BufferGeometry();
 geo.setAttribute('position',new BufferAttribute(pos,3));geo.setAttribute('normal',new BufferAttribute(normal,3));
 geo.setAttribute('uv',new BufferAttribute(uv,2));geo.setAttribute('lodData',new BufferAttribute(lodData,4));geo.setIndex(new BufferAttribute(indices,1));
 geo.userData.meshletVertexLocks=locks;geo.userData.meshletFullDetailOnly=d.fullDetail;
 let prepared;
 if(d.fullDetail)prepared=packTerrainGrid(pos,normal,uv,lodData,indices,segments);
 else{
  const clusters=createMeshlets(geo);
  for(let i=0;i<clusters.chunkArray.length;i+=8){clusters.chunkArray[i+3]+=maxDelta;clusters.chunkArray[i+7]=1;}
  prepared=packMeshlets(clusters);
 }
 const bounds=[ox,minHeight-Math.max(.3,Math.min(30,maxDelta+step*.15)),oz,ox+span,maxHeight,oz+span];
 postMessage({id,level,tx,tz,pos,normal,uv,index:indices,lodData,maxDelta,bounds,prepared,upgrade:d.upgrade,buildMs:performance.now()-started},[pos.buffer,normal.buffer,uv.buffer,indices.buffer,lodData.buffer,prepared.vertexArray.buffer,prepared.indexArray.buffer,prepared.chunkArray.buffer]);
};

