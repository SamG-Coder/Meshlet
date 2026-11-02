// A regular heightfield already has a spatial cluster layout. In full-resident
// mode, grouping 8 x 4 quads directly avoids welding and clustering it again.
// Each group contains exactly 64 triangles, matching the Three.js pipeline.
export function packTerrainGrid(pos, normal, uv, lod, sourceIndex, segments) {
 if(segments%8!==0)throw Error('Terrain grid size must be divisible by eight');
 const vertexCount=pos.length/3,vertexArray=new Float32Array(vertexCount*12);
 for(let i=0;i<vertexCount;i++){
  const v=i*12,p=i*3,t=i*2,l=i*4;
  vertexArray[v]=pos[p];vertexArray[v+1]=pos[p+1];vertexArray[v+2]=pos[p+2];vertexArray[v+3]=uv[t];
  vertexArray[v+4]=normal[p];vertexArray[v+5]=normal[p+1];vertexArray[v+6]=normal[p+2];vertexArray[v+7]=uv[t+1];
  vertexArray.set(lod.subarray(l,l+4),v+8);
 }
 const indexArray=new Uint32Array(sourceIndex.length);let at=0;
 for(let z=0;z<segments;z+=4)for(let x=0;x<segments;x+=8)
  for(let dz=0;dz<4;dz++)for(let dx=0;dx<8;dx++){
   const source=((z+dz)*segments+x+dx)*6;
   indexArray.set(sourceIndex.subarray(source,source+6),at);at+=6;
  }
 indexArray.set(sourceIndex.subarray(segments*segments*6),at);
 const chunkCount=indexArray.length/192,chunkArray=new Float32Array(chunkCount*8);
 for(let c=0;c<chunkCount;c++){
  let x0=Infinity,y0=Infinity,z0=Infinity,x1=-Infinity,y1=-Infinity,z1=-Infinity;
  for(let k=c*192;k<(c+1)*192;k++){
   const i=indexArray[k],p=i*3,ph=lod[i*4];
   x0=Math.min(x0,pos[p]);x1=Math.max(x1,pos[p]);z0=Math.min(z0,pos[p+2]);z1=Math.max(z1,pos[p+2]);
   y0=Math.min(y0,pos[p+1],ph);y1=Math.max(y1,pos[p+1],ph);
  }
  const x=(x0+x1)*.5,y=(y0+y1)*.5,z=(z0+z1)*.5;
  chunkArray.set([x,y,z,Math.hypot(x1-x,y1-y,z1-z)+.002,0,1,0,1],c*8);
 }
 const triangles=indexArray.length/3;
 return {vertexArray,indexArray,chunkArray,stride:3,chunkCount,maxChunks:chunkCount,lodCount:1,lodTriangleEnds:[triangles],lodOffsets:[[0,triangles,0,0],...[0,0,0,0,0].map(()=>[0,0,0,0])],layoutKey:'lodData:4|',attributeLayout:[['lodData',{offset:2,itemSize:4}]]};
}
