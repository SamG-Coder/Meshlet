import { StorageBufferAttribute, Vector4 } from 'three/webgpu';
import { storage, uniform, uniformArray, vec2 } from 'three/tsl';

// Only typed arrays and plain data cross the worker boundary; GPU nodes stay on the render thread.
export function packMeshlets(c) {
 return {vertexArray:c.vertexArray,indexArray:c.indexArray,chunkArray:c.chunkArray,stride:c.stride,chunkCount:c.chunkCount,maxChunks:c.maxChunks,lodCount:c.lodCount,lodTriangleEnds:c.lodTriangleEnds,lodOffsets:c.lodOffsetValues.map(v=>v.toArray()),layoutKey:c.layoutKey,attributeLayout:[...c.attributeLayout]};
}
export function unpackMeshlets(d,geometry) {
 const vertex=storage(new StorageBufferAttribute(d.vertexArray,4),'vec4').toReadOnly(),stride=uniform(d.stride,'uint');
 const element=(v,o)=>vertex.element(v.mul(stride).add(o));
 const chunks=storage(new StorageBufferAttribute(d.chunkArray,4),'vec4').toReadOnly();
 const offsets=d.lodOffsets.map(v=>new Vector4(...v));
 const layout=new Map(d.attributeLayout);
 return {...d,geometry,position:v=>element(v,0).xyz,normal:v=>element(v,1).xyz,uv:v=>vec2(element(v,0).w,element(v,1).w),hasAttribute:name=>layout.has(name),attribute:(name,v)=>{const a=layout.get(name),node=element(v,a.offset);return a.itemSize===1?node.x:a.itemSize===2?node.xy:a.itemSize===3?node.xyz:node;},indexBuffer:storage(new StorageBufferAttribute(d.indexArray,1),'uint').toReadOnly(),chunkBounds:c=>chunks.element(c.mul(2)),chunkCone:c=>chunks.element(c.mul(2).add(1)),lodOffsets:uniformArray(offsets,'vec4'),lodOffsetValues:offsets,attributeLayout:layout,instancedAttributes:[],boundingSphere:geometry.boundingSphere,boundingBox:geometry.boundingBox};
}
