import { DataArrayTexture, RGBAFormat, UnsignedByteType, RepeatWrapping, LinearFilter, LinearMipmapLinearFilter, SRGBColorSpace } from 'three/webgpu';
import { texture, uint, int, vec2, vec3, vec4, select, positionWorld, normalWorldGeometry, float, cameraPosition, cameraViewMatrix, smoothstep, mix, max, uv } from 'three/tsl';

export async function loadGroundVariants() {
 const root='/assets/ground-generated-v3/';
 const meta=await fetch(root+'manifest.json').then(r=>r.json());
 const buffers=await Promise.all(['color-array.bin','normal-array.bin','material-array.bin'].map(p=>fetch(root+p).then(r=>r.arrayBuffer())));
 const textures=buffers.map((b,i)=>{
  const t=new DataArrayTexture(new Uint8Array(b),meta.size,meta.size,meta.count);t.format=RGBAFormat;t.type=UnsignedByteType;
  t.wrapS=t.wrapT=RepeatWrapping;t.magFilter=LinearFilter;t.minFilter=LinearMipmapLinearFilter;t.generateMipmaps=true;t.anisotropy=8;if(i===0)t.colorSpace=SRGBColorSpace;t.needsUpdate=true;return t;
 });
 return {meta,color:textures[0],normal:textures[1],material:textures[2],height:{...meta,data:new Uint8Array(buffers[2])}};
}
function seed(cell) {
 let h=uint(int(cell.x)).mul(uint(374761393)).bitXor(uint(int(cell.y)).mul(uint(668265263)));
 h=h.bitXor(h.shiftRight(13)).mul(uint(1274126177));return h.bitXor(h.shiftRight(16));
}
const rotate=(p,t)=>select(t.equal(0),p,select(t.equal(1),vec2(p.y.negate(),p.x),select(t.equal(2),p.negate(),vec2(p.y,p.x.negate()))));
function stochastic(p,maps) {
 const cell=p.floor(),f=p.fract(),lower=f.x.add(f.y).lessThanEqual(1);
 const corners=[select(lower,cell,cell.add(1)),cell.add(vec2(1,0)),cell.add(vec2(0,1))];
 const bary=[select(lower,f.x.add(f.y).oneMinus(),f.x.add(f.y).sub(1)),select(lower,f.x,f.y.oneMinus()),select(lower,f.y,f.x.oneMinus())];
 const samples=corners.map((c,i)=>{
  const h=seed(c).toVar(),turn=h.shiftRight(8).bitAnd(3),layer=int(h.mod(maps.meta.count));
  const offset=vec2(h.shiftRight(16).bitAnd(255),h.shiftRight(24).bitAnd(255)).div(256);
  const coords=rotate(p,turn).add(offset),dx=rotate(p.dFdx(),turn),dy=rotate(p.dFdy(),turn);
  const mat=texture(maps.material,coords).grad(dx,dy).depth(layer).toVar();
  const color=texture(maps.color,coords).grad(dx,dy).depth(layer).rgb.toVar();
  const normal=texture(maps.normal,coords).grad(dx,dy).depth(layer).xyz.mul(2).sub(1).toVar();
  const gradient=rotate(normal.xy.div(max(normal.z,.3)),uint(4).sub(turn).mod(4));
  const weight=max(bary[i],0).pow(3).mul(mat.b.add(.25).pow(2));
  return {mat,color,gradient,weight};
 });
 const total=samples[0].weight.add(samples[1].weight).add(samples[2].weight).max(.00001);
 const blend=key=>samples[0][key].mul(samples[0].weight).add(samples[1][key].mul(samples[1].weight)).add(samples[2][key].mul(samples[2].weight)).div(total);
 return {color:blend('color'),mat:blend('mat'),gradient:blend('gradient')};
}
export function groundVariantNodes(maps,baseMap,terrainNormal,terrainOrm,geologyMap=null) {
 const p=positionWorld.div(maps.meta.metres);
 const baked=texture(terrainNormal,uv()).xyz.mul(2).sub(1);
 const n=vec3(baked.x,baked.z,baked.y.negate()).normalize();
 const raw=n.abs().pow(6),w=raw.div(raw.x.add(raw.y).add(raw.z));
 const sx=stochastic(p.zy,maps),sy=stochastic(p.xz,maps),sz=stochastic(p.xy,maps);
 const blend=key=>sx[key].mul(w.x).add(sy[key].mul(w.y)).add(sz[key].mul(w.z));
 const near=smoothstep(120,650,positionWorld.distance(cameraPosition)).oneMinus();
 const fines=geologyMap?texture(geologyMap,uv()).b:float(0);
 const relief=fines.mul(.55).oneMinus();
 const detail=blend('color').dot(vec3(.2126,.7152,.0722)).div(.34).clamp(.4,1.4);
 const tangent=vec3(0,sx.gradient.y,sx.gradient.x).mul(w.x).add(vec3(sy.gradient.x,0,sy.gradient.y).mul(w.y)).add(vec3(sz.gradient.x,sz.gradient.y,0).mul(w.z));
 const perturb=tangent.sub(n.mul(tangent.dot(n)));
 const worldNormal=n.add(perturb.mul(near).mul(.65).mul(relief)).normalize(),mat=blend('mat');
 return {
  terrainHeightDetailNode:mat.b,
  terrainBaseNormalNode:n,
  colorNode:texture(baseMap,uv()).rgb.mul(mix(1,detail,near.mul(relief))),
  normalNode:cameraViewMatrix.mul(vec4(worldNormal,0)).xyz.normalize(),
  roughnessNode:mix(.9,mat.g,near),
  aoNode:texture(terrainOrm,uv()).r.mul(mix(1,mat.r,near))
 };
}
