import { sampleVariantHeight } from './material-variation.js';
export function hash(x,z) {
 let h=Math.imul(x|0,374761393)^Math.imul(z|0,668265263);
 h=Math.imul(h^(h>>>13),1274126177);
 return ((h^(h>>>16))>>>0)/4294967296;
}
export function noise(x,z) {
 const ix=Math.floor(x),iz=Math.floor(z);let fx=x-ix,fz=z-iz;
 fx=fx*fx*(3-2*fx);fz=fz*fz*(3-2*fz);
 return (hash(ix,iz)*(1-fx)+hash(ix+1,iz)*fx)*(1-fz)+(hash(ix,iz+1)*(1-fx)+hash(ix+1,iz+1)*fx)*fz;
}
export function makeSurface(heights,fields,n,pn,world,detail=null,detailSize=2048) {
 function sample(x,z,c=0) {
  const a=c===0?heights:fields,res=c===0?n:pn,stride=c===0?1:11;
  const u=Math.max(0,Math.min(res-1,(x/world+.5)*(res-1))),v=Math.max(0,Math.min(res-1,(z/world+.5)*(res-1)));
  const ix=Math.min(res-2,Math.floor(u)),iz=Math.min(res-2,Math.floor(v)),fx=u-ix,fz=v-iz;
  const at=(i,j)=>a[(j*res+i)*stride+c];
  return (at(ix,iz)*(1-fx)+at(ix+1,iz)*fx)*(1-fz)+(at(ix,iz+1)*(1-fx)+at(ix+1,iz+1)*fx)*fz;
 }
 function height(x,z) {
  const base=sample(x,z),cut=sample(x,z,6),deposit=sample(x,z,7);
  const exposed=(.3+.7*(1-Math.exp(-cut*.12)))*Math.exp(-deposit*.1);
  let fragment=0;
  if(detail?.data){
   const fines=1-Math.exp(-sample(x,z,3)*2);
   fragment=sampleVariantHeight(detail,x,z)*(1-.55*fines);
  }else if(detail){
   const u=((x/2)%1+1)%1*detailSize,v=((z/2)%1+1)%1*detailSize;
   const ix=Math.floor(u),iz=Math.floor(v),fx=u-ix,fz=v-iz;
   const at=(a,b)=>detail[(b%detailSize)*detailSize+a%detailSize];
   fragment=((at(ix,iz)*(1-fx)+at(ix+1,iz)*fx)*(1-fz)+(at(ix,iz+1)*(1-fx)+at(ix+1,iz+1)*fx)*fz-.5)*.16;
  }
  return base+fragment+exposed*(noise(x*.35,z*.35)-.5)*.04;
 }
 // Match the parent's triangle diagonal. Bilinear interpolation would bow the child.
 function parentHeight(x,z,step) {
  const ix=Math.floor((x+world/2)/step),iz=Math.floor((z+world/2)/step);
  const ox=ix*step-world/2,oz=iz*step-world/2,fx=(x-ox)/step,fz=(z-oz)/step;
  const a=height(ox,oz),b=height(ox+step,oz),c=height(ox,oz+step),d=height(ox+step,oz+step);
  return fx+fz<=1?a+(b-a)*fx+(c-a)*fz:d+(c-d)*(1-fx)+(b-d)*(1-fz);
 }
 return {sample,height,parentHeight};
}
