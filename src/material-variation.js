export function materialSeed(x,y) {
 let h=(Math.imul(x|0,374761393)^Math.imul(y|0,668265263))>>>0;
 h=Math.imul(h^(h>>>13),1274126177)>>>0;
 return (h^(h>>>16))>>>0;
}
export function sampleVariantHeight(variants,x,z) {
 const {data,size,count,metres,heightRangeMetres}=variants;
 const px=x/metres,py=z/metres,ix=Math.floor(px),iy=Math.floor(py),fx=px-ix,fy=py-iy;
 const lower=fx+fy<=1;
 const corners=lower?[[ix,iy,1-fx-fy],[ix+1,iy,fx],[ix,iy+1,fy]]:[[ix+1,iy+1,fx+fy-1],[ix+1,iy,1-fy],[ix,iy+1,1-fx]];
 let value=0,total=0;
 for(const [cx,cy,bary] of corners){
  const h=materialSeed(cx,cy),turn=(h>>>8)&3,layer=h%count;
  let u=px,v=py;if(turn===1){u=-py;v=px;}else if(turn===2){u=-px;v=-py;}else if(turn===3){u=py;v=-px;}
  u+=((h>>>16)&255)/256;v+=((h>>>24)&255)/256;
  // Match GPU normalized-repeat sampling (texel centers lie at (i+.5)/size).
  const tx=(u-Math.floor(u))*size-.5,ty=(v-Math.floor(v))*size-.5;
  const ax=Math.floor(tx),ay=Math.floor(ty),dx=tx-ax,dy=ty-ay;
  const at=(a,b)=>data[((layer*size+(b+size)%size)*size+(a+size)%size)*4+2]/255;
  const height=(at(ax,ay)*(1-dx)+at(ax+1,ay)*dx)*(1-dy)+(at(ax,ay+1)*(1-dx)+at(ax+1,ay+1)*dx)*dy;
  const weight=Math.max(0,bary)**3*(.25+height)**2;
  value+=height*weight;total+=weight;
 }
 return value/Math.max(total,1e-9)*heightRangeMetres-(variants.heightCenterMetres??.035);
}
