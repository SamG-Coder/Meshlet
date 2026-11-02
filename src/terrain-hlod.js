export const tileKey=t=>t.level+'/'+t.tx+'/'+t.tz;
const box=t=>{const s=2**-t.level;return {t,x:t.tx*s,z:t.tz*s,r:(t.tx+1)*s,b:(t.tz+1)*s};};
function touches(a,b,corners=false){
 const x=Math.min(a.r,b.r)-Math.max(a.x,b.x),z=Math.min(a.b,b.b)-Math.max(a.z,b.z);
 return ((a.r===b.x||a.x===b.r)&&(corners?z>=0:z>0))||((a.b===b.z||a.z===b.b)&&(corners?x>=0:x>0));
}
export function balanceCover(entries){
 const cover=new Map(entries.map(t=>[tileKey(t),t]));
 let changed=true;
 while(changed){changed=false;const list=[...cover.values()].map(box);
  outer:for(let i=0;i<list.length;i++)for(let j=i+1;j<list.length;j++){
   const aa=list[i],bb=list[j],a=aa.t,b=bb.t;if(Math.abs(a.level-b.level)<=1||!touches(aa,bb,true))continue;
   const fine=a.level>b.level?a:b,coarse=fine===a?b:a,level=coarse.level+1;
   const shift=fine.level-level,parent={level,tx:Math.floor(fine.tx/2**shift),tz:Math.floor(fine.tz/2**shift)};
   for(const [id,t]of cover)if(t.level>=level&&Math.floor(t.tx/2**(t.level-level))===parent.tx&&Math.floor(t.tz/2**(t.level-level))===parent.tz)cover.delete(id);
   cover.set(tileKey(parent),parent);changed=true;break outer;
  }
 }
 return [...cover.values()];
}
export function coverEdgeFlags(entries){
 const flags=new Map(entries.map(t=>[tileKey(t),[0,0,0,0]]));
 const boxes=entries.map(box);
 for(let i=0;i<entries.length;i++)for(let j=i+1;j<entries.length;j++){
  const a=entries[i],b=entries[j],aa=boxes[i],bb=boxes[j];if(a.level===b.level||!touches(aa,bb))continue;
  const af=flags.get(tileKey(a)),bf=flags.get(tileKey(b)),sign=a.level>b.level?-1:1;
  if(aa.r===bb.x){af[1]=sign;bf[0]=-sign;}if(aa.x===bb.r){af[0]=sign;bf[1]=-sign;}
  if(aa.b===bb.z){af[3]=sign;bf[2]=-sign;}if(aa.z===bb.b){af[2]=sign;bf[3]=-sign;}
 }
 return flags;
}

