// Diffuse path tracing of the retained heightfield. No terrain synthesis.
struct V { float x,y,z; };
__device__ V v(float x,float y,float z){return {x,y,z};}
__device__ V add(V a,V b){return v(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ V sub(V a,V b){return v(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ V mul(V a,float b){return v(a.x*b,a.y*b,a.z*b);}
__device__ V prod(V a,V b){return v(a.x*b.x,a.y*b.y,a.z*b.z);}
__device__ float dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ V cross(V a,V b){return v(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ V norm(V a){return mul(a,rsqrtf(fmaxf(dot(a,a),1.e-20f)));}
__device__ float clamp(float a,float lo,float hi){return fminf(hi,fmaxf(lo,a));}
__device__ unsigned hash(unsigned x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return x;}
__device__ float random(unsigned &s){s=hash(s+0x9e3779b9U);return (s>>8)*0x1p-24f;}
__device__ float hsample(const float* h,int n,float x,float z){
  x=clamp(x,0.f,float(n-1)-.001f);z=clamp(z,0.f,float(n-1)-.001f);
  int ix=int(x),iz=int(z);float u=x-ix,w=z-iz;
  return (h[iz*n+ix]*(1-u)+h[iz*n+ix+1]*u)*(1-w)+(h[(iz+1)*n+ix]*(1-u)+h[(iz+1)*n+ix+1]*u)*w;
}
__device__ V normalAt(const float*h,int n,float dx,V p){
  float x=p.x/dx,z=p.z/dx;
  return norm(v((hsample(h,n,x-1,z)-hsample(h,n,x+1,z))/(2*dx),1,(hsample(h,n,x,z-1)-hsample(h,n,x,z+1))/(2*dx)));
}
__device__ V albedoAt(const float*a,int n,float world,V p){
  int x=int(clamp(p.x/world*float(n-1),0.f,float(n-1))),z=int(clamp(p.z/world*float(n-1),0.f,float(n-1)));
  int i=(z*n+x)*3;return v(a[i],a[i+1],a[i+2]);
}
__device__ float tri(V o,V d,V a,V b,V c){
  V e1=sub(b,a),e2=sub(c,a),p=cross(d,e2);float determinant=dot(e1,p);
  if(fabsf(determinant)<1e-8f)return 1e30f;
  float inv=1/determinant;V t=sub(o,a);float u=dot(t,p)*inv;if(u<-.0001f||u>1.0001f)return 1e30f;
  V q=cross(t,e1);float w=dot(d,q)*inv;if(w<-.0001f||u+w>1.0001f)return 1e30f;
  float hit=dot(e2,q)*inv;return hit>.01f?hit:1e30f;
}
__device__ float trace(const float*h,const float*pyramid,const int*offset,int n,int top,float dx,V o,V d,unsigned* stats){
  float world=dx*(n-1),t=.03f,tEnd=60000.f;
  if(d.x>1e-8f)tEnd=fminf(tEnd,(world-o.x)/d.x);else if(d.x< -1e-8f)tEnd=fminf(tEnd,-o.x/d.x);
  if(d.z>1e-8f)tEnd=fminf(tEnd,(world-o.z)/d.z);else if(d.z< -1e-8f)tEnd=fminf(tEnd,-o.z/d.z);
  for(int iteration=0;iteration<16384&&t<tEnd;iteration++){
    V pos=add(o,mul(d,t));bool moved=false;
    for(int level=top;level>=0;level--){
      int width=n>>level;float cell=dx*float(1<<level);
      int ix=int(floorf(pos.x/cell)),iz=int(floorf(pos.z/cell));
      if(ix<0||iz<0||ix>=width||iz>=width)return 1e30f;
      float tx=1e30f,tz=1e30f;
      if(fabsf(d.x)>1e-8f)tx=(((d.x>0?ix+1:ix)*cell)-o.x)/d.x;
      if(fabsf(d.z)>1e-8f)tz=(((d.z>0?iz+1:iz)*cell)-o.z)/d.z;
      float exit=fminf(tEnd,fminf(tx,tz));
      float highest=pyramid[offset[level]+iz*width+ix];
      if(fminf(pos.y,o.y+d.y*exit)>highest+.001f){t=fmaxf(t+.02f,exit+.02f);moved=true;break;}
      if(level==0){
        if(ix>=n-1||iz>=n-1)return 1e30f;
        // Intersect the same bilinear patch that supplies the fine render grid.
        // A two-triangle source proxy can lie above/below that surface on saddles.
        float h00=h[iz*n+ix],hx=h[iz*n+ix+1]-h00,hz=h[(iz+1)*n+ix]-h00;
        float hxz=h[(iz+1)*n+ix+1]-h00-hx-hz;
        float u=pos.x/dx-ix,w=pos.z/dx-iz,du=d.x/dx,dw=d.z/dx;
        float aa=-hxz*du*dw,bb=d.y-hx*du-hz*dw-hxz*(u*dw+w*du);
        float cc=pos.y-h00-hx*u-hz*w-hxz*u*w,local=1e30f;
        if(cc<-.01f)return t;
        if(fabsf(aa)<1e-10f){if(fabsf(bb)>1e-10f)local=-cc/bb;}
        else{
          float disc=bb*bb-4*aa*cc;
          if(disc>=0){float r1=(-bb-sqrtf(disc))/(2*aa),r2=(-bb+sqrtf(disc))/(2*aa);local=fminf(r1>=-.002f?r1:1e30f,r2>=-.002f?r2:1e30f);}
        }
        if(local>=-.002f&&local<=exit-t+.002f)return t+fmaxf(0.f,local);
        t=fmaxf(t+.02f,exit+.02f);moved=true;
      }
    }
    if(!moved)break;
  }
  if(t<tEnd)atomicAdd(stats,1U);
  return 1e30f;
}
__device__ V cosineDirection(V n,unsigned &rng){
  float u=random(rng),phi=6.283185307f*random(rng),r=sqrtf(u);
  V t=norm(cross(fabsf(n.y)<.95f?v(0,1,0):v(1,0,0),n)),b=cross(n,t);
  return norm(add(add(mul(t,r*cosf(phi)),mul(b,r*sinf(phi))),mul(n,sqrtf(1-u))));
}
__device__ V sky(V d,V sun){
  float h=powf(clamp(d.y,0.f,1.f),.45f),glow=powf(fmaxf(0.f,dot(d,sun)),12.f)*.25f;
  return add(add(mul(v(.78f,.94f,1.16f),1-h),mul(v(.12f,.30f,.63f),h)),mul(v(1.f,.9f,.72f),glow));
}
extern "C" __global__ void bake(const float*h,const float*pyr,const int*off,const float*albedo,int an,int n,int top,float dx,
 int size,int rowStart,int rows,int spp,int bounces,float sunX,float sunY,float sunZ,float*outSky,float*outSun,unsigned*stats){
  int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=size*rows)return;
  int x=id%size,z=id/size+rowStart;float world=dx*(n-1);V sun=norm(v(sunX,sunY,sunZ));
  V p=v((x+.5f)*world/size,0,(z+.5f)*world/size);p.y=hsample(h,n,p.x/dx,p.z/dx);
  V normal=normalAt(h,n,dx,p);V origin=add(p,mul(normal,.25f));
  V skySum=v(0,0,0),sunSum=v(0,0,0);unsigned rng=hash(x+z*size+701U);
  for(int sample=0;sample<spp;sample++){
    V o=origin,d=cosineDirection(normal,rng),beta=v(1,1,1);
    for(int bounce=0;bounce<bounces;bounce++){
      float hit=trace(h,pyr,off,n,top,dx,o,d,stats);
      if(hit>1e20f){skySum=add(skySum,prod(beta,sky(d,sun)));break;}
      V point=add(o,mul(d,hit)),nn=normalAt(h,n,dx,point);beta=prod(beta,albedoAt(albedo,an,world,point));
      float cosSun=fmaxf(0.f,dot(nn,sun));o=add(point,mul(nn,.25f));
      if(cosSun>0&&trace(h,pyr,off,n,top,dx,o,sun,stats)>1e20f)sunSum=add(sunSum,mul(beta,cosSun/3.1415926536f));
      d=cosineDirection(nn,rng);
    }
  }
  int at=id*4;float scale=3.1415926536f/spp;
  outSky[at]=skySum.x*scale;outSky[at+1]=skySum.y*scale;outSky[at+2]=skySum.z*scale;outSky[at+3]=1;
  outSun[at]=sunSum.x*scale;outSun[at+1]=sunSum.y*scale;outSun[at+2]=sunSum.z*scale;outSun[at+3]=1;
}
extern "C" __global__ void visibility(const float*h,const float*pyr,const int*off,int n,int top,float dx,int size,int rowStart,int rows,int spp,
 float sunX,float sunY,float sunZ,float*out,unsigned*stats){
  int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=size*rows)return;
  int x=id%size,z=id/size+rowStart;float world=dx*(n-1);V sun=norm(v(sunX,sunY,sunZ));
  V t=norm(cross(v(0,1,0),sun)),b=cross(sun,t);float visible=0;
  unsigned rng=hash(x+z*size+817U);
  for(int s=0;s<spp;s++){
    V p=v((x+.5f)*world/size,0,(z+.5f)*world/size);p.y=hsample(h,n,p.x/dx,p.z/dx);
    V nn=normalAt(h,n,dx,p);p=add(p,mul(nn,.25f));
    float r=sqrtf(random(rng))*.00465f,a=random(rng)*6.283185307f;
    V d=norm(add(sun,add(mul(t,r*cosf(a)),mul(b,r*sinf(a)))));
    visible+=(trace(h,pyr,off,n,top,dx,p,d,stats)>1e20f?1.f:0.f);
  }
  out[id]=visible/spp;
}
