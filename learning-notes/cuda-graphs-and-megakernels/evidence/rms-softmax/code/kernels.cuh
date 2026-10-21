#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <math_constants.h>
#ifndef STUDY_TRACE
#define STUDY_TRACE 0
#endif
struct Stamp { unsigned long long start,end; int sm,block,visits,slab; };
struct Payload {
    const float* x;
    float *y,*e,*tm,*ts,*rm,*z,*adjusted,*energy,*scale,*squared;
    unsigned *arrivals,*epochs;
    Stamp* stamps;
    int rows,n,tiles;
};
__device__ inline unsigned long long global_time(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ inline int sm_id(){unsigned s;asm volatile("mov.u32 %0, %%smid;":"=r"(s));return int(s);}
__device__ inline void begin_phase(Payload p,int phase,int tile,int slab=0){
    if constexpr(STUDY_TRACE){__syncthreads();if(threadIdx.x==0){auto& s=p.stamps[phase*p.rows*p.tiles+tile];s.start=global_time();s.sm=sm_id();s.block=blockIdx.x;s.slab=slab;atomicAdd(&s.visits,1);}__syncthreads();}}
__device__ inline void end_phase(Payload p,int phase,int tile){
    if constexpr(STUDY_TRACE){__syncthreads();if(threadIdx.x==0)p.stamps[phase*p.rows*p.tiles+tile].end=global_time();__syncthreads();}}
template<int T,bool Max> __device__ float block_reduce(float v,float* scratch){
    for(int d=16;d;d>>=1){float q=__shfl_down_sync(0xffffffff,v,d);v=Max?fmaxf(v,q):v+q;}
    if((threadIdx.x&31)==0)scratch[threadIdx.x/32]=v;
    __syncthreads();
    if(threadIdx.x<32){v=threadIdx.x<T/32?scratch[threadIdx.x]:(Max?-CUDART_INF_F:0.f);
        for(int d=16;d;d>>=1){float q=__shfl_down_sync(0xffffffff,v,d);v=Max?fmaxf(v,q):v+q;}
        if(threadIdx.x==0)scratch[0]=v;}
    __syncthreads();v=scratch[0];__syncthreads();return v;
}
// Each warp independently combines the tiny row metadata; all lanes get the result.
template<bool Max> __device__ float row_reduce(const float* data,int k){
    int lane=threadIdx.x&31;float v=Max?-CUDART_INF_F:0.f;
    for(int j=lane;j<k;j+=32)v=Max?fmaxf(v,data[j]):v+data[j];
    for(int d=16;d;d>>=1){float q=__shfl_down_sync(0xffffffff,v,d);v=Max?fmaxf(v,q):v+q;}
    return __shfl_sync(0xffffffff,v,0);
}
__device__ inline void row_initial(Payload p,int r,float& m,float& scale){
    m=row_reduce<true>(p.tm+r*p.tiles,p.tiles);
    float energy=row_reduce<false>(p.energy+r*p.tiles,p.tiles);
    scale=1.f/sqrtf(energy/float(p.n)+1e-5f);
}
template<int W,int T,bool Retain> __device__ void initial(Payload p,int tile,float* values,float* scratch,int slab=0){
    begin_phase(p,0,tile,slab);int r=tile/p.tiles,t=tile%p.tiles;float m=-CUDART_INF_F,ss=0;
    for(int j=threadIdx.x;j<W;j+=T){bool valid=t*W+j<p.n;float v=valid?p.x[size_t(r)*p.n+t*W+j]:0;
        if constexpr(Retain)values[j]=v;if(valid)m=fmaxf(m,v);ss+=v*v;}
    m=block_reduce<T,true>(m,scratch);ss=block_reduce<T,false>(ss,scratch);
    if(threadIdx.x==0){p.tm[tile]=m;p.energy[tile]=ss;}end_phase(p,0,tile);
}
template<int W,int T,bool Retain,bool Materialize> __device__ void exponentials(Payload p,int tile,float* values,float* scratch,int slab=0){
    begin_phase(p,1,tile,slab);int r=tile/p.tiles,t=tile%p.tiles;float m,scale;row_initial(p,r,m,scale);
    float mass=0;
    for(int j=threadIdx.x;j<W;j+=T){bool valid=t*W+j<p.n;size_t i=size_t(r)*p.n+t*W+j;float v=0;
        if(valid){float x;if constexpr(Retain)x=values[j];else x=p.x[i];v=expf((x-m)*scale);}
        if constexpr(Retain)values[j]=v;
        if constexpr(Materialize)if(valid)p.e[i]=v;mass+=v;}
    mass=block_reduce<T,false>(mass,scratch);
    if(threadIdx.x==0){p.ts[tile]=mass;if(t==0){p.rm[r]=m;p.scale[r]=scale;}}end_phase(p,1,tile);
}
// Source 0=global E, 1=recompute from X, 2=retained E.
template<int W,int T,int Source> __device__ void finish(Payload p,int tile,float* values,int slab=0){
    begin_phase(p,2,tile,slab);int r=tile/p.tiles,t=tile%p.tiles;
    float z=row_reduce<false>(p.ts+r*p.tiles,p.tiles),m=p.rm[r],scale=p.scale[r];
    if(threadIdx.x==0&&t==0)p.z[r]=z;
    for(int j=threadIdx.x;j<W;j+=T)if(t*W+j<p.n){size_t i=size_t(r)*p.n+t*W+j;float v;
        if constexpr(Source==0)v=p.e[i];else if constexpr(Source==1)v=expf((p.x[i]-m)*scale);else v=values[j];p.y[i]=v/z;}
    end_phase(p,2,tile);
}
// Nine explicit groups form the intentionally unfused educational baseline.
template<int W,int T> __global__ void baseline_square(Payload p){int tile=blockIdx.x,r=tile/p.tiles,t=tile%p.tiles;begin_phase(p,0,tile);
    for(int j=threadIdx.x;j<W;j+=T)if(t*W+j<p.n){size_t i=size_t(r)*p.n+t*W+j;float x=p.x[i];p.squared[i]=x*x;}}
template<int W,int T> __global__ void baseline_energy(Payload p){extern __shared__ float scratch[];int tile=blockIdx.x,r=tile/p.tiles,t=tile%p.tiles;float ss=0;
    for(int j=threadIdx.x;j<W;j+=T)if(t*W+j<p.n)ss+=p.squared[size_t(r)*p.n+t*W+j];ss=block_reduce<T,false>(ss,scratch);if(threadIdx.x==0)p.energy[tile]=ss;}
template<int W,int T> __global__ void baseline_max(Payload p){extern __shared__ float scratch[];int tile=blockIdx.x,r=tile/p.tiles,t=tile%p.tiles;float m=-CUDART_INF_F;
    for(int j=threadIdx.x;j<W;j+=T)if(t*W+j<p.n)m=fmaxf(m,p.x[size_t(r)*p.n+t*W+j]);m=block_reduce<T,true>(m,scratch);if(threadIdx.x==0)p.tm[tile]=m;end_phase(p,0,tile);}
__global__ void baseline_scale(Payload p){int r=blockIdx.x;float ss=row_reduce<false>(p.energy+r*p.tiles,p.tiles);if(threadIdx.x==0)p.scale[r]=1.f/sqrtf(ss/float(p.n)+1e-5f);}
__global__ void baseline_row_max(Payload p){int r=blockIdx.x;float m=row_reduce<true>(p.tm+r*p.tiles,p.tiles);if(threadIdx.x==0)p.rm[r]=m;}
template<int W,int T> __global__ void baseline_exp(Payload p){int tile=blockIdx.x,r=tile/p.tiles,t=tile%p.tiles;begin_phase(p,1,tile);
    float m=p.rm[r],scale=p.scale[r];for(int j=threadIdx.x;j<W;j+=T)if(t*W+j<p.n){size_t i=size_t(r)*p.n+t*W+j;p.e[i]=expf((p.x[i]-m)*scale);}}
template<int W,int T> __global__ void baseline_sum(Payload p){extern __shared__ float scratch[];int tile=blockIdx.x,r=tile/p.tiles,t=tile%p.tiles;float z=0;
    for(int j=threadIdx.x;j<W;j+=T)if(t*W+j<p.n)z+=p.e[size_t(r)*p.n+t*W+j];z=block_reduce<T,false>(z,scratch);if(threadIdx.x==0)p.ts[tile]=z;end_phase(p,1,tile);}
__global__ void baseline_denominator(Payload p){int r=blockIdx.x;float z=row_reduce<false>(p.ts+r*p.tiles,p.tiles);if(threadIdx.x==0)p.z[r]=z;}
template<int W,int T> __global__ void baseline_normalize(Payload p){int tile=blockIdx.x,r=tile/p.tiles,t=tile%p.tiles;begin_phase(p,2,tile);float z=p.z[r];
    for(int j=threadIdx.x;j<W;j+=T)if(t*W+j<p.n){size_t i=size_t(r)*p.n+t*W+j;p.y[i]=p.e[i]/z;}end_phase(p,2,tile);}
template<int W,int T> __global__ void local_a(Payload p){extern __shared__ float s[];initial<W,T,false>(p,blockIdx.x,nullptr,s);}
template<int W,int T,bool Materialize> __global__ void local_b(Payload p){extern __shared__ float s[];exponentials<W,T,false,Materialize>(p,blockIdx.x,nullptr,s);}
template<int W,int T,int Source> __global__ void local_c(Payload p){finish<W,T,Source>(p,blockIdx.x,nullptr);}
// Two launches: each tile repeats full-row initial statistics to avoid the first launch boundary.
template<int W,int T,bool Materialize> __global__ void local_ab(Payload p){extern __shared__ float s[];int tile=blockIdx.x,r=tile/p.tiles,t=tile%p.tiles;begin_phase(p,0,tile);
    float m=-CUDART_INF_F,ss=0;for(int j=threadIdx.x;j<p.n;j+=T){float x=p.x[size_t(r)*p.n+j];m=fmaxf(m,x);ss+=x*x;}
    m=block_reduce<T,true>(m,s);ss=block_reduce<T,false>(ss,s);float scale=1.f/sqrtf(ss/float(p.n)+1e-5f),mass=0;
    for(int j=threadIdx.x;j<W;j+=T)if(t*W+j<p.n){size_t i=size_t(r)*p.n+t*W+j;float v=expf((p.x[i]-m)*scale);mass+=v;if constexpr(Materialize)p.e[i]=v;}
    mass=block_reduce<T,false>(mass,s);if(threadIdx.x==0){p.ts[tile]=mass;if(t==0){p.rm[r]=m;p.scale[r]=scale;}}end_phase(p,0,tile);
}
// One launch: repeat two full-row passes per output tile; W=N gives a single CTA per row.
template<int W,int T> __global__ void streaming(Payload p){extern __shared__ float s[];int tile=blockIdx.x,r=tile/p.tiles,t=tile%p.tiles;
    float m=-CUDART_INF_F,ss=0;for(int j=threadIdx.x;j<p.n;j+=T){float x=p.x[size_t(r)*p.n+j];m=fmaxf(m,x);ss+=x*x;}
    m=block_reduce<T,true>(m,s);ss=block_reduce<T,false>(ss,s);float scale=1.f/sqrtf(ss/float(p.n)+1e-5f),z=0;
    for(int j=threadIdx.x;j<p.n;j+=T)z+=expf((p.x[size_t(r)*p.n+j]-m)*scale);z=block_reduce<T,false>(z,s);
    for(int j=threadIdx.x;j<W;j+=T)if(t*W+j<p.n){size_t i=size_t(r)*p.n+t*W+j;p.y[i]=expf((p.x[i]-m)*scale)/z;}}
// Reusable device-scope release/acquire row barrier. Cooperative admission
// guarantees space for every participating CTA; never launch this ordinarily.
__device__ inline void row_barrier(Payload p,int r,int phase){
    __syncthreads();
    if(threadIdx.x==0){
        cuda::atomic_ref<unsigned,cuda::thread_scope_device> epoch(p.epochs[r*2+phase]);
        cuda::atomic_ref<unsigned,cuda::thread_scope_device> count(p.arrivals[r*2+phase]);
        unsigned before=epoch.load(cuda::memory_order_relaxed);
        unsigned prior=count.fetch_add(1,cuda::memory_order_acq_rel);
        if(prior==unsigned(p.tiles-1)){
            count.store(0,cuda::memory_order_relaxed);
            epoch.store(before+1,cuda::memory_order_release);
        }else{
            while(epoch.load(cuda::memory_order_acquire)==before)__nanosleep(32);
        }
    }
    __syncthreads();
}
template<int W,int T> __global__ void row_retained(Payload p,int row_base,int slab_rows,int unused){
    extern __shared__ float values[];float* scratch=values+W;int tile=row_base*p.tiles+blockIdx.x,r=tile/p.tiles;
    initial<W,T,true>(p,tile,values,scratch,row_base);row_barrier(p,r,0);
    exponentials<W,T,true,false>(p,tile,values,scratch,row_base);row_barrier(p,r,1);
    finish<W,T,2>(p,tile,values,row_base);
}
// Same binary controls: mask1 reloads X, mask2 forces E through global storage.
template<int W,int T,bool Control> __global__ void retained(Payload p,int row_base,int slab_rows,int mask){
    extern __shared__ float values[];float* scratch=values+W;auto grid=cooperative_groups::this_grid();int tile=row_base*p.tiles+blockIdx.x;
    initial<W,T,true>(p,tile,values,scratch,row_base);grid.sync();
    int r=tile/p.tiles,t=tile%p.tiles;
    if constexpr(Control)if(mask&1){for(int j=threadIdx.x;j<W;j+=T)values[j]=t*W+j<p.n?p.x[size_t(r)*p.n+t*W+j]:0;__syncthreads();}
    exponentials<W,T,true,false>(p,tile,values,scratch,row_base);
    if constexpr(Control)if(mask&2){for(int j=threadIdx.x;j<W;j+=T)if(t*W+j<p.n)p.e[size_t(r)*p.n+t*W+j]=values[j];}
    grid.sync();
    if constexpr(Control)if(mask&2){for(int j=threadIdx.x;j<W;j+=T)values[j]=t*W+j<p.n?p.e[size_t(r)*p.n+t*W+j]:0;__syncthreads();}
    finish<W,T,2>(p,tile,values,row_base);
}
// Library control: unscaled row statistics, then scale logits, then cuDNN softmax.
template<int T> __global__ void library_rms(Payload p){extern __shared__ float s[];int r=blockIdx.x;float ss=0;
    for(int j=threadIdx.x;j<p.n;j+=T){float x=p.x[size_t(r)*p.n+j];ss+=x*x;}ss=block_reduce<T,false>(ss,s);if(threadIdx.x==0)p.scale[r]=1.f/sqrtf(ss/float(p.n)+1e-5f);}
template<int W,int T> __global__ void library_scale(Payload p){int tile=blockIdx.x,r=tile/p.tiles,t=tile%p.tiles;for(int j=threadIdx.x;j<W;j+=T)if(t*W+j<p.n){size_t i=size_t(r)*p.n+t*W+j;p.e[i]=p.x[i]*p.scale[r];}}
__global__ void evict_cache(float* buffer,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)buffer[i]+=1.f;
}
__global__ void check_replay(const float* y,const double* reference,int rows,int n,int* failures) {
    __shared__ double scratch[16];int r=blockIdx.x;double sum=0,l1=0;int bad=0;
    for(int j=threadIdx.x;j<n;j+=256){size_t i=size_t(r)*n+j;double a=y[i],b=reference[i],e=fabs(a-b);sum+=a;l1+=e;
        bad|=!isfinite(a)||a<0||e>1e-7+2e-5*fabs(b);}
    if(__syncthreads_or(bad)&&threadIdx.x==0)atomicAdd(failures,1);
    for(int d=16;d;d>>=1){sum+=__shfl_down_sync(0xffffffff,sum,d);l1+=__shfl_down_sync(0xffffffff,l1,d);}
    if((threadIdx.x&31)==0){scratch[threadIdx.x/32]=sum;scratch[8+threadIdx.x/32]=l1;}
    __syncthreads();
    if(threadIdx.x==0){sum=0;l1=0;for(int i=0;i<8;++i){sum+=scratch[i];l1+=scratch[8+i];}if(fabs(sum-1)>2e-5||l1>2e-5)atomicAdd(failures,1);}
}
