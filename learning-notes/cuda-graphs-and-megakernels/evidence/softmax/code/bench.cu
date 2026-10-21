#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_profiler_api.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <cudnn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CU(expr) do { auto cuda_status_=(expr); if(cuda_status_!=cudaSuccess) throw std::runtime_error(std::string(#expr)+": "+cudaGetErrorString(cuda_status_)); } while(0)
#define DN(expr) do { auto cudnn_status_=(expr); if(cudnn_status_!=CUDNN_STATUS_SUCCESS) throw std::runtime_error(std::string(#expr)+": "+cudnnGetErrorString(cudnn_status_)); } while(0)
constexpr int THREADS=256, TILE=1024, ROUNDS=21, BATCH=32;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
template<class T> T* allocate(size_t n){T* p;CU(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> void save(const std::string& path,const std::vector<T>& data){std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));if(!f)throw std::runtime_error("write failed: "+path);}
template<class T> void dump(const std::string& path,const T* gpu,size_t n){std::vector<T> h(n);CU(cudaMemcpy(h.data(),gpu,n*sizeof(T),cudaMemcpyDeviceToHost));save(path,h);}

template<bool Maximum> __device__ float reduce(float v){
    __shared__ float scratch[8];
    for(int step=16;step;step>>=1){float q=__shfl_down_sync(0xffffffff,v,step);v=Maximum?fmaxf(v,q):v+q;}
    if((threadIdx.x&31)==0)scratch[threadIdx.x/32]=v;
    __syncthreads();
    if(threadIdx.x<32){v=threadIdx.x<8?scratch[threadIdx.x]:(Maximum?-CUDART_INF_F:0.f);
        for(int step=16;step;step>>=1){float q=__shfl_down_sync(0xffffffff,v,step);v=Maximum?fmaxf(v,q):v+q;}
        if(threadIdx.x==0)scratch[0]=v;}
    __syncthreads();v=scratch[0];__syncthreads();return v;
}
__global__ void row_max(const float* x,float* m,int n){
    float v=-CUDART_INF_F;for(int c=threadIdx.x;c<n;c+=THREADS)v=fmaxf(v,x[size_t(blockIdx.x)*n+c]);
    v=reduce<true>(v);if(threadIdx.x==0)m[blockIdx.x]=v;
}
__global__ void row_exp(const float* x,const float* m,float* e,int n){
    for(int c=threadIdx.x;c<n;c+=THREADS)e[size_t(blockIdx.x)*n+c]=expf(x[size_t(blockIdx.x)*n+c]-m[blockIdx.x]);
}
__global__ void row_sum(const float* e,float* z,int n){
    float v=0.f;for(int c=threadIdx.x;c<n;c+=THREADS)v+=e[size_t(blockIdx.x)*n+c];
    v=reduce<false>(v);if(threadIdx.x==0)z[blockIdx.x]=v;
}
__global__ void row_div(const float* e,const float* z,float* y,int n){
    for(int c=threadIdx.x;c<n;c+=THREADS)y[size_t(blockIdx.x)*n+c]=e[size_t(blockIdx.x)*n+c]/z[blockIdx.x];
}
__global__ void row_fused(const float* x,float* y,int n){
    float value[16],m=-CUDART_INF_F;
    #pragma unroll
    for(int j=0;j<16;++j){int c=j*THREADS+threadIdx.x;value[j]=c<n?x[size_t(blockIdx.x)*n+c]:-CUDART_INF_F;m=fmaxf(m,value[j]);}
    m=reduce<true>(m);float z=0.f;
    #pragma unroll
    for(int j=0;j<16;++j){value[j]=j*THREADS+threadIdx.x<n?expf(value[j]-m):0.f;z+=value[j];}
    z=reduce<false>(z);
    #pragma unroll
    for(int j=0;j<16;++j){int c=j*THREADS+threadIdx.x;if(c<n)y[size_t(blockIdx.x)*n+c]=value[j]/z;}
}

struct Stamp {unsigned long long start,end;int sm,worker,visits;};
struct Payload {
    const float* x;float* y;float* tile_m;float* tile_z;float* row_m;float* row_z;
    const int* row_offsets;const int* tile_offsets;const int* tile_rows;
    int* queue;int* ready;int* done;int* head;int* tail;int* next_a;int* completed_c;
    Stamp* stamps;int rows,tiles;
};
__device__ unsigned long long timer(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ unsigned actual_sm(){unsigned s;asm volatile("mov.u32 %0, %%smid;":"=r"(s));return s;}
template<bool Trace> __device__ void begin_task(Payload p,int id){
    if constexpr(Trace){if(threadIdx.x==0){p.stamps[id].start=timer();p.stamps[id].sm=actual_sm();p.stamps[id].worker=blockIdx.x;atomicAdd(&p.stamps[id].visits,1);}__syncthreads();}
}
template<bool Trace> __device__ void end_task(Payload p,int id){
    __syncthreads();if constexpr(Trace){if(threadIdx.x==0)p.stamps[id].end=timer();__syncthreads();}
}
template<bool Trace> __device__ void task_a(Payload p,int tile){
    begin_task<Trace>(p,tile);int r=p.tile_rows[tile];int offset=p.row_offsets[r]+(tile-p.tile_offsets[r])*TILE;
    float values[4],m=-CUDART_INF_F;
    #pragma unroll
    for(int j=0;j<4;++j){values[j]=p.x[offset+j*THREADS+threadIdx.x];m=fmaxf(m,values[j]);}
    m=reduce<true>(m);float z=0;
    #pragma unroll
    for(int j=0;j<4;++j)z+=expf(values[j]-m);
    z=reduce<false>(z);if(threadIdx.x==0){p.tile_m[tile]=m;p.tile_z[tile]=z;}end_task<Trace>(p,tile);
}
template<bool Trace> __device__ void task_b(Payload p,int r){
    int id=p.tiles+r;begin_task<Trace>(p,id);float m=-CUDART_INF_F;
    for(int t=p.tile_offsets[r]+threadIdx.x;t<p.tile_offsets[r+1];t+=THREADS)m=fmaxf(m,p.tile_m[t]);
    m=reduce<true>(m);float z=0;
    for(int t=p.tile_offsets[r]+threadIdx.x;t<p.tile_offsets[r+1];t+=THREADS)z+=p.tile_z[t]*expf(p.tile_m[t]-m);
    z=reduce<false>(z);if(threadIdx.x==0){p.row_m[r]=m;p.row_z[r]=z;}end_task<Trace>(p,id);
}
template<bool Trace> __device__ void task_c(Payload p,int tile){
    int id=p.tiles+p.rows+tile;begin_task<Trace>(p,id);int r=p.tile_rows[tile];int offset=p.row_offsets[r]+(tile-p.tile_offsets[r])*TILE;
    for(int c=threadIdx.x;c<TILE;c+=THREADS)p.y[offset+c]=expf(p.x[offset+c]-p.row_m[r])/p.row_z[r];
    end_task<Trace>(p,id);
}
template<bool Trace> __global__ void phase_a(Payload p){task_a<Trace>(p,blockIdx.x);}
template<bool Trace> __global__ void phase_b(Payload p){task_b<Trace>(p,blockIdx.x);}
template<bool Trace> __global__ void phase_c(Payload p){task_c<Trace>(p,blockIdx.x);}
__global__ void initialize_queue(Payload p){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<p.tiles+p.rows){p.ready[i]=0;p.queue[i]=-1;}
    if(i<p.rows)p.done[i]=0;if(i==0){*p.head=0;*p.tail=0;*p.next_a=0;*p.completed_c=0;}
}
using Atomic=cuda::atomic_ref<int,cuda::thread_scope_device>;
template<bool Trace> __global__ void persistent(Payload p,int mode){
    if(mode==0){
        auto grid=cooperative_groups::this_grid();
        for(int t=blockIdx.x;t<p.tiles;t+=gridDim.x)task_a<Trace>(p,t);
        grid.sync();
        for(int r=blockIdx.x;r<p.rows;r+=gridDim.x)task_b<Trace>(p,r);
        grid.sync();
        for(int t=blockIdx.x;t<p.tiles;t+=gridDim.x)task_c<Trace>(p,t);
        return;
    }
    __shared__ int id,slot;
    while(true){
        if(threadIdx.x==0){
            while(true){
                if(Atomic(*p.completed_c).load(cuda::memory_order_acquire)==p.tiles){id=-1;break;}
                int q=Atomic(*p.head).load(cuda::memory_order_relaxed);
                if(q<Atomic(*p.tail).load(cuda::memory_order_relaxed)){
                    if(Atomic(*p.head).compare_exchange_weak(q,q+1,cuda::memory_order_relaxed)){
                        while(Atomic(p.ready[q]).load(cuda::memory_order_acquire)==0)__nanosleep(32);
                        id=p.queue[q];break;
                    }
                    continue;
                }
                int a=Atomic(*p.next_a).load(cuda::memory_order_relaxed);
                if(a<p.tiles&&Atomic(*p.next_a).compare_exchange_weak(a,a+1,cuda::memory_order_relaxed)){id=a;break;}
                __nanosleep(32);
            }
        }
        __syncthreads();if(id<0)break;
        if(id<p.tiles){
            task_a<Trace>(p,id);
            if(threadIdx.x==0){int r=p.tile_rows[id];
                if(Atomic(p.done[r]).fetch_add(1,cuda::memory_order_acq_rel)==p.tile_offsets[r+1]-p.tile_offsets[r]-1){
                    int q=Atomic(*p.tail).fetch_add(1,cuda::memory_order_relaxed);p.queue[q]=p.tiles+r;Atomic(p.ready[q]).store(1,cuda::memory_order_release);}}
        }else if(id<p.tiles+p.rows){
            int r=id-p.tiles;task_b<Trace>(p,r);int count=p.tile_offsets[r+1]-p.tile_offsets[r];
            if(threadIdx.x==0)slot=Atomic(*p.tail).fetch_add(count,cuda::memory_order_relaxed);
            __syncthreads();
            for(int j=threadIdx.x;j<count;j+=THREADS){int q=slot+j;p.queue[q]=p.tiles+p.rows+p.tile_offsets[r]+j;Atomic(p.ready[q]).store(1,cuda::memory_order_release);}
        }else{task_c<Trace>(p,id-p.tiles-p.rows);if(threadIdx.x==0)Atomic(*p.completed_c).fetch_add(1,cuda::memory_order_release);}
        __syncthreads();
    }
}

struct Variant {std::string name;std::function<void()> launch;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;};
void add_pair(std::vector<Variant>& variants,const std::string& name,std::function<void()> launch,cudaStream_t s,const std::string& out,std::ofstream& meta,int expected){
    launch();CU(cudaGetLastError());CU(cudaStreamSynchronize(s)); // library initialization before capture
    auto t0=Clock::now();cudaGraph_t g;cudaGraphExec_t e;
    CU(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));launch();CU(cudaStreamEndCapture(s,&g));auto t1=Clock::now();
    CU(cudaGraphInstantiate(&e,g,0));auto t2=Clock::now();
    CU(cudaGraphLaunch(e,s));CU(cudaStreamSynchronize(s));auto t3=Clock::now();
    size_t count=0;CU(cudaGraphGetNodes(g,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);CU(cudaGraphGetNodes(g,nodes.data(),&count));int kernels=0,memsets=0;
    for(auto node:nodes){cudaGraphNodeType type;CU(cudaGraphNodeGetType(node,&type));kernels+=type==cudaGraphNodeTypeKernel;memsets+=type==cudaGraphNodeTypeMemset;}
    if(expected>=0&&(int(count)!=expected||kernels!=expected))throw std::runtime_error("unexpected graph topology: "+name);
    CU(cudaGraphDebugDotPrint(g,(out+"/"+name+".dot").c_str(),cudaGraphDebugDotFlagsVerbose));
    meta<<"{\"kind\":\"graph\",\"variant\":\""<<name<<"\",\"nodes\":"<<count<<",\"kernel_nodes\":"<<kernels<<",\"memset_nodes\":"<<memsets<<",\"capture_us\":"<<elapsed(t0,t1)<<",\"instantiate_us\":"<<elapsed(t1,t2)<<",\"first_replay_sync_us\":"<<elapsed(t2,t3)<<"}\n";
    variants.push_back({name+"_stream",launch});variants.push_back({name+"_graph",[=](){CU(cudaGraphLaunch(e,s));},g,e});
}
void resources(const void* f,const std::string& name,std::ofstream& meta){cudaFuncAttributes a;CU(cudaFuncGetAttributes(&a,f));int active;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,f,THREADS,0));
    meta<<"{\"kind\":\"resource\",\"kernel\":\""<<name<<"\",\"registers\":"<<a.numRegs<<",\"local_bytes\":"<<a.localSizeBytes<<",\"static_shared_bytes\":"<<a.sharedSizeBytes<<",\"active_blocks_per_sm\":"<<active<<"}\n";}
void measure(std::vector<Variant>& v,cudaStream_t s,std::ofstream& samples){
    for(auto& z:v){for(int k=0;k<10;++k)z.launch();CU(cudaStreamSynchronize(s));}
    cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));std::mt19937 gen(20261019);std::vector<int> order(v.size());std::iota(order.begin(),order.end(),0);
    for(int round=0;round<ROUNDS;++round){std::shuffle(order.begin(),order.end(),gen);
        auto idle0=Clock::now();CU(cudaStreamSynchronize(s));auto idle1=Clock::now();
        samples<<"{\"variant\":\"empty_sync\",\"round\":"<<round<<",\"host_single_us\":"<<elapsed(idle0,idle1)<<"}\n";
        for(int pos=0;pos<int(order.size());++pos){auto& z=v[order[pos]];auto t0=Clock::now();z.launch();CU(cudaStreamSynchronize(s));auto t1=Clock::now();
            CU(cudaEventRecord(a,s));auto t2=Clock::now();for(int k=0;k<BATCH;++k)z.launch();auto t3=Clock::now();CU(cudaEventRecord(b,s));CU(cudaEventSynchronize(b));auto t4=Clock::now();float ms;CU(cudaEventElapsedTime(&ms,a,b));
            samples<<"{\"variant\":\""<<z.name<<"\",\"round\":"<<round<<",\"order\":"<<pos<<",\"batch\":"<<BATCH<<",\"host_single_us\":"<<elapsed(t0,t1)<<",\"host_enqueue_us\":"<<elapsed(t2,t3)/BATCH<<",\"host_batch_us\":"<<elapsed(t2,t4)/BATCH<<",\"event_interval_us\":"<<ms*1000/BATCH<<"}\n";
        }
    }CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));
}
std::vector<float> fixture(const std::vector<int>& offsets,int seed){
    std::vector<float> h(offsets.back());
    for(int r=0;r<int(offsets.size())-1;++r)for(int i=offsets[r];i<offsets[r+1];++i){uint32_t z=uint32_t(i)+uint32_t(seed+1)*0x9e3779b9u;z=(z^(z>>16))*0x7feb352du;z=(z^(z>>15))*0x846ca68bu;z^=z>>16;
        float value=(int(z%16385)-8192)/1024.f;int pattern=(r+seed)%4;
        if(pattern==0)value=0;if(pattern==1)value+=80.f;if(pattern==2&&i==offsets[r])value=20.f;h[i]=value;}
    return h;
}
void destroy(std::vector<Variant>& v){for(auto& z:v){if(z.exec)CU(cudaGraphExecDestroy(z.exec));if(z.graph)CU(cudaGraphDestroy(z.graph));}}
void row_case(const std::string& out,int rows,int n,const std::string& mode,const std::string& selected,cudaStream_t s,std::ofstream& meta,std::ofstream& samples){
    if(n<1||n>4096)throw std::runtime_error("row width outside protocol");size_t total=size_t(rows)*n;
    float *x=allocate<float>(total),*y=allocate<float>(total),*e=allocate<float>(total),*m=allocate<float>(rows),*z=allocate<float>(rows);CU(cudaMemsetAsync(x,0,total*4,s));
    cudnnHandle_t handle;DN(cudnnCreate(&handle));DN(cudnnSetStream(handle,s));cudnnTensorDescriptor_t desc;DN(cudnnCreateTensorDescriptor(&desc));DN(cudnnSetTensor4dDescriptor(desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,rows,n,1,1));
    std::vector<Variant> variants;
    add_pair(variants,"staged",[=](){row_max<<<rows,THREADS,0,s>>>(x,m,n);row_exp<<<rows,THREADS,0,s>>>(x,m,e,n);row_sum<<<rows,THREADS,0,s>>>(e,z,n);row_div<<<rows,THREADS,0,s>>>(e,z,y,n);},s,out,meta,4);
    add_pair(variants,"fused",[=](){row_fused<<<rows,THREADS,0,s>>>(x,y,n);},s,out,meta,1);
    add_pair(variants,"cudnn",[=](){float alpha=1,beta=0;DN(cudnnSoftmaxForward(handle,CUDNN_SOFTMAX_ACCURATE,CUDNN_SOFTMAX_MODE_INSTANCE,&alpha,desc,x,&beta,desc,y));},s,out,meta,-1);
    for(auto pair:std::vector<std::pair<const void*,std::string>>{{(const void*)row_max,"row_max"},{(const void*)row_exp,"row_exp"},{(const void*)row_sum,"row_sum"},{(const void*)row_div,"row_div"},{(const void*)row_fused,"row_fused"}})resources(pair.first,pair.second,meta);
    std::vector<int> offsets(rows+1);for(int r=0;r<=rows;++r)offsets[r]=r*n;save(out+"/offsets.i32",offsets);
    for(int f=0;f<(mode=="profile"?1:2);++f){int seed=(mode=="validate"?0:2)+f;auto h=fixture(offsets,seed);CU(cudaMemcpyAsync(x,h.data(),total*4,cudaMemcpyHostToDevice,s));CU(cudaStreamSynchronize(s));
        if(mode=="profile"){CU(cudaProfilerStart());for(auto& v:variants)if(selected=="all"||v.name==selected)v.launch();CU(cudaStreamSynchronize(s));CU(cudaProfilerStop());break;}
        save(out+"/input_"+std::to_string(seed)+".f32",h);
        for(auto& v:variants){v.launch();CU(cudaGetLastError());CU(cudaStreamSynchronize(s));dump(out+"/"+v.name+"_"+std::to_string(seed)+".f32",y,total);}
        dump(out+"/maximum_"+std::to_string(seed)+".f32",m,rows);dump(out+"/denominator_"+std::to_string(seed)+".f32",z,rows);
        if(mode=="bench"&&f==0)measure(variants,s,samples);
    }
    destroy(variants);DN(cudnnDestroyTensorDescriptor(desc));DN(cudnnDestroy(handle));for(auto p:{x,y,e,m,z})CU(cudaFree(p));
}
template<bool Trace> void dag_case(const std::string& out,int rows,const std::string& pattern,const std::string& mode,const std::string& selected,cudaStream_t s,const cudaDeviceProp& prop,std::ofstream& meta,std::ofstream& samples){
    if(rows%8)throw std::runtime_error("row count must be divisible by eight");
    std::vector<int> lengths(rows,8192);if(pattern=="ragged"){for(int r=0;r<rows;++r)lengths[r]=r%8==0?36864:4096;std::mt19937 gen(20261019);std::shuffle(lengths.begin(),lengths.end(),gen);}else if(pattern!="balanced")throw std::runtime_error("bad pattern");
    std::vector<int> offsets(rows+1),tile_offsets(rows+1),tile_rows;
    for(int r=0;r<rows;++r){offsets[r+1]=offsets[r]+lengths[r];tile_offsets[r+1]=tile_offsets[r]+lengths[r]/TILE;for(int t=0;t<lengths[r]/TILE;++t)tile_rows.push_back(r);}
    int tiles=tile_rows.size(),total=offsets.back(),tasks=2*tiles+rows;
    Payload p{};p.rows=rows;p.tiles=tiles;p.x=allocate<float>(total);p.y=allocate<float>(total);p.tile_m=allocate<float>(tiles);p.tile_z=allocate<float>(tiles);p.row_m=allocate<float>(rows);p.row_z=allocate<float>(rows);
    p.row_offsets=allocate<int>(rows+1);p.tile_offsets=allocate<int>(rows+1);p.tile_rows=allocate<int>(tiles);p.queue=allocate<int>(tiles+rows);p.ready=allocate<int>(tiles+rows);p.done=allocate<int>(rows);p.head=allocate<int>(1);p.tail=allocate<int>(1);p.next_a=allocate<int>(1);p.completed_c=allocate<int>(1);p.stamps=allocate<Stamp>(tasks);
    CU(cudaMemcpyAsync((void*)p.row_offsets,offsets.data(),(rows+1)*4,cudaMemcpyHostToDevice,s));CU(cudaMemcpyAsync((void*)p.tile_offsets,tile_offsets.data(),(rows+1)*4,cudaMemcpyHostToDevice,s));CU(cudaMemcpyAsync((void*)p.tile_rows,tile_rows.data(),tiles*4,cudaMemcpyHostToDevice,s));CU(cudaMemsetAsync((void*)p.x,0,total*4,s));CU(cudaMemsetAsync(p.stamps,0,tasks*sizeof(Stamp),s));CU(cudaStreamSynchronize(s));
    int active;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,persistent<Trace>,THREADS,0));int workers=std::min(tiles,prop.multiProcessorCount*active);
    auto coop=[=](int which){Payload arg=p;void* args[]={&arg,&which};CU(cudaLaunchCooperativeKernel((void*)persistent<Trace>,dim3(workers),dim3(THREADS),args,0,s));};
    std::vector<Variant> variants;
    add_pair(variants,"separate",[=](){phase_a<Trace><<<tiles,THREADS,0,s>>>(p);phase_b<Trace><<<rows,THREADS,0,s>>>(p);phase_c<Trace><<<tiles,THREADS,0,s>>>(p);},s,out,meta,3);
    add_pair(variants,"phased",[=](){coop(0);},s,out,meta,1);
    add_pair(variants,"dag",[=](){initialize_queue<<<(tiles+rows+THREADS-1)/THREADS,THREADS,0,s>>>(p);coop(1);},s,out,meta,2);
    meta<<"{\"kind\":\"dag\",\"rows\":"<<rows<<",\"elements\":"<<total<<",\"tiles\":"<<tiles<<",\"tasks\":"<<tasks<<",\"workers\":"<<workers<<",\"active_blocks_per_sm\":"<<active<<",\"instrumented\":"<<(Trace?"true":"false")<<"}\n";
    resources((const void*)persistent<Trace>,"persistent",meta);resources((const void*)phase_a<Trace>,"phase_a",meta);resources((const void*)phase_b<Trace>,"phase_b",meta);resources((const void*)phase_c<Trace>,"phase_c",meta);
    save(out+"/offsets.i32",offsets);save(out+"/tile_offsets.i32",tile_offsets);save(out+"/tile_rows.i32",tile_rows);
    for(int f=0;f<(mode=="profile"?1:2);++f){int seed=(mode=="validate"?0:2)+f;auto h=fixture(offsets,seed);CU(cudaMemcpyAsync((void*)p.x,h.data(),total*4,cudaMemcpyHostToDevice,s));CU(cudaStreamSynchronize(s));
        if(mode=="profile"){CU(cudaProfilerStart());for(auto& v:variants)if(selected=="all"||v.name==selected)v.launch();CU(cudaStreamSynchronize(s));CU(cudaProfilerStop());break;}
        save(out+"/input_"+std::to_string(seed)+".f32",h);
        for(auto& v:variants){if constexpr(Trace)CU(cudaMemsetAsync(p.stamps,0,tasks*sizeof(Stamp),s));v.launch();CU(cudaGetLastError());CU(cudaStreamSynchronize(s));std::string prefix=out+"/"+v.name+"_"+std::to_string(seed);
            dump(prefix+".f32",p.y,total);dump(prefix+"_tile_max.f32",p.tile_m,tiles);dump(prefix+"_tile_sum.f32",p.tile_z,tiles);dump(prefix+"_row_max.f32",p.row_m,rows);dump(prefix+"_row_sum.f32",p.row_z,rows);
            if(v.name.rfind("dag",0)==0){int head,tail,next_a,completed_c;CU(cudaMemcpy(&head,p.head,4,cudaMemcpyDeviceToHost));CU(cudaMemcpy(&tail,p.tail,4,cudaMemcpyDeviceToHost));CU(cudaMemcpy(&next_a,p.next_a,4,cudaMemcpyDeviceToHost));CU(cudaMemcpy(&completed_c,p.completed_c,4,cudaMemcpyDeviceToHost));if(tail!=tiles+rows||head!=tail||next_a!=tiles||completed_c!=tiles)throw std::runtime_error("queue bounds invalid");dump(prefix+"_done.i32",p.done,rows);meta<<"{\"kind\":\"queue\",\"variant\":\""<<v.name<<"\",\"fixture\":"<<seed<<",\"head\":"<<head<<",\"tail\":"<<tail<<",\"next_a\":"<<next_a<<",\"completed_c\":"<<completed_c<<"}\n";}
            if constexpr(Trace){std::vector<Stamp> stamps(tasks);CU(cudaMemcpy(stamps.data(),p.stamps,tasks*sizeof(Stamp),cudaMemcpyDeviceToHost));std::ofstream file(prefix+"_trace.csv");file<<"id,stage,row,tile,sm,worker,visits,start_ns,end_ns\n";
                for(int id=0;id<tasks;++id){int stage=id<tiles?0:id<tiles+rows?1:2;int tile=stage==0?id:stage==2?id-tiles-rows:-1;int row=stage==1?id-tiles:tile_rows[tile];auto q=stamps[id];file<<id<<","<<stage<<","<<row<<","<<tile<<","<<q.sm<<","<<q.worker<<","<<q.visits<<","<<q.start<<","<<q.end<<"\n";}}
        }
        if(mode=="bench"&&f==0)measure(variants,s,samples);
    }
    destroy(variants);for(auto q:{p.x,(const float*)p.y,(const float*)p.tile_m,(const float*)p.tile_z,(const float*)p.row_m,(const float*)p.row_z})CU(cudaFree((void*)q));
    for(auto q:{p.row_offsets,p.tile_offsets,p.tile_rows,(const int*)p.queue,(const int*)p.ready,(const int*)p.done,(const int*)p.head,(const int*)p.tail,(const int*)p.next_a,(const int*)p.completed_c})CU(cudaFree((void*)q));CU(cudaFree(p.stamps));
}
int main(int argc,char** argv){try{
    if(argc<7)throw std::runtime_error("usage: bench OUT row|dag ROWS COLS balanced|ragged validate|bench|trace|profile [VARIANT]");
    std::string out=argv[1],kind=argv[2],pattern=argv[5],mode=argv[6],selected=argc>7?argv[7]:"all";int rows=std::stoi(argv[3]),cols=std::stoi(argv[4]);std::filesystem::create_directories(out);
    cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,0));if(std::string(prop.name).find("H100")==std::string::npos||!prop.cooperativeLaunch)throw std::runtime_error("exact cooperative H100 required");
    int driver,runtime;CU(cudaDriverGetVersion(&driver));CU(cudaRuntimeGetVersion(&runtime));
    std::ofstream meta(out+"/metadata.jsonl"),samples(out+"/samples.jsonl");meta<<std::setprecision(12);samples<<std::setprecision(12);
    meta<<"{\"kind\":\"device\",\"name\":\""<<prop.name<<"\",\"sms\":"<<prop.multiProcessorCount<<",\"l2_bytes\":"<<prop.l2CacheSize<<",\"global_bytes\":"<<prop.totalGlobalMem<<",\"driver\":"<<driver<<",\"runtime\":"<<runtime<<",\"cudnn\":"<<cudnnGetVersion()<<",\"cc\":"<<prop.major*10+prop.minor<<",\"threads\":"<<THREADS<<"}\n";
    meta<<"{\"kind\":\"case\",\"operation\":\""<<kind<<"\",\"rows\":"<<rows<<",\"cols\":"<<cols<<",\"pattern\":\""<<pattern<<"\",\"mode\":\""<<mode<<"\"}\n";
    cudaStream_t stream;CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    if(kind=="row")row_case(out,rows,cols,mode,selected,stream,meta,samples);
    else if(kind=="dag"&&mode=="trace")dag_case<true>(out,rows,pattern,mode,selected,stream,prop,meta,samples);
    else if(kind=="dag")dag_case<false>(out,rows,pattern,mode,selected,stream,prop,meta,samples);
    else throw std::runtime_error("unknown operation");
    CU(cudaGetLastError());CU(cudaStreamSynchronize(stream));CU(cudaStreamDestroy(stream));std::cout<<"completed "<<kind<<" "<<mode<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
