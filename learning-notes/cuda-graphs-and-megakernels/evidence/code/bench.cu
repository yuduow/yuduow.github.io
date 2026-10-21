#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
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

#define CU(x) do { auto cuda_check_status=(x); if(cuda_check_status!=cudaSuccess) throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(cuda_check_status)); } while(0)
using Clock=std::chrono::steady_clock;
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
template<class T> void save(const std::string& p,const std::vector<T>& v){std::ofstream f(p,std::ios::binary);f.write((const char*)v.data(),v.size()*sizeof(T));if(!f)throw std::runtime_error("write "+p);}
template<class T> T* alloc(size_t n){T* p;CU(cudaMalloc(&p,n*sizeof(T)));return p;}

__global__ void add_one(const float* x,float* a,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]=__fadd_rn(x[i],1.f);}
__global__ void square(const float* a,float* b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=__fmul_rn(a[i],a[i]);}
__global__ void half_value(const float* b,float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=__fmul_rn(b[i],.5f);}
__global__ void fused_pointwise(const float* x,float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float a=__fadd_rn(x[i],1.f);float b=__fmul_rn(a,a);y[i]=__fmul_rn(b,.5f);}}

struct Stamp {unsigned long long a0,a1,b0,b1;unsigned int sm,worker,sm_b,worker_b;};
__device__ unsigned long long timer(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ unsigned smid(){unsigned s;asm volatile("mov.u32 %0, %%smid;":"=r"(s));return s;}
__device__ uint32_t iterate(uint32_t v,int count,uint32_t a,uint32_t b){
    #pragma unroll 1
    for(int i=0;i<count;++i)v=v*a+b;return v;
}
template<bool Trace> __device__ void stage_a(int job,int worker,const uint32_t* x,volatile uint32_t* tmp,const int* work,Stamp* trace){
    if constexpr(Trace) {if(threadIdx.x==0){trace[job].a0=timer();trace[job].sm=smid();trace[job].worker=worker;}}
    int i=job*256+threadIdx.x;tmp[i]=iterate(x[i],work[job],1664525u,1013904223u);
    __syncthreads();
    if constexpr(Trace) {if(threadIdx.x==0)trace[job].a1=timer();}
}
template<bool Trace> __device__ void stage_b(int job,const volatile uint32_t* tmp,uint32_t* y,Stamp* trace){
    if constexpr(Trace) {if(threadIdx.x==0){trace[job].b0=timer();trace[job].sm_b=smid();trace[job].worker_b=blockIdx.x;}}
    int i=job*256+threadIdx.x;y[i]=iterate(tmp[i],512,22695477u,1u);
    __syncthreads();
    if constexpr(Trace) {if(threadIdx.x==0)trace[job].b1=timer();}
}
template<bool Trace> __global__ void ragged_a(const uint32_t* x,volatile uint32_t* tmp,const int* work,Stamp* t){stage_a<Trace>(blockIdx.x,blockIdx.x,x,tmp,work,t);}
template<bool Trace> __global__ void ragged_b(const volatile uint32_t* tmp,uint32_t* y,Stamp* t){stage_b<Trace>(blockIdx.x,tmp,y,t);}
template<bool Trace> __global__ void ragged_fused(const uint32_t* x,volatile uint32_t* tmp,uint32_t* y,const int* work,Stamp* t){stage_a<Trace>(blockIdx.x,blockIdx.x,x,tmp,work,t);__syncthreads();stage_b<Trace>(blockIdx.x,tmp,y,t);}
template<bool Trace> __global__ void ragged_queue(const uint32_t* x,volatile uint32_t* tmp,uint32_t* y,const int* work,Stamp* t,int jobs,unsigned* next){
    __shared__ unsigned job;
    while(true){if(threadIdx.x==0)job=atomicAdd(next,1);__syncthreads();if(job>=jobs)break;
        stage_a<Trace>(job,blockIdx.x,x,tmp,work,t);__syncthreads();stage_b<Trace>(job,tmp,y,t);__syncthreads();}
}

struct Variant {std::string name;std::function<void()> launch;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;};
void graph_variant(std::vector<Variant>& v,std::string name,std::function<void()> launch,cudaStream_t s,const std::string& root,std::ofstream& meta,int expected){
    v.push_back({name+"_stream",launch});
    auto start=Clock::now();cudaGraph_t g;cudaGraphExec_t e;
    CU(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));launch();CU(cudaStreamEndCapture(s,&g));
    auto captured=Clock::now();CU(cudaGraphInstantiate(&e,g,0));auto instantiated=Clock::now();
    size_t nodes=0;CU(cudaGraphGetNodes(g,nullptr,&nodes));if(nodes!=(size_t)expected)throw std::runtime_error("unexpected graph node count");
    CU(cudaGraphDebugDotPrint(g,(root+"/"+name+".dot").c_str(),cudaGraphDebugDotFlagsVerbose));
    CU(cudaGraphLaunch(e,s));CU(cudaStreamSynchronize(s));auto warmed=Clock::now();
    meta<<"{\"kind\":\"graph\",\"case\":\""<<name<<"\",\"nodes\":"<<nodes<<",\"capture_us\":"<<us(start,captured)<<",\"instantiate_us\":"<<us(captured,instantiated)<<",\"first_replay_sync_us\":"<<us(instantiated,warmed)<<"}\n";
    v.push_back({name+"_graph",[=](){CU(cudaGraphLaunch(e,s));},g,e});
}
void measure(std::vector<Variant>& v,cudaStream_t s,const std::string& label,int count,std::ofstream& samples){
    for(auto& x:v){for(int i=0;i<10;++i)x.launch();CU(cudaStreamSynchronize(s));}
    cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));
    std::mt19937 rng(20261018);std::vector<int> order(v.size());std::iota(order.begin(),order.end(),0);
    for(int round=0;round<21;++round){std::shuffle(order.begin(),order.end(),rng);
        for(int pos=0;pos<(int)order.size();++pos){auto& x=v[order[pos]];
            auto t0=Clock::now();x.launch();CU(cudaStreamSynchronize(s));auto t1=Clock::now();
            CU(cudaEventRecord(a,s));auto t2=Clock::now();for(int i=0;i<count;++i)x.launch();auto t3=Clock::now();
            CU(cudaEventRecord(b,s));CU(cudaEventSynchronize(b));auto t4=Clock::now();float ms;CU(cudaEventElapsedTime(&ms,a,b));
            samples<<"{\"case\":\""<<label<<"\",\"variant\":\""<<x.name<<"\",\"round\":"<<round<<",\"order\":"<<pos<<",\"batch\":"<<count<<",\"host_single_us\":"<<us(t0,t1)<<",\"host_enqueue_us\":"<<us(t2,t3)/count<<",\"host_batch_us\":"<<us(t2,t4)/count<<",\"event_interval_us\":"<<ms*1000/count<<"}\n";
        }
    }CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));
}
void destroy(std::vector<Variant>& v){for(auto& x:v){if(x.exec)CU(cudaGraphExecDestroy(x.exec));if(x.graph)CU(cudaGraphDestroy(x.graph));}}
template<class T> void dump_device(const std::string& p,const T* d,size_t n){std::vector<T> h(n);CU(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));save(p,h);}

void pointwise(const std::string& root,cudaStream_t s,std::ofstream& meta,std::ofstream& samples,bool profile,bool timed,const std::string& select){
    for(int n:{256,32768,1048576,16777216}){
        if(profile&&n!=16777216)continue;std::string label="pointwise_"+std::to_string(n);std::string dir=root+"/"+label;std::filesystem::create_directories(dir);
        float *x=alloc<float>(n),*a=alloc<float>(n),*b=alloc<float>(n),*y=alloc<float>(n);
        CU(cudaMemsetAsync(x,0,n*4,s));CU(cudaStreamSynchronize(s));
        std::vector<Variant> v;int blocks=(n+255)/256;
        graph_variant(v,label+"_separate",[=](){add_one<<<blocks,256,0,s>>>(x,a,n);square<<<blocks,256,0,s>>>(a,b,n);half_value<<<blocks,256,0,s>>>(b,y,n);},s,dir,meta,3);
        graph_variant(v,label+"_fused",[=](){fused_pointwise<<<blocks,256,0,s>>>(x,y,n);},s,dir,meta,1);
        for(int fixture=0;fixture<4;++fixture){if(profile&&fixture!=2)continue;std::vector<float> h(n);for(int i=0;i<n;++i){uint32_t z=(uint32_t)i*1664525u+(fixture+1)*1013904223u;h[i]=fixture==1?0.f:((int)(z%2049)-1024)/1024.f;}
            if(profile){CU(cudaMemcpyAsync(x,h.data(),n*4,cudaMemcpyHostToDevice,s));CU(cudaStreamSynchronize(s));for(auto& z:v){if(select.empty()||z.name.find(select)!=std::string::npos){CU(cudaProfilerStart());for(int i=0;i<3;++i)z.launch();CU(cudaStreamSynchronize(s));CU(cudaProfilerStop());}}continue;}
            CU(cudaMemcpyAsync(x,h.data(),n*4,cudaMemcpyHostToDevice,s));CU(cudaStreamSynchronize(s));save(dir+"/input_"+std::to_string(fixture)+".f32",h);
            for(auto& z:v){z.launch();CU(cudaStreamSynchronize(s));dump_device(dir+"/"+z.name+"_fixture"+std::to_string(fixture)+".f32",y,n);}
            if(fixture==2&&timed)measure(v,s,label,n==16777216?32:128,samples);
        }
        destroy(v);CU(cudaFree(x));CU(cudaFree(a));CU(cudaFree(b));CU(cudaFree(y));
    }
}
template<bool Trace> void ragged_case(const std::string& root,cudaStream_t s,const cudaDeviceProp& prop,std::ofstream& meta,std::ofstream& samples,bool balanced,bool timed=false){
    int jobs=4*prop.multiProcessorCount+prop.multiProcessorCount/2+3,n=jobs*256;
    int shmem=((prop.sharedMemPerMultiprocessor/2+255)/256)*256+256;
    if(shmem>(int)prop.sharedMemPerBlockOptin)throw std::runtime_error("one-slot reservation unsupported");
    CU(cudaFuncSetAttribute(ragged_a<Trace>,cudaFuncAttributeMaxDynamicSharedMemorySize,shmem));
    CU(cudaFuncSetAttribute(ragged_b<Trace>,cudaFuncAttributeMaxDynamicSharedMemorySize,shmem));
    CU(cudaFuncSetAttribute(ragged_fused<Trace>,cudaFuncAttributeMaxDynamicSharedMemorySize,shmem));
    CU(cudaFuncSetAttribute(ragged_queue<Trace>,cudaFuncAttributeMaxDynamicSharedMemorySize,shmem));
    for(auto f:{(const void*)ragged_a<Trace>,(const void*)ragged_b<Trace>,(const void*)ragged_fused<Trace>,(const void*)ragged_queue<Trace>}){int active;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,f,256,shmem));if(active!=1)throw std::runtime_error("expected one active block per SM");}
    std::string label=balanced?"tiles_balanced":"tiles_ragged";std::string dir=root+"/"+label;std::filesystem::create_directories(dir);
    meta<<"{\"kind\":\"tiles\",\"case\":\""<<label<<"\",\"jobs\":"<<jobs<<",\"lanes\":256,\"shmem_bytes\":"<<shmem<<",\"active_blocks_per_sm\":1,\"trace\":"<<(Trace?"true":"false")<<"}\n";
    auto *x=alloc<uint32_t>(n),*tmp=alloc<uint32_t>(n),*y=alloc<uint32_t>(n),*next=alloc<unsigned>(1);auto* work=alloc<int>(jobs);auto* t=alloc<Stamp>(jobs);
    CU(cudaMemsetAsync(x,0,n*4,s));
    std::vector<int> w(jobs);for(int j=0;j<jobs;++j)w[j]=balanced?1024:(j%8==0?8192:256);save(dir+"/work.i32",w);CU(cudaMemcpyAsync(work,w.data(),jobs*4,cudaMemcpyHostToDevice,s));CU(cudaStreamSynchronize(s));
    std::vector<Variant> v;
    graph_variant(v,label+"_separate",[=](){ragged_a<Trace><<<jobs,256,shmem,s>>>(x,tmp,work,t);ragged_b<Trace><<<jobs,256,shmem,s>>>(tmp,y,t);},s,dir,meta,2);
    graph_variant(v,label+"_fused",[=](){ragged_fused<Trace><<<jobs,256,shmem,s>>>(x,tmp,y,work,t);},s,dir,meta,1);
    graph_variant(v,label+"_queue",[=](){CU(cudaMemsetAsync(next,0,sizeof(unsigned),s));ragged_queue<Trace><<<prop.multiProcessorCount,256,shmem,s>>>(x,tmp,y,work,t,jobs,next);},s,dir,meta,2);
    for(int fixture=0;fixture<2;++fixture){std::vector<uint32_t> h(n);for(int i=0;i<n;++i)h[i]=(uint32_t)i*747796405u+(fixture+1)*2891336453u;save(dir+"/input_"+std::to_string(fixture)+".u32",h);CU(cudaMemcpyAsync(x,h.data(),n*4,cudaMemcpyHostToDevice,s));CU(cudaStreamSynchronize(s));
        for(auto& z:v){CU(cudaMemsetAsync(t,0,jobs*sizeof(Stamp),s));z.launch();CU(cudaStreamSynchronize(s));dump_device(dir+"/"+z.name+"_fixture"+std::to_string(fixture)+".u32",y,n);dump_device(dir+"/"+z.name+"_fixture"+std::to_string(fixture)+"_intermediate.u32",tmp,n);
            if constexpr(Trace){std::vector<Stamp> htrace(jobs);CU(cudaMemcpy(htrace.data(),t,jobs*sizeof(Stamp),cudaMemcpyDeviceToHost));std::ofstream f(dir+"/"+z.name+"_fixture"+std::to_string(fixture)+"_trace.csv");f<<"job,a_sm,b_sm,a_worker,b_worker,a0_ns,a1_ns,b0_ns,b1_ns\n";for(int j=0;j<jobs;++j){auto q=htrace[j];f<<j<<","<<q.sm<<","<<q.sm_b<<","<<q.worker<<","<<q.worker_b<<","<<q.a0<<","<<q.a1<<","<<q.b0<<","<<q.b1<<"\n";}}
        }if constexpr(!Trace){if(fixture==0&&timed)measure(v,s,label,128,samples);}
    }destroy(v);CU(cudaFree(x));CU(cudaFree(tmp));CU(cudaFree(y));CU(cudaFree(next));CU(cudaFree(work));CU(cudaFree(t));
}
int main(int argc,char** argv){try{
    if(argc<2)throw std::runtime_error("usage: bench OUTPUT [trace|profile] [variant-substring]");std::string root=argv[1],mode=argc>2?argv[2]:"bench",select=argc>3?argv[3]:"";std::filesystem::create_directories(root);
    cudaDeviceProp p;CU(cudaGetDeviceProperties(&p,0));if(std::string(p.name).find("H100")==std::string::npos)throw std::runtime_error("exact H100 required");int driver,runtime;CU(cudaDriverGetVersion(&driver));CU(cudaRuntimeGetVersion(&runtime));
    std::ofstream meta(root+"/metadata.jsonl"),samples(root+"/samples.jsonl");meta<<std::setprecision(12);samples<<std::setprecision(12);
    meta<<"{\"kind\":\"device\",\"name\":\""<<p.name<<"\",\"sms\":"<<p.multiProcessorCount<<",\"l2_bytes\":"<<p.l2CacheSize<<",\"global_bytes\":"<<p.totalGlobalMem<<",\"driver\":"<<driver<<",\"runtime\":"<<runtime<<",\"cc_major\":"<<p.major<<",\"cc_minor\":"<<p.minor<<"}\n";
    cudaStream_t s;CU(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    if(mode=="trace"){ragged_case<true>(root,s,p,meta,samples,false);ragged_case<true>(root,s,p,meta,samples,true);}
    else {pointwise(root,s,meta,samples,mode=="profile",mode=="bench",select);if(mode=="bench"||mode=="validate"){ragged_case<false>(root,s,p,meta,samples,false,mode=="bench");ragged_case<false>(root,s,p,meta,samples,true,mode=="bench");}}
    CU(cudaGetLastError());CU(cudaStreamSynchronize(s));CU(cudaStreamDestroy(s));std::cout<<"completed "<<mode<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
