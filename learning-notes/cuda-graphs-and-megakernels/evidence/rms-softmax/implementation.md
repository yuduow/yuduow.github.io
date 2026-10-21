# RMS-Scaled Softmax: Implementation

[Back to the article](../../). These excerpts are copied unchanged from the saved benchmark source. They depend on the surrounding helpers and are not standalone programs.

## CUDA Graph capture and replay

`Spec::launch` chooses graph replay or the original launch sequence. `make` captures that sequence, instantiates it, and checks the resulting kernel nodes.

From [bench.cu](code/bench.cu), lines 32–41.

```cpp
struct Graph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}
};
struct Spec {
    std::string family,label;int w=0,t=0,launches=0,slab_rows=0;json resources;
    std::function<void()> run;std::shared_ptr<Graph> graph;
    void launch(bool replay=true)const{if(replay)CU(cudaGraphLaunch(graph->exec,stream));else run();}
    cudaStream_t stream=nullptr;
};
```

From [bench.cu](code/bench.cu), lines 131–148.

```cpp
Spec make(Context& c,const std::string& family,int w,int t,const std::string& label){
    Spec s;
#define CASE(W,T) if(w==W&&t==T)s=make_typed<W,T>(c,family,label);else
    CASE(4096,64) CASE(4096,128) CASE(4096,256) CASE(8192,64) CASE(8192,128) CASE(8192,256) CASE(16384,64) CASE(16384,128) CASE(16384,256) CASE(131072,64) CASE(131072,128) CASE(131072,256)
    throw std::runtime_error("geometry outside protocol");
#undef CASE
    s.run();CU(cudaGetLastError());CU(cudaStreamSynchronize(c.stream));s.graph=std::make_shared<Graph>();auto a=Clock::now();
    CU(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeThreadLocal));s.run();CU(cudaStreamEndCapture(c.stream,&s.graph->graph));auto b=Clock::now();
    CU(cudaGraphInstantiate(&s.graph->exec,s.graph->graph,0));auto d=Clock::now();s.launch();CU(cudaStreamSynchronize(c.stream));auto e=Clock::now();
    size_t count=0;CU(cudaGraphGetNodes(s.graph->graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);CU(cudaGraphGetNodes(s.graph->graph,nodes.data(),&count));
    int kernels=0;json cooperative=json::array(),node_types=json::array();
    for(auto node:nodes){cudaGraphNodeType type;CU(cudaGraphNodeGetType(node,&type));node_types.push_back(int(type));if(type==cudaGraphNodeTypeKernel){++kernels;cudaKernelNodeAttrValue value{};CU(cudaGraphKernelNodeGetAttribute(node,cudaKernelNodeAttributeCooperative,&value));cooperative.push_back(value.cooperative);}}
    if(s.launches>=0&&(kernels!=s.launches||int(count)!=s.launches))throw std::runtime_error("graph node mismatch "+label);
    if(s.slab_rows>0)for(auto value:cooperative)if(value.get<int>()!=1)throw std::runtime_error("cooperative graph attribute lost");
    s.launches=kernels;fs::create_directories(c.out/"graphs");CU(cudaGraphDebugDotPrint(s.graph->graph,(c.out/"graphs"/(label+".dot")).c_str(),cudaGraphDebugDotFlagsVerbose));
    c.metadata<<json{{"kind","variant"},{"rows",c.rows},{"columns",c.n},{"family",family},{"label",label},{"W",w},{"T",t},{"slab_rows",s.slab_rows},{"kernel_nodes",kernels},{"nodes",count},{"node_types",node_types},{"cooperative",cooperative},{"resources",s.resources},{"capture_us",micros(a,b)},{"instantiate_us",micros(b,d)},{"first_replay_sync_us",micros(d,e)}}.dump()<<"\n";
    return s;
}
```

## Megakernel and row barrier

`row_retained` keeps one shared-memory tile across the three phases. Its two calls to `row_barrier` wait for the other blocks processing the same row.

From [kernels.cuh](code/kernels.cuh), lines 102–125.

```cpp
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
```

## Tile computation

These are the phase helpers called by the megakernel. Small row statistics pass through global memory; the tile values remain in the shared-memory buffer.

From [kernels.cuh](code/kernels.cuh), lines 39–69.

```cpp
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
```

## Cooperative launch and row groups

This excerpt from the variant dispatcher computes how many rows fit the cooperative launch limit, then launches one row group at a time.

From [bench.cu](code/bench.cu), lines 114–122.

```cpp
    }else if(family=="retained"||family=="row_retained"||family=="roundtrip_off"||family=="roundtrip_on"||family=="input_reload"||family=="both_reload"){
        cudaDeviceProp device;CU(cudaGetDeviceProperties(&device,0));if(shared>device.sharedMemPerBlockOptin)throw AdmissionError("tile exceeds opt-in shared-memory capacity");
        bool row=family=="row_retained",control=family!="retained"&&!row;const void* f=row?(void*)row_retained<W,T>:control?(void*)retained<W,T,true>:(void*)retained<W,T,false>;
        auto r=resource(f,T,shared,true);int active=r["active_blocks_per_sm"],per_slab=active*c.sm/p.tiles;
        if(per_slab<1)throw AdmissionError("one row exceeds compiled cooperative residency");
        s.resources={{"cooperative",r}};s.slab_rows=std::min(per_slab,c.rows);s.launches=(c.rows+s.slab_rows-1)/s.slab_rows;
        int slab=s.slab_rows,flag=family=="input_reload"?1:family=="roundtrip_on"?2:family=="both_reload"?3:0;
        s.run=[=](){for(int row=0;row<p.rows;row+=slab){int count=std::min(slab,p.rows-row);Payload pp=p;int base=row,force=flag;
            void* args[]={&pp,&base,&count,&force};CU(cudaLaunchCooperativeKernel(f,dim3(count*p.tiles),dim3(T),args,shared,stream));}};
```

