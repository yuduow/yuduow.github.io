#include "kernels.cuh"
#include <cuda_profiler_api.h>
#include <cudnn.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using json=nlohmann::json;namespace fs=std::filesystem;using Clock=std::chrono::steady_clock;
#define CU(x) do{auto status_=(x);if(status_!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(status_));}while(0)
#define DN(x) do{auto status_=(x);if(status_!=CUDNN_STATUS_SUCCESS)throw std::runtime_error(std::string(#x)+": "+cudnnGetErrorString(status_));}while(0)
double micros(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
template<class T>T* alloc(size_t n){T* p;CU(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>void save(fs::path p,const std::vector<T>& a){fs::create_directories(p.parent_path());std::ofstream f(p,std::ios::binary);f.write((const char*)a.data(),a.size()*sizeof(T));if(!f)throw std::runtime_error("write "+p.string());}
template<class T>std::vector<T> download(const T* p,size_t n){std::vector<T>a(n);CU(cudaMemcpy(a.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return a;}
void write_json(fs::path p,const json& j){std::ofstream f(p);f<<std::setw(2)<<j<<"\n";if(!f)throw std::runtime_error("write json "+p.string());}
json read_json(fs::path p){std::ifstream f(p);json j;f>>j;return j;}
struct AdmissionError:std::runtime_error{using runtime_error::runtime_error;};
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
struct TileReference {std::vector<double> tm,ts,energy;};
struct Context {
    int rows,n,sm;size_t q;fs::path root,out,input_path;cudaStream_t stream;Payload p{};
    cudnnHandle_t handle;cudnnTensorDescriptor_t desc;
    std::vector<float> x;std::vector<double> reference,rm,z,scale;std::map<int,TileReference> tile_refs;
    std::ofstream& metadata;std::ofstream& validations;int fixture_counter=0;
    Context(int rows_,int n_,fs::path root_,fs::path out_,std::ofstream& m,std::ofstream& v):rows(rows_),n(n_),q(size_t(rows)*n),root(root_),out(out_),metadata(m),validations(v){
        fs::create_directories(out);CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cudaDeviceProp d;CU(cudaGetDeviceProperties(&d,0));sm=d.multiProcessorCount;
        if(std::string(d.name).find("H100")==std::string::npos||!d.cooperativeLaunch)throw std::runtime_error("Exact H100 with cooperative launch required");
        p.arrivals=alloc<unsigned>(rows*2);p.epochs=alloc<unsigned>(rows*2);CU(cudaMemsetAsync(p.arrivals,0,rows*2*sizeof(unsigned),stream));CU(cudaMemsetAsync(p.epochs,0,rows*2*sizeof(unsigned),stream));p.x=alloc<float>(q);p.y=alloc<float>(q);p.e=alloc<float>(q);
        size_t max_tiles=size_t(rows)*((n+4095)/4096);p.tm=alloc<float>(max_tiles);p.ts=alloc<float>(max_tiles);p.adjusted=alloc<float>(max_tiles);
        p.energy=alloc<float>(max_tiles);p.scale=alloc<float>(rows);p.squared=alloc<float>(q);p.rm=alloc<float>(rows);p.z=alloc<float>(rows);p.stamps=alloc<Stamp>(max_tiles*3);p.rows=rows;p.n=n;
        DN(cudnnCreate(&handle));DN(cudnnSetStream(handle,stream));DN(cudnnCreateTensorDescriptor(&desc));
        DN(cudnnSetTensor4dDescriptor(desc,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,rows,n,1,1));
    }
    ~Context(){cudaStreamSynchronize(stream);for(auto a:{(void*)p.x,(void*)p.y,(void*)p.e,(void*)p.tm,(void*)p.ts,(void*)p.adjusted,(void*)p.rm,(void*)p.z,(void*)p.stamps,(void*)p.energy,(void*)p.scale,(void*)p.squared,(void*)p.arrivals,(void*)p.epochs})cudaFree(a);cudnnDestroyTensorDescriptor(desc);cudnnDestroy(handle);cudaStreamDestroy(stream);}
    void input(std::string kind,int seed,bool persist=true){
        x.resize(q);reference.resize(q);rm.assign(rows,-INFINITY);z.assign(rows,0);scale.assign(rows,0);tile_refs.clear();
        for(int r=0;r<rows;++r)for(int j=0;j<n;++j){size_t i=size_t(r)*n+j;uint32_t h=uint32_t(i)+uint32_t(seed+1)*0x9e3779b9u;h=(h^(h>>16))*0x7feb352du;h=(h^(h>>15))*0x846ca68bu;h^=h>>16;
            float a=(int(h%16385)-8192)/1024.f;
            if(kind=="constant")a=3;
            else if(kind=="dominant")a=j==(r*8191)%n?1024.f:0.f;
            else if(kind=="alternating")a=j%2?12.f:-12.f;
            else if(kind=="tile_offsets")a+=((j/4096)%7)*8-24;
            else if(kind=="common_offset")a+=1000;
            else if(kind=="zero")a=0;
            else if(kind=="near_zero")a*=1e-4f;
            else if(kind!="random")throw std::runtime_error("fixture kind");
            x[i]=a;rm[r]=std::max(rm[r],double(a));}
        for(int r=0;r<rows;++r){double energy=0;for(int j=0;j<n;++j){double v=x[size_t(r)*n+j];energy+=v*v;}scale[r]=1/std::sqrt(energy/n+1e-5);for(int j=0;j<n;++j){size_t i=size_t(r)*n+j;reference[i]=std::exp((double(x[i])-rm[r])*scale[r]);z[r]+=reference[i];}
            for(int j=0;j<n;++j)reference[size_t(r)*n+j]/=z[r];}
        CU(cudaMemcpyAsync((void*)p.x,x.data(),q*4,cudaMemcpyHostToDevice,stream));CU(cudaStreamSynchronize(stream));
        input_path=out/("input-"+kind+"-s"+std::to_string(seed)+".f32");if(persist)save(input_path,x);
    }
    TileReference& tiles(int w){
        if(!tile_refs.count(w)){int k=(n+w-1)/w;TileReference ref;ref.tm.assign(rows*k,-INFINITY);ref.ts.assign(rows*k,0);ref.energy.assign(rows*k,0);
            for(int r=0;r<rows;++r)for(int t=0;t<k;++t){int id=r*k+t;for(int j=t*w;j<std::min(n,(t+1)*w);++j)ref.tm[id]=std::max(ref.tm[id],double(x[size_t(r)*n+j]));
                for(int j=t*w;j<std::min(n,(t+1)*w);++j){double v=x[size_t(r)*n+j];ref.energy[id]+=v*v;ref.ts[id]+=std::exp((v-rm[r])*scale[r]);}}
            tile_refs.emplace(w,std::move(ref));}
        return tile_refs.at(w);
    }
    void poison(){for(auto pair:std::vector<std::pair<float*,size_t>>{{p.y,q},{p.e,q},{p.squared,q},{p.energy,size_t(rows)*((n+4095)/4096)},{p.scale,size_t(rows)},{p.tm,size_t(rows)*((n+4095)/4096)},{p.ts,size_t(rows)*((n+4095)/4096)},{p.adjusted,size_t(rows)*((n+4095)/4096)},{p.rm,size_t(rows)},{p.z,size_t(rows)}})CU(cudaMemsetAsync(pair.first,0xff,pair.second*4,stream));CU(cudaStreamSynchronize(stream));}
};

json resource(const void* f,int t,int dynamic,bool full=false){
    if(dynamic>49152)CU(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,dynamic));
    if(full)CU(cudaFuncSetAttribute(f,cudaFuncAttributePreferredSharedMemoryCarveout,100));
    cudaFuncAttributes a;CU(cudaFuncGetAttributes(&a,f));int active;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,f,t,dynamic));
    return {{"registers",a.numRegs},{"local_bytes",a.localSizeBytes},{"static_shared_bytes",a.sharedSizeBytes},{"dynamic_shared_bytes",dynamic},{"active_blocks_per_sm",active}};
}
template<int W,int T>Spec make_typed(Context& c,std::string family,std::string label){
    Spec s;s.family=family;s.label=label;s.w=W;s.t=T;s.stream=c.stream;
    Payload p=c.p;p.tiles=(c.n+W-1)/W;int g=c.rows*p.tiles;auto stream=c.stream;
    constexpr int shared=W*4+256;
    if(family=="baseline"){
        s.resources={{"square",resource((void*)baseline_square<W,T>,T,0)},{"energy",resource((void*)baseline_energy<W,T>,T,256)},{"max",resource((void*)baseline_max<W,T>,T,256)},{"exp",resource((void*)baseline_exp<W,T>,T,0)},{"sum",resource((void*)baseline_sum<W,T>,T,256)},{"normalize",resource((void*)baseline_normalize<W,T>,T,0)}};
        s.run=[=](){baseline_square<W,T><<<g,T,0,stream>>>(p);baseline_energy<W,T><<<g,T,256,stream>>>(p);baseline_max<W,T><<<g,T,256,stream>>>(p);
            baseline_scale<<<p.rows,T,0,stream>>>(p);baseline_row_max<<<p.rows,T,0,stream>>>(p);baseline_exp<W,T><<<g,T,0,stream>>>(p);baseline_sum<W,T><<<g,T,256,stream>>>(p);
            baseline_denominator<<<p.rows,T,0,stream>>>(p);baseline_normalize<W,T><<<g,T,0,stream>>>(p);};s.launches=9;
    }else if(family=="local_materialize2"||family=="local_recompute2"){
        bool rec=family=="local_recompute2";const void* a=rec?(void*)local_ab<W,T,false>:(void*)local_ab<W,T,true>;
        const void* cfun=rec?(void*)local_c<W,T,1>:(void*)local_c<W,T,0>;
        s.resources={{"AB",resource(a,T,256)},{"C",resource(cfun,T,0)}};
        s.run=[=](){if(rec){local_ab<W,T,false><<<g,T,256,stream>>>(p);local_c<W,T,1><<<g,T,0,stream>>>(p);}
            else{local_ab<W,T,true><<<g,T,256,stream>>>(p);local_c<W,T,0><<<g,T,0,stream>>>(p);}};s.launches=2;
    }else if(family=="local_materialize"||family=="local_recompute"){
        bool rec=family=="local_recompute";const void* bfun=rec?(void*)local_b<W,T,false>:(void*)local_b<W,T,true>;
        const void* cfun=rec?(void*)local_c<W,T,1>:(void*)local_c<W,T,0>;
        s.resources={{"A",resource((void*)local_a<W,T>,T,256)},{"B",resource(bfun,T,256)},{"C",resource(cfun,T,0)}};
        s.run=[=](){local_a<W,T><<<g,T,256,stream>>>(p);if(rec){local_b<W,T,false><<<g,T,256,stream>>>(p);local_c<W,T,1><<<g,T,0,stream>>>(p);}
            else{local_b<W,T,true><<<g,T,256,stream>>>(p);local_c<W,T,0><<<g,T,0,stream>>>(p);}};s.launches=3;
    }else if(family=="retained"||family=="row_retained"||family=="roundtrip_off"||family=="roundtrip_on"||family=="input_reload"||family=="both_reload"){
        cudaDeviceProp device;CU(cudaGetDeviceProperties(&device,0));if(shared>device.sharedMemPerBlockOptin)throw AdmissionError("tile exceeds opt-in shared-memory capacity");
        bool row=family=="row_retained",control=family!="retained"&&!row;const void* f=row?(void*)row_retained<W,T>:control?(void*)retained<W,T,true>:(void*)retained<W,T,false>;
        auto r=resource(f,T,shared,true);int active=r["active_blocks_per_sm"],per_slab=active*c.sm/p.tiles;
        if(per_slab<1)throw AdmissionError("one row exceeds compiled cooperative residency");
        s.resources={{"cooperative",r}};s.slab_rows=std::min(per_slab,c.rows);s.launches=(c.rows+s.slab_rows-1)/s.slab_rows;
        int slab=s.slab_rows,flag=family=="input_reload"?1:family=="roundtrip_on"?2:family=="both_reload"?3:0;
        s.run=[=](){for(int row=0;row<p.rows;row+=slab){int count=std::min(slab,p.rows-row);Payload pp=p;int base=row,force=flag;
            void* args[]={&pp,&base,&count,&force};CU(cudaLaunchCooperativeKernel(f,dim3(count*p.tiles),dim3(T),args,shared,stream));}};
    }else if(family=="streaming"){
        s.resources={{"redundant_full_row",resource((void*)streaming<W,T>,T,256)}};s.run=[=](){streaming<W,T><<<g,T,256,stream>>>(p);};s.launches=1;
    }else if(family=="cudnn"){
        s.resources={{"rms",resource((void*)library_rms<T>,T,256)},{"scale",resource((void*)library_scale<W,T>,T,0)}};
        auto h=c.handle;auto d=c.desc;s.run=[=](){library_rms<T><<<p.rows,T,256,stream>>>(p);library_scale<W,T><<<g,T,0,stream>>>(p);float alpha=1,beta=0;DN(cudnnSoftmaxForward(h,CUDNN_SOFTMAX_ACCURATE,CUDNN_SOFTMAX_MODE_INSTANCE,&alpha,d,p.e,&beta,d,p.y));};s.launches=-1;
    }else throw std::runtime_error("unknown family "+family);
    return s;
}
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

void validate(Context& c,const Spec& s,std::string label,bool dump_e=false,bool replay=true){
    c.poison();s.launch(replay);CU(cudaStreamSynchronize(c.stream));auto y=download(c.p.y,c.q);double max_error=0,max_l1=0,max_sum=0;
    for(int r=0;r<c.rows;++r){double l1=0,sum=0;for(int j=0;j<c.n;++j){size_t i=size_t(r)*c.n+j;double a=y[i],b=c.reference[i],e=std::abs(a-b);
        if(!std::isfinite(a)||a<0||e>1e-7+2e-5*std::abs(b))throw std::runtime_error("output error "+label+" index "+std::to_string(i));max_error=std::max(max_error,e);l1+=e;sum+=a;}
        max_l1=std::max(max_l1,l1);max_sum=std::max(max_sum,std::abs(sum-1));if(l1>2e-5||std::abs(sum-1)>2e-5)throw std::runtime_error("row error "+label);}
    auto arrivals=download(c.p.arrivals,c.rows*2);if(std::any_of(arrivals.begin(),arrivals.end(),[](unsigned a){return a!=0;}))throw std::runtime_error("row barrier left incomplete");auto x_back=download(c.p.x,c.q);if(std::memcmp(x_back.data(),c.x.data(),c.q*4))throw std::runtime_error("input mutated");
    fs::path base=c.out/("check-"+label);save(base.string()+"-y.f32",y);
    json arrays;int k=(c.n+s.w-1)/s.w;
    if(s.family!="cudnn"&&s.family!="streaming"){
        auto& ref=c.tiles(s.w);
        std::vector<std::tuple<std::string,const float*,size_t,const std::vector<double>*>> checks={{"ts",c.p.ts,size_t(c.rows)*k,&ref.ts},{"rm",c.p.rm,size_t(c.rows),&c.rm},{"scale",c.p.scale,size_t(c.rows),&c.scale},{"z",c.p.z,size_t(c.rows),&c.z}};
        if(s.family!="local_materialize2"&&s.family!="local_recompute2"){checks.push_back({"tm",c.p.tm,size_t(c.rows)*k,&ref.tm});checks.push_back({"energy",c.p.energy,size_t(c.rows)*k,&ref.energy});}
        for(auto [name,p,n,target]:checks){
            auto a=download(p,n);for(size_t i=0;i<n;++i)if(!std::isfinite(a[i])||std::abs(double(a[i])-(*target)[i])>(name=="z"||name=="energy"?0:1e-7)+2e-5*std::abs((*target)[i]))throw std::runtime_error("intermediate error "+label+" "+name);
            fs::path file=base.string()+"-"+name+".f32";save(file,a);arrays[name]=fs::relative(file,c.root).string();}
        if(dump_e&&(s.family=="baseline"||s.family=="local_materialize"||s.family=="local_materialize2"||s.family=="roundtrip_on"||s.family=="both_reload")){
            auto a=download(c.p.e,c.q);for(int r=0;r<c.rows;++r)for(int j=0;j<c.n;++j){size_t i=size_t(r)*c.n+j;double b=std::exp((double(c.x[i])-c.rm[r])*c.scale[r]);if(!std::isfinite(a[i])||std::abs(double(a[i])-b)>1e-7+2e-5*std::abs(b))throw std::runtime_error("E intermediate error");}
            fs::path file=base.string()+"-e.f32";save(file,a);arrays["e"]=fs::relative(file,c.root).string();}
        if(dump_e&&s.family=="baseline"){auto a=download(c.p.squared,c.q);for(size_t i=0;i<c.q;++i){double v=c.x[i];if(!std::isfinite(a[i])||std::abs(double(a[i])-v*v)>1e-7+2e-5*v*v)throw std::runtime_error("squared intermediate error");}fs::path file=base.string()+"-squared.f32";save(file,a);arrays["squared"]=fs::relative(file,c.root).string();}
    }

    c.validations<<json{{"label",label},{"captured_execution",replay},{"family",s.family},{"rows",c.rows},{"columns",c.n},{"W",s.w},{"T",s.t},{"input",fs::relative(c.input_path,c.root).string()},{"output",fs::relative(base.string()+"-y.f32",c.root).string()},{"intermediates",arrays},{"max_absolute_error",max_error},{"max_row_l1",max_l1},{"max_row_sum_error",max_sum},{"input_unchanged",true},{"passed",true}}.dump()<<"\n";c.validations.flush();
}
std::vector<std::string> families={"baseline","local_materialize","local_recompute","local_materialize2","local_recompute2","retained","row_retained","roundtrip_off","roundtrip_on","input_reload","both_reload","streaming","cudnn"};
std::vector<Spec> selected(Context& c,const json& selection){
    std::vector<Spec> specs;
    for(auto family:families){std::string geometry=family;if(family=="roundtrip_off"||family=="roundtrip_on"||(family=="input_reload"||family=="both_reload"))geometry="retained";
        if(family=="cudnn")geometry="baseline";auto g=selection.at(geometry);specs.push_back(make(c,family,g["W"],g["T"],family));}
    return specs;
}
void measure(Context& c,const std::vector<Spec>& specs,std::string best,int allocation){
    struct Item{int i;bool graph;std::string label;};std::vector<Item> items;
    for(int i=0;i<int(specs.size());++i){items.push_back({i,true,specs[i].family+"_graph"});if(specs[i].family=="baseline"||specs[i].family==best||(specs[i].family=="retained"||specs[i].family=="row_retained"))items.push_back({i,false,specs[i].family+"_stream"});}
    std::ofstream f(c.out/"timings.jsonl");cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));std::mt19937 rng(1000+allocation);std::vector<int>order(items.size());std::iota(order.begin(),order.end(),0);
    for(auto item:items){for(int j=0;j<10;++j)specs[item.i].launch(item.graph);CU(cudaStreamSynchronize(c.stream));}
    for(int round=0;round<21;++round){std::shuffle(order.begin(),order.end(),rng);auto idle0=Clock::now();CU(cudaStreamSynchronize(c.stream));auto idle1=Clock::now();f<<json{{"variant","empty_sync"},{"round",round},{"host_single_us",micros(idle0,idle1)}}.dump()<<"\n";
        for(int pos=0;pos<int(order.size());++pos){auto item=items[order[pos]];auto& s=specs[item.i];for(int j=0;j<10;++j)s.launch(item.graph);CU(cudaStreamSynchronize(c.stream));
            auto t0=Clock::now();s.launch(item.graph);CU(cudaStreamSynchronize(c.stream));auto t1=Clock::now();CU(cudaEventRecord(a,c.stream));auto t2=Clock::now();for(int j=0;j<32;++j)s.launch(item.graph);auto t3=Clock::now();CU(cudaEventRecord(b,c.stream));CU(cudaEventSynchronize(b));auto t4=Clock::now();float ms;CU(cudaEventElapsedTime(&ms,a,b));
            f<<json{{"variant",item.label},{"round",round},{"order",pos},{"batch",32},{"host_single_us",micros(t0,t1)},{"host_enqueue_us",micros(t2,t3)/32},{"host_batch_us",micros(t2,t4)/32},{"event_interval_us",ms*1000/32},{"kernel_nodes",s.launches}}.dump()<<"\n";}}
    if(c.rows==16||c.rows==64){size_t bytes=std::max(size_t(256)*1024*1024,size_t(4)*([&](){cudaDeviceProp d;CU(cudaGetDeviceProperties(&d,0));return d.l2CacheSize;})());float* buffer=alloc<float>(bytes/4);CU(cudaMemsetAsync(buffer,0,bytes,c.stream));CU(cudaStreamSynchronize(c.stream));
        for(int round=0;round<15;++round){std::shuffle(order.begin(),order.end(),rng);for(int i:order){auto item=items[i];evict_cache<<<1024,256,0,c.stream>>>(buffer,bytes/4);CU(cudaStreamSynchronize(c.stream));auto t0=Clock::now();specs[item.i].launch(item.graph);CU(cudaStreamSynchronize(c.stream));auto t1=Clock::now();f<<json{{"variant",item.label},{"round",round},{"cache_policy","external_eviction"},{"eviction_bytes",bytes},{"host_single_us",micros(t0,t1)}}.dump()<<"\n";}}CU(cudaFree(buffer));}
    CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));
}
void stress(Context& c,const std::vector<Spec>& specs){
    std::vector<float*> inputs;std::vector<double*> refs;for(int i=0;i<8;++i){c.input("random",700+i);float* x=alloc<float>(c.q);double* r=alloc<double>(c.q);CU(cudaMemcpyAsync(x,c.x.data(),c.q*4,cudaMemcpyHostToDevice,c.stream));CU(cudaMemcpyAsync(r,c.reference.data(),c.q*8,cudaMemcpyHostToDevice,c.stream));CU(cudaStreamSynchronize(c.stream));inputs.push_back(x);refs.push_back(r);}
    int* failures=alloc<int>(1);json results=json::array();
    CU(cudaMemsetAsync(failures,0,4,c.stream));CU(cudaMemsetAsync(c.p.y,0,c.q*4,c.stream));
    check_replay<<<c.rows,256,0,c.stream>>>(c.p.y,refs[7],c.rows,c.n,failures);CU(cudaStreamSynchronize(c.stream));
    int negative=download(failures,1)[0];if(negative<c.rows)throw std::runtime_error("replay validator failed its negative control");
    write_json(c.out/"checker-selftest.json",{{"intentionally_zeroed_output_detected",true},{"failure_conditions",negative}});
    for(auto& s:specs)if(s.family!="cudnn"){
        CU(cudaMemsetAsync(failures,0,4,c.stream));for(int i=0;i<1000;++i){CU(cudaMemcpyAsync((void*)c.p.x,inputs[i%8],c.q*4,cudaMemcpyDeviceToDevice,c.stream));s.launch();check_replay<<<c.rows,256,0,c.stream>>>(c.p.y,refs[i%8],c.rows,c.n,failures);}CU(cudaStreamSynchronize(c.stream));int count=download(failures,1)[0];if(count)throw std::runtime_error("changing-input replay failure "+s.family);
        c.input("random",707);validate(c,s,"stress-final-"+s.family);results.push_back({{"family",s.family},{"changing_input_replays",1000},{"failures",count}});}
    for(auto p:inputs)CU(cudaFree(p));for(auto p:refs)CU(cudaFree(p));CU(cudaFree(failures));write_json(c.out/"stress.json",results);
}
void save_trace(Context& c,const Spec& s,int rep){
    int k=(c.n+s.w-1)/s.w;size_t count=size_t(c.rows)*k*3;CU(cudaMemsetAsync(c.p.stamps,0,count*sizeof(Stamp),c.stream));s.launch();CU(cudaStreamSynchronize(c.stream));auto stamps=download(c.p.stamps,count);
    std::ofstream f(c.out/("trace-"+s.family+"-"+std::to_string(rep)+".csv"));f<<"phase,row,tile,start_ns,end_ns,sm,block,visits,slab\n";
    bool two=s.family=="local_materialize2"||s.family=="local_recompute2";
    for(size_t id=0;id<count;++id){int phase=id/(c.rows*k);if(two&&phase==1)continue;int local=id%(c.rows*k),row=local/k,tile=local%k;auto a=stamps[id];f<<char('A'+phase)<<","<<row<<","<<tile<<","<<a.start<<","<<a.end<<","<<a.sm<<","<<a.block<<","<<a.visits<<","<<a.slab<<"\n";}

}
int main(int argc,char** argv){
    fs::path out=argc>1?argv[1]:"/tmp/softmax";fs::create_directories(out);std::string mode=argc>2?argv[2]:"pilot";
    try{
        std::ofstream meta(out/"metadata.jsonl"),validation(out/"validation.jsonl");cudaDeviceProp d;CU(cudaGetDeviceProperties(&d,0));int runtime,driver,reserved;CU(cudaRuntimeGetVersion(&runtime));CU(cudaDriverGetVersion(&driver));CU(cudaDeviceGetAttribute(&reserved,cudaDevAttrReservedSharedMemoryPerBlock,0));
        write_json(out/"device.json",{{"name",d.name},{"sm_count",d.multiProcessorCount},{"uuid",std::vector<unsigned char>(d.uuid.bytes,d.uuid.bytes+16)},{"pci_bus",d.pciBusID},{"shared_bytes_per_sm",d.sharedMemPerMultiprocessor},{"shared_bytes_per_block_optin",d.sharedMemPerBlockOptin},{"reserved_shared_bytes_per_block",reserved},{"registers_per_sm",d.regsPerMultiprocessor},{"l2_bytes",d.l2CacheSize},{"runtime",runtime},{"driver",driver},{"cooperative_launch",d.cooperativeLaunch},{"cudnn_version",cudnnGetVersion()},{"trace_build",bool(STUDY_TRACE)}});
        json config;if(argc>3)config=read_json(argv[3]);
        if(mode=="pilot"){
            json selections;
            for(int rows:{1,16,32,64}){Context c(rows,131072,out,out/("b"+std::to_string(rows)),meta,validation);c.input("random",11);json chosen;std::ofstream tune(c.out/"tuning.jsonl");
                for(auto family:{"baseline","local_materialize","local_recompute","local_materialize2","local_recompute2","retained","row_retained","streaming"}){double best=INFINITY;json winner;
                    for(int w:{4096,8192,16384,131072})for(int t:{64,128,256}){
                        std::string label=std::string(family)+"-w"+std::to_string(w)+"-t"+std::to_string(t);try{auto s=make(c,family,w,t,label);validate(c,s,"pilot-"+label);for(int j=0;j<10;++j)s.launch();CU(cudaStreamSynchronize(c.stream));std::vector<double> samples;
                            for(int round=0;round<5;++round){auto a=Clock::now();for(int j=0;j<16;++j)s.launch();CU(cudaStreamSynchronize(c.stream));samples.push_back(micros(a,Clock::now())/16);}
                            auto sorted=samples;std::sort(sorted.begin(),sorted.end());double score=sorted[2];tune<<json{{"family",family},{"W",w},{"T",t},{"samples_us",samples},{"median_us",score},{"kernel_nodes",s.launches},{"valid",true}}.dump()<<"\n";
                            if(score<best){best=score;winner={{"W",w},{"T",t},{"pilot_median_us",score},{"kernel_nodes",s.launches}};}
                        }catch(const AdmissionError& e){tune<<json{{"family",family},{"W",w},{"T",t},{"valid",false},{"reason",e.what()}}.dump()<<"\n";}}
                    if(winner.is_null())throw std::runtime_error("no legal candidate");chosen[family]=winner;}
                std::string best_local="local_materialize";
                for(auto name:{"local_recompute","local_materialize2","local_recompute2","streaming"})if(chosen[name]["pilot_median_us"].get<double>()<chosen[best_local]["pilot_median_us"].get<double>())best_local=name;
                chosen["local_best"]=best_local;chosen["mega_best"]=chosen["row_retained"]["pilot_median_us"].get<double>()<chosen["retained"]["pilot_median_us"].get<double>()?"row_retained":"retained";
                selections[std::to_string(rows)]=chosen;std::cout<<"selected batch "<<rows<<": "<<chosen.dump()<<std::endl;
            }
            write_json(out/"selected-configs.json",selections);
            for(auto [rows,n]:std::vector<std::pair<int,int>>{{16,131072},{1,1},{3,257},{2,8193},{2,131075}}){Context c(rows,n,out,out/("structured-b"+std::to_string(rows)+"-n"+std::to_string(n)),meta,validation);c.input("random",23);auto specs=selected(c,selections["16"]);
                for(auto kind:{"random","constant","dominant","alternating","tile_offsets","common_offset","zero","near_zero"}){c.input(kind,23);for(auto& s:specs){validate(c,s,std::string(kind)+"-"+s.family,true);if(rows==16&&n==131072)validate(c,s,std::string(kind)+"-direct-"+s.family,false,false);}}
                if(rows==16&&n==131072)stress(c,specs);}
        }else if(mode=="measurement"){
            int allocation=argc>4?std::stoi(argv[4]):1;std::vector<int> batches={1,16,32,64};if(allocation==2)std::reverse(batches.begin(),batches.end());
            for(int rows:batches){Context c(rows,131072,out,out/("b"+std::to_string(rows)),meta,validation);c.input("random",41);auto selection=config.at(std::to_string(rows));auto specs=selected(c,selection);for(auto& s:specs)validate(c,s,"preflight-"+s.family,true);
                measure(c,specs,selection["local_best"],allocation);c.input("random",53);for(auto& s:specs)validate(c,s,"heldout-"+s.family,true);}
        }else if(mode=="trace"){
            for(int rows:{16,64}){Context c(rows,131072,out,out/("b"+std::to_string(rows)),meta,validation);c.input("random",41);auto specs=selected(c,config.at(std::to_string(rows)));
                for(auto& s:specs)if(s.family!="cudnn"&&s.family!="streaming"){validate(c,s,"trace-"+s.family);for(int rep=0;rep<3;++rep)save_trace(c,s,rep);}}
        }else if(mode=="profile"){
            int rows=std::stoi(argv[4]);std::string family=argv[5];Context c(rows,131072,out,out/("b"+std::to_string(rows)),meta,validation);c.input("random",41);std::string geometry=family;if(family=="roundtrip_on"||family=="roundtrip_off"||(family=="input_reload"||family=="both_reload"))geometry="retained";if(family=="cudnn")geometry="baseline";auto g=config.at(std::to_string(rows)).at(geometry);auto s=make(c,family,g["W"],g["T"],family);validate(c,s,"profile-"+family);for(int j=0;j<10;++j)s.launch();CU(cudaStreamSynchronize(c.stream));CU(cudaProfilerStart());s.launch();CU(cudaStreamSynchronize(c.stream));CU(cudaProfilerStop());
        }else if(mode=="sanitizer"){
            int rows=std::stoi(argv[4]),n=std::stoi(argv[5]);Context c(rows,n,out,out/"case",meta,validation);c.input("tile_offsets",23);
            for(auto family:{"baseline","retained","row_retained","roundtrip_off","roundtrip_on","input_reload","both_reload","local_materialize","local_recompute","local_materialize2","local_recompute2","streaming","cudnn"}){std::string geometry=family;if(geometry=="roundtrip_on"||geometry=="roundtrip_off"||geometry=="input_reload"||geometry=="both_reload")geometry="retained";if(geometry=="cudnn")geometry="baseline";auto g=config.at("16").at(geometry);auto s=make(c,family,g["W"],g["T"],family);validate(c,s,std::string("sanitizer-")+family,true);}
        }else throw std::runtime_error("mode");
        write_json(out/"status.json",{{"status","complete"},{"mode",mode}});return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;write_json(out/"status.json",{{"status","failed"},{"mode",mode},{"error",e.what()}});return 1;}
}
