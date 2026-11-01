# Tensor programs → H100 kernels

## October 21: Luminal FP16 Repair

The [repair and H100 evidence report](luminal-fp16-results.md) documents a corrected numerical adapter and a 31-line backend patch, a ten-workload screen, default/expanded search, paired upstream comparisons, and an exact-plan replay through timing and both profilers. Nine workloads pass; the two-projection join remains numerically rejected in both binaries. Existing October 18 captures below remain unmodified upstream evidence.


This experiment asks which small program best supports the post: **Part A follows tensor semantics to kernel boundaries; Part B follows storage, scheduling, implementation selection, code generation, and execution.** The comparison concerns observable decisions, rather than assigning each compiler an exclusive philosophy.

**Use `y = x @ wt.T + b` as the main walkthrough.** It adds one layout question to an affine layer, yet produces different, inspectable implementations. The native FP16 paths in tinygrad, MAX, Inductor, and TVM all passed validation and have Tensor Core instruction evidence. Luminal supplies an informative, separately labeled widened-operand control; its pinned native FP16 backend failed before candidate measurement. There is no demonstrated all-five FP16 Tensor Core example.

The October 18 campaign is complete: ten-program screens, deep captures, repeated search comparisons, evidence verification, and compute teardown. The [campaign ledger](native-v2-campaign.json) records allocations, plan adjustments, and teardown state. The [derived results](native-v2-results.json) link each observation to its retained source. Historical October 17 results appear below and are not pooled with the new runs.

## October 21 Example Sufficiency Review

The existing transposed-weight affine layer is sufficient for the two-part post with **tinygrad, Inductor, and TVM as the three main cases**. The [selection review](example-selection-review.json) records the evidence and local checks. Their mechanisms differ at both levels, without requiring mutually exclusive compiler categories:

| Compiler | Semantic Transformation | Scheduling and Code Generation Decision |
|---|---|---|
| tinygrad | Matmul expands into UOps for views, multiplication, casts, and reduction; indexing represents the transpose | Kernel splitting and selected axis/Tensor Core options lead to one generated CUDA matrix-and-bias kernel |
| Inductor | FX retains matmul and precision conversions; the weight transpose becomes a strided layout | Default selection calls a library matrix implementation and generates a Triton epilogue; autotuning selects a fused Triton template with explicit tile and pipeline parameters |
| TVM | Relax tensor operators legalize into loop functions; graph fusion forms executable regions | DLight records tiling, cache placement, thread binding, and tensorization; later passes lower storage and synchronization into generated CUDA |

Each has a validated native FP16 default on the same H100, retained intermediate representations, generated code or an explicit library boundary, and matrix-instruction evidence. The native matrix kernels use Tensor Cores, although their instruction families, tile organization, and complete execution plans differ. MAX remains useful for the graph-routine/buffer/dispatch boundary, but these captures do not expose its internal matrix schedule. Luminal remains a separate widened-operand control: its native FP16 failure occurred in the preliminary plain GEMM gate, before attempting the final native-FP16 transposed-weight program.

The program needs only three operations: a logical transpose creates an indexing-versus-materialization question; matmul creates reduction and matrix-instruction scheduling questions; broadcast bias creates fusion and projection-rounding questions. Bare GEMM removes two of these decisions. Adding ReLU, normalization, another projection, or branches increases explanation cost without resolving the missing MAX internals or Luminal backend path. These observations support a bounded comparison of implementation decisions, not a claim that one workload characterizes an entire compiler.

One small existing control strengthens the explanation: compare `p = x @ w; return p + b` with `return p + b, p`. All input bytes match across the seven fixtures. Returning `p` changes its observability and lifetime. In the saved H100 defaults, tinygrad and TVM change from one kernel to two; Inductor retains two kernels but allocates a separate bias result instead of overwriting the projection. A two-output fused kernel would also be legal, so the observed split is an implementation choice. Use this as a short controlled extension, keeping the transposed-weight layer as the main walkthrough.

### Local CPU Results

[`check_example_cpu.py`](check_example_cpu.py) now executes the affine, transposed-weight, and exposed-projection cases at the explicitly separate inspection dimensions M=8, N=24, K=16. tinygrad, **compiled CPU Inductor**, and TVM passed 84 output checks across seven fixtures per case. The reference also passed analytical multiplication/rounding probes and a wrong-dtype rejection check. CPU generated code is retained; these results do not establish H100 performance or arithmetic for different schedules.

The same driver constructed and replayed five CUDA-target DLight schedules on the host at the actual H100 dimensions M=N=1024, K=768. All three scheduled modules match the October 18 H100 artifacts byte for byte. This checks schedule reconstruction without device execution. [Local results and environment](artifacts/local-example-review-20261021/environment.json), [reference](artifacts/local-example-review-20261021/reference/results.json), [tinygrad](artifacts/local-example-review-20261021/tinygrad/results.json), [Inductor](artifacts/local-example-review-20261021/inductor/results.json), [TVM CPU](artifacts/local-example-review-20261021/tvm/results.json), [CUDA-target schedule replay](artifacts/local-example-review-20261021/tvm-cuda-schedule/results.json).

### Prepared GPU Follow-Up

No new workload sweep is needed. The one useful remaining capture is a **matched tuned-Inductor execution profile**, because the current post has its validated caller and PTX but no tuned Nsight counters. The [bounded plan](article-gpu-followup.json) runs default and original tuning for `weight_view` on one H100 allocation, each through trace, two uninstrumented benchmarks, Systems, and Compute. It retains the full exposed Triton IR chain, including `.llir`, and requires selected-artifact identity before connecting a timing to a profile. It permits 180 seconds per worker and 40 minutes for the suite; ten workers leave ten minutes of the suite cap for preparation and postprocessing. These are maximum limits, not runtime estimates. Broader search budgets were already studied separately.

Preview the exact command and validate harness hashes without contacting Modal:

```sh
.venv-research/bin/python experiments/cuda_case/run_article_gpu_followup.py
```

The explicit `--run` flag starts the prepared GPU job. It has **not been run**. A failed trace blocks dependent phases; differing selected kernels remain separate observations. New timings must stay with their same-allocation control. The plan checkpoints artifacts, releases the GPU function, moves packaging to the CPU-only review job, and requires confirmation of stopped GPU containers.

## What the plan review changed

The original adapters obscured an important distinction. Widening tinygrad's matrix operands prevented its FP16 matrix matcher from applying. Inductor's explicit FP32-output API excluded its ordinary Triton GEMM candidates. The follow-up retains FP16 matrix operands while testing the requested accumulation and rounding behavior independently. Its native GEMM gate admitted four backends before the ten-program screen; the Luminal failure received one larger-budget retry, then moved to a separate control cohort.

A failed trace now prevents dependent timing and profiling. Search configurations run sequentially with fresh processes and shuffled order on one physical GPU. Expensive search is concentrated on the finalist after complete default coverage. H100 allocations release after checkpointing; CPU-only jobs package the review copies. Native progress logs, timeouts, rejected results, and the interrupted pilot remain available.

The pilot also exposed measurement instrumentation: tinygrad's search logging enabled both runtime bookkeeping and profiling in its compiled submission program. The corrected comparison resets runtime `DEBUG` and passes `profile=False` to `compile_linear`. The pilot's tuned latency is excluded; its search, compilation, and validation evidence remains. Its reported 296 µs must not be described as a scheduling regression.

## One program and one numerical contract

```python
def layer(x, wt, b):
    p = x @ wt.T
    return p + b
```

The main shape is `x: [1024, 768]`, `wt: [1024, 768]`, and `b: [1024]`, all FP16. The requested matrix arithmetic uses FP32 products/accumulation, rounds the projection to FP16, then adds the bias and rounds the output to FP16. A single-rounding FP32 fused epilogue is a different computation. Fusion is allowed if it preserves the declared value boundaries. Reassociation is allowed within the numerical admission check; bit-identical reductions are not required.

Authors supply operations, precision, shapes, and outputs—not tiles, thread mappings, shared-memory stages, or kernels. Each adapter expresses that contract through its available API. Consequently, code inspection and the product-accumulation and projection-rounding sentinels accompany ordinary random, structured, zero, and cancellation fixtures. The tolerance remains `abs(actual-reference) <= 2e-4 + 2e-3*abs(reference)`. Passing these finite fixtures is evidence for the tested programs, not a proof for all inputs.

[`next_workloads.py`](next_workloads.py) defines ten controlled candidates at the same matrix shape:

| Case | Author computation | Decision investigated | Native default screen |
|---|---|---|---|
| `gemm` | `x @ w` | Matrix eligibility and accumulation | 4/4 pass |
| `bias` | `x @ w + b` | Projection rounding and broadcast epilogue | 4/4 pass |
| `weight_view` | `x @ wt.T + b` | Indexing/descriptor versus transpose storage | 4/4 pass |
| `input_scale` | `(x * s) @ w` | Input-load fusion versus producer storage | 4/4 pass |
| `output_scale` | `(x @ w) * s` | Multiplicative epilogue | 4/4 pass |
| `residual` | `x @ w + residual` | Full-size second input | 4/4 pass |
| `public_projection` | `p=x@w; return p+b, p` | Observable intermediate and buffer reuse | 4/4 pass |
| `relu` | `relu(x @ w + b)` | Activation placement | 4/4 pass |
| `projection_join` | `(x @ w) + (x @ v)` | Independent producers and cancellation | 0/4 admitted; diagnosed below |
| `projection_chain` | `(x @ w) @ v` | Producer–consumer matrix boundary | 4/4 pass |

The [native screen](artifacts/20261018T063712Z-screen-analysis/review.json) contains 36 passes and four failed numerical admissions. The [Luminal control screen](artifacts/20261018T064926Z-screen-analysis/review.json) admits nine of ten default programs plus an extended-search `weight_view` run. Screen success alone does not establish matrix instructions for every candidate: `bias`, `weight_view`, and `public_projection` received the full deep treatment.

## Why the transposed-weight example earns its extra operation

The [native deep run](artifacts/20261018T065152Z-deep-analysis/review.json) validates all 60 phases for the three promoted programs: four compilers × three programs × trace, two benchmarks, Systems, and Compute. All their contractions have matrix-instruction evidence. For `weight_view`, the implementation decisions differ in concrete ways:

| Backend | Semantic representation | Commitment and emitted implementation | Observed execution |
|---|---|---|---|
| tinygrad | `PERMUTE`, reshapes, `MUL`, cast, `REDUCE`, cast, `ADD` UOps | Tensor Core option and axis splits in the kernel's recorded options; transpose becomes load indexing and bias stays in the kernel | One generated CUDA kernel; `HMMA.16816.F32` |
| MAX | `rmo.mo.transpose`, `rmo.matmul`, `rmo.add` in a tensor graph | Dumped graph/partition stages and target operations; selected matrix implementation consumes the transposed layout | Matrix kernel, then elementwise bias; `HGMMA.64x64x16.F32` |
| Inductor | FX `permute`, `mm`, explicit precision casts, `add` | Generated wrapper passes a strided view to `extern_kernels.mm`; pointwise scheduling emits Triton bias | Library matrix kernel, then Triton bias; `HGMMA.64x64x16.F32` |
| TVM / DLight | Relax `permute_dims`, `matmul`, casts, `add` | Legalized tensor loops, fused functions, replayable DLight schedule, storage/indexing passes, CUDA | Materialized transpose, then fused matrix-and-bias kernel; `HMMA.16816.F32` |

MAX also illustrates a boundary that needs care: its captured MOGG stage contains one `mogg.routine` with `fusionInfo = #matmul_static_broadcast_to_add`, while the execution capture contains two GPU kernels. A graph routine is not necessarily a final kernel boundary. Part A should stop at the inspected region/partition representation; Part B must verify what the selected implementation actually launches.

This is why two launches are not one explanation: TVM's first launch rearranges weights; Inductor's and MAX's second launch applies bias. The final instruction family also does not reveal where the earlier choices were represented.

Concrete entry points for the article figures:

- **tinygrad:** [semantic UOps](artifacts/20261018T065152Z-deep-analysis/weight_view/tinygrad-default/trace-0/01-semantic.uop.txt), [scheduled UOps](artifacts/20261018T065152Z-deep-analysis/weight_view/tinygrad-default/trace-0/02-scheduled.uop.txt), [rewrite examples](artifacts/20261018T065152Z-deep-analysis/weight_view/tinygrad-default/trace-0/rewrite-examples.json), [selected options](artifacts/20261018T065152Z-deep-analysis/weight_view/tinygrad-default/trace-0/programs.json), and [CUDA source](artifacts/20261018T065152Z-deep-analysis/weight_view/tinygrad-default/trace-0/kernel-0-1204.txt). A UOp is a computation node carrying an operation, type, inputs, and operation-specific information; the same node machinery serves different lowering stages. The source uses `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. The observed launch is grid 8×16, block 32×4, with 92 registers per thread and no shared memory.
- **MAX:** [input graph](artifacts/20261018T065152Z-deep-analysis/weight_view/max-default/trace-0/01-semantic.mlir), [partition excerpt](artifacts/20261018T065152Z-deep-analysis/weight_view/max-default/trace-0/max-excerpts/weight_view.mogg-pre-mgp-0.mlir), and [target-stage excerpt](artifacts/20261018T065152Z-deep-analysis/weight_view/max-default/trace-0/max-excerpts/weight_view.mgp-2.mlir). [Excerpt provenance](artifacts/20261018T065152Z-deep-analysis/weight_view/max-default/trace-0/max-excerpts/provenance.json) preserves original file hashes and line ranges. The selected matrix launch uses grid 2×64, 384 threads, 168 registers per thread, and 213,268 dynamic shared-memory bytes. Internal library tile choices are not inferred solely from a kernel name. This run uses `tile_based_fusion=False`, also the installed API default; it is not a statement about every MAX configuration.
- **Inductor:** [same-function diagnostic FX graph](artifacts/20261018T065152Z-deep-analysis/weight_view/inductor-default/trace-0/01-fx.py); the actual [compiled FX graph](artifacts/20261018T065152Z-deep-analysis/weight_view/inductor-default/trace-0/torch-debug/torch_compile_debug/run_2026_10_18_06_59_53_413986-pid_2301/torchinductor/model__0_inference_0.0/fx_graph_readable.py), [scheduler IR](artifacts/20261018T065152Z-deep-analysis/weight_view/inductor-default/trace-0/torch-debug/torch_compile_debug/run_2026_10_18_06_59_53_413986-pid_2301/torchinductor/model__0_inference_0.0/ir_post_fusion.txt), and [generated wrapper](artifacts/20261018T065152Z-deep-analysis/weight_view/inductor-default/trace-0/torch-debug/torch_compile_debug/run_2026_10_18_06_59_53_413986-pid_2301/torchinductor/model__0_inference_0.0/output_code.py) are retained separately. The wrapper uses `reinterpret_tensor(wt, (768,1024), (1,768), 0)` rather than a transpose kernel. Its matrix launch has the same observed geometry/resource counts as MAX's; the generated bias launch uses 128 threads. The diagnostic `make_fx` capture is explicitly distinguished from the graph actually compiled by Inductor.
- **TVM:** [Relax](artifacts/20261018T065152Z-deep-analysis/weight_view/tvm-default/trace-0/01-relax.py), [legalized loops](artifacts/20261018T065152Z-deep-analysis/weight_view/tvm-default/trace-0/02-legalized.py), [fused functions](artifacts/20261018T065152Z-deep-analysis/weight_view/tvm-default/trace-0/03-fused.py), [schedule](artifacts/20261018T065152Z-deep-analysis/weight_view/tvm-default/trace-0/schedules/fused_matmul_cast1_add_cast2.py), [replay verification](artifacts/20261018T065152Z-deep-analysis/weight_view/tvm-default/trace-0/schedule-replay.json), and [CUDA](artifacts/20261018T065152Z-deep-analysis/weight_view/tvm-default/trace-0/cuda-compile-0.cu). The scheduled matrix kernel has grid 8×8, block 32×16, 122 registers per thread, and 73,728 dynamic shared-memory bytes. The transpose uses 1,024 threads per block. This is DLight rule scheduling, not MetaSchedule search.

The early tinygrad UOp graph has an FP16-typed `MUL` followed by an FP32 cast. The selected matrix implementation passes the product-accumulation probe, while the scalar join fallback does not. The requested reduction dtype alone therefore does not establish the multiplication policy; retain both the typed graph and the selected instruction evidence in the walkthrough.

`bias` is the simpler baseline. `public_projection` is the controlled extension for Part B: tinygrad and TVM change from one launch to two when the projection becomes an output. Inductor retains two launches but stops reusing the projection buffer for the epilogue. Making a value observable therefore need not add a kernel; it can change storage ownership. The artifacts do not support a general claim that public outputs always force boundaries.

## Measurements on the same H100

The native deep run used GPU `GPU-c9c1ab46-ffa3-b7df-cd52-1668300090a6`, H100 80GB HBM3, driver 610.57.04. Software versions and toolchains are in its [environment](artifacts/20261018T065152Z-deep-analysis/environment.json) and [build manifest](artifacts/20261018T065152Z-deep-analysis/build-manifest.json). Dependency freezes match the original campaign: tinygrad `8e21758`, Luminal `77d3060`, MAX 26.7.0rc1, PyTorch 2.14.1+cu130 / Triton 3.8, and TVM 0.27.0.post1.

Values are **ranges of two fresh-process median synchronized host latencies, µs**. Each process measures 15 warmed blocks with resident inputs and completion synchronization. These are invocation costs, including submission/allocation behavior; they are not isolated kernel durations or confidence intervals. Profiles are collected separately. Different allocations and the widened-control cohort are not pooled into this table.

| Native backend | Bias | Transposed weights + bias | Return bias result and projection |
|---|---:|---:|---:|
| tinygrad | 70.24–71.22 | 64.26–64.91 | 79.89–81.92 |
| MAX | 76.70–80.64 | 66.63–67.09 | 78.82–80.40 |
| Inductor | 51.51–51.53 | 53.21–55.25 | 54.01–54.33 |
| TVM / DLight | 49.91–49.95 | 57.66–58.20 | 53.39–53.72 |

The common main shape gives about 1.61 billion contraction FLOPs, but dividing that by synchronized host latency would not measure Tensor Core efficiency. Use the separate timeline and resource captures to explain the launches. Static matrix opcodes in the profiled kernel establish the instruction path, not dynamic instruction counts.

## Luminal: a useful control with an explicit limitation

The [native gate](artifacts/20261018T062110Z-gate-analysis/review.json) and [extended-budget retry](artifacts/20261018T062616Z-preflight-analysis/review.json) fail before a timed executable candidate exists: extraction cannot find a plan that writes the bound output. The pinned CUDA backend's [matrix-site rules](https://github.com/luminal-ai/luminal/blob/77d3060f01dc9d6f649687cf16a651e3f288aa0c/crates/luminal_cuda_lite/src/ops/cublaslt/egg/cublaslt_marker_site.egg) and [dtype dispatch](https://github.com/luminal-ai/luminal/blob/77d3060f01dc9d6f649687cf16a651e3f288aa0c/crates/luminal_cuda_lite/src/ops/cublaslt/device_call.rs) expose F32/BF16 matrix paths, not this native F16 path. Giving it a larger extraction budget does not add the missing implementation. The logs also retain arena pruning; neither exclusion nor extraction failure is a poor measured ranking.

The control widens matrix operands to FP32 and keeps the same source operations, FP16 inputs/outputs, and numerical boundaries. Its [logical graph](artifacts/20261018T071206Z-deep-analysis/weight_view/luminal-default-legacy-seed0/trace-0/01-logical.egg) contains index maps, casts, multiplication and reduction. Its computation e-graph and extracted plans are separate artifacts. The selected plans expose layout choices and materialization, and the matrix call takes an F32 cuBLAS path. Names containing `xmma` are not Tensor Core evidence; sampled instructions here are scalar FFMA.

In the [saturated computation e-graph](artifacts/20261018T071206Z-deep-analysis/weight_view/luminal-default-legacy-seed0/trace-0/02-saturated-egraph.json), e-class `LogicalTensor-69` contains both `function-0-LogicalReduceSum` and `function-0-LogicalHelperMatrixMultiply`. The helper's typed site rule admits a cuBLASLt implementation; later descriptor/layout terms and extraction determine the executable plan. This is concrete contraction recognition evidence, although the snapshot alone does not record the order of rule firings. From one extended trace, independently valid [candidate 1](artifacts/20261018T071206Z-deep-analysis/weight_view/luminal-tuned-extended-seed0/trace-0/candidate-1/plan.txt) and [candidate 10](artifacts/20261018T071206Z-deep-analysis/weight_view/luminal-tuned-extended-seed0/trace-0/candidate-10/plan.txt) differ by an explicit copy/materialization. Those plan alternatives should not be labeled new algebraic identities.

For the matched `weight_view` trace, benchmark, and Systems captures, expanded search removes a `CopyGeneric` that transposes the FP32 projection. System allocations fall from five to four, selected plan operations from 16 to 13, and GPU launches from six to five. Matched benchmark medians are 74.76 µs and 65.75 µs. These are two actual executable plans, not two e-graph snapshots. All **40 retained alternatives** across the four deep control configurations pass the complete fixture manifest, totaling 336 output checks.

The [identity review](native-v2-luminal-identity.json) hashes the selected node DAG, outputs, buffer layouts/bindings, and selected CUDA sources. Baseline `weight_view` matches trace, both benchmarks, and Systems; extended matches trace, the first benchmark, and Systems. Their Compute captures select different plans. Do not attach those counters to the illustrated trace. Hash-map entries are sorted without discarding any buffer or layout fields; vendor-library JIT binaries are not fingerprinted by this check.

## Search effort versus outcome

Search settings are native effort controls, not equivalent amounts of work:

| Path | Default | Original tuning | Expanded tuning |
|---|---|---|---|
| tinygrad | BEAM 0 | BEAM 2 | BEAM 8 |
| Luminal | 2 generations × 4 | 4 × 8, 3 trials | 16 × 32, 5 trials |
| Inductor | Default selection | Max autotune, default space | Max autotune, exhaustive space |

The parent deadline is 600 seconds per worker and includes compilation, capture, validation, and measurement. Luminal's 512 MiB arena and 64 MiB intermediate caps remain fixed; its candidate execution-trial timeout is two seconds. Every fresh paired block shares one physical GPU allocation; only Luminal exposes the seeded extraction used here. Luminal completed all three blocks in one allocation. Native blocks may be split across allocations to retain the full per-worker allowance; compare each tuned result with its own same-allocation default. MAX and DLight retain their native rules rather than receiving invented search knobs.

The [Luminal repeated comparison](artifacts/20261018T071630Z-screen-analysis/review.json) is complete:

| Seed | Default µs / plans / search s | Original tuning µs / plans / search s | Expanded tuning µs / plans / search s |
|---|---|---|---|
| 0 | 75.70 / 8 / 1.15 | 72.44 / 30 / 1.30 | 66.93 / 107 / 7.27 |
| 1 | 70.91 / 8 / 1.07 | 65.22 / 32 / 1.86 | 66.66 / 145 / 7.36 |
| 2 | 71.68 / 8 / 1.26 | 71.08 / 28 / 1.61 | 65.49 / 118 / 7.22 |

Expanded search beats the baseline in all three paired observations, but not every original-tuning result. It measures 107–145 plans, not 512 distinct successful programs. These three blocks do not establish statistical significance or general search scalability.

<!-- NATIVE_SEARCH_RESULTS_START -->
The native paired comparison is complete. Blocks 0 and 1 share allocation A; block 2 uses allocation B after the earlier block-boundary stop. Every paired comparison stays on one physical GPU. All expanded searches received their full 600-second worker limit; the interrupted and deferred duplicate block-2 jobs remain excluded. Different host timings between allocations are visible rather than pooled.

| Backend | Block / allocation | Default µs | Original tuning: µs / compile s | Expanded tuning: µs / compile s |
|---|---|---:|---:|---:|
| tinygrad | 0 / A | 83.75 | 75.91 / 152.8 | Timeout at 600 s (worker limit) |
| tinygrad | 1 / A | 82.91 | 83.66 / 89.3 | Timeout at 600 s (worker limit) |
| tinygrad | 2 / B | 69.27 | 74.22 / 116.3 | Timeout at 600 s (worker limit) |
| inductor | 0 / A | 140.65 | 92.81 / 13.6 | 91.79 / 183.8 |
| inductor | 1 / A | 140.96 | 93.04 / 12.2 | 95.60 / 184.2 |
| inductor | 2 / B | 65.79 | 50.76 / 10.3 | 53.83 / 93.1 |

For tinygrad, BEAM 2 passes all fixtures in each block but improves the common host metric in only one of three pairs. It logs 226, 211, and 217 candidate timing evaluations. BEAM 8 logs 697, 693, and 693 timing evaluations before its three 600-second timeouts; these are censored searches with no accepted final latency. They do not establish that its transformation space was exhausted. The final timing path has runtime `DEBUG=0` and HCQ profiling disabled.

Inductor's original tuning reports 22 standalone GEMM choices; exhaustive tuning reports 1,401. Both tuning levels emit a fused Triton implementation in all three blocks, while their standalone GEMM logs rank the library call first. Expanded search is slightly faster than original tuning in block 0 and slower in blocks 1 and 2. More candidates and more compilation time did not consistently improve the independently measured invocation. These counts are native search statistics, not independent correctness checks of every candidate.
<!-- NATIVE_SEARCH_RESULTS_END -->

The retained tuned Inductor callers make the implementation change explicit. The [original-tuning source](artifacts/20261018T071742Z-screen-analysis/weight_view/inductor-tuned-legacy-seed0/screen-0/retained-cache-code/inductor/tm/ctmduzmnepgtywrw5pf2pep4exubrk3yvzhloey7oblhupsgy5h2.py) uses a 64×128 output tile, K steps of 128, four warps, and four pipeline stages. The [expanded-search source](artifacts/20261018T071742Z-screen-analysis/weight_view/inductor-tuned-extended-seed0/screen-0/retained-cache-code/inductor/ub/cubglrq22n5qfaxc3ixf7bkgra5vzumstdalbvtquhbfik2ijcmy.py) keeps that output tile and warp count, but uses K steps of 64 and five stages. Both round the FP32 accumulator to FP16 and widen it again before adding bias. Their selected PTX contains `wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16`. These are selected-source/PTX observations from the validated runs; a tuned Nsight timeline or counter capture was not collected.

Compiler-reported cost scopes differ. tinygrad's timer includes scheduling, compilation/search, UOp and candidate-code export, linking, and first execution; TVM includes schedule replay verification; MAX includes initialization and upload. Inductor reports native autotuning/precompile costs as well as adapter initialization. Use these labels when discussing amortization. Candidate scores optimize backend-specific timing objectives, which need not minimize the common warm synchronized host metric. In the first native Inductor pair, the standalone 1,401-choice GEMM search ranks the library call first, yet the final tuned caller launches a fused Triton matrix-plus-bias kernel. Inspect the final caller as well as the standalone autotuning log; later fusion selection can change the implementation.

## Why the join was not promoted

Four native defaults and the widened Luminal control miss the fixed admission tolerance on `projection_join`. A separate [diagnostic](artifacts/join-diagnostic/diagnosis.json), generated by [`diagnose_join.py`](diagnose_join.py), checks fixture hashes and reproduces a mismatch using independent CPU FP32 matmuls. A tiny accumulation difference crosses an FP16 rounding midpoint; cancellation in the subsequent sum makes the final absolute difference slightly exceed tolerance. At one output, CPU FP32 and the GPU both return `-0.019287109375` against the rounded FP64 reference `-0.01953125`, a difference of 0.00024414 against an allowed 0.00023906.

Thus the small MAX/Inductor/TVM misses do not establish compiler bugs; the admission tolerance is too strict for this particular boundary/cancellation case. The threshold was not loosened after seeing the outputs. The failed cases remain excluded from accepted timing and promotion. tinygrad also fails the independent product-accumulation sentinel: its selected scalar join code rounds a product to FP16 before widening. That is a separate precision issue, not explained away by the tolerance diagnosis.

## Evidence, reproduction, and limits

The [final verification](native-v2-verification.json) checks 13,442 retained-copy hashes and 6,847 entries from original worker inventories. All 27 timing-only normalizations are accounted for, with no unexplained mismatches. The campaign has 168 validated phase/process records; failed admissions, timeouts, and superseded jobs remain separate. All accepted results pass their complete fixture checks. The [teardown record](native-v2-teardown.json) confirms zero active Modal apps and containers. The evidence volume is preserved.

The [derived report](native-v2-results.json) verifies retained-file hashes, immutable fixture hashes, and each worker's observed GPU identity against its allocation. It separates planned-but-replaced jobs, timeouts, validation-gated skips, and completed phases. The [original-index audit](native-v2-index-provenance.json) also checks retained files against the worker inventories. It accounts for 27 Luminal timing reports whose six derived statistics fields were added after indexing: removing only those fields reproduces the indexed bytes exactly. Raw samples and compiler evidence are unchanged. Analysis now writes a separate `timing-summary.json`. Counts include repeated fixtures and are not counts of independent proofs. Raw captures stay private in Modal volume `tensor-program-h100-evidence`; compact analysis copies record every omission and preserve original evidence paths. Raw Nsight databases can contain process-environment metadata and are not publication artifacts.

The [original proposal](next-round-plan.json) remains the planning record; the [campaign ledger](native-v2-campaign.json) records the actual adapted sequence. To reproduce the main evidence after image preparation:

```sh
.venv-modal/bin/modal run experiments/cuda_case/modal_app.py \
  --suite native-matrix-v2 --phase gate --workloads gemm --shape matrix \
  --minutes 45 --case-seconds 180 --download none
.venv-modal/bin/modal run experiments/cuda_case/modal_app.py \
  --suite native-matrix-v2 --phase screen --workloads all --shape matrix \
  --compilers tinygrad,max,inductor,tvm --minutes 45 --case-seconds 180 --download none
.venv-modal/bin/modal run experiments/cuda_case/modal_app.py \
  --suite native-matrix-v2 --phase deep --workloads bias,weight_view,public_projection --shape matrix \
  --compilers tinygrad,max,inductor,tvm --minutes 45 --case-seconds 180 --download none
```

For the separate Luminal control, use `--suite matrix-v2-widened-control --compilers luminal`. For a finalist search comparison use `--phase screen --workloads weight_view --variants default,tuned --search-effort compare --repetitions 3 --case-seconds 600`. Run search backends separately if all requested repeats cannot fit a 45-minute allocation. Package completed runs on CPU with `review_app.py --run-id <run-id>`; `--selected-cache-only` keeps generated Inductor callers and their referenced Triton IR while retaining full candidate exports remotely. Regenerate the local report with `collect_native_campaign.py`.

The example supports a concrete comparison of representation, candidate availability, selection, storage, and code generation. It does not cover tails, dynamic shapes, large-model planning, independent-branch overlap, or general maintenance cost. No agent kernel experiment or BF16 substitution was performed. The useful conclusion is conditional: the same mathematics and matrix hardware leave substantial compiler decisions visible, while the unavailable native Luminal path must remain an explicit limitation.

## Historical campaign — October 17, 2026

**The original campaign is complete: ten candidates screened across five compilers, followed by 115 successful main-size phases on one H100.** The deep run contains 23 workload/configuration groups: four programs across five baseline configurations, plus affine tuning variants for tinygrad, Luminal, and Inductor. Each group has one trace, two fresh-process timing runs, an Nsight Systems run, and an Nsight Compute run. Experiment wall time was 1,621 seconds, excluding image setup and final archive packaging. The [final verification](verification.json) records 815 selected-output checks, 240 alternative-output checks, intact fixture hashes, and the same GPU UUID across all 115 phases. These counts include repeated fixtures and are not a correctness proof.

Start with the [complete review](artifacts/20261017T192707Z-deep-analysis/review.json), [run manifest](artifacts/20261017T192707Z-deep-analysis/environment.json), and [software/build manifest](artifacts/20261017T192707Z-deep-analysis/build-manifest.json). The [artifact index for the local analysis copy](artifacts/20261017T192707Z-deep-analysis/local-copy.json) records retained files and omissions. Native traces and raw profiles are retained in the private Modal Volume `tensor-program-h100-evidence`, under `20261017T192707Z-deep`; the full archive contains approximately 4.25 GB of uncompressed evidence.

The original screen (`20261017T190543Z-screen`) passed its five numerical fixtures for all 50 workload/compiler combinations. Its Inductor results are superseded by the corrected ten-case screen (`20261017T192338Z-screen`). Those screens used separate allocations and are not combined into a timing comparison. The corrected main run below uses one physical GPU throughout. Earlier failed setup runs and the intentionally stopped precision-review run remain recorded in `run-notes.json`.

### Measurements

Every entry below is the **range of two fresh-process median resident-input synchronized host latencies, in microseconds**. Each median comes from 15 warmed timing blocks. The ranges are not confidence intervals. Inputs, outputs, and declared intermediate rounding follow the numerical contract below. Matrix cases use M=K=N=1024; RMSNorm uses 1024 rows of width 1024.

| Configuration | Affine | Residual + ReLU | Also return projection | RMSNorm |
|---|---:|---:|---:|---:|
| tinygrad | 188.5–189.1 | 188.5–188.9 | 198.3–200.3 | 46.0–46.4 |
| Luminal | 78.2–78.7 | 169.9 | 173.3–174.1 | 166.8–166.9 |
| MAX | 67.6–68.1 | 68.6–69.8 | 84.3–87.6 | 57.8–60.2 |
| Inductor | 52.4–58.8 | 54.9–61.2 | 55.4–56.6 | 38.9–40.0 |
| TVM / DLight | 62.1–62.3 | 62.5–62.6 | 66.7–67.0 | 21.4–21.5 |

These are completed invocation costs, including each runtime's submission, allocation behavior, and synchronization. Luminal's timing loop is Rust; the others use Python. They do not rank GPU arithmetic efficiency. No main-run timing series exceeded the predeclared 15% within-process drift threshold; the largest observed drift was about 5.9%. Fresh-process variation still matters: the baseline Inductor affine medians were 52.43 and 58.79 µs.

Actual **kernel launches per invocation**, from separate ten-invocation Nsight Systems captures:

| Configuration | Affine | Residual + ReLU | Also return projection | RMSNorm |
|---|---:|---:|---:|---:|
| tinygrad | 1 | 1 | 2 | 2 |
| Luminal | 5 | 36 | 35 | 17 |
| MAX | 2 | 2 | 2 | 3 |
| Inductor | 2 | 2 | 2 | 1 |
| TVM / DLight | 1 | 1 | 2 | 3 |

This table describes each profiling process's selected implementation. Measured search can select a different plan in a different process. Exact plan hashes, schedules, generated code, and the individual benchmark records are retained; kernel counts are never substituted for timing. In particular, Luminal's 36 versus 35 launches across the residual/shared-output cases is an observed result, not proof that adding an output inherently eliminates a kernel.

Tuning results for affine, with baseline and tuned settings kept separate:

| Affine variant | Baseline host µs | Tuned host µs | Baseline compile/search s | Tuned compile/search s |
|---|---:|---:|---:|---:|
| tinygrad | 188.5–189.1 | 131.4–153.4 | 0.32–0.33 | 12.39–13.87 |
| Luminal | 78.2–78.7 | 76.2–76.9 | 1.08–1.16 | 1.28–1.46 |
| Inductor | 52.4–58.8 | 54.1–56.1 | 4.47–4.49 | 4.70–4.89 |

Compile/search columns use the adapter-specific scope recorded with each result; compare them within a compiler, not across compilers. tinygrad's fresh tuned processes chose different schedules, producing a 16.7% spread between their medians despite stable timing within each process. Luminal's default affine trace, both benchmark processes, and Systems profile have identical selected-plan hashes; its Compute profile differs. The tuned Luminal processes also selected different plans, so their counters and latencies must not be assembled into one fictitious implementation. Inductor retained a library matmul in both variants; the explicit output-dtype path excludes its usual Triton GEMM templates. These runs establish what the configured searches did, not a universal benefit from enabling tuning.

### Which workloads to use in the post

**Keep affine as the common starting program.** It is sufficient to expose primitive contraction representations, structured matrix operations, extracted implementation alternatives, rule-selected schedules, library boundaries, and generated kernels. The other nine candidates answer controlled follow-up questions; they should not all appear in the main reading path.

| Candidate | Evidence and editorial disposition |
|---|---|
| Affine | Main walkthrough. Full semantic, implementation, timing, and device captures for all five compilers. |
| Affine + ReLU | Screened successfully. Useful to isolate activation recognition, but redundant with the residual extension for the main narrative. |
| Residual + ReLU | Retain as an epilogue extension. The main-size run exposes markedly different fusion outcomes. |
| Also return projection | Retain as the controlled boundary experiment. Compared with residual-only output, tinygrad and TVM go from one actual launch to two. An extra output does not mathematically require an extra kernel; these are the observed choices. |
| Two layers | Screened successfully. Adds a second contraction and producer–consumer storage, but also another algorithmic region. Defer from the introductory example. |
| Gated projections | Screened successfully. tinygrad's captured screen-size source combines two reductions into one GPU program; TVM produces two scheduled functions. Interesting optional follow-up, without a main-size execution-profile claim. |
| Transposed weight | Screened successfully. TVM exposes a separate transpose schedule. Useful for a layout-focused sidebar; not needed to teach the primary boundary. |
| Softmax | Screened successfully. Adds maximum, exponential, and sum stages. RMSNorm is the smaller reduction example. |
| RMSNorm | Optional second example for Part B: one reduction plus elementwise work, with actual launch counts of 1, 2, 3, or 17 across the configured paths. |
| Norm + projection | Screened successfully. Combines both algorithmic regions and is harder to explain before the reader understands either. Defer. |

### What the native artifacts establish

**tinygrad.** The [rewrite examples](artifacts/20261017T192707Z-deep-analysis/affine/tinygrad-default/trace-0/rewrite-examples.json) retain actual matched node IDs and source locations. Relevant stages include `apply rangeify`, `split kernels`, `remove reduces`, and `range to special`. The recorded semantic region contains multiplication and reduction; scheduled UOps introduce explicit ranges and calls. The [selected program metadata](artifacts/20261017T192707Z-deep-analysis/affine/tinygrad-default/trace-0/programs.json) records the optimizer actions and launch mapping. The baseline affine profile uses a 16×32 grid with 8×16 threads per block. Its scalar FP32 reduction uses FFMA in the captured SASS, rather than HMMA/HGMMA matrix instructions. Explicit operand widening is part of this adapter and affects matrix-instruction eligibility; this is not a claim about all tinygrad FP16 matmuls.

**Luminal.** The [recorded computation](artifacts/20261017T192707Z-deep-analysis/affine/luminal-default/trace-0/01-logical.egg) contains mapped operands, `LogicalMul`, and `LogicalReduceSum`. The computation e-graph—not a shape-only graph—contains `LogicalReduceSum` and `LogicalHelperMatrixMultiply` in the same class, with separate backend descriptor and `LayoutTensorOpCublasLt` nodes. The pinned [site rule](https://github.com/luminal-ai/luminal/blob/77d3060f01dc9d6f649687cf16a651e3f288aa0c/crates/luminal_cuda_lite/src/ops/cublaslt/egg/cublaslt_marker_site.egg) connects the recognized helper to the library vocabulary. Main-size affine retained eight valid alternatives, including [candidate 1](artifacts/20261017T192707Z-deep-analysis/affine/luminal-default/trace-0/candidate-1/plan.txt) and [candidate 2](artifacts/20261017T192707Z-deep-analysis/affine/luminal-default/trace-0/candidate-2/plan.txt), which differ in explicit materializations. All 40 retained plans across the five Luminal trace configurations passed the five original independent fixtures.

The [pruning report](artifacts/20261017T192707Z-deep-analysis/affine/luminal-default/trace-0/02-pruning-replay.txt) records six oversized tensors and 378 removed nodes; its largest tensor is 4,294,967,296 bytes, above the 64 MiB intermediate cap. That is feasibility exclusion, not poor cost ranking. The selected default affine path launches two input casts, a strict-FP32 cuBLASLt GEMM, an output cast, and an add. Its reported internal arena is 46,137,600 bytes, excluding caller-owned boundary allocations; this is not total device memory usage.

For residual + ReLU, the saturated graph contains plain cuBLASLt alternatives but no bias- or ReLU-decorated cuBLASLt alternative. The selected trace plan contains casts, comparisons, selects, constants, copies, and materializations. The frontend's NaN-preserving maximum decomposition and the explicit FP16 value boundaries are relevant to explaining that result. Do not present this case as a fused cuBLAS epilogue that merely lost the search.

**MAX.** The [MOGG excerpt](artifacts/20261017T192707Z-deep-analysis/affine/max-default/trace-0/max-excerpts/affine.mogg-pre-mgp-0.mlir) groups matmul and bias into one routine. The [generated Mojo](artifacts/20261017T192707Z-deep-analysis/affine/max-default/trace-0/max-ir/affine.mojo) calls `MatmulGPU.execute` with `has_epilogue_fusion = True`. Nevertheless, the [actual timeline](artifacts/20261017T192707Z-deep-analysis/affine/max-default/nsys-3/timeline-activities.json) contains a vendor GEMM and a separate elementwise kernel. A fused compiler routine therefore does not establish a single physical GPU kernel. MGP exposes output-buffer allocation and runtime calls; internal vendor workspace and selection remain a separate scope. Captured HGMMA instructions use FP32 accumulators, and the FP16 boundary probe passes. The exact internal graph-fusion rule is not exposed by these dumps.

**Inductor.** The actual compiler debug dumps distinguish an external `mm_dtype` call from a generated Triton epilogue. The epilogue explicitly narrows and widens the projection before adding the bias, preserving the FP16 boundary. The [installed-version matmul lowering](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/kernel/mm.py#L437) guards the usual Triton template choices with `out_dtype is None`; our native FP32-output adapter does not satisfy that guard. The two affine launches are consequently a library GEMM and a generated pointwise kernel in both configurations. RMSNorm instead yields one generated reduction kernel, making it the better example of Inductor's reduction and elementwise scheduling here.

**TVM / DLight.** The [Relax graph](artifacts/20261017T192707Z-deep-analysis/affine/tvm-default/trace-0/01-relax.py), [legalized operations](artifacts/20261017T192707Z-deep-analysis/affine/tvm-default/trace-0/02-legalized.py), and [fused function](artifacts/20261017T192707Z-deep-analysis/affine/tvm-default/trace-0/03-fused.py) make the change from tensor operators to a schedulable implementation explicit. The [recorded schedule](artifacts/20261017T192707Z-deep-analysis/affine/tvm-default/trace-0/04-schedule-trace.py) is replayed from disk and checked for structural equality. The [scheduled form](artifacts/20261017T192707Z-deep-analysis/affine/tvm-default/trace-0/05-scheduled.py) introduces WMMA scopes, shared-memory staging, thread bindings, and tensorization. Actual affine execution is one kernel with an 8×8 grid, 32×16 threads per block, 122 registers per thread, and 73,728 bytes of dynamic shared memory. Its [machine-code excerpt](artifacts/20261017T192707Z-deep-analysis/affine/tvm-default/ncu-4/machine-code-excerpts.json) contains `HMMA.16816.F32`. This is DLight rule scheduling, not a MetaSchedule search experiment.

RMSNorm supplies a more direct scheduling comparison than the library-backed matmuls. [tinygrad's first kernel](artifacts/20261017T192707Z-deep-analysis/rmsnorm/tinygrad-default/trace-0/kernel-0-79.txt) assigns 16 threads to each row, gives each thread 64 values, stages 16 partial sums in shared memory, and writes one FP32 row statistic; a second kernel scales the elements. [Inductor's generated Triton](artifacts/20261017T192707Z-deep-analysis/rmsnorm/inductor-default/trace-0/torch-debug/torch_compile_debug/run_2026_10_17_19_52_16_535003-pid_7919/torchinductor/model__0_inference_0.0/output_code.py) keeps the row reduction and scaling in one kernel, observed with 32 threads per row. [TVM's scheduled functions](artifacts/20261017T192707Z-deep-analysis/rmsnorm/tvm-default/trace-0/05-scheduled.py) materialize the FP32 input and squared values separately, then use 1,024 threads per row for the reduction/normalization kernel. [MAX's regions](artifacts/20261017T192707Z-deep-analysis/rmsnorm/max-default/trace-0/max-excerpts/rmsnorm.mogg-pre-mgp-0.mlir) separate the row statistic, the gamma cast, and the final scaling. Their similar or different kernel counts therefore conceal different storage and work-distribution decisions. These exact programs, rather than generic reduction diagrams, should supply the optional Part B figure.

### Evidence boundaries for the two-part article

Part A can follow the single tensor definition into each compiler's native semantic representation and first explicit partition. Part B should then show storage, mapping, selected code, and actual launches. MAX demonstrates why the first partition must not automatically be labeled the final GPU kernel boundary: later implementation choices can still split the work. Bufferization, partitioning, and schedule selection can also interact rather than form a universal one-way pipeline.

The original campaign recommended affine for the five main figures; the October 18 evidence above supersedes that choice with transposed weights. Show the residual/shared-output pair only when explaining observable intermediates, and use RMSNorm when a matrix operation would hide reduction scheduling inside a library. Keep original node IDs and code intact; any cropped graph or excerpt must retain its source path and omitted context. MAX excerpts already have line-range and hash provenance. The review includes short verbatim SASS excerpts from the actually profiled kernels; instruction presence is not a dynamic instruction count.

The results support these bounded implementation stories. They do not establish general compiler rankings, production-model performance, independent-branch concurrency, search scalability, tail handling, or correctness on all inputs. The `narrow` and `tail` shapes are implemented but were not executed in this campaign. Profiling is separate from timing, and Compute captures at most ten kernel launches, so a long Luminal pipeline is only partially sampled by hardware counters. Full arithmetic policy still needs to be checked whenever the example, adapter, precision boundary, or compiler revision changes.


## Setup and reproduction of the original campaign

1. Create/sign into a [Modal account](https://modal.com/), add the payment method required for GPU use, and confirm available compute credits. In **Usage & Billing**, set the **out-of-pocket spend limit to $0**. The account's credit balance and spending controls remain authoritative; the harness time limit is an additional guard.
2. From the repository root, create the client environment if necessary:

   ```sh
   uv venv --python 3.12 .venv-modal
   uv pip install --python .venv-modal/bin/python modal==1.6.2 numpy==2.5.4
   .venv-modal/bin/modal setup
   .venv-modal/bin/modal profile list
   ```

3. Run the small preflight. Image construction happens on CPU before a GPU is allocated. Compiler packages are isolated in separate Python environments. A failed installation is retained in `build-manifest.json` and becomes a failed preflight, never a substitute backend.

   A CPU-only image and profiler installation check is also available before GPU billing is enabled: `.venv-modal/bin/modal run experiments/cuda_case/prepare_app.py`. This does not test CUDA execution or profiler counter permissions.

   ```sh
   .venv-modal/bin/modal run experiments/cuda_case/modal_app.py \
     --phase preflight --workloads affine --shape inspect \
     --minutes 12 --case-seconds 120
   ```

4. Screen all ten programs on one H100 allocation:

   ```sh
   .venv-modal/bin/modal run experiments/cuda_case/modal_app.py \
     --phase screen --workloads all --shape screen \
     --minutes 35 --case-seconds 100
   ```

5. Deep-capture the four candidates selected from screening. Affine isolates contraction and bias; the residual/shared-output pair tests whether making a value observable changes a boundary; RMSNorm adds reduction coordination. Only affine receives additional tuning variants:

   ```sh
   .venv-modal/bin/modal run experiments/cuda_case/modal_app.py \
     --phase deep --workloads affine,residual_relu,shared_projection,rmsnorm --shape main \
     --variants default,tuned --tuned-workloads affine --minutes 45 --case-seconds 150
   ```

Each deep case gets a trace process, two fresh benchmark processes, an Nsight Systems process, and an Nsight Compute process. These phases run sequentially on the same physical GPU. Counter access is tested by the actual profiler; installation alone does not establish access. To repeat profiling without the other phases, use `--phase profile`.

Use `--compilers tinygrad,max,inductor,tvm` or another comma-separated subset when isolating a failure. There are no automatic task retries, minimum warm containers, or permanent deployments. `H100!` prevents Modal's automatic substitution of H200 for an ordinary H100 request. See [Modal GPU selection](https://modal.com/docs/guide/gpu) and [budget controls](https://modal.com/docs/guide/budgets).

Artifacts are checkpointed after screening cases, trace captures, and completed profiling groups to the `tensor-program-h100-evidence` Modal Volume, and downloaded into `artifacts/<UTC run ID>/`. The download excludes disposable compiler caches; selected cache code is copied into retained artifacts first. If a client disconnects, retrieve the archive using `modal volume get tensor-program-h100-evidence <run-id>.tar.gz <local-path>`. A hard interruption may leave only the checkpointed directory; inspect `modal volume ls` before assuming an archive exists.

If local disk space cannot accommodate the full archive and extraction, the runner downloads a compact review archive (excluding fixtures and files over 2 MiB) when space permits. Its `local-copy.json` explicitly lists omissions; full evidence stays remote. `sync_summary.py <run-id>` retrieves checkpointed reports while a run is active. `fetch_evidence.py <run-id>` can create a bounded partial local extraction from a completed full archive. Missing files in these review copies must not be treated as missing compiler evidence.

After completion, `.venv-modal/bin/modal run experiments/cuda_case/review_app.py --run-id <run-id>` performs CPU-only analysis and downloads a separate `<run-id>-analysis/` directory, capped at 100 MiB before compression. It excludes validation arrays and raw profiler databases and keeps a manifest of omissions. **Keep full Nsight captures private:** they can contain process-environment metadata, including container identity information. The derived timeline JSON exports only CUDA activities and their referenced kernel names. Future worker launches also remove Modal identity variables from the inherited environment. The analysis bundle is the intended review surface; a raw capture is not a publication artifact.

## Original ten candidates

| Program | Smallest new question |
|---|---|
| `affine`: `x @ w + b` | How is the contraction represented and the bias placed? |
| `affine_relu` | Does activation alter epilogue fusion or library eligibility? |
| `residual_relu` | Does a second epilogue input change the boundary or output write? |
| `shared_projection` | What changes when the projection is also a public output? |
| `two_layer` | How is an intermediate between two contractions stored and scheduled? |
| `gated_projections` | How are independent producers and their elementwise join ordered? |
| `transposed_weight` | Is transpose a view, a copy, or an implementation parameter? |
| `softmax` | How are dependent reductions and their broadcasts combined? |
| `rmsnorm` | Where is a reduction accumulator placed and how is its result shared? |
| `norm_projection` | Can a reduction producer be integrated with a contraction consumer? |

There is one computation definition per case. The JSON manifest is **author input**, not an invented common compiler IR. Each adapter calls its compiler's tensor operations and captures the resulting native representation. The actual human-readable definitions and investigation questions are in `workloads.py`.

Shapes `(M,K,N)`:

| Label | Dimensions | Purpose |
|---|---|---|
| `inspect` | 8, 16, 16 | Installation and numerical inspection only |
| `screen` | 128, 256, 256 | Bounded initial search for informative mechanisms |
| `main` | 1024, 1024, 1024 | Main GPU case; captures must be generated again at this size |
| `narrow` | 32, 1024, 1024 | Sensitivity to small batch dimension |
| `tail` | 1000, 1024, 1024 | Sensitivity to incomplete output tiles |

An 8-row graph or kernel is never presented as evidence of what a 1024-row run produced. The ten candidates cover several controlled changes, not ten independent estimates of performance and not every compiler capability.

## Original campaign numerical contract

Inputs and outputs use FP16. Matmul requests FP32 accumulation and rounds its result to FP16 before the epilogue. Every declared node rounds to its declared dtype; normalization and softmax use explicit FP32 interiors. This preserves the source program's FP16 value boundaries even when a compiler fuses operations. It is not a single-rounding FP32 fused-epilogue contract.

The original campaign adapters differ from the native follow-up above. Inductor disables TF32 and reduced-precision reduction, uses native `torch.mm(..., out_dtype=torch.float32)`, narrows its result, and enables `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1`. This prevents the observed `mm + bias → addmm` substitution from removing a required rounding boundary. TVM uses an FP32 matmul result followed by a cast. tinygrad and Luminal explicitly widen operands before matmul; their materialization, absorption, and effect on matrix-instruction eligibility are part of the evidence. The initial tinygrad preflight used only `matmul(dtype=float32)`: its generated scalar path rounded products to FP16, so that adapter was corrected before screening. MAX has no accumulation-dtype argument in this adapter; its arithmetic policy remains a code-review gate. Numerical agreement alone does not certify any of these policies.

`reference.py` independently evaluates exact stored inputs with NumPy FP64 arithmetic and rounds at the declared boundaries. Fixed seeds produce search, held-out random, zero, structured, and cancellation-oriented fixtures. Outputs must have matching shapes and FP16 dtype, finite values, and satisfy the predeclared check:

```text
abs(actual - reference) <= atol + 0.002 * abs(reference)
atol = 0.0002   # affine / normalization cases
atol = 0.000002 # softmax: probabilities can be around 1/1024
```

These tolerances were tightened during local harness review, before any H100 execution. Softmax needs a smaller absolute tolerance so an all-zero output cannot pass merely because each probability is small. They admit bounded numerical variation on these fixtures; they do not justify arbitrary arithmetic substitutions or guarantee all-input correctness. Do not widen them after seeing a GPU failure. Diagnose a failure and retain the original result. Search sees only the search fixture. Validation reuses the selected compiled implementation with changed inputs, so a stale captured result cannot pass all fixtures. Fixtures and references have SHA-256 manifests and are outside the compiler adapter's editable implementation. The controller verifies their hashes after each worker.

After the Inductor code-review finding, an additional held-out probe was added to affine, activation, residual, shared-projection, and transposed-weight cases. One nonzero product is `1.0048828125 × 1.0927734375 = 1.098109245300293`; its FP16 result is `1.09765625`. Adding bias `−1.09765625` must give zero. Bypassing the intermediate rounding leaves `0.00045299530029296875`, which fails the unchanged tolerance. All five compilers passed this probe at inspection size. Retained Luminal alternatives are independently checked against the five original fixtures; the additional probe currently checks the selected implementation only. Earlier runs retain their original fixture set and validity limits.

## Measurement

- All inputs are uploaded before measurement. Outputs are copied back only for validation. Luminal uses external device bindings so runtime staging cannot silently include input/output transfers in the timed path.
- The common metric is **resident-input synchronized host invocation latency**, including runtime submission and completion. It is not kernel duration. Luminal's timed loop is Rust; other adapters use Python. Host-language and runtime differences are explicitly part of this metric.
- Warm up 10 invocations; use 3 pilots to select a block size targeting 10 ms, capped at 200 invocations. Collect 15 blocks with a completion wait after each invocation. Retain every sample, median, p10/p90, and a deterministic bootstrap interval of the block median. Flag >15% first-to-last block drift. These intervals describe this run, not variation across machines.
- Repeated inputs and warm caches are deliberate. No L2 flush, throughput batching, outer user-added CUDA graph, or concurrent compiler benchmark is introduced. Native runtime graph submission remains visible as a design choice.
- Compilation/search and first execution are measured separately with each adapter's actual timing scope. They are not interchangeable compile-time measurements. Luminal's candidate search timings and tinygrad/Inductor tuning are additionally retained.
- Profiles are separate processes. Nsight Systems records the warmed CUDA-profiler range of 10 invocations; Nsight Compute records a bounded kernel sample when counters are permitted. Instrumented times do not enter the uninstrumented benchmark table.
- Record GPU UUID, driver, toolkit, device state, source revisions, package freezes, command, variant, trace files, and artifact hashes. Do not merge timing from different GPU allocations into a head-to-head comparison. Compare fresh-process repeats and implementation fingerprints before publication.

## Evidence required for the post

| Compiler | Semantic and boundary evidence | Implementation evidence |
|---|---|---|
| tinygrad | Tensor UOps, scheduled UOps, retained rewrite history | Compiled UOps, optimizer variant, generated CUDA/binaries, launch metadata; distinguish selected programs from search trials |
| Luminal | Logical graph, saturated computation e-graph, independently replayed mandatory memory-pruning snapshot | Selected plan, ranked extracted plans, independent candidate tests, search/refusal costs, CUDA sources or library-op identity, graph statistics |
| MAX | Native graph MLIR and compiler dump files actually exposed by the installed build | MEF executable, available lowering/JIT artifacts and observed device activity; closed or missing internal stages remain unknown |
| Inductor | FX graph, compiler debug dumps, fusion/scheduling diagnostics | Generated wrapper, external calls or Triton kernels, retained Triton IR/PTX/cubin, autotuning logs |
| TVM | Relax, legalized TensorIR, fused functions | Actual DLight schedule and structural replay check, scheduled IR, target module source, VM execution |

`analyze.py` produces `report.md`, `summary.csv`, an evidence matrix, and raw normalized timeline activities. It checks coverage, not interpretation. For each consequential transformation, a reviewer must record the before/after artifact, rule/pass or matcher, preconditions, newly fixed decision, unresolved choices, and runtime confirmation. Source excerpts must be tied to the recorded revision.

Choose the smallest candidate that satisfies these gates across the selected compilers. A Luminal alternatives figure requires two **meaningfully different, independently validated** extracted plans; two genomes or two timing values are insufficient. Distinguish an unavailable alternative from an available alternative that lost selection. A library call is a valid result. Fewer kernels do not by themselves establish a better implementation.

The baseline cases are rule-selected for tinygrad, untuned Inductor, bounded device search for Luminal, MAX with tile-based fusion disabled, and DLight for TVM. `tuned` adds BEAM=2, Inductor max-autotune without an outer CUDA graph, or a larger Luminal search budget. These are disclosed configurations, not claims that the projects have the same default search policy. Keep baselines and tuned results separate.

## Local checks

`check_local.py` checks the fixture oracle against analytical cases and tests rejection of stale outputs, wrong dtype, and nonfinite values. Native CPU adapter checks validate graph construction and reference agreement where FP16 CPU execution is supported. MAX's current CPU backend rejects FP16, so a successful MAX graph construction does not imply successful CPU execution. All CPU artifacts are labeled accordingly. CUDA compilation, execution, profiling permissions, and final arithmetic policy require the H100 runs.

No article result is ready for publication merely because setup, a local check, or a screenshot succeeds. Read each run's status and retain failures.
