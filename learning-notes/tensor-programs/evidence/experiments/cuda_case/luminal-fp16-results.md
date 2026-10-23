# Luminal FP16 Backend Repair and H100 Evidence

The research patch makes native FP16 matrix storage usable with explicit FP32 computation in the pinned CUDA backend. The transposed-weight affine workload passes the unchanged numerical contract. This is patched behavior against `77d3060f01dc9d6f649687cf16a651e3f288aa0c`, not a claim about the unmodified upstream release.

## What Was Fixed

Two separate issues had been conflated:

1. The original adapter recorded an FP16 multiply/reduce and then cast its result to FP32. Luminal's frontend defines matmul in the operands' dtype; this spelling did not express the requested FP32 arithmetic.
2. The backend had no FP16 cuBLASLt admission rule or FP16 matrix-descriptor mapping. Increasing the search budget could not introduce a missing implementation.

The corrected adapter records:

```python
# Pseudocode for the adapter; the executable harness is Rust.
p = x.cast(F32).matmul(wt.permute([1, 0]).cast(F32)).cast(F16)
y = p + bias
```

The [patch](patches/luminal-fp16-f32-compute.patch) matches exact FP16-to-FP32 operand widening, offers a cuBLASLt implementation consuming the original FP16 storage, and preserves the FP32 result and subsequent FP16 projection cast. A transpose can commute with that widening. The device dispatcher now accepts `CUDA_R_16F` matrix descriptors while retaining `CUBLAS_COMPUTE_32F`.

The original FP32-storage alternative remains in the e-graph. The rule does not reinterpret a genuinely FP16 reduction, narrow arbitrary FP32 operands, or fuse bias across the projection's FP16 rounding boundary. [Frontend contract](https://github.com/luminal-ai/luminal/blob/77d3060f01dc9d6f649687cf16a651e3f288aa0c/src/frontend/matmul.rs), [adapter](luminal/src/main.rs), [regression checks](luminal/tests/fp16_admission.rs).

## Validation and Experimental Design

Three CPU regression tests pass: native/widened alternatives exist for ordinary and transposed weights; result-only widening is not admitted as FP32 arithmetic; arbitrary FP32 operands are not narrowed.

H100 runs use M = N = 1024, K = 768, the existing deterministic fixtures, and the unchanged tolerance `abs(actual-reference) <= 2e-4 + 2e-3*abs(reference)`. The independent reference evaluates stored FP16 inputs in FP64 and rounds at declared FP16 node boundaries. Plain GEMM has six applicable fixtures; bias workloads add the projection-rounding fixture. The product/accumulation fixture rejects an implementation that rounds products to FP16 before summation.

Default search requests 2 generations × 4 candidates with up to 3 timing trials. Expanded search requests 16 × 32 with up to 5 trials. Both retain the same 512 MiB arena and 64 MiB intermediate limits. Duplicate fingerprints do not fill population slots; generation sampling attempts are bounded. Report actual distinct plans profiled rather than treating the budget as an evaluation count.

Uninstrumented measurements time synchronized resident-input invocations in Rust, with Python IPC outside the timer. Each process uses 10 warmup calls, 3 pilot calls, and 15 blocks of repeated calls. Nsight captures run separately. Clock locking was unavailable on the first profiling host; subsequent Nsight Compute captures use `--clock-control none`. The failed capture remains in the evidence.

## What the First Successful Gate Shows

The GEMM e-graph contains two FP32/FP32/FP32 matrix sites and two FP16/FP16/FP32 sites. The saturated graph has 3,732 nodes; pruning leaves 3,354, retaining both implementation families. Eight distinct plans were measured and all eight retained plans pass the six applicable fixtures. Two use native FP16 matrix storage; the other six materialize FP32 inputs, with different additional layout/copy choices.

The selected library descriptor has FP16 A and B, FP32 D, and m = n = 1024, k = 768. The successful instruction capture contains:

```text
HGMMA.64x64x16.F32 R24, gdesc[UR16].tnspA, R24
```

This is an instruction in a kernel captured executing on the H100; the descriptor establishes the input storage types. Static instruction listings are not dynamic instruction counts. [Gate artifacts](artifacts/20261021T194131Z-luminal-patched-explicit-f32-gate-analysis/), [successful profile](artifacts/20261021T194325Z-luminal-patched-explicit-f32-profile-analysis/).

## Where Implementation Decisions Are Made

The frontend records casts, broadcasts, multiplication, and reduction. Core rules recognize a matrix helper. Backend rules make library-call alternatives available; layout rules and extraction select compatible producers. Bufferization establishes storage and lifetimes. Generated CUDA implements remaining pointwise/materialization operations, while the matrix operation becomes a typed library call.

Luminal measures these extracted implementation plans. Its cuBLASLt bridge requests one heuristic algorithm for each resolved descriptor/workspace combination; it does not explicitly search the library's tile sizes. The library supplies the matrix kernel. CUDA graphs handle runtime submission. These are separate decisions and separate evidence layers. [Backend dispatch](https://github.com/luminal-ai/luminal/blob/77d3060f01dc9d6f649687cf16a651e3f288aa0c/crates/luminal_cuda_lite/src/ops/cublaslt/device_call.rs), [search](https://github.com/luminal-ai/luminal/blob/77d3060f01dc9d6f649687cf16a651e3f288aa0c/crates/luminal_cuda_lite/src/search.rs), [measurement](https://github.com/luminal-ai/luminal/blob/77d3060f01dc9d6f649687cf16a651e3f288aa0c/crates/luminal_cuda_lite/src/profile.rs).

## Ten-Workload Screen

Expanded search, seed 0; identical shape and numerical fixtures. Each row is one completed search, so these counts describe this run rather than the full available implementation space.

| Workload | Distinct plans measured | Search seconds | Validation |
|---|---:|---:|---|
| `bias` | 142 | 7.82 | Pass |
| `gemm` | 82 | 5.74 | Pass |
| `input_scale` | 123 | 7.96 | Pass |
| `output_scale` | 147 | 8.47 | Pass |
| `projection_chain` | 113 | 7.55 | Pass |
| `projection_join` | 137 | 6.85 | Rejected: numerical tolerance |
| `public_projection` | 139 | 7.96 | Pass |
| `relu` | 467 | 17.94 | Pass |
| `residual` | 99 | 6.31 | Pass |
| `weight_view` | 121 | 7.49 | Pass |

All ten selected programs use FP16 matrix operands and FP32 matrix results in their parsed library plans; the join remains numerically rejected. Instruction captures establish Tensor Core execution for the profiled accepted cases, not independently for every screened program. [Screen analysis](artifacts/luminal-fp16-fix-20261021/screen-analytics.json).

## Paired Upstream and Patched Measurements

Three seeds, fresh processes, shuffled backend order within each pair, on one physical H100. Both binaries receive the corrected explicit-FP32 logical program and pass all seven fixtures. These are synchronized Rust `CudaRuntime.execute` host latencies, not GPU-only kernel durations. The reported center is the median of three process medians; the bracket is their observed range.

| Backend | Search | Host latency, µs [range] | Median search seconds |
|---|---|---:|---:|
| Upstream | Default | 74.74 [73.77, 76.64] | 1.70 |
| Upstream | Expanded | 67.03 [66.79, 67.93] | 8.44 |
| Patched | Default | 30.85 [28.42, 35.85] | 1.51 |
| Patched | Expanded | 21.75 [21.66, 27.64] | 6.83 |

Expanded search produced lower latency than default in all three patched pairs, with additional search cost. Three pairs do not establish a significance result or a general compiler ranking. [Samples, counts, search costs, and GPU identity](artifacts/luminal-fp16-fix-20261021/paired-comparison.json).

## One Identical Implementation Through Timing and Profiling

Fresh searches sometimes selected different implementations: the default transposed-weight Compute run selected FP32 storage, and expanded runs changed operand orientation or pointwise operand order. These artifacts are retained separately; differing captures are not treated as one identical implementation.

For the final walkthrough, the replay tool reconstructs the already selected, validated transposed-weight winner from its retained e-graph and genome. It refuses to proceed unless the BufferIR summary, typed operation mapping, and every generated CUDA file match the original capture byte for byte. All five replay phases pass those checks and all seven fixtures: 35 output checks. The same normalized implementation signature appears in both timing processes and both profilers.

The replay follows the actual choices:

1. Explicit FP32 operand casts define multiplication and reduction precision in the semantic graph. Matrix recognition and the new rule expose native FP16 storage as an implementation alternative. This trace contains 4,086 saturated nodes and 3,708 after pruning, with two F32/F32/F32 and two F16/F16/F32 matrix sites retained.
2. The extracted program contains a cuBLASLt operation, an FP32-to-FP16 cast, and a bias-add operation. Allocation/free operations in the plan do not imply additional GPU kernels.
3. A CPU replay of the pinned descriptor resolver binds A to the stored weight and B to the stored input: both are row-major FP16 `[1024,768]`, with `trans_a=false` and `trans_b=true`. The FP32 destination is column-major `[1024,1024]`. Thus the call computes `wt @ x.T = p.T`, whose column-major bytes are already `p` in row-major order. No input or output transpose-copy kernel is required.
4. The contiguous cast kernel rounds the projection to FP16. The separate add kernel broadcasts bias over rows.
5. The GPU timeline contains exactly these three kernels per invocation. The captured matrix kernel contains `HGMMA.64x64x16.F32`.

The `typed_op` artifact stores `LtMatmulSpec`, whose transpose flags precede the executor’s row/column-order conversion. The final flags above come from executing `plan_call` and `bind_destination` on the retained plan on CPU; they are not presented as an intercepted CUDA API log. [Resolved call and arena plan](artifacts/luminal-fp16-fix-20261021/resolved-call.json).

The buffer plan shows concrete reuse:

| Region | Byte offset | Bytes | Use |
|---|---:|---:|---|
| Parameters | 0 | 8 | First allocation starts at the next 256-byte boundary |
| FP32 projection | 256 | 4,194,304 | Matrix result consumed by the cast |
| cuBLASLt workspace | 4,194,560 | 33,554,432 | During the matrix call |
| FP16 projection | 4,194,560 | 2,097,152 | Reuses the workspace region after the matrix call |

Peak arena storage is 37,748,992 bytes, including the workspace and alignment. Input and output boundary allocations are separate. The equal offsets above come from the retained plan’s CPU arena reconstruction, not from an estimate.

The frozen-plan timing medians are **19.66 µs** and **19.04 µs** in two fresh processes. This replay uses `CudaExecutable.execute_external`, a narrower host-wrapper scope than the paired search-run table; these timing populations are not pooled. cuBLASLt resolves its heuristic from the fixed descriptors on the replay device, and the matrix kernel name agrees across both profilers.

| Captured kernel | Threads/block | Registers/thread | Dynamic shared bytes | Systems median, µs |
|---|---:|---:|---:|---:|
| `nvjet_hss_64x128_64x8_2x1_v_bz_TNN` | 384 | 168 | 213,284 | 5.456 |
| Projection cast (`k`) | 256 | 16 | 0 | 3.488 |
| Bias add (`k`) | 256 | 16 | 0 | 3.776 |

The two `k` kernels are distinguished by graph node, dependency order, and generated source, not by their equal names. Durations are instrumented observations; the kernel-name components are not used as proof of a tile configuration. [Frozen replay artifacts](artifacts/20261021T200328Z-luminal-frozen-replay-deep-analysis/), [replay input and provenance](replay-input/weight_view/replay-provenance.json), [replay code](luminal/src/replay.rs).

These October 21 measurements are a Luminal follow-up, not a same-allocation rerun of the October 18 cross-compiler comparison. They should not silently replace one row in that comparison.

## Rejected Workload

`(x @ w) + (x @ v)` remains rejected under the fixed tolerance. Search and execution complete, but random and held-out outputs fail validation. A paired check on the same physical GPU reproduces numerical rejection in both the unpatched and patched binaries; none of their 16 retained finalists passes all fixtures. The patch therefore does not establish support for this workload under the article's contract. No timing from this rejected case is admitted to performance comparisons.

In the patched screen its maximum absolute error is 0.00048828125, with maximum scaled errors 1.127 and 1.196 on the two failing fixtures. The corresponding upstream paired values are 1.021 and 1.054. These observations establish a tolerance failure, not a new extraction failure or a proven general explanation for the arithmetic discrepancy. The original screen controller mislabeled the nonzero validation exit `process_failed`; derived analysis uses the retained validation results, and the controller now preserves numerical-failure status.

[Upstream paired trace](artifacts/20261021T194808Z-luminal-upstream-explicit-f32-preflight-analysis/), [patched paired trace](artifacts/20261021T194847Z-luminal-patched-explicit-f32-preflight-analysis/).

## Reproduction and Raw Evidence

[Patch notes and commands](patches/README.md), [campaign manifest](luminal-fp16-campaign.json), [dedicated Modal runner](luminal_fp16_app.py), [analysis script](analyze_luminal_fp16.py).

Every run retains source and binary hashes, the patch, harness, numerical fixtures and their hashes, selected plans, candidate data where captured, and environment metadata. The complete raw artifacts remain in the private Modal Volume `tensor-program-h100-evidence`. Local analysis copies explicitly inventory omissions, including raw binary arrays and profiler databases. Successful finite fixtures and three host checks do not prove correctness for all shapes, values, or architectures.

## Completion and Coverage

The campaign retained 13 runs and 64 worker phases: 60 validated phases, three numerical rejections of the same join workload, and one failed initial profiler capture. Accepted phases pass 489 output checks. Of 112 retained candidate plans, 80 pass all applicable fixtures; the other 32 belong to the rejected upstream/patched join ablation. Thirty deep-capture phases cover bias, transposed weights, and public projection under both budgets; five additional phases replay one fixed transposed-weight implementation.

[Complete derived analytics](artifacts/luminal-fp16-fix-20261021/complete-analytics.json) links every observation to its raw artifact directory. The H100 jobs have ended. Final inspection records zero running containers and all listed experiment apps stopped: [apps](artifacts/luminal-fp16-fix-20261021/final-apps.json), [containers](artifacts/luminal-fp16-fix-20261021/final-containers.json).
