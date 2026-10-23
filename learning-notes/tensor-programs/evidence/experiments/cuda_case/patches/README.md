# Luminal FP16 Storage with FP32 Computation

Research patch against `luminal-ai/luminal@77d3060f01dc9d6f649687cf16a651e3f288aa0c`. This is not an upstream feature claim.

The old native eligibility probe recorded FP16 multiply/reduce followed by a result cast. The frontend explicitly defines multiplication and reduction in the operands' dtype; that spelling does not express our requested FP32 arithmetic. It also fails extraction at the tested shape: no FP16 cuBLASLt site exists and the generic path loses oversized intermediates during pruning.

The corrected adapter records `x.cast(F32).matmul(w.cast(F32)).cast(F16)`. The backend patch adds an implementation alternative consuming the original FP16 storage, producing an FP32 result with `CUBLAS_COMPUTE_32F`. It also commutes exact widening with a rank-two transpose, and admits `CUDA_R_16F` matrix descriptors. It does not equate an FP16 reduction with FP32 arithmetic, narrow arbitrary FP32 inputs, or remove the FP16 projection boundary before bias.

FP16-to-FP32 conversion is exact. A product of two finite FP16 values is representable in FP32; reduction reassociation and FMA are permitted by this experiment's numerical contract. The original materialized-FP32 alternative remains available to search. The patch does not add a general FP16-output cuBLASLt rule.

## Evidence Gates

1. Host saturation regression checks: native and widened alternatives for normal/transposed weights; reject result-only widening and narrowing arbitrary FP32 inputs.
2. H100 GEMM: six applicable independent fixtures, including the product/accumulation probe; bias cases add a seventh projection-rounding fixture; inspect selected dtypes and Nsight machine instructions.
3. Screen all ten existing workload candidates with expanded search (16 generations × 32 population, up to five timing trials (early stopping is permitted)). These are budgets, not unique valid-plan counts.
4. Deep captures for the useful cases: before/after e-graphs, pruning, selected and alternative plans, generated CUDA, typed library descriptors, independent finalist validation, uninstrumented timing, Nsight Systems timeline, Nsight Compute resources and SASS.
5. Paired upstream/patched and default/expanded measurements where informative, with GPU identity and all search/refusal costs retained.

All fixtures, tolerances, shapes, and reference calculations remain unchanged. The expanded budget is an upper bound: duplicate plan fingerprints do not consume population slots, and each generation has a bounded number of sampling attempts. Record `plans_profiled`, not merely generations × population. Luminal measures graph/implementation candidates; its cuBLASLt wrapper requests one heuristic algorithm, so this is not an explicit tile-size search within the library. Record upstream failure separately from corrected-adapter baseline and patched results. Passing finite tests does not prove all shapes or special values.

## Reproduction

```sh
# From the research repository root, with the pinned clean Luminal checkout:
git -C .deps/luminal apply --check ../../experiments/cuda_case/patches/luminal-fp16-f32-compute.patch
git -C .deps/luminal apply ../../experiments/cuda_case/patches/luminal-fp16-f32-compute.patch
cargo test --manifest-path experiments/cuda_case/luminal/Cargo.toml --test fp16_admission -- --nocapture
.venv-modal/bin/modal run experiments/cuda_case/luminal_fp16_app.py --phase gate --workloads gemm --minutes 8 --case-seconds 180
```

The CPU test expects the patch applied to `.deps/luminal`. The dedicated Modal image applies the saved patch to a clean pinned checkout and builds both upstream and patched binaries. Every run retains the exact patch hash, binary hashes, source revision, harness, and environment. Existing upstream campaign evidence is immutable.

The H100 function has zero minimum containers, one maximum container, no retries, and a two-second scale-down window. Raw evidence is checkpointed to `tensor-program-h100-evidence`; packaging runs on CPU through `review_app.py`. Verify no active apps or containers at completion.

The cuBLASLt datatype/compute combination is documented by [NVIDIA CUDA 12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublasltmatmul).

## Retained-Plan Replay

The final walkthrough uses a validated winner from the expanded transposed-weight trace. `prepare_luminal_replay.py` derives its producer choices from the saved search outcome and verifies that rank one reproduces the selected plan. The saved input is in `replay-input/weight_view`; the runtime refuses a changed plan, typed mapping, or generated CUDA source.

```sh
.venv-modal/bin/modal run experiments/cuda_case/luminal_replay_app.py
cargo run --quiet --manifest-path experiments/cuda_case/luminal/Cargo.toml --bin inspect_replay -- experiments/cuda_case/replay-input/weight_view
```

The replay image extends the recorded Modal image `im-g96azfHcjBNw3SJlVOOApS`, built by `luminal_fp16_app.py`; that base recipe and patch are retained if the image must be rebuilt. Replay executes the retained plan through `CudaExecutable.execute_external`, while the paired search runs use `CudaRuntime.execute`. Their host timings have different wrapper scopes and must not be pooled. The library heuristic is resolved from the retained descriptors on the replay device; actual library kernel names are checked across profilers.

Package multiple runs with `review_batch.py RUN_ID ...` to use one CPU app and avoid app-creation bursts. No GPU is requested by that script.
