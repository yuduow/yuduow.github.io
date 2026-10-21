# Softmax submission fusion and dependency scheduling

Protocol fixed before GPU execution on October 19, 2026. This replaces the earlier proposal to use a hand-written MLP as the immediate next experiment. An MLP introduces Tensor Core implementation quality, library selection, and additional numerical policies before it cleanly tests scheduling. These two softmax experiments share a familiar numerical operation while testing different mechanisms. A small task scheduler is an illustration of megakernel mechanisms, not a benchmark of a model-scale megakernel or an inference engine.

## Questions and controls

1. Can CUDA Graph replay reduce submission/completion cost for identical kernels?
2. What changes when a row's intermediates stay within one block?
3. Can a consumer start after its own producers finish, before unrelated producers finish? Does that actually improve completion latency after scheduling costs are included?

FP32 is an arithmetic policy, not a separate conversion workload. Compute stable softmax: `m = max(x)`, `e = exp(x-m)`, `z = sum(e)`, `y = e/z`. Custom kernels use the same `expf`, reduction helpers, and compiler policy (`--fmad=false`, no fast-math flag). cuDNN is checked numerically rather than required to be bitwise identical. Finite inputs only; NaN, infinity, and empty rows are outside this experiment.

## Experiment A

Compare four explicit stages (row maximum; subtract and exponentiate; row sum; divide), one fused row kernel, and cuDNN 9.10.2.21 `CUDNN_SOFTMAX_ACCURATE` in instance mode. Test each with ordinary stream calls and graph replay. All calls originate in the same C++ harness, use the same nonblocking stream and fixed GPU pointers, and exclude allocation/transfers from timing. Warm cuDNN before capture. Record actual graph node types/counts rather than assuming its API call is a single kernel.

The four-stage code is an educational decomposition, not an optimized-library baseline. Its array passes and the fused kernel's local reuse explain a mechanism; cache traffic and latency must be measured separately. The cuDNN reference makes implementation quality visible.

Timed `(rows, columns)` pairs are `(1,128)`, `(32,1024)`, `(4096,128)`, `(4096,1024)`, `(4096,4096)`, `(16384,1024)`. An additional `(7,781)` case checks masks. This smaller fixed sweep spans tiny inputs and 64 MiB inputs without generating an unnecessarily large artifact bundle. Read actual L2 size from the device. Fused rows use 256 threads and up to 16 values per thread; inspect emitted code and local-memory usage for spills.

## Experiment B

Packed rows are divided into 1024-element tiles. A computes each tile's maximum `m_t` and shifted exponential sum `z_t`. B combines a row's tile summaries stably: `m = max(m_t)` and `z = sum(z_t * exp(m_t-m))`. C rereads its input tile and writes `exp(x-m)/z`. The graph has a genuine cross-block fan-in (all A tiles for one row feed its B task) and fan-out (B feeds that row's C tiles).

Compare the same A/B/C helpers and intermediate arrays in:

- Three ordinary kernels, with full A→B→C phase boundaries.
- One cooperative persistent kernel retaining global phase barriers.
- One cooperative persistent kernel consuming a ready-task FIFO, publishing B only when its row's A tiles finish and C only after that B finishes.

Measure all three under ordinary stream submission and CUDA Graph replay. The task queue needs a separate initialization kernel; include it in both timing and graph node counts. The phased persistent kernel does not pay an unnecessary queue reset. Thus the two persistent paths share the compiled kernel and its resource allocation but have different scheduling/reset costs; their timing difference is not a pure dependency-latency estimate.

Use 128, 512, and 2048 rows. Balanced rows have length 8192. Ragged rows contain one length-36864 row and seven length-4096 rows per eight, then a fixed seeded permutation. Both distributions contain exactly 8192 elements and eight tiles per row on average. They have equal total input/output elements, tile work, and row tasks. All implementations use the same packed layout and row order within a case. Report each case and do not select only favorable shapes.

No unused shared-memory reservation restricts occupancy. Set worker count from the cooperative kernel's occupancy API result and recorded SM count, capped by producer task count. The phased control uses cooperative grid barriers. The DAG path uses device-scope acquire/release atomics: A writes its summary before an acquire/release row counter; the last producer publishes B through a release-ready flag; consumers acquire that flag. B publishes its C tasks after its result writes. Workers prefer the ready B/C FIFO; when it is empty, they atomically claim another A tile. The FIFO holds exactly all B+C tasks and is freshly initialized for every invocation. Termination requires every C tile to finish. Verify exact queue head/tail, A-claim and C-completion counts, and per-task trace visits. Never assume a block index identifies an SM.

Pre-execution review amendment: the initial draft put every A task into the FIFO before any B/C tasks. That would delay ready consumers behind nearly all producer tasks and weaken the intended scheduling demonstration. The ready-consumer preference above was adopted before compilation or any GPU execution. No measured result informed this change.

## Correctness and measurement

Each shape is preflighted on two deterministic fixtures before its timings are admitted. Benchmark execution then uses two different fixture IDs, including a changed input after timing, with captured pointers fixed. Save inputs and every variant's output. Fixtures include bounded random values, constant rows, a dominant element, and a large common offset. The independent CPU oracle uses float64 stable softmax on the exact saved float32 inputs. Require `abs(error) <= 1e-7 + 2e-5 * abs(reference)`, finite nonnegative outputs, and row sums within `2e-5` of one. Do not relax tolerances after a failure. Also require bitwise equality of the scheduling variants, which share the tile algorithm, and retain intermediate summaries for inspection.

Ten warmups precede 21 shuffled rounds. Record single-invocation host completion latency; host enqueue time for a batch of 32; batch completion; and CUDA-event interval per invocation. Event intervals include host-starvation gaps. Timings exclude compilation, allocation, transfers, graph setup, validation, and profiling. Record capture, instantiation, and first replay separately; keep graph inspection/DOT writing outside those intervals. Record a no-work stream-synchronization baseline separately; do not subtract it from results. Report medians, p10/p90, and paired per-round ratios from one GPU allocation, not population significance.

Trace builds are separate from performance builds. Record globaltimer start/end, actual SM ID, worker ID, and visit count for every task. Verify complete unique task coverage, B after all of its own A tasks, C after its own B, and full-phase ordering for the separate/phased controls. Count early B/C starts for the DAG without requiring them for acceptance. Trace intervals illustrate ordering, not SM utilization.

Use Nsight Systems node-level graph tracing on representative cases. Collect separate Nsight Compute counters for DRAM/L2 traffic, registers, local/shared memory, and occupancy when permitted. Preserve profiler errors and raw captures. Profiling is separate from timing; record its cache/replay policy and do not translate logical array bytes into HBM traffic. A lack of speedup or a slower task scheduler is a valid result.

## H100 and artifacts

H100 supports the required ordinary CUDA kernels, graph replay, cooperative grids, and device atomics. No Blackwell-only instruction or multi-GPU operation is needed. Request Modal `H100!` to prevent an automatic H200 upgrade, assert the device model, and save memory/SM/driver/runtime/clock information. This establishes behavior on the allocated H100, not portability of measured speedups to other GPUs.

Compile in the CPU image before allocation. Run one exact H100 with a 900-second hard limit, one container, zero retries, and no warm pool. Validate before timing, checkpoint each case to a dedicated Modal Volume, and stop on required failures. Hashing, archival, and report packaging run in a separate CPU function so they do not occupy the GPU. Full source, compiler output, executable, PTX/SASS, configuration, inputs, outputs, graph DOTs, raw timing JSONL, task CSVs, metadata, and profiler files remain in the Volume. Download a compact review bundle by default because local disk space is limited; include hashes and a precise list of omitted raw files.

Run: `.venv-modal/bin/modal run experiments/graphs_softmax/modal_app.py`.

Sources: [Triton softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html), [cuDNN softmax](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-ops-library.html#cudnnsoftmaxforward), [Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html), [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html), [Modal GPU selection](https://modal.com/docs/guide/gpu).
