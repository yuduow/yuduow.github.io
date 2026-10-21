# CUDA Graphs and fused kernels experiment

Protocol fixed before GPU execution on October 18, 2026. These examples accompany the [corrected article](../../posts/cuda-graphs-megakernels/article.md). They test mechanisms, not LLM serving performance.

## Example A: submission and intermediate storage

For contiguous FP32 x, compute three explicitly rounded operations:

```text
a[i] = round32(x[i] + 1)
b[i] = round32(a[i] * a[i])
y[i] = round32(b[i] * 0.5)
```

Compare a 2 by 2 design: three separate kernels or one locally fused kernel; ordinary stream launches or replay of a captured CUDA Graph. Captured kernel node counts must be 3 and 1. Inputs, output pointers, arithmetic, stream, and 256-thread block geometry are fixed within each comparison. Fusion keeps a and b in thread-local values; there are no reductions or inter-thread dependencies. This is ordinary fusion, not a model-scale persistent runtime.

N = 256, 32,768, 1,048,576, and 16,777,216 elements. The largest input alone occupies 64 MiB; record actual L2 capacity. Three kernels perform 24N logical bytes of array reads/writes; the fused kernel performs 8N. These are algorithmic global-memory bytes, not measured HBM bytes. Repeated inputs and warm caches are deliberate. Hardware DRAM counters, if available, determine actual device-memory transfers.

Validate all four variants on three deterministic fixtures before timing and again on a held-out fixture afterward. Inputs are bounded dyadic FP32 numbers, enabling exact comparison with an independent NumPy oracle that rounds each operation. Preserve input/output binary files and hashes. No tolerance changes after failure.

## Example B: tile dependencies and ragged work

Each job owns 256 uint32 values. Stage A applies a per-job number of iterations of `v = 1664525*v + 1013904223` modulo 2^32. Stage B applies 512 iterations of `v = 22695477*v + 1`. A[j] is needed only by B[j]. Most A jobs take 256 iterations; every eighth takes 8,192. Also run a balanced 1,024-iteration control. Use 4*S + floor(S/2) + 3 jobs, where S is the recorded SM count, deliberately leaving a partial nominal wave.

The balanced and ragged cases have different total A work. Compare implementations within each case; their difference is not an estimate of the cost of imbalance alone. For the accepted 132-SM run, 597 jobs imply 748,032 ragged A iterations summed across jobs versus 611,328 balanced iterations, before multiplying by 256 lanes.

Compare six variants: separate A/B kernels, one fused CTA per job, and a persistent atomic work queue with S worker CTAs; each under stream and graph submission. The queue counter is reset by a device memset on every invocation, including graph replay; record this as a non-kernel graph node and a real overhead. All variants deliberately store and reload A through a volatile global-memory array. They therefore retain the intermediate even with one kernel. There are no cross-CTA producer/consumer waits: a worker completes both stages for its job. A block barrier separates stages. This intentionally simple persistent queue is not an interpreter or a reproduction of Hazy/MPK.

Reserve just over half the device's per-SM shared memory per CTA in all three implementations and verify one active CTA per SM with CUDA's occupancy API. The unused reservation creates a controlled one-slot scheduling example; it is not an optimized implementation. Do not infer physical SM placement from block indices. Record `%smid` in traces.

Validate all variants against a separately derived affine-composition oracle modulo 2^32, including changed seeds after timing. Save complete intermediate/output arrays. In a separate instrumented run, record `%globaltimer` stage intervals, job, worker, and actual SM for every job. Verify complete job coverage, A-before-B for each job, absence of B-before-all-A in the separate variant, and measure whether fused variants exhibit B-before-unrelated-A-completion. Instrumented durations do not enter performance tables. Queue scheduling, tail behavior, and speed remain empirical outcomes.

## Measurement and artifacts

- One exact Modal `H100!`, one container, no retries, no warm pool, 600-second function limit. Sequential commands each have narrower timeouts. CPU image construction precedes allocation. Upload only these public experimental sources and the existing public NVIDIA profiler installer.
- CUDA 12.8.1 development image; nvcc `-O3 -std=c++17 -arch=sm_90 -lineinfo --fmad=false`. Save compiler output, PTX, SASS, executable, source snapshots, GPU identity, driver/runtime versions, resources, and pre/post device state.
- Ten warmups per variant. Twenty-one rounds with shuffled variant order, fixed host seed. Two distinct measurements: one synchronized host invocation per round (no timing events), and a batch of 128 invocations (32 for largest A) bracketed by CUDA events. Save host enqueue time, host batch completion time, and event interval per invocation. Event intervals include any GPU idle time waiting for host submission; they are not isolated kernel durations. Never add enqueue and GPU times.
- Graph capture, instantiation, and first replay occur outside timing. Save setup timings and graph DOT files. No allocation, host/device input copy, validation, or disk IO inside timed regions. B's counter reset remains included.
- Keep all samples. Report medians and p10/p90; paired ratios compare each round's same-allocation results. No claims of cross-machine significance from these rounds.
- Separate Nsight Systems captures establish launch topology. Separate Nsight Compute samples for large A inspect DRAM traffic and resources if permissions allow. Keep errors instead of inventing counters. Full reports remain private because profiler files can contain process metadata. Export only derived CUDA activity and metric data for illustrations.
- Archive on a dedicated Modal Volume, download locally, independently validate arrays and traces, and hash every artifact. Record teardown. No deployment or continuous GPU service.

Run after reviewing this protocol:

```sh
.venv-modal/bin/modal run experiments/graphs_megakernels/modal_app.py
```

Re-analyze an existing download without GPU execution:

```sh
.venv-modal/bin/python experiments/graphs_megakernels/analyze.py experiments/graphs_megakernels/artifacts/RUN_ID
```

Suggested illustrations: 2 by 2 latency panels across N; logical bytes beside measured DRAM bytes; actual per-SM job intervals for B; graph DOT nodes beside emitted kernel entry points. A measured lack of speedup is a valid outcome. Neither example establishes PDL performance, cross-SM semaphore correctness, LLM bandwidth utilization, or production megakernel performance.

## Completed evidence

The accepted run is `20261018T211922Z`: [results](RESULTS.md), [full local artifact directory](artifacts/20261018T211922Z/), [local verification](artifacts/20261018T211922Z/verification.json), and [remote raw manifest](artifacts/20261018T211922Z/manifest.json). It passed 272 exact array checks, 24 trace checks, and all 450 original file hashes. The 588 timing samples represent 28 configurations with 21 rounds each. Figures are generated from saved data, not hand-drawn performance traces.

Full evidence and the archive also remain on `cuda-graphs-megakernel-toy-evidence`. The earlier `20261018T211809Z` run passed preflight, then failed before timing because a copied executable occupied the intended benchmark directory. It is retained remotely and excluded from results. Two earlier CPU image compilation failures are retained in [run notes](run-notes.json). No failed run is pooled with accepted measurements.

Profiling succeeded. Systems' default graph mode records complete graph activities rather than node durations; named Compute samples and graph dumps supply the per-node implementation evidence. Compute's default cache flushes differ from the warm-cache benchmark. The results report describes these boundaries and the broader scope of an unused first-replay metadata field.
