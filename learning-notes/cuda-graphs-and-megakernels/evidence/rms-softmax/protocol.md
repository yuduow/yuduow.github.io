# Methods: one RMS-scaled softmax workload

## Operator and inputs

The input is a 2D tensor with 131,072 elements per row. The output is `softmax(x / sqrt(mean(x*x) + 1e-5))`, using subtraction of the row maximum before the exponential. This differs from ordinary softmax. The tested finite-input contract bounds the input magnitude at 1,024. The source and configuration retain the actual input cases, compiler flags and fixed validation tolerances.

The workload has two sets of row-wide reductions in the tested decomposition: sum of squares and maximum before exponentiation, then sum of exponentials before normalization. This decomposition is an implementation choice, not a proof that every correct algorithm needs three phases.

The NumPy function in the article defines the mathematical operator; the timed kernels are CUDA C++. Input and output tensors use contiguous row-major float32 storage, and CUDA reductions use float arithmetic. Numerical agreement is tested within the fixed tolerances, not required to be bitwise identical to NumPy. The positive inverse RMS allows subtraction of the unscaled row maximum before multiplication.

Warm timing uses the deterministic hash-generated `random` fixture, seed 41, with values in [-8, 8] at 1/1,024 increments. The same device-resident input is replayed between warmup and measurement. The broader correctness fixtures are not all benchmarked input distributions. The [frozen input generator](code/bench.cu) specifies these details.

## Platform and compilation

The two measurement sessions used separately provisioned NVIDIA H100 80GB HBM3 GPUs with distinct device UUIDs, each with 132 SMs and 50 MiB L2. The archived labels “allocation 1” and “allocation 2” refer to these GPU sessions, not CUDA memory allocation or single timing samples. The image pins CUDA 12.8.1 development tools on Ubuntu 24.04 and cuDNN 9.10.2.21. Saved device records report runtime 12080 and driver 13030; these are the API's numeric version encodings. The build uses `nvcc -O3 -std=c++17 -arch=sm_90 --fmad=false`; `--use_fast_math` is not enabled. The [image definition](code/modal_app.py), [build script](code/build.py) and device records below preserve the recorded environment.

## Implementations and controls

- Unfused baseline: nine CUDA kernels, direct stream launches and capture of the same nine nodes.
- Local fusion, three kernels: sum of squares and maximum in one tiled pass, exponentials and their tile sums in a second, normalization in a third. Both materialized and recomputed exponential paths are tested. Recompute is the pilot-selected local family at all four shapes.
- Local fusion, two kernels: each output tile repeats whole-row statistics to remove one boundary. Both materialized and recomputed exponential paths are retained.
- Local fusion, one kernel: each output tile repeats whole-row statistics and normalization work. A full-row tile was also considered in the pilot. These controls show that launch count alone does not determine performance.
- Megakernel with grid-wide barriers: one cooperative block owns a tile across two grid synchronizations, reusing one shared-memory buffer for the input and then exponentials.
- Megakernel with row-scoped barriers: the same storage strategy with device-scope release/acquire row-scoped barriers. Cooperative launch admission bounds the grid to the compiled residency budget.
- Same-binary controls: retain input values and then exponentials in the same shared-memory buffer, reload the input, write and reload exponentials through global memory, or both. Runtime flags keep the compiled code, resources, tile geometry and launch count fixed. Added accesses include shared-memory staging; this is not an isolated HBM-latency experiment.
- RMS preprocessing plus cuDNN is a diagnostic control. The preprocessing was not exhaustively tuned; the article makes no library-speedup claim.

Tile widths and thread counts were tuned per family in the pilot. Local kernels were not padded to reserve the retained path's larger shared-memory allocation. The final selection was frozen before two separate H100 80 GB sessions were measured. The megakernel with grid-wide barriers is selected at one row; the megakernel with row-scoped barriers at 16, 32 and 64 rows. Both megakernel variants use two cooperative launches at 64 rows. Frozen source, configuration and exact resources are in the [source listing](code/) and measurement metadata below.

The source calls the selected local fusion variant `local_recompute`, the megakernel with grid-wide barriers `retained`, and the megakernel with row-scoped barriers `row_retained`. These identifiers remain unchanged in archived code and data.

## Timing

Warm measurements use 21 shuffled rounds per variant, ten warmup invocations per variant per round, and batches of 32. The two sessions use the same frozen code, selected configurations and input fixture; the shape order is reversed in the second session and variant order is shuffled independently. Detailed results report each session separately. Their quantiles describe variability across rounds, not confidence intervals across a GPU population.

The main article uses the arithmetic mean of the two session medians for each shape, variant and metric. Whiskers span those two medians; they are neither within-session percentiles nor confidence intervals. Speedups divide these average latencies rather than averaging speedup ratios. For the retention controls, each session first computes the median of round-paired latency differences; the figure then averages those two medians and shows their range. This is a descriptive summary of two sessions, not a fleet-wide performance estimate. Raw samples, per-session p10/median/p90 and [figure provenance](figure-provenance.json) remain available.

- Host completion time: one invocation followed by stream synchronization.
- Host enqueue time: submission of a batch, divided by batch size; excludes the completion wait.
- Host batch completion time: submission of 32 invocations, recording the stop event, and waiting for that event, divided by batch size. The start-event record occurs before this host timer starts.
- CUDA event interval: a bracketed batch divided by batch size. Host supply gaps can appear inside this interval; it is not pure kernel compute time.

Memory allocation, transfers, compilation, graph capture/instantiation and profiler runs are outside warm timing. Capture/instantiate/first-replay records are retained in variant metadata. Separate external-eviction diagnostics touch a buffer of at least 256 MiB before each invocation, outside the interval, for 15 rounds. This does not guarantee that all cache state is cold.

## Validation and provenance

The original independent CPU audit compared direct and graph outputs and intermediate tensors with references computed from saved inputs in FP64, using fixed tolerances. It also checked row sums, nonnegativity, changing-input replays, trace dependencies, sanitizer results, frozen hashes and archive readback. This website's [fresh recheck](validation-recheck.json) explicitly separates locally repeated checks from those archived full-size checks.

The [source protocol](code/PROTOCOL.md), [configuration](code/config.json), [run summary](data/summary.json), [GPU ledger](data/gpu-ledger.json) and [freeze manifest](data/frozen-manifest.json) retain the original acceptance rules. The primary 16-row 1.25× event-speedup target was not achieved; numerical correctness and evidence requirements passed. A target miss is not a validation failure and is not concealed by selecting a different headline shape.

[Session 1 metadata](data/measurement-1/data/metadata.jsonl) · [Session 2 metadata](data/measurement-2/data/metadata.jsonl) · [Session 1 device](data/measurement-1/data/device.json) · [Session 2 device](data/measurement-2/data/device.json)
