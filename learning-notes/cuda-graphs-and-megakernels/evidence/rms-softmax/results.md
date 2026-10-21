# Results: RMS-scaled softmax on H100

[Complete timing tables](full-results/) include every measured variant in both studies, every shape, both GPU sessions, direct launches and cache-eviction diagnostics. Archived labels A1/A2 or allocation 1/2 mean separate GPU measurement sessions, each with 21 warm rounds per variant. Download [all raw samples](full-results/raw-timing-samples.csv) or [p10/median/p90 quantiles](full-results/all-timing-quantiles.csv). The [methods](protocol.html) define each metric and the [claim audit](claim-audit.html) sets the interpretation limits.

## Article comparison

All values below are per-session medians in microseconds. The main article averages the two medians and shows their range; these tables preserve the individual sessions. Each input row has 131,072 elements. Local fusion and megakernel configurations were frozen from pilot results. CUDA Graphs, local fusion and megakernel columns all use graph replay; the unfused baseline uses direct launches. The first four timing columns report host completion time; the last two report the CUDA event interval.

| Rows | Session | Unfused baseline | CUDA Graphs | Local fusion | Megakernel | Local fusion (event) | Megakernel (event) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 37.031 | 28.157 | 16.557 | 14.881 | 11.999 | 10.196 |
| 1 | 2 | 36.789 | 29.924 | 17.405 | 15.269 | 12.376 | 10.152 |
| 16 | 1 | 44.958 | 39.171 | 19.982 | 19.201 | 15.313 | 14.454 |
| 16 | 2 | 44.393 | 38.431 | 19.908 | 19.242 | 14.782 | 14.149 |
| 32 | 1 | 69.855 | 64.692 | 27.904 | 25.683 | 23.275 | 19.007 |
| 32 | 2 | 69.990 | 64.289 | 27.170 | 24.643 | 22.050 | 19.525 |
| 64 | 1 | 143.703 | 137.057 | 62.174 | 55.701 | 56.424 | 49.479 |
| 64 | 2 | 143.014 | 134.625 | 61.377 | 55.188 | 55.229 | 48.102 |

## What the results establish

The primary 16-row case achieves 1.045–1.059× CUDA event interval speedup and 1.035–1.041× host completion speedup over the fastest tested local family. The predeclared 1.25× primary event target was missed. Speedups are ratios of per-session medians, not pooled measurements or a claim of statistical significance.

Recomputation beats materialization in both sessions at all four measured shapes, for both host completion time and CUDA event intervals after warmup. The one-/two-kernel local alternatives are slower in these trials. This is evidence about the tested candidates, not global optimality of the three-kernel approach.

The pilot-selected row-scoped path is not uniformly faster than the grid-wide variant on held-out measurements. At 16 rows, for example, grid-wide event medians are 14.410 / 14.139 µs and row-scoped medians are 14.454 / 14.149 µs. Both remain visible in the full tables; the article does not reselect winners after measurement.

The grid-wide and row-scoped kernels have different [register counts and active-block limits](data/resource-summary.json), and use different tile widths at 32 and 64 rows. Their timing difference does not isolate the effect of barrier scope.

At 32 rows, the row-scoped kernel uses 32 registers per thread and 16,640 dynamic shared bytes; its occupancy query permits up to eight active blocks per SM. Grid-wide uses 34 registers and 33,024 dynamic shared bytes, with a limit of six active blocks per SM. These are queried capacity limits, not measurements of actual block residency or utilization. Selected tile widths are 4,096 and 8,192 respectively. At 64 rows both megakernel variants need two launches, as recorded in the metadata.

The grid-wide same-binary retention controls add input reloads and/or writes and reloads of exponentials through global memory. Each session computes medians of paired per-round host differences, not differences of unpaired percentile endpoints. Input reload: 1.488 / 1.286 µs; exponential writes and reloads: 2.893 / 2.854 µs; both: 3.993 / 3.910 µs. The figure averages each pair of medians and shows their range. The source and compiled-resource records support the fixed-binary comparison. These controls include shared staging and do not isolate physical HBM traffic or explain the complete cross-implementation gap.

Direct launch of a single cooperative kernel can outperform graph replay here. The selected row-scoped 16-row variant has ordinary host completion medians of 18.434 / 18.774 µs, versus graph medians of 19.201 / 19.242 µs. The four-way graph plot is a controlled comparison, not a claim that the last graph-replayed point is the fastest possible execution mode.

## Validation scope

The fresh CPU pass recomputed 544 warm timing metric groups from raw records, checked 208 available saved small-array outputs against independent references, and rechecked all 66 dependency traces in the final study. Maximum fresh small-array row L1 error was 9.15e-08. The [machine-readable recheck](validation-recheck.json) names every verified source and distinguishes these checks from the archived full-size numerical audit.

The full-size audit, intermediate checks, changing-input stress, sanitizers and profiler records are summarized in the [accepted run summary](data/summary.json). All accepted outputs met the original fixed tolerances. Full binary arrays and raw profiler databases remain archived. The [readback record](data/archive-readback-verification.json) verifies the saved archives; the [teardown record](data/gpu-teardown.json) records the stopped experiment apps. No new GPU allocation was started for this post update.

The initial grid-wide study is retained in a separate section of the full tables, not pooled into the final study's error bars. Setup/first-replay timings are in metadata. GPU execution-ledger seconds are not billing time.
