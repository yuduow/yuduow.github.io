# Frozen protocol

For finite float32 X with abs(X)<=1024 and eps=1e-5:

    scale = 1 / sqrt(sum(X*X, axis=row) / N + eps)
    E = exp((X - max(X, axis=row)) * scale)
    Y = E / sum(E, axis=row)

N=131072; primary batch=16; other batches=1,32,64. Independent float64 oracle uses saved input bytes. Fixed element atol=1e-7, rtol=2e-5, row-sum and row-L1 <=2e-5; no fast math or FMA contraction. Structured fixtures and tails include zero, constant, outlier, common offset, near zero, and a 33-tile row. Output/input/intermediates, both direct and captured execution, changing-input graph replay, traces, and sanitizers are checked.

Nine-kernel baseline: square; tile energy; tile max; row RMS scale; row max; scaled exponentials; tile mass; row mass; normalize. Local3 combines energy/max in A, computes tile exponentials/mass in B, and normalizes in C; materialize or recompute E. All tiny row summaries are redundantly reduced per warp, avoiding an unnecessary row-owner phase. Local2 scans the entire row for energy/max per tile in A, then normalizes in B. Local1 independently scans the whole row twice per output tile, then emits that tile, with W=N as a full-row streaming control. RMS preprocessing plus cuDNN is an additional library-assisted control.

Retained: tile X to shared plus initial statistics; grid barrier; replicate tiny summaries, overwrite shared X with E plus tile mass; grid barrier; replicate mass summary and emit Y. Only two genuine grid barriers. Legal cooperative residency is queried from compiled occupancy and actual SM count; oversized batches use independent row slabs. W=131072 is allowed for local candidates and rejected for retained when beyond runtime opt-in shared capacity.

Same-binary runtime masks 0/1/2/3 add X reload, E global roundtrip, or both, at exactly the same geometry/resources. X reload includes a shared write; E roundtrip includes shared staging. These are controlled additions inside this binary, not claims of pure HBM savings or sole causes across different implementations.

Tune W=4096,8192,16384,131072 and T=64,128,256 independently per family and batch. Five rounds of 16 graph calls select pilot median completion; freeze before two allocation measurements. Primary comparator includes all local3/local2/local1 candidates. Warm steady state: 21 shuffled rounds, 10 warmups per variant/round, 32 calls per batch; record host completion, enqueue, batched completion, and CUDA event interval (which can include host supply gaps). Record cold setup separately, external cache-eviction diagnostic for batches16/64, and per-round paired deltas. Hypothesis: >=1.25x primary event-interval win on both allocations, not required for scientific completion.

Evidence: dependency traces for tile A/B/C (local2 has A/C only), sanitizer memcheck/racecheck/synccheck on tail and primary, Nsight Systems node counts, Nsight Compute traffic/instructions/resources. Trace timestamps are instrumented ordering evidence, not latency measurements. Keep actual HBM and L2 counters separate from logical traffic. Capture all failures, preserve immutable sources, checksum full raw archive and selected review files, verify stopped apps and zero containers. Maximum recorded GPU-phase time including failures:900 seconds. Build and package on CPU.

## Follow-up frozen before collection

Same workload, primary shape, tolerances, local controls and measurement method as run 20261020T022150Z. Add row_retained: same two dependencies, device-scope release/acquire row barriers instead of whole-grid barriers. Require cooperative admission; no ordinary launch. Every CTA publishes all initial statistics before its row barrier, and all tile mass / row debug values before the second. The last arriving CTA acquires the atomic RMW release sequence, resets arrival count, then releases a new epoch. Waiters acquire that epoch. CTA barriers publish all threads before arrival and after release. Separate counters/epochs for both phases and every row persist safely across sequential graph replays; validation checks arrival counters return to zero. Sanitizers, changing-input replay and traces cover the new path; traces assert row dependencies, without requiring unrelated rows to wait. Both megakernel strategies are independently tuned and the pilot chooses mega_best; preserve/report both, even if row scope loses. The parent study and its elapsed GPU phases count toward the same 900-second budget. No changes to input arithmetic or primary local comparison.
