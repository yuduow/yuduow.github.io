# RMS-scaled softmax: evidence and reproduction

The article now uses one constructed workload through CUDA Graphs, local fusion and a megakernel that retains tiles across row-wide dependencies. The final study is `20261020T023055Z`; the initial grid-wide study `20261020T022150Z` is retained separately in the full appendix.

- [Results and interpretation](results.html) and [all timing tables](full-results/).
- [Experiment methods](protocol.html) and [claim audit](claim-audit.html).
- [Every timing sample](full-results/raw-timing-samples.csv) and [every timing quantile](full-results/all-timing-quantiles.csv), including direct launches, slower candidates and external-eviction diagnostics.
- [Frozen experiment source](code/) and [pilot-selected configurations](data/selected-configs.json).
- [Fresh CPU recheck](validation-recheck.json), [figure values and hashes](figure-provenance.json) and [selected website bundle manifest](bundle-manifest.json).
- [GPU teardown record](data/gpu-teardown.json), [archive readback verification](data/archive-readback-verification.json) and [archive index](data/archive-index.json).

This website carries a selected evidence bundle. Full-size binary arrays, raw profiler captures, sanitizer logs and complete experiment manifests remain in the verified research archives; they are not all included here. No new GPU work was needed for this editorial update.
