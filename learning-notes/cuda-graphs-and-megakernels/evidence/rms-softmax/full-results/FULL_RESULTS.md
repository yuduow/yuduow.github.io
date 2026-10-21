# Full RMS-scaled softmax results

Both frozen studies are included. Every row has 131,072 values. Values are microseconds per invocation.

Within each study, A1 and A2 identify two separately provisioned GPU measurement sessions, not CUDA memory allocations or individual timing samples. Each session has 21 warm rounds per variant. These tables keep the sessions separate; the main article averages the two session medians from the final study, with whiskers spanning those medians.

Warm results: 21 shuffled rounds, 32 invocations per timed batch. Host completion times one invocation plus stream synchronization; event intervals time batches and can contain host supply gaps. Host enqueue excludes completion; host batch completion amortizes the host wait across the batch. Allocations, transfers, compilation, graph setup and profiling are outside these warm measurements.

The baseline is a deliberately explicit nine-kernel educational decomposition. It is not a stock-framework baseline. Local and retained paths also use graph replay unless labeled ordinary launches. Local3 means three kernels; Local2/Local1 reduce launches by repeating work. The cuDNN pipeline includes the benchmark RMS preprocessing and is not exhaustively tuned.

[Every timing sample](raw-timing-samples.csv) · [Every measured p10/median/p90](all-timing-quantiles.csv)

The saved studies retain setup/capture/instantiate/first-replay records, device records, compiled resources, pilot candidates, traces and profiler counters. This website includes selected evidence; complete records remain in the research archives. The CSVs here contain all steady-state and external-eviction timing samples, including empty synchronization calls. No new GPU run was performed.

## row_scope_followup: 20261020T023055Z

[Full evidence and figures](/Users/1duo/Work/YuduoTechBlog/experiments/graphs_scaled_softmax_row/artifacts/20261020T023055Z/report/READOUT.md)

### 1 row(s): all warm variants

| Variant | Kernels | Event A1 | Event A2 | Host completion A1 | Host completion A2 |
|:---|---:|---:|---:|---:|---:|
| Unfused, ordinary launches | 9 | 32.164 | 31.334 | 37.031 | 36.789 |
| Unfused + graph | 9 | 23.372 | 24.623 | 28.157 | 29.924 |
| Local3: materialize E + graph | 3 | 13.914 | 13.845 | 18.486 | 18.914 |
| Local3: recompute E + graph | 3 | 11.999 | 12.376 | 16.557 | 17.405 |
| Local3: recompute E, ordinary launches | 3 | 13.707 | 13.597 | 18.589 | 19.147 |
| Local2: repeat full-row stats, materialize E + graph | 2 | 21.360 | 21.363 | 25.976 | 26.458 |
| Local2: repeat full-row stats, recompute E + graph | 2 | 19.522 | 19.622 | 24.102 | 24.741 |
| Local1: repeat full-row work + graph | 1 | 42.469 | 42.616 | 47.129 | 47.708 |
| Full-grid retained + graph | 1 | 10.196 | 10.152 | 14.881 | 15.269 |
| Full-grid retained, ordinary launches | 1 | 9.233 | 9.258 | 14.168 | 14.721 |
| Row-scoped retained + graph | 1 | 11.008 | 11.012 | 15.674 | 16.087 |
| Row-scoped retained, ordinary launches | 1 | 9.819 | 9.844 | 14.776 | 15.351 |
| Same-binary control: retain X and E | 1 | 10.306 | 10.201 | 14.969 | 15.432 |
| Same-binary control: reload X | 1 | 10.896 | 10.865 | 15.610 | 15.934 |
| Same-binary control: roundtrip E | 1 | 11.077 | 11.005 | 15.805 | 16.127 |
| Same-binary control: reload X and roundtrip E | 1 | 11.675 | 11.659 | 16.320 | 16.804 |
| RMS preprocessing + cuDNN + graph | 3 | 144.551 | 146.912 | 149.324 | 151.983 |
| Empty stream synchronization | 0 | — | — | 1.356 | 1.438 |

| Variant | Host enqueue A1 | Host enqueue A2 | Host batch completion A1 | Host batch completion A2 |
|:---|---:|---:|---:|---:|
| Unfused, ordinary launches | 12.564 | 15.545 | 32.360 | 31.530 |
| Unfused + graph | 1.532 | 1.833 | 23.555 | 24.813 |
| Local3: materialize E + graph | 1.518 | 1.816 | 14.115 | 14.035 |
| Local3: recompute E + graph | 1.514 | 1.830 | 12.199 | 12.569 |
| Local3: recompute E, ordinary launches | 4.124 | 5.037 | 13.889 | 13.790 |
| Local2: repeat full-row stats, materialize E + graph | 1.515 | 1.814 | 21.561 | 21.557 |
| Local2: repeat full-row stats, recompute E + graph | 1.519 | 1.816 | 19.712 | 19.815 |
| Local1: repeat full-row work + graph | 1.514 | 1.809 | 42.665 | 42.808 |
| Full-grid retained + graph | 1.527 | 1.825 | 10.396 | 10.348 |
| Full-grid retained, ordinary launches | 1.449 | 1.692 | 9.411 | 9.455 |
| Row-scoped retained + graph | 1.524 | 1.832 | 11.214 | 11.211 |
| Row-scoped retained, ordinary launches | 1.426 | 1.696 | 9.998 | 10.035 |
| Same-binary control: retain X and E | 1.522 | 1.814 | 10.507 | 10.392 |
| Same-binary control: reload X | 1.536 | 1.820 | 11.092 | 11.058 |
| Same-binary control: roundtrip E | 1.516 | 1.825 | 11.279 | 11.203 |
| Same-binary control: reload X and roundtrip E | 1.525 | 1.812 | 11.877 | 11.862 |
| RMS preprocessing + cuDNN + graph | 1.513 | 1.794 | 144.754 | 147.106 |

### 16 row(s): all warm variants

| Variant | Kernels | Event A1 | Event A2 | Host completion A1 | Host completion A2 |
|:---|---:|---:|---:|---:|---:|
| Unfused, ordinary launches | 9 | 39.767 | 39.218 | 44.958 | 44.393 |
| Unfused + graph | 9 | 34.258 | 32.807 | 39.171 | 38.431 |
| Local3: materialize E + graph | 3 | 16.783 | 16.146 | 21.417 | 21.210 |
| Local3: recompute E + graph | 3 | 15.313 | 14.782 | 19.982 | 19.908 |
| Local3: recompute E, ordinary launches | 3 | 16.391 | 16.385 | 21.485 | 21.909 |
| Local2: repeat full-row stats, materialize E + graph | 2 | 30.705 | 28.958 | 35.384 | 34.068 |
| Local2: repeat full-row stats, recompute E + graph | 2 | 26.070 | 26.054 | 30.835 | 31.165 |
| Local1: repeat full-row work + graph | 1 | 53.383 | 52.598 | 58.145 | 57.702 |
| Full-grid retained + graph | 1 | 14.410 | 14.139 | 19.200 | 19.341 |
| Full-grid retained, ordinary launches | 1 | 13.398 | 13.162 | 18.493 | 18.720 |
| Row-scoped retained + graph | 1 | 14.454 | 14.149 | 19.201 | 19.242 |
| Row-scoped retained, ordinary launches | 1 | 13.405 | 13.220 | 18.434 | 18.774 |
| Same-binary control: retain X and E | 1 | 14.386 | 14.167 | 19.251 | 19.306 |
| Same-binary control: reload X | 1 | 15.931 | 15.357 | 20.708 | 20.469 |
| Same-binary control: roundtrip E | 1 | 17.356 | 16.952 | 22.134 | 22.157 |
| Same-binary control: reload X and roundtrip E | 1 | 18.446 | 18.042 | 23.297 | 23.164 |
| RMS preprocessing + cuDNN + graph | 3 | 152.003 | 151.210 | 157.144 | 156.491 |
| Empty stream synchronization | 0 | — | — | 1.370 | 1.430 |

| Variant | Host enqueue A1 | Host enqueue A2 | Host batch completion A1 | Host batch completion A2 |
|:---|---:|---:|---:|---:|
| Unfused, ordinary launches | 12.626 | 15.142 | 39.978 | 39.414 |
| Unfused + graph | 1.544 | 1.805 | 34.444 | 33.002 |
| Local3: materialize E + graph | 1.529 | 1.804 | 16.987 | 16.343 |
| Local3: recompute E + graph | 1.530 | 1.796 | 15.518 | 14.980 |
| Local3: recompute E, ordinary launches | 4.126 | 4.947 | 16.574 | 16.575 |
| Local2: repeat full-row stats, materialize E + graph | 1.517 | 1.778 | 30.910 | 29.156 |
| Local2: repeat full-row stats, recompute E + graph | 1.522 | 1.787 | 26.271 | 26.248 |
| Local1: repeat full-row work + graph | 1.520 | 1.771 | 53.595 | 52.795 |
| Full-grid retained + graph | 1.528 | 1.793 | 14.614 | 14.331 |
| Full-grid retained, ordinary launches | 1.437 | 1.671 | 13.583 | 13.358 |
| Row-scoped retained + graph | 1.525 | 1.790 | 14.655 | 14.340 |
| Row-scoped retained, ordinary launches | 1.440 | 1.674 | 13.588 | 13.410 |
| Same-binary control: retain X and E | 1.526 | 1.799 | 14.582 | 14.384 |
| Same-binary control: reload X | 1.524 | 1.790 | 16.136 | 15.553 |
| Same-binary control: roundtrip E | 1.526 | 1.795 | 17.552 | 17.144 |
| Same-binary control: reload X and roundtrip E | 1.528 | 1.793 | 18.648 | 18.235 |
| RMS preprocessing + cuDNN + graph | 1.516 | 1.770 | 152.203 | 151.406 |

External-eviction diagnostic: a separate buffer of at least 256 MiB is touched before each invocation, outside the measured interval. This is a cache-eviction experiment, not a guarantee that every cache or metadata entry is cold. Fifteen rounds per variant.

| Variant | Eviction host completion A1 | Eviction host completion A2 |
|:---|---:|---:|
| Unfused, ordinary launches | 53.453 | 52.519 |
| Unfused + graph | 48.112 | 45.917 |
| Local3: materialize E + graph | 27.985 | 27.300 |
| Local3: recompute E + graph | 25.848 | 26.189 |
| Local3: recompute E, ordinary launches | 25.873 | 26.707 |
| Local2: repeat full-row stats, materialize E + graph | 52.544 | 51.255 |
| Local2: repeat full-row stats, recompute E + graph | 47.744 | 47.921 |
| Local1: repeat full-row work + graph | 68.549 | 68.458 |
| Full-grid retained + graph | 24.978 | 24.985 |
| Full-grid retained, ordinary launches | 22.718 | 22.908 |
| Row-scoped retained + graph | 24.337 | 24.823 |
| Row-scoped retained, ordinary launches | 22.488 | 22.697 |
| Same-binary control: retain X and E | 24.798 | 25.102 |
| Same-binary control: reload X | 26.142 | 26.163 |
| Same-binary control: roundtrip E | 27.702 | 27.954 |
| Same-binary control: reload X and roundtrip E | 28.754 | 28.941 |
| RMS preprocessing + cuDNN + graph | 173.625 | 172.691 |

### 32 row(s): all warm variants

| Variant | Kernels | Event A1 | Event A2 | Host completion A1 | Host completion A2 |
|:---|---:|---:|---:|---:|---:|
| Unfused, ordinary launches | 9 | 66.524 | 66.367 | 69.855 | 69.990 |
| Unfused + graph | 9 | 58.353 | 57.318 | 64.692 | 64.289 |
| Local3: materialize E + graph | 3 | 34.255 | 34.622 | 39.760 | 40.725 |
| Local3: recompute E + graph | 3 | 23.275 | 22.050 | 27.904 | 27.170 |
| Local3: recompute E, ordinary launches | 3 | 23.731 | 23.287 | 28.818 | 28.669 |
| Local2: repeat full-row stats, materialize E + graph | 2 | 61.734 | 60.887 | 67.975 | 66.992 |
| Local2: repeat full-row stats, recompute E + graph | 2 | 42.365 | 38.239 | 47.727 | 44.337 |
| Local1: repeat full-row work + graph | 1 | 62.954 | 61.730 | 69.067 | 67.122 |
| Full-grid retained + graph | 1 | 20.420 | 20.053 | 25.136 | 25.113 |
| Full-grid retained, ordinary launches | 1 | 19.406 | 19.112 | 24.433 | 24.598 |
| Row-scoped retained + graph | 1 | 19.007 | 19.525 | 25.683 | 24.643 |
| Row-scoped retained, ordinary launches | 1 | 18.826 | 18.566 | 24.008 | 24.098 |
| Same-binary control: retain X and E | 1 | 20.653 | 19.985 | 25.364 | 25.181 |
| Same-binary control: reload X | 1 | 22.931 | 22.174 | 27.686 | 27.388 |
| Same-binary control: roundtrip E | 1 | 31.849 | 31.527 | 38.455 | 38.658 |
| Same-binary control: reload X and roundtrip E | 1 | 33.733 | 33.402 | 40.331 | 40.420 |
| RMS preprocessing + cuDNN + graph | 3 | 185.073 | 183.295 | 189.775 | 189.275 |
| Empty stream synchronization | 0 | — | — | 1.390 | 1.388 |

| Variant | Host enqueue A1 | Host enqueue A2 | Host batch completion A1 | Host batch completion A2 |
|:---|---:|---:|---:|---:|
| Unfused, ordinary launches | 12.491 | 14.843 | 66.735 | 66.585 |
| Unfused + graph | 1.526 | 1.795 | 58.557 | 57.545 |
| Local3: materialize E + graph | 1.515 | 1.777 | 34.457 | 34.846 |
| Local3: recompute E + graph | 1.520 | 1.782 | 23.478 | 22.244 |
| Local3: recompute E, ordinary launches | 4.104 | 4.858 | 23.919 | 23.481 |
| Local2: repeat full-row stats, materialize E + graph | 1.515 | 1.772 | 61.944 | 61.115 |
| Local2: repeat full-row stats, recompute E + graph | 1.522 | 1.767 | 42.569 | 38.437 |
| Local1: repeat full-row work + graph | 1.514 | 1.767 | 63.150 | 61.954 |
| Full-grid retained + graph | 1.518 | 1.779 | 20.611 | 20.250 |
| Full-grid retained, ordinary launches | 1.423 | 1.652 | 19.603 | 19.306 |
| Row-scoped retained + graph | 1.516 | 1.780 | 19.212 | 19.724 |
| Row-scoped retained, ordinary launches | 1.425 | 1.641 | 19.016 | 18.756 |
| Same-binary control: retain X and E | 1.520 | 1.776 | 20.852 | 20.178 |
| Same-binary control: reload X | 1.521 | 1.778 | 23.119 | 22.374 |
| Same-binary control: roundtrip E | 1.516 | 1.763 | 32.057 | 31.751 |
| Same-binary control: reload X and roundtrip E | 1.516 | 1.778 | 33.934 | 33.626 |
| RMS preprocessing + cuDNN + graph | 1.512 | 1.770 | 185.277 | 183.519 |

### 64 row(s): all warm variants

| Variant | Kernels | Event A1 | Event A2 | Host completion A1 | Host completion A2 |
|:---|---:|---:|---:|---:|---:|
| Unfused, ordinary launches | 9 | 141.588 | 139.626 | 143.703 | 143.014 |
| Unfused + graph | 9 | 131.313 | 128.051 | 137.057 | 134.625 |
| Local3: materialize E + graph | 3 | 79.868 | 78.401 | 85.879 | 84.732 |
| Local3: recompute E + graph | 3 | 56.424 | 55.229 | 62.174 | 61.377 |
| Local3: recompute E, ordinary launches | 3 | 58.877 | 57.949 | 62.515 | 62.292 |
| Local2: repeat full-row stats, materialize E + graph | 2 | 108.622 | 107.824 | 114.373 | 114.274 |
| Local2: repeat full-row stats, recompute E + graph | 2 | 77.895 | 75.944 | 83.699 | 82.439 |
| Local1: repeat full-row work + graph | 1 | 123.241 | 120.762 | 130.650 | 126.713 |
| Full-grid retained + graph | 2 | 50.946 | 50.175 | 57.488 | 57.239 |
| Full-grid retained, ordinary launches | 2 | 51.893 | 51.219 | 56.558 | 56.195 |
| Row-scoped retained + graph | 2 | 49.479 | 48.102 | 55.701 | 55.188 |
| Row-scoped retained, ordinary launches | 2 | 50.072 | 49.439 | 54.648 | 54.172 |
| Same-binary control: retain X and E | 2 | 51.003 | 50.067 | 57.634 | 57.136 |
| Same-binary control: reload X | 2 | 56.342 | 55.443 | 62.947 | 62.865 |
| Same-binary control: roundtrip E | 2 | 63.078 | 62.552 | 69.928 | 69.693 |
| Same-binary control: reload X and roundtrip E | 2 | 67.981 | 66.549 | 74.761 | 73.875 |
| RMS preprocessing + cuDNN + graph | 3 | 359.474 | 357.358 | 365.669 | 363.215 |
| Empty stream synchronization | 0 | — | — | 1.394 | 1.407 |

| Variant | Host enqueue A1 | Host enqueue A2 | Host batch completion A1 | Host batch completion A2 |
|:---|---:|---:|---:|---:|
| Unfused, ordinary launches | 12.406 | 13.744 | 141.802 | 139.831 |
| Unfused + graph | 1.539 | 1.680 | 131.522 | 128.284 |
| Local3: materialize E + graph | 1.520 | 1.662 | 80.072 | 78.622 |
| Local3: recompute E + graph | 1.522 | 1.662 | 56.626 | 55.450 |
| Local3: recompute E, ordinary launches | 4.102 | 4.559 | 59.088 | 58.151 |
| Local2: repeat full-row stats, materialize E + graph | 1.517 | 1.661 | 108.822 | 108.042 |
| Local2: repeat full-row stats, recompute E + graph | 1.516 | 1.667 | 78.097 | 76.199 |
| Local1: repeat full-row work + graph | 1.513 | 1.661 | 123.442 | 120.980 |
| Full-grid retained + graph | 1.522 | 1.663 | 51.150 | 50.373 |
| Full-grid retained, ordinary launches | 2.868 | 3.139 | 52.106 | 51.420 |
| Row-scoped retained + graph | 1.526 | 1.667 | 49.683 | 48.325 |
| Row-scoped retained, ordinary launches | 2.869 | 3.141 | 50.281 | 49.636 |
| Same-binary control: retain X and E | 1.520 | 1.661 | 51.206 | 50.241 |
| Same-binary control: reload X | 1.524 | 1.663 | 56.546 | 55.673 |
| Same-binary control: roundtrip E | 1.523 | 1.659 | 63.281 | 62.773 |
| Same-binary control: reload X and roundtrip E | 1.521 | 1.654 | 68.187 | 66.769 |
| RMS preprocessing + cuDNN + graph | 1.522 | 1.658 | 359.673 | 357.580 |

External-eviction diagnostic: a separate buffer of at least 256 MiB is touched before each invocation, outside the measured interval. This is a cache-eviction experiment, not a guarantee that every cache or metadata entry is cold. Fifteen rounds per variant.

| Variant | Eviction host completion A1 | Eviction host completion A2 |
|:---|---:|---:|
| Unfused, ordinary launches | 146.788 | 146.005 |
| Unfused + graph | 139.224 | 136.988 |
| Local3: materialize E + graph | 88.512 | 88.111 |
| Local3: recompute E + graph | 67.764 | 67.310 |
| Local3: recompute E, ordinary launches | 68.012 | 67.913 |
| Local2: repeat full-row stats, materialize E + graph | 116.557 | 116.578 |
| Local2: repeat full-row stats, recompute E + graph | 88.724 | 87.880 |
| Local1: repeat full-row work + graph | 136.646 | 135.583 |
| Full-grid retained + graph | 58.015 | 57.625 |
| Full-grid retained, ordinary launches | 56.330 | 56.122 |
| Row-scoped retained + graph | 57.229 | 55.995 |
| Row-scoped retained, ordinary launches | 55.624 | 54.672 |
| Same-binary control: retain X and E | 57.929 | 57.594 |
| Same-binary control: reload X | 63.867 | 63.579 |
| Same-binary control: roundtrip E | 69.695 | 69.946 |
| Same-binary control: reload X and roundtrip E | 75.031 | 74.714 |
| RMS preprocessing + cuDNN + graph | 369.658 | 363.848 |

## initial_full_grid: 20261020T022150Z

[Full evidence and figures](/Users/1duo/Work/YuduoTechBlog/experiments/graphs_scaled_softmax/artifacts/20261020T022150Z/report/READOUT.md)

### 1 row(s): all warm variants

| Variant | Kernels | Event A1 | Event A2 | Host completion A1 | Host completion A2 |
|:---|---:|---:|---:|---:|---:|
| Unfused, ordinary launches | 9 | 30.064 | 35.387 | 35.864 | 44.444 |
| Unfused + graph | 9 | 22.848 | 24.043 | 28.826 | 29.353 |
| Local3: materialize E + graph | 3 | 13.322 | 14.087 | 18.945 | 19.279 |
| Local3: recompute E + graph | 3 | 11.860 | 12.158 | 17.460 | 17.386 |
| Local3: recompute E, ordinary launches | 3 | 13.204 | 14.223 | 18.927 | 23.116 |
| Local2: repeat full-row stats, materialize E + graph | 2 | 20.662 | 21.450 | 26.293 | 27.013 |
| Local2: repeat full-row stats, recompute E + graph | 2 | 19.129 | 19.819 | 24.819 | 25.399 |
| Local1: repeat full-row work + graph | 1 | 42.493 | 42.573 | 48.059 | 48.062 |
| Full-grid retained + graph | 1 | 10.190 | 10.461 | 15.931 | 15.744 |
| Full-grid retained, ordinary launches | 1 | 9.189 | 9.461 | 14.890 | 17.441 |
| Same-binary control: retain X and E | 1 | 10.152 | 10.452 | 15.930 | 15.800 |
| Same-binary control: reload X | 1 | 10.644 | 10.875 | 16.352 | 16.228 |
| Same-binary control: roundtrip E | 1 | 10.979 | 11.174 | 16.756 | 16.543 |
| Same-binary control: reload X and roundtrip E | 1 | 11.481 | 11.874 | 17.137 | 17.290 |
| RMS preprocessing + cuDNN + graph | 3 | 145.778 | 144.481 | 151.362 | 150.226 |
| Empty stream synchronization | 0 | — | — | 1.378 | 1.724 |

| Variant | Host enqueue A1 | Host enqueue A2 | Host batch completion A1 | Host batch completion A2 |
|:---|---:|---:|---:|---:|
| Unfused, ordinary launches | 19.172 | 35.017 | 30.253 | 35.586 |
| Unfused + graph | 2.715 | 1.498 | 23.035 | 24.250 |
| Local3: materialize E + graph | 2.653 | 1.477 | 13.512 | 14.310 |
| Local3: recompute E + graph | 2.650 | 1.484 | 12.048 | 12.385 |
| Local3: recompute E, ordinary launches | 6.365 | 11.442 | 13.391 | 14.424 |
| Local2: repeat full-row stats, materialize E + graph | 2.655 | 1.467 | 20.854 | 21.672 |
| Local2: repeat full-row stats, recompute E + graph | 2.633 | 1.485 | 19.316 | 20.041 |
| Local1: repeat full-row work + graph | 2.653 | 1.480 | 42.677 | 42.776 |
| Full-grid retained + graph | 2.670 | 1.510 | 10.377 | 10.681 |
| Full-grid retained, ordinary launches | 2.137 | 3.832 | 9.376 | 9.659 |
| Same-binary control: retain X and E | 2.653 | 1.492 | 10.348 | 10.676 |
| Same-binary control: reload X | 2.656 | 1.506 | 10.830 | 11.093 |
| Same-binary control: roundtrip E | 2.657 | 1.489 | 11.170 | 11.397 |
| Same-binary control: reload X and roundtrip E | 2.670 | 1.500 | 11.665 | 12.096 |
| RMS preprocessing + cuDNN + graph | 2.634 | 1.477 | 145.971 | 144.684 |

### 16 row(s): all warm variants

| Variant | Kernels | Event A1 | Event A2 | Host completion A1 | Host completion A2 |
|:---|---:|---:|---:|---:|---:|
| Unfused, ordinary launches | 9 | 40.110 | 42.478 | 46.145 | 48.038 |
| Unfused + graph | 9 | 32.749 | 31.710 | 39.966 | 38.344 |
| Local3: materialize E + graph | 3 | 16.052 | 16.151 | 21.600 | 21.411 |
| Local3: recompute E + graph | 3 | 14.881 | 14.777 | 20.513 | 19.950 |
| Local3: recompute E, ordinary launches | 3 | 16.726 | 16.620 | 22.366 | 24.376 |
| Local2: repeat full-row stats, materialize E + graph | 2 | 29.247 | 29.290 | 34.738 | 34.817 |
| Local2: repeat full-row stats, recompute E + graph | 2 | 25.865 | 25.856 | 31.460 | 31.217 |
| Local1: repeat full-row work + graph | 1 | 53.167 | 53.121 | 58.843 | 58.767 |
| Full-grid retained + graph | 1 | 14.239 | 14.402 | 19.992 | 19.728 |
| Full-grid retained, ordinary launches | 1 | 13.520 | 13.635 | 19.301 | 21.636 |
| Same-binary control: retain X and E | 1 | 14.384 | 14.588 | 20.157 | 19.885 |
| Same-binary control: reload X | 1 | 15.729 | 15.943 | 21.418 | 21.218 |
| Same-binary control: roundtrip E | 1 | 18.092 | 18.446 | 23.908 | 23.868 |
| Same-binary control: reload X and roundtrip E | 1 | 19.607 | 19.706 | 25.314 | 25.068 |
| RMS preprocessing + cuDNN + graph | 3 | 152.041 | 150.116 | 157.793 | 155.874 |
| Empty stream synchronization | 0 | — | — | 1.418 | 1.646 |

| Variant | Host enqueue A1 | Host enqueue A2 | Host batch completion A1 | Host batch completion A2 |
|:---|---:|---:|---:|---:|
| Unfused, ordinary launches | 18.797 | 33.972 | 40.303 | 42.689 |
| Unfused + graph | 2.675 | 1.482 | 32.945 | 31.942 |
| Local3: materialize E + graph | 2.634 | 1.453 | 16.235 | 16.379 |
| Local3: recompute E + graph | 2.637 | 1.432 | 15.070 | 14.987 |
| Local3: recompute E, ordinary launches | 6.254 | 11.078 | 16.910 | 16.815 |
| Local2: repeat full-row stats, materialize E + graph | 2.620 | 1.458 | 29.434 | 29.510 |
| Local2: repeat full-row stats, recompute E + graph | 2.620 | 1.477 | 26.049 | 26.061 |
| Local1: repeat full-row work + graph | 2.622 | 1.426 | 53.352 | 53.341 |
| Full-grid retained + graph | 2.636 | 1.465 | 14.422 | 14.610 |
| Full-grid retained, ordinary launches | 2.141 | 3.836 | 13.708 | 13.833 |
| Same-binary control: retain X and E | 2.631 | 1.430 | 14.572 | 14.799 |
| Same-binary control: reload X | 2.629 | 1.454 | 15.919 | 16.148 |
| Same-binary control: roundtrip E | 2.634 | 1.438 | 18.274 | 18.674 |
| Same-binary control: reload X and roundtrip E | 2.630 | 1.463 | 19.804 | 19.924 |
| RMS preprocessing + cuDNN + graph | 2.619 | 1.441 | 152.235 | 150.330 |

External-eviction diagnostic: a separate buffer of at least 256 MiB is touched before each invocation, outside the measured interval. This is a cache-eviction experiment, not a guarantee that every cache or metadata entry is cold. Fifteen rounds per variant.

| Variant | Eviction host completion A1 | Eviction host completion A2 |
|:---|---:|---:|
| Unfused, ordinary launches | 54.166 | 55.019 |
| Unfused + graph | 47.274 | 46.664 |
| Local3: materialize E + graph | 27.638 | 27.587 |
| Local3: recompute E + graph | 26.398 | 26.380 |
| Local3: recompute E, ordinary launches | 27.046 | 29.220 |
| Local2: repeat full-row stats, materialize E + graph | 52.181 | 51.741 |
| Local2: repeat full-row stats, recompute E + graph | 48.838 | 48.406 |
| Local1: repeat full-row work + graph | 69.880 | 68.781 |
| Full-grid retained + graph | 26.380 | 26.210 |
| Full-grid retained, ordinary launches | 24.294 | 26.137 |
| Same-binary control: retain X and E | 26.548 | 26.480 |
| Same-binary control: reload X | 27.744 | 27.421 |
| Same-binary control: roundtrip E | 30.265 | 29.833 |
| Same-binary control: reload X and roundtrip E | 32.044 | 31.110 |
| RMS preprocessing + cuDNN + graph | 174.271 | 172.156 |

### 32 row(s): all warm variants

| Variant | Kernels | Event A1 | Event A2 | Host completion A1 | Host completion A2 |
|:---|---:|---:|---:|---:|---:|
| Unfused, ordinary launches | 9 | 66.791 | 67.925 | 70.847 | 73.647 |
| Unfused + graph | 9 | 57.040 | 57.916 | 64.038 | 64.800 |
| Local3: materialize E + graph | 3 | 34.431 | 35.208 | 41.069 | 41.790 |
| Local3: recompute E + graph | 3 | 21.910 | 21.798 | 27.242 | 27.337 |
| Local3: recompute E, ordinary launches | 3 | 23.638 | 23.620 | 29.427 | 31.426 |
| Local2: repeat full-row stats, materialize E + graph | 2 | 61.909 | 60.827 | 68.997 | 67.683 |
| Local2: repeat full-row stats, recompute E + graph | 2 | 36.388 | 37.920 | 42.113 | 44.907 |
| Local1: repeat full-row work + graph | 1 | 60.614 | 63.759 | 66.189 | 69.824 |
| Full-grid retained + graph | 1 | 19.824 | 20.535 | 25.571 | 26.137 |
| Full-grid retained, ordinary launches | 1 | 18.939 | 19.541 | 24.814 | 27.645 |
| Same-binary control: retain X and E | 1 | 19.847 | 20.644 | 25.542 | 26.106 |
| Same-binary control: reload X | 1 | 22.162 | 22.990 | 27.731 | 28.303 |
| Same-binary control: roundtrip E | 1 | 31.720 | 31.579 | 39.306 | 39.023 |
| Same-binary control: reload X and roundtrip E | 1 | 33.495 | 33.761 | 41.159 | 41.149 |
| RMS preprocessing + cuDNN + graph | 3 | 209.458 | 193.972 | 216.045 | 200.674 |
| Empty stream synchronization | 0 | — | — | 1.407 | 1.702 |

| Variant | Host enqueue A1 | Host enqueue A2 | Host batch completion A1 | Host batch completion A2 |
|:---|---:|---:|---:|---:|
| Unfused, ordinary launches | 19.044 | 34.240 | 66.984 | 68.136 |
| Unfused + graph | 2.709 | 1.478 | 57.244 | 58.151 |
| Local3: materialize E + graph | 2.641 | 1.454 | 34.625 | 35.434 |
| Local3: recompute E + graph | 2.687 | 1.475 | 22.097 | 22.022 |
| Local3: recompute E, ordinary launches | 6.371 | 11.480 | 23.826 | 23.829 |
| Local2: repeat full-row stats, materialize E + graph | 2.635 | 1.464 | 62.111 | 61.057 |
| Local2: repeat full-row stats, recompute E + graph | 2.668 | 1.472 | 36.581 | 38.147 |
| Local1: repeat full-row work + graph | 2.651 | 1.462 | 60.799 | 63.970 |
| Full-grid retained + graph | 2.679 | 1.466 | 20.011 | 20.742 |
| Full-grid retained, ordinary launches | 2.168 | 3.887 | 19.135 | 19.740 |
| Same-binary control: retain X and E | 2.690 | 1.453 | 20.040 | 20.862 |
| Same-binary control: reload X | 2.677 | 1.466 | 22.348 | 23.219 |
| Same-binary control: roundtrip E | 2.684 | 1.482 | 31.918 | 31.805 |
| Same-binary control: reload X and roundtrip E | 2.668 | 1.463 | 33.705 | 33.983 |
| RMS preprocessing + cuDNN + graph | 2.641 | 1.476 | 209.652 | 194.205 |

### 64 row(s): all warm variants

| Variant | Kernels | Event A1 | Event A2 | Host completion A1 | Host completion A2 |
|:---|---:|---:|---:|---:|---:|
| Unfused, ordinary launches | 9 | 142.286 | 141.194 | 144.783 | 146.706 |
| Unfused + graph | 9 | 130.838 | 128.254 | 137.514 | 135.771 |
| Local3: materialize E + graph | 3 | 80.088 | 79.413 | 86.734 | 86.446 |
| Local3: recompute E + graph | 3 | 56.405 | 55.017 | 62.818 | 61.969 |
| Local3: recompute E, ordinary launches | 3 | 59.093 | 58.005 | 62.927 | 64.474 |
| Local2: repeat full-row stats, materialize E + graph | 2 | 108.536 | 108.189 | 115.299 | 115.472 |
| Local2: repeat full-row stats, recompute E + graph | 2 | 76.568 | 75.549 | 82.842 | 82.509 |
| Local1: repeat full-row work + graph | 1 | 123.467 | 122.137 | 130.531 | 130.597 |
| Full-grid retained + graph | 2 | 50.830 | 50.162 | 57.979 | 58.119 |
| Full-grid retained, ordinary launches | 2 | 51.720 | 51.551 | 56.963 | 59.089 |
| Same-binary control: retain X and E | 2 | 51.009 | 50.192 | 58.314 | 58.008 |
| Same-binary control: reload X | 2 | 56.196 | 55.639 | 63.437 | 63.274 |
| Same-binary control: roundtrip E | 2 | 62.693 | 62.277 | 70.045 | 70.031 |
| Same-binary control: reload X and roundtrip E | 2 | 67.313 | 66.678 | 74.957 | 74.481 |
| RMS preprocessing + cuDNN + graph | 3 | 357.723 | 358.603 | 363.986 | 366.168 |
| Empty stream synchronization | 0 | — | — | 1.481 | 1.686 |

| Variant | Host enqueue A1 | Host enqueue A2 | Host batch completion A1 | Host batch completion A2 |
|:---|---:|---:|---:|---:|
| Unfused, ordinary launches | 14.750 | 33.630 | 142.489 | 141.404 |
| Unfused + graph | 1.780 | 1.505 | 131.062 | 128.476 |
| Local3: materialize E + graph | 1.751 | 1.465 | 80.314 | 79.648 |
| Local3: recompute E + graph | 1.768 | 1.475 | 56.657 | 55.242 |
| Local3: recompute E, ordinary launches | 4.910 | 11.210 | 59.306 | 58.211 |
| Local2: repeat full-row stats, materialize E + graph | 1.752 | 1.453 | 108.763 | 108.422 |
| Local2: repeat full-row stats, recompute E + graph | 1.751 | 1.433 | 76.796 | 75.779 |
| Local1: repeat full-row work + graph | 1.756 | 1.490 | 123.694 | 122.375 |
| Full-grid retained + graph | 1.770 | 1.448 | 51.053 | 50.393 |
| Full-grid retained, ordinary launches | 3.357 | 7.714 | 51.920 | 51.767 |
| Same-binary control: retain X and E | 1.761 | 1.434 | 51.235 | 50.418 |
| Same-binary control: reload X | 1.761 | 1.448 | 56.418 | 55.865 |
| Same-binary control: roundtrip E | 1.754 | 1.444 | 62.916 | 62.516 |
| Same-binary control: reload X and roundtrip E | 1.749 | 1.447 | 67.536 | 66.913 |
| RMS preprocessing + cuDNN + graph | 1.752 | 1.444 | 357.952 | 358.846 |

External-eviction diagnostic: a separate buffer of at least 256 MiB is touched before each invocation, outside the measured interval. This is a cache-eviction experiment, not a guarantee that every cache or metadata entry is cold. Fifteen rounds per variant.

| Variant | Eviction host completion A1 | Eviction host completion A2 |
|:---|---:|---:|
| Unfused, ordinary launches | 147.964 | 149.623 |
| Unfused + graph | 139.837 | 138.018 |
| Local3: materialize E + graph | 89.317 | 88.633 |
| Local3: recompute E + graph | 68.512 | 68.122 |
| Local3: recompute E, ordinary launches | 68.917 | 70.729 |
| Local2: repeat full-row stats, materialize E + graph | 117.076 | 116.987 |
| Local2: repeat full-row stats, recompute E + graph | 88.691 | 88.831 |
| Local1: repeat full-row work + graph | 136.731 | 134.977 |
| Full-grid retained + graph | 58.387 | 58.156 |
| Full-grid retained, ordinary launches | 56.820 | 59.064 |
| Same-binary control: retain X and E | 58.614 | 57.985 |
| Same-binary control: reload X | 63.865 | 64.048 |
| Same-binary control: roundtrip E | 69.907 | 70.071 |
| Same-binary control: reload X and roundtrip E | 75.231 | 75.177 |
| RMS preprocessing + cuDNN + graph | 364.795 | 366.509 |

## Method and interpretation

The semantic operator shared by every measured implementation is:

```python
import numpy as np

def rms_scaled_softmax(x, eps=1e-5):
    inv_rms = 1 / np.sqrt(
        np.mean(x * x, axis=1, keepdims=True) + eps
    )
    exp_scores = np.exp(
        (x - np.max(x, axis=1, keepdims=True)) * inv_rms
    )
    return exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )
```

RMS scaling changes the operator relative to ordinary softmax. This is a constructed teaching workload, not a claim that production attention normally contains this exact sequence. Tiny cross-block statistics still use global memory; the retained path keeps the large tile buffers in shared memory.

The latest study selects the full-grid retained kernel for one row and row-scoped retention for 16/32/64 rows using the pilot. Both variants and ordinary-launch controls remain visible above. At 64 rows both retained implementations use two cooperative launches. A graph and a single kernel are distinct concepts; a single-kernel graph need not beat its ordinary launch.

The predeclared 1.25× primary-case event speedup was not achieved. The fastest tested local family also had the lowest held-out event medians among the tested local alternatives. The one-/two-kernel alternatives remain valuable controls because fewer kernels can require more repeated work.

Resource queries constrain the interpretation at 32 rows: the row-scoped binary reports 32 registers/thread and an occupancy limit of eight active blocks/SM; the full-grid binary reports 34 and a limit of six. These are capacity limits, not observed occupancy. Their selected tile widths differ, so the comparison does not isolate barrier cost.

The initial study used only full-grid synchronization. Its measurements are preserved separately rather than pooled with the row-scoped follow-up. The latest reported GPU ledger includes both studies; all GPU work was already torn down and archived.
