# Interactive Article Figures

Published article enhancements, promoted from the reviewed private preview.

- Figure 2: switch direct submission and graph replay. Show 9 CPU→GPU submission arrows versus 1, an illustrative overlap between enqueuing and execution, and a single completion return after K9. The benchmark calls `cudaStreamSynchronize` once after the whole chain; it never waits for each kernel before submitting the next. Steps illustrate a valid ordering, not a measured schedule, API duration, or hardware notification protocol. The same 9 kernels remain inspectable. Capture and instantiation precede the repeated invocation shown.
  - Semantic references: archived `code/bench.cu` (`measure` and `Spec::launch`), plus [NVIDIA’s overlapping-launch and graph example](https://developer.nvidia.com/blog/cuda-graphs/#overlapping-kernel-launch-and-execution).
- Figure 4: step through 3 compute phases and 2 barriers. Local fusion discards tile values between kernels, while the megakernel's same shared-memory buffer retains x and then E. Normalization writes y to global memory without claiming to overwrite retained E. Counts are source-level reads/evaluations per element, not HBM transactions. Row metadata stays in global memory.
- Figure 5: choose only the 4 measured row counts. Show the mean of the two session medians and inspect those medians. The default 0–160 µs axis stays fixed across row counts; an explicit control fits the current selection. Ranges span session medians, not confidence intervals. Graph replay stays enabled for both fusion steps.

`data.json` copies `completion_summaries` from `../evidence/rms-softmax/figure-provenance.json`; its `source_sha256` identifies that exact file. No measurements are synthesized or rerun. All numerical values remain available in the archived results.

`build_blog.py` adds this stylesheet/module only when the prototype files exist. It renders the original static figures first. JavaScript enhances only Figures 2, 4 and 5 after successful initialization; each offers Explore/Original and an SVG link. Printing uses the static figures and their matching captions. Without JavaScript or the data file, the article remains readable.

Animation is opt-in, stops when hidden/offscreen or switched to Original, and never loops automatically. Reduced-motion preferences remove CSS motion and turn Play into a manual Next control. All interactions use native buttons and checkboxes.
