"""Copy a documented, hash-checked evidence selection into the draft blog."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("post", type=Path)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    run = here/"artifacts/20261019T175825Z"
    report = run.with_name(run.name+"-report")
    profiles = run.with_name(run.name+"-profile-exports")
    dest = args.post/"evidence/softmax"
    source_hashes = {e["path"]: e["sha256"] for e in json.loads((run/"manifest.json").read_text())}

    def copy(source, relative):
        if source.is_relative_to(run) and str(source.relative_to(run)) in source_hashes:
            assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hashes[str(source.relative_to(run))]
        target = dest/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    for name in ["bench.cu", "analyze.py", "build.py", "modal_app.py"]:
        copy(run/name, "code/"+name)
    for name in ["article_figures.py", "site_evidence.py", "profile_exports.py", "report.py"]:
        copy(here/name, "code/"+name)
    for name in ["config.json", "manifest.json", "summary.json", "build.json", "compile.log", "bench.ptx", "bench.sass", "device-before.stdout", "device-after.stdout"]:
        copy(run/name, "artifacts/"+name)
    for pattern in ["bench/*/samples.jsonl", "bench/*/metadata.jsonl", "bench/*/analysis.json", "bench/*/*.dot", "trace/*/*_trace.csv", "trace/*/tile_offsets.i32", "trace/*/analysis.json", "trace/*/metadata.jsonl", "memcheck-*.stdout", "memcheck-*.stderr", "synccheck-*.stdout", "synccheck-*.stderr"]:
        for p in sorted(run.glob(pattern)):
            copy(p, "artifacts/"+str(p.relative_to(run)))
    for name in ["timings.csv", "trace-validation.json", "figure-provenance.json", "local-review.json"]:
        copy(report/name, name)
    for p in sorted(report.iterdir()):
        if p.suffix in (".svg", ".png"):
            copy(p, "figures/"+p.name)
    copy(profiles/"profile-summary.json", "profile-summary.json")
    for p in profiles.glob("*.csv"):
        copy(p, "profile-exports/"+p.name)
    for name in ["softmax-figure-provenance.json", "softmax-recheck.json"]:
        copy(args.post/"assets"/name, name)
    copy(run/"README.md", "protocol.md")
    text = (report/"README.md").read_text()
    replacements = {
        "../20261019T175825Z/README.md": "protocol.html",
        "../20261019T175825Z/bench.cu": "code/bench.cu",
        "../20261019T175825Z/analyze.py": "code/analyze.py",
        "../20261019T175825Z/config.json": "artifacts/config.json",
        "../20261019T175825Z/summary.json": "artifacts/summary.json",
        "../20261019T175825Z/manifest.json": "artifacts/manifest.json",
        "../20261019T175825Z-profile-exports/profile-summary.json": "profile-summary.json",
    }
    for name in ["row-completion", "dag-completion", "task-ordering", "profile-traffic"]:
        replacements["("+name+".png)"] = "(figures/"+name+".png)"
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("## Evaluation", "The article's new four-way figures use these same raw samples. In the full-sweep figures below, **4 stages / Stream** (row example) and **3 kernels / Stream** (tiled example) are the ordinary-launch baselines. Fusion and megakernel-style scheduling are complementary to graph replay, not successive guaranteed speedups.\n\n## Evaluation", 1)
    (dest/"results.md").write_text(text)
    index = """# Softmax evidence and reproduction

Run **20261019T175825Z**, collected October 19, 2026 on one NVIDIA H100 80GB HBM3. The article keeps its original February 22 date.

- [Protocol](protocol.html): workload definitions, controls, fixed tolerances, and measurement scopes.
- [Full results](results.html): all shapes, cuDNN reference, timing distributions, and profiling limitations.
- [Article figure provenance](softmax-figure-provenance.json): exact configuration mapping and input/output hashes.
- [Fresh timing and trace recheck](softmax-recheck.json): 12 timing cases and 24 dependency traces recomputed from saved samples.
- [All timing summaries](timings.csv), [saved validation summary](artifacts/summary.json), and [profiler counter exports](profile-summary.json).
- [Selected site bundle manifest](bundle-manifest.json), distinct from the [complete raw-run manifest](artifacts/manifest.json).

The baseline uses ordinary stream launches with no graph replay or fusion; normal compiler optimizations still apply. The tiled comparison labels the cooperative phase-barrier kernel as fusion, and the ready-task kernel as a **megakernel toy**. Both can run inside a graph. This toy is not a model-scale megakernel performance result, and its queue initialization is included.

The complete raw run retains 1,741 files / 9.37 GB in Modal Volume `cuda-graphs-softmax-evidence`, directory `/20261019T175825Z/`, with an archive at `/20261019T175825Z.tar.gz`. This site contains selected source, samples, traces, and text exports; large binary arrays and original profiler databases remain in that archive. Local review verified 1,372 files, with 369 remote-only files explicitly recorded. No GPU rerun was needed for the article restructuring.

## Reproduction

The CUDA source, analyzer, build script, and Modal runner below are the captured run versions. Figure and export scripts are subsequent CPU postprocessing. With the complete review bundle, run `python article_figures.py RUN_DIRECTORY OUTPUT_DIRECTORY` to rebuild the four-way figures. It verifies hashes, recomputes timing summaries, and checks trace coverage and ordering. The selected website bundle is not a substitute for the full review bundle when rerunning that command.

## Selected files

"""
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.suffix != ".html" and p.name not in ("index.md", "bundle-manifest.json"):
            rel = str(p.relative_to(dest))
            index += f"- [{rel}]({rel})\n"
    (dest/"index.md").write_text(index)
    print(f"Prepared {dest}")


if __name__ == "__main__":
    main()
