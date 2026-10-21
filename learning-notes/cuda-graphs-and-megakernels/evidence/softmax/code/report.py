"""Build figures and tidy tables from the verified raw review bundle."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

from analyze import check_traces, records, summarize, verify_manifest

COLORS = ["#2865ad", "#d77923", "#28806a"]


def output_figure(fig, path):
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def bars(ax, timing, families, labels):
    positions = np.arange(len(families))
    for i, mode in enumerate(["stream", "graph"]):
        summaries = [timing[f+"_"+mode]["host_single_us"] for f in families]
        values = np.array([s["median"] for s in summaries])
        low = values - np.array([s["p10"] for s in summaries])
        high = np.array([s["p90"] for s in summaries]) - values
        xs = positions+(i-.5)*.32
        ax.bar(xs, values, .3, color=COLORS[i], label="Stream" if i == 0 else "Graph replay",
               yerr=np.vstack([low, high]), capsize=2, error_kw={"linewidth": .8})
        for x, v in zip(xs, values):
            ax.annotate(f"{v:.1f}", (x, v), xytext=(-2 if i == 0 else 2, 5), textcoords="offset points",
                        ha="right" if i == 0 else "left", fontsize=8)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Completion latency (µs)")
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1]*1.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    validation = verify_manifest(root)
    config = json.loads((root/"config.json").read_text())
    output = root.parent / (root.name+"-report")
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    data = {}
    table = []
    for path in sorted((root/"bench").iterdir()):
        summary = summarize(path)
        saved = json.loads((path/"analysis.json").read_text())["timings"]
        if summary != saved:
            raise ValueError("saved summary disagrees with raw samples")
        data[path.name] = summary["timing_variants"]
        for variant, metrics in summary["timing_variants"].items():
            for metric, stats in metrics.items():
                table.append({"case": path.name, "variant": variant, "metric": metric, **stats})
    with (output/"timings.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8), layout="constrained")
    for ax, (rows, cols) in zip(axes.flat, config["row_cases"]):
        bars(ax, data[f"row-{rows}-{cols}-balanced"], ["staged", "fused", "cudnn"], ["4 stages", "Fused row", "cuDNN"])
        size = rows*cols*4
        size_label = f"{size/2**20:g} MiB" if size >= 2**20 else f"{size/1024:g} KiB" if size >= 1024 else f"{size} B"
        ax.set_title(f"{rows:,} × {cols:,}   |   input {size_label}", fontsize=11)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Softmax: submission and fusion\nH100 · medians and p10–p90 across 21 shuffled rounds", fontsize=15)
    fig.supxlabel("One invocation through stream synchronization; excludes allocation, transfers, and graph setup.", fontsize=9)
    output_figure(fig, output/"row-completion")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8), layout="constrained")
    for i, pattern in enumerate(config["dag_patterns"]):
        for j, rows in enumerate(config["dag_rows"]):
            ax = axes[i, j]
            bars(ax, data[f"dag-{rows}-0-{pattern}"], ["separate", "phased", "dag"], ["3 kernels", "Persistent\nphased", "Ready-task\nqueue"])
            stats = [v["host_single_us"] for k, v in data[f"dag-{rows}-0-{pattern}"].items() if k != "empty_sync"]
            ax.set_yscale("log")
            ax.set_ylim(min(s["p10"] for s in stats)*.65, max(s["p90"] for s in stats)*2)
            ax.set_ylabel("Completion latency (µs; log scale)")
            ax.set_title(f"{rows:,} rows · {pattern} · {rows*8192*4/2**20:g} MiB input", fontsize=11)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Tiled softmax: fewer launches versus scheduling cost\nEqual total work for balanced and ragged rows · 21 shuffled rounds", fontsize=15)
    fig.supxlabel("Same A/B/C helpers and arrays. Queue initialization is included. Trace instrumentation is disabled.", fontsize=9)
    output_figure(fig, output/"dag-completion")

    trace_results = []
    for path in sorted((root/"trace").iterdir()):
        trace_results.extend({"case": path.name, **r} for r in check_traces(path))
    (output/"trace-validation.json").write_text(json.dumps(trace_results, indent=2)+"\n")
    path = root/"trace"/f"dag-{config['trace_rows']}-0-ragged"
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    for ax, name, title in zip(axes, ["separate_graph", "phased_graph", "dag_graph"],
                               ["Three kernels", "Persistent kernel with phase barriers", "Persistent kernel with a ready-task queue"]):
        with (path/(name+"_2_trace.csv")).open() as stream:
            tasks = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        origin = min(t["start_ns"] for t in tasks)
        for stage, color in enumerate(COLORS):
            segments = [[((t["start_ns"]-origin)/1000, t["row"]), ((t["end_ns"]-origin)/1000, t["row"])]
                        for t in tasks if t["stage"] == stage]
            ax.add_collection(LineCollection(segments, colors=color, linewidths=.9))
        last_a = max(t["end_ns"] for t in tasks if t["stage"] == 0)
        ax.axvline((last_a-origin)/1000, color=COLORS[0], alpha=.5, linestyle="--", linewidth=.8)
        ax.autoscale_view()
        ax.set_ylim(-3, config["trace_rows"]+2)
        ax.set_ylabel("Row")
        ax.set_xlabel("Microseconds from first task start (each panel has its own scale)")
        ax.set_title(title, loc="left", fontsize=12)
    axes[0].legend(handles=[Line2D([0],[0],color=c,lw=3,label=l) for c,l in zip(COLORS,["A: tile statistics", "B: row reduction", "C: normalize tiles"])],
                   loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(1,1.16), fontsize=9)
    fig.suptitle("Measured task ordering · ragged softmax, 128 rows\nEvery task interval is plotted; overlapping tiles on a row can obscure one another.", fontsize=14)
    fig.supxlabel("Instrumented runs illustrate dependencies, not utilization or benchmark latency. Dashed line: final A task ends.", fontsize=9)
    output_figure(fig, output/"task-ordering")

    profiles = root.parent/(root.name+"-profile-exports")
    verify_manifest(profiles)
    counters = json.loads((profiles/"profile-summary.json").read_text())
    source_hashes = {e["path"]: e["sha256"] for e in json.loads((root/"manifest.json").read_text())}
    for group in ["compute", "systems"]:
        for name, entry in counters[group].items():
            if entry["source_sha256"] != source_hashes[name]:
                raise ValueError("Counter export does not match captured report")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for ax, metric, title in zip(axes, ["dram__bytes_read.sum", "lts__t_bytes.sum"], ["DRAM read bytes", "L2 transaction bytes"]):
        values = [counters["compute"]["compute-row-"+v+"_graph.ncu-repz"]["traffic_bytes"][metric]/2**20 for v in ["staged", "fused", "cudnn"]]
        ax.bar(range(3), values, color=COLORS)
        for i, value in enumerate(values):
            ax.annotate(f"{value:.1f}", (i, value), xytext=(0,5), textcoords="offset points", ha="center")
        ax.set_xticks(range(3), ["4 stages", "Fused row", "cuDNN"])
        ax.set_ylabel("MiB per profiled invocation")
        ax.set_title(title)
        ax.set_ylim(0, max(values)*1.2)
    fig.suptitle("Separate counter capture · 4,096 × 4,096 softmax", fontsize=14)
    fig.supxlabel("Nsight Compute kernel replay, cache-control=all. Sums across stages; not traffic from the warm timing trials.", fontsize=9)
    output_figure(fig, output/"profile-traffic")

    inputs = [root/"manifest.json", profiles/"manifest.json", profiles/"profile-summary.json", Path(__file__), *sorted((root/"bench").glob("*/samples.jsonl")),
              *sorted((root/"trace").glob("*/*_trace.csv"))]
    provenance = {"run_id": root.name, "review_verification": validation, "matplotlib": matplotlib.__version__, "numpy": np.__version__,
                  "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]}
    provenance["outputs"] = [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.iterdir()) if p.is_file() and p.name not in ("figure-provenance.json", "manifest.json")]
    (output/"figure-provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    main()
