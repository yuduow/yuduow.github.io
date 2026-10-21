"""Article-specific views of the immutable H100 softmax run; no GPU execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
import numpy as np

from analyze import check_traces, summarize, verify_manifest

COLORS = ["#64748b", "#2563eb", "#087f73", "#8054c7"]
VARIANT_COLORS = {
    "staged_stream": COLORS[0], "separate_stream": COLORS[0],
    "staged_graph": COLORS[1], "separate_graph": COLORS[1],
    "fused_stream": "#329b8c", "fused_graph": COLORS[2],
    "phased_graph": COLORS[2], "dag_graph": COLORS[3],
}
VARIANT_NOTES = {
    "staged_stream": "4 kernels · ordinary launches",
    "staged_graph": "4 kernels · one graph replay",
    "fused_stream": "1 kernel · ordinary launch",
    "fused_graph": "1 fused kernel · graph replay",
    "separate_stream": "3 kernels · ordinary launches",
    "separate_graph": "3 kernels · one graph replay",
    "phased_graph": "1 kernel · global barriers",
    "dag_graph": "Queue setup + persistent kernel",
}
STAGES = ["#2563eb", "#d18b14", "#087f73"]
INK, MUTED, LINE = "#172338", "#596b82", "#e5ebf3"
FONT_DIR = Path(__file__).resolve().parent / "fonts"


def tick_label(value, _):
    return f"{value:,.0f}" if abs(value-round(value)) < 1e-10 else f"{value:,.1f}"


def save(fig, path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [*fig.texts]
    for legend in fig.legends:
        labels.extend(legend.get_texts())
    for ax in fig.axes:
        labels.extend([*ax.texts, ax.title, ax.xaxis.label, ax.yaxis.label])
    for label in labels:
        if label.get_text():
            box = label.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, label.get_text()
    visible = [(label.get_text(), label.get_window_extent(renderer)) for label in labels if label.get_text()]
    for i, (text, box) in enumerate(visible):
        for other_text, other_box in visible[i+1:]:
            assert not box.overlaps(other_box), f"Overlapping labels: {text!r} and {other_text!r}"
    for ext in ("png", "svg"):
        fig.savefig(path.with_suffix("."+ext), dpi=200, facecolor="white",
                    metadata={"Date": None} if ext == "svg" else {})
    plt.close(fig)


def comparison(ax, data, variants, labels, log=False, maximum=None, mobile=False):
    maximum = maximum or max(data[v]["host_single_us"]["p90"] for v in variants)*1.12
    ax.set_ylim(len(labels)-.23, -.38)
    ax.set_yticks([])
    if log:
        ax.set_xscale("log")
        ax.set_xlim(8, maximum)
        ax.xaxis.set_major_locator(FixedLocator([10, 100, 1000]))
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.set_xlim(0, maximum)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.xaxis.set_major_formatter(FuncFormatter(tick_label))
    ax.grid(axis="x", color=LINE, linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=8, labelsize=11 if mobile else 10, colors=MUTED)
    for i, (variant, label) in enumerate(zip(variants, labels)):
        color = VARIANT_COLORS[variant]
        s = data[variant]["host_single_us"]
        v = s["median"]
        errors = [[v-s["p10"]], [s["p90"]-v]]
        y = i+.45
        # Labels and values share a baseline, separate from the data marks.
        ax.text(0, i-.12, label, transform=ax.get_yaxis_transform(), color=INK,
                fontsize=13 if mobile else 12, va="bottom", fontweight=600,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": .5})
        ax.text(1, i-.12, f"{v:,.2f}", transform=ax.get_yaxis_transform(), color=color,
                fontsize=14 if mobile else 13, ha="right", va="bottom", fontweight=600,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": .5})
        ax.text(0, i+.25, VARIANT_NOTES[variant], transform=ax.get_yaxis_transform(), color=MUTED,
                fontsize=10.2 if mobile else 10, va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": .5})
        if log:
            ax.hlines(y, 8, maximum, color="#f0f3f8", linewidth=2)
            ax.errorbar(v, y, xerr=errors, fmt="o", color=color, capsize=3,
                        markersize=6.5, elinewidth=1.5, zorder=3)
        else:
            ax.barh(y, maximum, color="#f0f3f8", height=.12, zorder=1)
            ax.barh(y, v, color=color, height=.12, xerr=errors, capsize=3,
                    error_kw={"ecolor": INK, "elinewidth": 1.2, "capthick": 1.2}, zorder=2)


def comparison_figure(data, cases, variants, labels, titles, output, mobile=False, log=False):
    fig = plt.figure(figsize=(4, 10.8) if mobile else (7.6, 6.2))
    fig.text(.075, .965, "FIGURE 2 / TILED SOFTMAX" if log else "FIGURE 1 / ROW SOFTMAX",
             color=MUTED, fontsize=10, fontweight=600)
    title = "The toy queue is slower" if log else "Same math, different execution"
    if mobile:
        title = "The toy queue\nis slower" if log else "Same math,\ndifferent execution"
    fig.text(.075, .95 if mobile else .94, title, color=INK, fontsize=21 if mobile else 22, fontweight=600, va="top", linespacing=1)
    subtitle = "Completion time (µs) · lower is better"
    fig.text(.075, .85 if mobile else .835, subtitle, color=MUTED, fontsize=10.5)
    fig.text(.075, .829 if mobile else .795,
             "Log scale: each major step is 10×" if log else "Each panel has its own linear scale", color=MUTED, fontsize=10)
    maximum = max(data[c][v]["host_single_us"]["p90"] for c in cases for v in variants)*1.2 if log else None
    positions = [(.075, y, .85, .25) for y in [.52, .12]] if mobile else [
        (.075, .16, .39, .50), (.535, .16, .39, .50)]
    for position, case, title in zip(positions, cases, titles):
        ax = fig.add_axes(position)
        comparison(ax, data[case], variants, labels, log=log, maximum=maximum, mobile=mobile)
        ax.text(0, 1.12 if mobile else 1.19, title, transform=ax.transAxes, fontsize=13.5 if mobile else 12,
                color=INK, fontweight=600)
        if log:
            ratio = data[case]["dag_graph"]["host_single_us"]["median"] / data[case]["separate_graph"]["host_single_us"]["median"]
            note = f"Queue takes {ratio:.1f}× as long as graph"
            color = COLORS[3]
        elif case == cases[0]:
            saved = data[case]["staged_stream"]["host_single_us"]["median"] - data[case]["staged_graph"]["host_single_us"]["median"]
            note = f"Graph replay saves {saved:.2f} µs"
            color = COLORS[1]
        else:
            ratio = data[case]["staged_graph"]["host_single_us"]["median"] / data[case]["fused_graph"]["host_single_us"]["median"]
            note = f"Fusion + graph: {ratio:.2f}× faster than graph"
            if not mobile:
                note = f"Fusion + graph: {ratio:.2f}× faster\nthan the four-kernel graph"
            color = COLORS[2]
        ax.text(0, -.23 if mobile else 1.04, note, transform=ax.transAxes, fontsize=10.5 if mobile else 10,
                color=color, fontweight=600)
    fig.text(.075, .022, "H100 · median · whiskers: 10th–90th percentiles",
             color=MUTED, fontsize=9.5)
    save(fig, output)


def readiness_figure(fixture, variants, labels, output, mobile=False):
    """Show exactly the ordering statistic discussed in the article, one dot per row."""
    fig = plt.figure(figsize=(4, 10.2) if mobile else (7.6, 6.2))
    fig.text(.075, .96, "FIGURE 3 / ROW DEPENDENCIES", color=MUTED, fontsize=10, fontweight=600)
    fig.text(.075, .925, "Which rows start B early?", color=INK,
             fontsize=20 if mobile else 24, fontweight=600, va="top")
    fig.text(.075, .875 if mobile else .83,
             "128 uneven rows · before the last A finishes", color=MUTED,
             fontsize=9.5 if mobile else 11)
    positions = [(.075, y, .85, .13) for y in [.625, .445, .265, .085]] if mobile else [
        (.075, .43, .40, .22), (.545, .43, .38, .22),
        (.075, .10, .40, .22), (.545, .10, .38, .22)]
    statistics = {}
    for (left, bottom, width, height), variant, label in zip(positions, variants, labels):
        with (fixture/(variant+"_2_trace.csv")).open() as stream:
            tasks = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(stream)]
        last_a = max(t["end_ns"] for t in tasks if t["stage"] == 0)
        b_tasks = sorted((t for t in tasks if t["stage"] == 1), key=lambda t: t["row"])
        assert [t["row"] for t in b_tasks] == list(range(128))
        early = np.array([t["start_ns"] < last_a for t in b_tasks])
        # Validate the local dependency separately: early means before other rows finish,
        # never before this row's producers finish.
        for b in b_tasks:
            assert b["start_ns"] >= max(t["end_ns"] for t in tasks if t["stage"] == 0 and t["row"] == b["row"])
        count = int(early.sum())
        statistics[variant] = {"early_rows": count, "rows": len(b_tasks), "last_a_ns": last_a}
        ax = fig.add_axes([left, bottom, width, height])
        ax.set_xlim(-.6, 15.6); ax.set_ylim(7.7, -.7); ax.set_axis_off()
        x, y = np.arange(128)%16, np.arange(128)//16
        color = VARIANT_COLORS[variant]
        ax.scatter(x[~early], y[~early], s=26 if mobile else 25, facecolors="#f1f5f9",
                   edgecolors="#b9c5d4", linewidths=.65)
        ax.scatter(x[early], y[early], s=26 if mobile else 25, facecolors=color, edgecolors=color, linewidths=.65)
        ax.text(0, 1.26, label, transform=ax.transAxes, fontsize=13 if mobile else 12.5,
                color=INK, fontweight=600)
        ax.text(0, 1.02, f"{count} / 128 rows started B early", transform=ax.transAxes,
                fontsize=11 if mobile else 11.5, color=color if count else MUTED)
    handles = [Line2D([0],[0],marker="o",linestyle="",markerfacecolor=COLORS[3],markeredgecolor=COLORS[3],label="Starts B early" if mobile else "B starts before last A"),
               Line2D([0],[0],marker="o",linestyle="",markerfacecolor="#f1f5f9",markeredgecolor="#b9c5d4",label="At/after last A" if mobile else "B starts at/after last A")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.055, .86 if mobile else .80),
               ncol=2, frameon=False, fontsize=10, handletextpad=.5, labelcolor=MUTED)
    fig.text(.075, .025, "One dot = one row · rows 0–127, left to right\nOrdering from traces; completion times are in Figure 2.",
             fontsize=9 if mobile else 10, color=MUTED, linespacing=1.5)
    save(fig, output)
    return statistics


def timeline_figure(fixture, variants, labels, output, mobile=False):
    fig = plt.figure(figsize=(4.8, 12) if mobile else (8.6, 10.8))
    fig.text(.07, .966, "SUPPORTING TRACE / TASK ORDERING", color=MUTED, fontsize=10, fontweight=600)
    fig.text(.07, .926, "Task ordering inside the GPU", color=INK,
             fontsize=17 if mobile else 23, fontweight=600)
    fig.text(.07, .893, "128 ragged rows · instrumented task intervals", color=MUTED, fontsize=10.3)
    handles = [Line2D([0], [0], color=c, lw=3, label=l) for c, l in
               zip(STAGES, ["A · tile statistics", "B · combine row", "C · normalize tiles"])]
    handles.append(Line2D([0], [0], color=MUTED, lw=1.2, ls="--", label="Last A ends"))
    if mobile:
        handles = [handles[0], handles[2], handles[1], handles[3]]
    fig.legend(handles=handles, frameon=False, ncol=2 if mobile else 4, loc="upper left",
               bbox_to_anchor=(.055, .867), fontsize=9.6, handlelength=1.5, columnspacing=1.4, labelcolor=MUTED)
    for n, (y, variant, label) in enumerate(zip([.66, .475, .29, .105], variants, labels), 1):
        ax = fig.add_axes([.13, y, .80, .13])
        with (fixture/(variant+"_2_trace.csv")).open() as stream:
            tasks = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(stream)]
        origin = min(t["start_ns"] for t in tasks)
        end = max(t["end_ns"] for t in tasks)
        for stage, color in enumerate(STAGES):
            segments = [[((t["start_ns"]-origin)/1000, t["row"]), ((t["end_ns"]-origin)/1000, t["row"])] for t in tasks if t["stage"] == stage]
            ax.add_collection(LineCollection(segments, colors=color, linewidths=.85, alpha=.88))
        last_a = max(t["end_ns"] for t in tasks if t["stage"] == 0)
        ax.axvline((last_a-origin)/1000, color=MUTED, linestyle=(0, (3, 3)), linewidth=1)
        ax.set_xlim(0, (end-origin)/1000*1.03)
        ax.set_ylim(130, -3)
        ax.set_yticks([0, 64, 127])
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FuncFormatter(tick_label))
        ax.set_title(f"{n:02d}  {label}", loc="left", fontsize=11.6 if mobile else 13, fontweight=600, color=INK, pad=12)
        ax.set_ylabel("Row", color=MUTED, fontsize=10, labelpad=5)
        ax.tick_params(length=0, colors=MUTED, labelsize=9.5, pad=6)
        ax.grid(axis="x", color=LINE, linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[:].set_visible(False)
    fig.text(.07, .035, "Time from first task (µs) · independent panel scales\nTrace intervals do not measure benchmark latency.",
             color=MUTED, fontsize=9.2, linespacing=1.5)
    save(fig, output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    root, out = args.run.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    verification = verify_manifest(root)
    data = {}
    for path in sorted((root/"bench").iterdir()):
        actual = summarize(path)
        assert actual == json.loads((path/"analysis.json").read_text())["timings"], path
        data[path.name] = actual["timing_variants"]
    traces = [{"case": p.name, **r} for p in sorted((root/"trace").iterdir()) for r in check_traces(p)]
    for font in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(font)
    assert font_manager.findfont(font_manager.FontProperties(family="Outfit Article"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Outfit Article", "font.size": 10, "font.weight": 400,
                         "text.color": INK, "svg.hashsalt": "softmax-article-modern", "axes.unicode_minus": False})
    row_variants = ["staged_stream", "staged_graph", "fused_stream", "fused_graph"]
    row_labels = ["Baseline", "CUDA Graphs", "Fusion", "Fusion + graph"]
    row_cases = ["row-1-128-balanced", "row-4096-4096-balanced"]
    dag_variants = ["separate_stream", "separate_graph", "phased_graph", "dag_graph"]
    dag_labels = ["Baseline", "CUDA Graphs", "Fusion + graph", "Megakernel toy + graph"]
    dag_cases = ["dag-128-0-balanced", "dag-128-0-ragged"]
    fixture = root/"trace/dag-128-0-ragged"
    for mobile, suffix in [(False, ""), (True, "-mobile")]:
        comparison_figure(data, row_cases, row_variants, row_labels,
                          ["1 row × 128 elements", "4,096 rows × 4,096 elements"], out/("softmax-row-baseline"+suffix), mobile=mobile)
        comparison_figure(data, dag_cases, dag_variants, dag_labels,
                          ["128 rows · equal lengths", "128 rows · uneven lengths"], out/("softmax-four-way"+suffix), mobile=mobile, log=True)
        readiness = readiness_figure(fixture, dag_variants, dag_labels,
                        out/("softmax-row-readiness"+suffix), mobile=mobile)
        timeline_figure(fixture, dag_variants, dag_labels,
                        out/("softmax-four-way-timeline"+suffix), mobile=mobile)

    inputs = [root/"manifest.json", Path(__file__), *sorted(FONT_DIR.glob("*")), *sorted((root/"bench").glob("*/samples.jsonl")), *sorted((root/"trace").glob("*/*_trace.csv"))]
    mapping = {"softmax-row-baseline": {"cases": row_cases, "variants": dict(zip(row_labels,row_variants))},
               "softmax-four-way": {"cases": dag_cases, "variants": dict(zip(dag_labels,dag_variants))},
               "softmax-four-way-timeline": {"case": "dag-128-0-ragged", "fixture": 2, "variants": dict(zip(dag_labels,dag_variants))},
               "softmax-row-readiness": {"case": "dag-128-0-ragged", "fixture": 2, "variants": dict(zip(dag_labels,dag_variants)), "statistics": readiness}}
    provenance = {"run_id": root.name, "verification": verification, "matplotlib": matplotlib.__version__, "numpy": np.__version__,
                  "metric": "host_single_us", "error_bars": "p10 and p90 of 21 rounds; not confidence intervals", "plots": mapping,
                  "presentation": {"font": "Outfit Article, static weights 400 and 600 from the site's Outfit font; SIL Open Font License",
                                   "colors": "Gray: baseline; blue: graph; teal: fusion (with or without graph); violet: queued megakernel",
                                   "responsive": "Each figure has desktop and -mobile layouts from identical data",
                                   "tiled_comparison": "Shared logarithmic x limits across the two panels",
                                   "row_readiness": "One dot per row, ascending row IDs, filled only if B starts strictly before the global final A end. All local producer dependencies checked.",
                                   "timeline": "Supporting artifact: all task intervals retained, independent linear panel scales; row 0 at the top"},
                  "inputs": [{"path": str(p.relative_to(root)) if p.is_relative_to(root) else p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
                  "outputs": [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.glob("softmax-*.?*")) if p.suffix in (".png", ".svg")]}
    (out/"softmax-figure-provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (out/"softmax-recheck.json").write_text(json.dumps({"run_id": root.name, "verification": verification, "timings": data, "traces": traces}, indent=2)+"\n")
    print(json.dumps({"output": str(out), "timing_cases": len(data), "trace_checks": len(traces), "verification": verification}))


if __name__ == "__main__":
    main()
