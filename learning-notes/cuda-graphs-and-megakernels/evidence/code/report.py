"""Build figures and a bounded interpretation from verified saved artifacts."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
POST = HERE.parents[1] / "posts/cuda-graphs-megakernels"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_manifest(root):
    records = json.loads((root / "manifest.json").read_text())
    for record in records:
        path = root / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("artifact integrity: " + record["path"])
    return len(records)


def cuda_activities(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    rows = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        if table in tables:
            for item in db.execute(f"SELECT * FROM {table} ORDER BY start"):
                r = dict(item)
                # Restrict export to performance data, excluding host process metadata.
                entry = {k: r[k] for k in ("start", "end", "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ", "registersPerThread", "staticSharedMemory", "dynamicSharedMemory", "graphNodeId", "graphId", "graphExecId") if k in r}
                entry["type"] = table
                if "demangledName" in r:
                    name = db.execute("SELECT value FROM StringIds WHERE id=?", (r["demangledName"],)).fetchone()
                    entry["kernel"] = name[0] if name else None
                rows.append(entry)
    db.close()
    return rows


def metrics(path):
    if not path.exists():
        return []
    text = path.read_text()
    start = text.find('"ID"')
    if start < 0:
        return []
    records = list(csv.DictReader(io.StringIO(text[start:])))
    units = next((r for r in records if not r.get("ID")), {})
    keys = ("dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum", "launch__registers_per_thread", "launch__shared_mem_per_block")
    return [{"id": int(r["ID"]), "kernel": r["Kernel Name"], "metrics": {k: {"value": float(r[k].replace(",", "")), "unit": units.get(k, "")} for k in keys if r.get(k)}} for r in records if r.get("ID", "").isdigit()]


def main(root):
    raw_count = verify_manifest(root)
    analysis = json.loads((root / "bench/analysis.json").read_text())
    trace = json.loads((root / "trace/analysis.json").read_text())
    device = json.loads((root / "bench/metadata.jsonl").read_text().splitlines()[0])
    status = json.loads((root / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("campaign incomplete")
    rows = analysis["timings"]
    selected = {r["variant"]: r for r in rows}
    samples = [json.loads(line) for line in (root / "bench/samples.jsonl").read_text().splitlines()]
    ratios = []
    for case in sorted({s["case"] for s in samples}):
        base = case + "_separate_stream"
        baseline = {s["round"]: s for s in samples if s["variant"] == base}
        for variant in sorted({s["variant"] for s in samples if s["case"] == case}):
            group = [s for s in samples if s["variant"] == variant]
            ratios.append({"case": case, "variant": variant, "baseline": base, "paired_host_speedup_median": float(np.median([baseline[s["round"]]["host_single_us"] / s["host_single_us"] for s in group]))})
    activities = {p.name: cuda_activities(p) for p in sorted(root.glob("systems*.sqlite"))}
    metric_rows = {variant: metrics(root / ("metrics-" + variant + ".stdout")) for variant in ("separate_graph", "fused_graph")}
    (root / "cuda-activities.json").write_text(json.dumps(activities, indent=2) + "\n")
    (root / "paired-ratios.json").write_text(json.dumps(ratios, indent=2) + "\n")
    (root / "dram-metrics.json").write_text(json.dumps(metric_rows, indent=2) + "\n")
    assets = POST / "assets"
    assets.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    ns = [256, 32768, 1048576, 16777216]
    styles = [("separate_stream", "Separate · stream", "#4477aa"), ("separate_graph", "Separate · graph", "#66aacc"), ("fused_stream", "Fused · stream", "#cc6677"), ("fused_graph", "Fused · graph", "#aa3377")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), layout="constrained")
    for axis, metric, title in zip(axes, ("host_single_us", "event_interval_us"), ("Synchronized host invocation", "Batched CUDA event interval per invocation")):
        for key, label, color in styles:
            vals = [selected[f"pointwise_{n}_{key}"][metric] for n in ns]
            y = np.array([r["median"] for r in vals])
            errors = [y - [r["p10"] for r in vals], np.array([r["p90"] for r in vals]) - y]
            axis.errorbar(ns, y, yerr=errors, marker="o", capsize=3, label=label, color=color)
        axis.set(xscale="log", yscale="log", title=title, xlabel="FP32 elements N", ylabel="µs per invocation")
        axis.set_xticks(ns, ["256", "32K", "1M", "16M"])
        axis.grid(alpha=.15)
    axes[0].legend(fontsize=9)
    fig.suptitle("Submission and local fusion change different parts of the same workflow", fontweight="bold")
    for suffix in ("svg", "png"):
        fig.savefig(assets / ("pointwise-latency." + suffix), dpi=160)
    plt.close(fig)

    paths = [root / f"trace/tiles_ragged/tiles_ragged_{variant}_graph_fixture0_trace.csv" for variant in ("separate", "fused", "queue")]
    data = [[{k: int(v) for k, v in r.items()} for r in csv.DictReader(p.open())] for p in paths]
    sms = sorted({r[k] for group in data for r in group for k in ("a_sm", "b_sm")})
    rank = {sm: i for i, sm in enumerate(sms)}
    origins = [min(r["a0_ns"] for r in group) for group in data]
    xmax = max((max(r["b1_ns"] for r in group) - origin) / 1000 for group, origin in zip(data, origins))
    fig, axes = plt.subplots(1, 3, figsize=(15, 7.5), sharey=True, layout="constrained")
    for ax, group, origin, title in zip(axes, data, origins, ("Separate A and B kernels", "Fused block per job", "Persistent worker queue")):
        for r in group:
            for stage, color in (("a", "#4477aa"), ("b", "#ee9977")):
                begin, end = r[stage + "0_ns"], r[stage + "1_ns"]
                ax.broken_barh([((begin - origin) / 1000, (end - begin) / 1000)], (rank[r[stage + "_sm"]] - .45, .9), facecolors=color, linewidth=0)
        ax.set(title=title, xlim=(0, xmax * 1.025), ylim=(-1, len(sms)), xlabel="µs from first recorded A start")
        ax.grid(axis="x", alpha=.15)
    axes[0].set_ylabel("Actual SM ID (dense sorted display)")
    ticks = list(range(0, len(sms), max(1, len(sms) // 8)))
    axes[0].set_yticks(ticks, [str(sms[t]) for t in ticks])
    fig.legend(handles=[Patch(facecolor="#4477aa", label="A producer"), Patch(facecolor="#ee9977", label="B consumer")], loc="outside lower center", ncol=2)
    fig.suptitle("Recorded tile intervals: ready work can proceed while other producers remain", fontweight="bold")
    for suffix in ("svg", "png"):
        fig.savefig(assets / ("ragged-timeline." + suffix), dpi=160)
    plt.close(fig)

    lines = ["# H100 toy example results", "", f"Saved run `{root.name}` on {device['name']}, {device['sms']} SMs, {device['l2_bytes']:,} L2 bytes. Runtime {device['runtime']}; driver API {device['driver']}. Function work elapsed {status['elapsed_seconds']:.1f} seconds, excluding final hashing and archive construction. Hardware/clock state is retained in device-before/after files.", "", f"All {raw_count} originally indexed files passed SHA-256 and size verification. Independent CPU oracles checked preflight, benchmark, and trace outputs. The failed earlier run is retained separately on the Modal Volume and contributes no timings.", "", "## Pointwise example", "", "Medians over 21 shuffled rounds. Single = submission through stream completion. Enqueue = batched host submission only. Event = batched device timeline interval per invocation, including possible host starvation; not an isolated kernel duration. p10/p90 and all samples remain in analysis.json and samples.jsonl.", "", "| N | Variant | Host single µs | Host enqueue µs | Event interval µs |", "|---:|---|---:|---:|---:|"]
    for n in ns:
        for key, _, _ in styles:
            row = selected[f"pointwise_{n}_{key}"]
            lines.append(f"| {n:,} | {key} | {row['host_single_us']['median']:.3f} | {row['host_enqueue_us']['median']:.3f} | {row['event_interval_us']['median']:.3f} |")
    lines += ["", "![Pointwise timings](../../posts/cuda-graphs-megakernels/assets/pointwise-latency.png)", "", "The graph node checks confirm 3 kernel nodes for separate pointwise and 1 for fused. Arithmetic and pointers stay fixed within comparisons. Logical global-memory array traffic is 24N versus 8N bytes. DRAM metrics below must not be equated to those logical counts.", "", "## Ragged and balanced scheduling", "", "| Case | Variant | Host single µs | Event interval µs |", "|---|---|---:|---:|"]
    for row in rows:
        if row["case"].startswith("tiles_"):
            lines.append(f"| {row['case']} | {row['variant'].removeprefix(row['case'] + '_')} | {row['host_single_us']['median']:.3f} | {row['event_interval_us']['median']:.3f} |")
    lines += ["", "Separate B waits for all A. Fused and queue variants retain the global intermediate while allowing each job's B after its own A. Occupancy checks fix one active CTA per SM through an artificial shared-memory reservation. Queue reset is included in every invocation and graph; it is a memset node in addition to the persistent kernel.", "", "| Trace fixture 0 | Jobs starting B before the last A completes | Instrumented span µs |", "|---|---:|---:|"]
    for r in trace["traces"]:
        if "tiles_ragged" in r["path"] and "_graph_fixture0" in r["path"]:
            lines.append(f"| {Path(r['path']).name} | {r['b_started_before_last_a']} / {r['jobs']} | {r['instrumented_span_us']:.3f} |")
    lines += ["", "![Recorded stage intervals](../../posts/cuda-graphs-megakernels/assets/ragged-timeline.png)", "", "Each panel is a separate instrumented launch aligned to its first A timestamp, with a common horizontal scale. Each mark comes from thread-zero globaltimer observations around a stage ending with a block barrier. Bars are recorded intervals, not a direct SM utilization counter. All observed SMs and jobs are included; gaps are not an independently measured breakdown of idle causes. Instrumentation, changed scheduling, and tracing overhead exclude these spans from the latency comparison.", "", "## Profiling evidence", "", f"Nsight Systems exported {len(activities)} SQLite captures and {sum(len(v) for v in activities.values())} CUDA activity records. Only CUDA activity records and directly referenced kernel names are exported in cuda-activities.json. Raw reports remain private.", ""]
    for key, records in metric_rows.items():
        lines.append(f"- {key}: {len(records)} exported metric rows. Raw CSV is metrics-{key}.stdout; counters were collected separately from timing.")
    lines += ["", "Systems records 9 separate-stream kernel activities, 3 fused-stream kernel activities, and 6 whole-graph activities. Its default graph capture records graph executions rather than individual node durations. Graph DOT/API checks and Compute's named graph-kernel samples supply the node evidence; do not infer missing node times from the Systems capture.", "", "Compute counters use the default all-cache-flush policy per replay, separate from the warm-cache benchmark. Auto-scaled CSV units are preserved per metric; Mbyte is decimal MB. Kernel-scoped store counters may omit dirty cache lines written back outside that kernel's interval. The counters cannot establish a 3× HBM ratio in the warm loop. [Nsight Compute cache control](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#profiling).", "", "Missing or failed profiling commands are recorded in commands.json and their stderr; availability of a report alone does not prove useful counters. Separate-graph Compute samples cover one three-kernel invocation; fused samples cover three one-kernel invocations, so compare per-invocation traffic using kernel IDs rather than summing both files indiscriminately.", "", "Graph setup metadata retains a field named first_replay_sync_us. Its original interval also includes graph queries and DOT serialization after instantiation; it is not a clean first-replay latency and is excluded from all performance claims. Capture and instantiation fields retain their narrower scopes. Original indexed files remain unchanged; local revalidation writes analysis-local.json.", "", "## Scope", "", "These measurements establish this toy workload on one allocation. They do not reproduce historical LLM benchmarks, isolate universal launch constants, benchmark PDL, or validate a cross-SM semaphore runtime. The persistent queue has a deliberately simple scheduler and resource limit. Its result is not a general comparison of persistent versus nonpersistent execution.", ""]
    (HERE / "RESULTS.md").write_text("\n".join(lines))
    provenance = {"run": str(root.relative_to(HERE)), "verified_raw_files": raw_count, "figures": {name: {"sha256": digest(assets / name), "sources": [str((root / p).relative_to(HERE)) for p in (("bench/samples.jsonl",) if name.startswith("pointwise") else tuple(str(x.relative_to(root)) for x in paths))]} for name in ("pointwise-latency.svg", "pointwise-latency.png", "ragged-timeline.svg", "ragged-timeline.png")}}
    (POST / "experiment-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"raw_files_verified": raw_count, "timing_groups": len(rows), "traces": len(trace["traces"]), "systems_captures": len(activities), "metric_rows": {k: len(v) for k, v in metric_rows.items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root.resolve())
