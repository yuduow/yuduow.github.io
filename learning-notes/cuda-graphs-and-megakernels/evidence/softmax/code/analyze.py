"""Independent float64 references, DAG order checks, and raw-sample summaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ATOL, RTOL, SUM_ATOL = 1e-7, 2e-5, 2e-5


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def close(actual, expected, label):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"shape/finite failure: {label}")
    error = np.abs(actual.astype(np.float64) - expected)
    if not np.all(error <= ATOL + RTOL * np.abs(expected)):
        raise ValueError(f"numerical failure: {label}, max abs error {error.max()}")
    return float(error.max(initial=0))


def validate_case(root):
    meta = records(root / "metadata.jsonl")
    case = next(x for x in meta if x["kind"] == "case")
    offsets = np.fromfile(root / "offsets.i32", dtype="<i4")
    if offsets.size != case["rows"] + 1 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("invalid offsets")
    dag = case["operation"] == "dag"
    names = [f"{v}_{s}" for v in (["separate", "phased", "dag"] if dag else ["staged", "fused", "cudnn"]) for s in ["stream", "graph"]]
    checks = []
    intermediate_checks = 0
    for source in sorted(root.glob("input_*.f32")):
        seed = source.stem.split("_")[-1]
        x = np.fromfile(source, dtype="<f4")
        if x.size != offsets[-1]:
            raise ValueError("input size")
        reference = np.empty(x.size, dtype=np.float64)
        row_max = np.empty(len(offsets)-1)
        row_sum = np.empty_like(row_max)
        for r, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
            row = x[a:b].astype(np.float64)
            row_max[r] = row.max()
            values = np.exp(row-row_max[r])
            row_sum[r] = values.sum()
            reference[a:b] = values / row_sum[r]
        first = None
        stream_outputs = {}
        if dag:
            tile_x = x.reshape(-1, 1024).astype(np.float64)
            tile_max = tile_x.max(axis=1)
            tile_sum = np.exp(tile_x-tile_max[:, None]).sum(axis=1)
            del tile_x
        for name in names:
            prefix = root / f"{name}_{seed}"
            path = prefix.with_suffix(".f32")
            y = np.fromfile(path, dtype="<f4")
            maximum_error = close(y, reference, path)
            sums = np.add.reduceat(y.astype(np.float64), offsets[:-1])
            sum_error = float(np.max(np.abs(sums-1)))
            if sum_error > SUM_ATOL or np.any(y < 0):
                raise ValueError(f"probability invariant: {path}")
            if name.endswith("_stream"):
                stream_outputs[name.removesuffix("_stream")] = y.copy()
            elif not np.array_equal(stream_outputs[name.removesuffix("_graph")].view(np.uint32), y.view(np.uint32)):
                raise ValueError(f"stream/graph bitwise mismatch: {path}")
            if dag:
                if first is None:
                    first = y.copy()
                elif not np.array_equal(first.view(np.uint32), y.view(np.uint32)):
                    raise ValueError(f"same-algorithm bitwise mismatch: {path}")
                for suffix, expected in [("tile_max", tile_max), ("tile_sum", tile_sum), ("row_max", row_max), ("row_sum", row_sum)]:
                    actual = np.fromfile(str(prefix)+f"_{suffix}.f32", dtype="<f4")
                    close(actual, expected, str(prefix)+suffix)
                    intermediate_checks += 1
                if name.startswith("dag_"):
                    counts = np.fromfile(str(prefix)+"_done.i32", dtype="<i4")
                    if not np.array_equal(counts, np.diff(offsets)//1024):
                        raise ValueError("producer counter mismatch")
            checks.append({"file": path.name, "elements": int(y.size), "max_abs_error": maximum_error,
                           "max_row_sum_error": sum_error, "same_algorithm_bitwise_checked": dag})
    if len(checks) != 12:
        raise ValueError(f"expected 12 output checks: {root}, got {len(checks)}")
    return {"case": case, "output_checks": checks, "intermediate_checks": intermediate_checks,
            "atol": ATOL, "rtol": RTOL, "row_sum_atol": SUM_ATOL}


def check_traces(root):
    results = []
    for path in sorted(root.glob("*_trace.csv")):
        data = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(path.open())]
        offsets = np.fromfile(root / "tile_offsets.i32", dtype="<i4")
        tiles, rows = int(offsets[-1]), len(offsets)-1
        expected_count = 2*tiles+rows
        if len(data) != expected_count or {d["id"] for d in data} != set(range(expected_count)):
            raise ValueError(f"missing task: {path}")
        if any(d["visits"] != 1 or not 0 < d["start_ns"] <= d["end_ns"] for d in data):
            raise ValueError(f"duplicate task/invalid interval: {path}")
        by_id = {d["id"]: d for d in data}
        for r in range(rows):
            b = by_id[tiles+r]
            a_end = max(by_id[t]["end_ns"] for t in range(offsets[r], offsets[r+1]))
            if b["start_ns"] < a_end:
                raise ValueError(f"B before its A producers: {path}, row {r}")
            for t in range(offsets[r], offsets[r+1]):
                c = by_id[tiles+rows+t]
                if c["start_ns"] < b["end_ns"]:
                    raise ValueError(f"C before its B: {path}, row {r}")
        last_a = max(d["end_ns"] for d in data if d["stage"] == 0)
        last_b = max(d["end_ns"] for d in data if d["stage"] == 1)
        early_b = sum(d["stage"] == 1 and d["start_ns"] < last_a for d in data)
        early_c_a = sum(d["stage"] == 2 and d["start_ns"] < last_a for d in data)
        early_c_b = sum(d["stage"] == 2 and d["start_ns"] < last_b for d in data)
        if not path.name.startswith("dag_") and (early_b or early_c_b):
            raise ValueError(f"global phase violation: {path}")
        results.append({"file": path.name, "tasks": len(data), "rows": rows, "tiles": tiles,
                        "b_before_last_a": early_b, "c_before_last_a": early_c_a,
                        "c_before_last_b": early_c_b,
                        "span_us": (max(d["end_ns"] for d in data)-min(d["start_ns"] for d in data))/1000,
                        "observed_sms": sorted({d["sm"] for d in data})})
    return results


def summarize(root):
    samples = records(root / "samples.jsonl")
    result = {}
    for name in sorted({s["variant"] for s in samples}):
        group = [s for s in samples if s["variant"] == name]
        if len(group) != 21 or {s["round"] for s in group} != set(range(21)):
            raise ValueError("missing/repeated timing round")
        result[name] = {}
        for metric in ["host_single_us", "host_enqueue_us", "host_batch_us", "event_interval_us"]:
            if metric not in group[0]:
                continue
            values = np.array([s[metric] for s in group])
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError("invalid timing")
            result[name][metric] = dict(zip(["p10", "median", "p90"], map(float, np.percentile(values, [10, 50, 90]))))
    if samples:
        for round_id in range(21):
            group = [s for s in samples if s["round"] == round_id and s["variant"] != "empty_sync"]
            if len(group) != 6 or {s["order"] for s in group} != set(range(6)):
                raise ValueError("incomplete shuffled round")
    pairs = {}
    indexed = {(s["variant"], s["round"]): s for s in samples}
    for name in result:
        if name.endswith("_stream"):
            graph = name.removesuffix("_stream")+"_graph"
            ratios = [indexed[name, r]["host_single_us"]/indexed[graph, r]["host_single_us"] for r in range(21)]
            pairs[name+"/"+graph] = dict(zip(["p10", "median", "p90"], map(float, np.percentile(ratios, [10, 50, 90]))))
    return {"samples": len(samples), "timing_variants": result, "paired_completion_ratios": pairs}


def analyze_case(root):
    result = {"validation": validate_case(root), "traces": check_traces(root), "timings": summarize(root)}
    (root / "analysis.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def verify_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    present, absent = [], []
    for entry in manifest:
        path = root / entry["path"]
        if not path.exists():
            absent.append(entry["path"])
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path}")
        present.append(entry["path"])
    review = root / "review-selection.json"
    if absent and (not review.exists() or set(absent) != {x["path"] for x in json.loads(review.read_text())["omitted"]}):
        raise ValueError("unexplained missing artifacts")
    return {"verified_files": len(present), "remote_only_files": len(absent)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()
    if args.review:
        result = {"hashes": verify_manifest(args.path), "cases": {}}
        for p in sorted((args.path / "bench").iterdir()):
            actual = summarize(p)
            expected = json.loads((p / "analysis.json").read_text())["timings"]
            if actual != expected:
                raise ValueError(f"summary changed: {p}")
            result["cases"][p.name] = {"samples": actual["samples"], "timing_variants": len(actual["timing_variants"])-1}
        result["traces_rechecked"] = sum(len(check_traces(p)) for p in (args.path / "trace").iterdir())
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(analyze_case(args.path), indent=2))


if __name__ == "__main__":
    main()
