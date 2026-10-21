"""Independent exact oracles, trace checks, and sample summaries; no GPU use."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def affine_power(a, b, n):
    """Compose affine maps by squaring, independently of device iteration."""
    mul, add = 1, 0
    while n:
        if n & 1:
            mul, add = (a * mul) & 0xffffffff, (a * add + b) & 0xffffffff
        a, b = (a * a) & 0xffffffff, (a * b + b) & 0xffffffff
        n >>= 1
    return np.uint32(mul), np.uint32(add)


def validate(root):
    checks = []
    for case in sorted(root.glob("pointwise_*")):
        for input_path in sorted(case.glob("input_*.f32")):
            fixture = input_path.stem.split("_")[-1]
            x = np.fromfile(input_path, dtype="<f4")
            a = np.add(x, np.float32(1), dtype=np.float32)
            ref = np.multiply(np.multiply(a, a, dtype=np.float32), np.float32(.5), dtype=np.float32)
            outputs = list(case.glob(f"*_fixture{fixture}.f32"))
            if len(outputs) != 4:
                raise ValueError(f"missing variants: {case} fixture {fixture}")
            for output in outputs:
                y = np.fromfile(output, dtype="<f4")
                if y.shape != ref.shape or not np.isfinite(y).all() or not np.array_equal(y, ref):
                    raise ValueError(f"incorrect pointwise output: {output}")
                checks.append({"path": str(output.relative_to(root)), "elements": len(y), "exact": True})
    for case in sorted(root.glob("tiles_*")):
        work = np.fromfile(case / "work.i32", dtype="<i4")
        bmul, badd = affine_power(22695477, 1, 512)
        for input_path in sorted(case.glob("input_*.u32")):
            fixture = input_path.stem.split("_")[-1]
            x = np.fromfile(input_path, dtype="<u4").reshape(len(work), 256)
            ref_a = np.empty_like(x)
            for j, iterations in enumerate(work):
                mul, add = affine_power(1664525, 1013904223, int(iterations))
                ref_a[j] = x[j] * mul + add
            ref_y = ref_a * bmul + badd
            outputs = list(case.glob(f"*_fixture{fixture}.u32"))
            if len(outputs) != 6:
                raise ValueError(f"missing tile variants: {case} fixture {fixture}")
            for output in outputs:
                for path, ref in ((output, ref_y), (output.with_name(output.stem + "_intermediate.u32"), ref_a)):
                    actual = np.fromfile(path, dtype="<u4")
                    if actual.shape != ref.ravel().shape or not np.array_equal(actual, ref.ravel()):
                        raise ValueError(f"incorrect tile data: {path}")
                    checks.append({"path": str(path.relative_to(root)), "elements": len(actual), "exact": True})
    if not checks:
        raise ValueError("no validation artifacts")
    return checks


def trace_checks(root):
    results = []
    for path in sorted(root.glob("tiles_*/*_trace.csv")):
        rows = list(csv.DictReader(path.open()))
        rows = [{k: int(v) for k, v in r.items()} for r in rows]
        jobs = np.fromfile(path.parent / "work.i32", dtype="<i4").size
        if len(rows) != jobs or {r["job"] for r in rows} != set(range(jobs)):
            raise ValueError(f"trace job coverage: {path}")
        for r in rows:
            if not 0 < r["a0_ns"] <= r["a1_ns"] <= r["b0_ns"] <= r["b1_ns"]:
                raise ValueError(f"dependency order: {path} job {r['job']}")
        last_a = max(r["a1_ns"] for r in rows)
        early_b = sum(r["b0_ns"] < last_a for r in rows)
        if "_separate_" in path.name and early_b:
            raise ValueError("separate kernels violated full completion dependency")
        results.append({"path": str(path.relative_to(root)), "jobs": jobs, "b_started_before_last_a": early_b,
                        "instrumented_span_us": (max(r["b1_ns"] for r in rows) - min(r["a0_ns"] for r in rows)) / 1000,
                        "observed_sm_ids": sorted({r[k] for r in rows for k in ("a_sm", "b_sm")})})
    return results


def summaries(root):
    path = root / "samples.jsonl"
    samples = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    result = []
    for case, variant in sorted({(s["case"], s["variant"]) for s in samples}):
        group = [s for s in samples if s["case"] == case and s["variant"] == variant]
        if len(group) != 21 or {s["round"] for s in group} != set(range(21)):
            raise ValueError("incomplete timing group")
        entry = {"case": case, "variant": variant, "rounds": len(group)}
        for metric in ("host_single_us", "host_enqueue_us", "host_batch_us", "event_interval_us"):
            values = [s[metric] for s in group]
            if not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError("invalid timing sample")
            entry[metric] = dict(zip(("p10", "median", "p90"), map(float, np.percentile(values, (10, 50, 90)))))
        result.append(entry)
    return result


def analyze(root):
    result = {"validation": validate(root), "traces": trace_checks(root), "timings": summaries(root)}
    (root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    outputs = {}
    for name in ("preflight", "bench", "trace"):
        if (root / name).exists():
            r = analyze(root / name)
            outputs[name] = {"checks": len(r["validation"]), "trace_checks": len(r["traces"]), "timing_groups": len(r["timings"])}
    if not outputs:
        r = analyze(root)
        outputs["root"] = {"checks": len(r["validation"]), "trace_checks": len(r["traces"]), "timing_groups": len(r["timings"])}
    (root / "verification.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
