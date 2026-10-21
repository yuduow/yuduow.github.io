"""Bounded H100 execution, CPU archival, and a compact local review bundle."""
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
app = modal.App("cuda-graphs-softmax-study")
volume = modal.Volume.from_name("cuda-graphs-softmax-evidence", create_if_missing=True)
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
         .apt_install("build-essential", "libglib2.0-0", "libnuma1", "libtinfo6")
         .pip_install("numpy==2.5.4", "nvidia-cudnn-cu12==9.10.2.21", index_url="https://pypi.org/simple")
         .add_local_file(HERE.parent / "cuda_case/install_profilers.py", "/workspace/install_profilers.py", copy=True)
         .run_commands("python /workspace/install_profilers.py"))
for name in ("bench.cu", "analyze.py", "README.md", "config.json", "build.py", "modal_app.py"):
    image = image.add_local_file(HERE / name, "/workspace/" + name, copy=True)
image = image.run_commands("python /workspace/build.py")


@app.function(image=image, gpu="H100!", cpu=4, memory=16384, timeout=900, retries=0,
              min_containers=0, max_containers=1, scaledown_window=2, volumes={"/evidence": volume})
def run():
    import datetime
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    import traceback
    sys.path.insert(0, "/workspace")
    from analyze import analyze_case

    config = json.loads(Path("/workspace/config.json").read_text())
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("/evidence") / run_id
    out.mkdir()
    for source in Path("/workspace").iterdir():
        if source.is_file():
            shutil.copy2(source, out / ("bench-executable" if source.name == "bench" else source.name))
    env = dict(os.environ)
    libs = json.loads((out / "build.json").read_text())["library_paths"]
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    history = []
    started = time.monotonic()

    def command(args, name, timeout=60, required=True):
        before = time.monotonic()
        print(f"{time.monotonic()-started:.1f}s {name}", flush=True)
        result = {"name": name, "command": list(map(str, args)), "required": required}
        try:
            p = subprocess.run(result["command"], capture_output=True, text=True, env=env, timeout=timeout)
            result["returncode"] = p.returncode
            stdout, stderr = p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            result["timeout"] = timeout
            stdout, stderr = e.stdout or b"", e.stderr or b""
        for suffix, value in [("stdout", stdout), ("stderr", stderr)]:
            (out / f"{name}.{suffix}").write_text(value.decode(errors="replace") if isinstance(value, bytes) else value)
        result["seconds"] = time.monotonic() - before
        history.append(result)
        (out / "commands.json").write_text(json.dumps(history, indent=2))
        volume.commit()
        if required and result.get("returncode") != 0:
            raise RuntimeError(f"failed {name}; see preserved stdout/stderr")
        return result

    def case(kind, rows, columns, pattern, mode, group=None):
        name = f"{kind}-{rows}-{columns}-{pattern}"
        path = out / (group or mode) / name
        command(["/workspace/bench", path, kind, rows, columns, pattern, mode], f"{group or mode}-{name}")
        result = analyze_case(path)
        volume.commit()
        print(f"validated {mode} {name}: {len(result['validation']['output_checks'])} outputs", flush=True)
        return path

    status, error = "failed", None
    try:
        command(["nvidia-smi", "-q"], "device-before")
        command(["nvcc", "--version"], "nvcc-version")
        command([sys.executable, "-m", "pip", "freeze"], "python-packages")
        command(["nsys", "--version"], "nsys-version", required=False)
        command(["ncu", "--version"], "ncu-version", required=False)
        command(["nsys", "profile", "--help"], "nsys-profile-help", required=False)
        command(["compute-sanitizer", "--version"], "sanitizer-version")
        # Two small cases exercise masks and cross-block queue synchronization.
        for kind, rows, cols, pattern in [("row", 7, 781, "balanced"), ("dag", 8, 0, "ragged")]:
            path = out / "sanitizer" / kind
            command(["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99",
                     "/workspace/bench", path, kind, rows, cols, pattern, "validate"], "memcheck-"+kind, 90)
            analyze_case(path)
            volume.commit()
        path = out / "sanitizer" / "dag-synccheck"
        command(["compute-sanitizer", "--tool", "synccheck", "--error-exitcode", "99",
                 "/workspace/bench", path, "dag", 8, 0, "ragged", "validate"], "synccheck-dag", 90)
        analyze_case(path)
        volume.commit()
        for rows, cols in config["validation_only_row_cases"]:
            case("row", rows, cols, "balanced", "validate", "validation-only")
        specs = [("row", r, c, "balanced") for r, c in config["row_cases"]]
        specs += [("dag", r, 0, p) for r in config["dag_rows"] for p in config["dag_patterns"]]
        for spec in specs:
            case(*spec, "validate", "preflight")
            case(*spec, "bench")
        for pattern in config["dag_patterns"]:
            case("dag", config["trace_rows"], 0, pattern, "trace")
        profiles = [("row", *config["profile_row_shape"], "balanced"),
                    ("dag", config["profile_dag_rows"], 0, "ragged")]
        for kind, rows, cols, pattern in profiles:
            command(["nsys", "profile", "--trace=cuda", "--cuda-graph-trace=node", "--sample=none",
                     "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                     "--force-overwrite=true", "-o", out / ("systems-"+kind), "/workspace/bench",
                     out / ("systems-data-"+kind), kind, rows, cols, pattern, "profile"], "nsys-"+kind, 90, False)
        metrics = "dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,launch__registers_per_thread,launch__shared_mem_per_block,sm__warps_active.avg.pct_of_peak_sustained_active"
        for kind, rows, cols, pattern in profiles:
            variants = ["staged_graph", "fused_graph", "cudnn_graph"] if kind == "row" else ["separate_graph", "dag_graph"]
            for variant in variants:
                if time.monotonic()-started > 700:
                    history.append({"name": "ncu-"+kind+"-"+variant, "skipped": "preserve GPU time bound", "required": False})
                    continue
                tag = kind+"-"+variant
                command(["ncu", "--profile-from-start", "off", "--target-processes", "all", "--cache-control", "all",
                         "--replay-mode", "kernel", "--metrics", metrics, "--force-overwrite", "-o", out / ("compute-"+tag),
                         "/workspace/bench", out / ("compute-data-"+tag), kind, rows, cols, pattern, "profile", variant],
                        "ncu-"+tag, 90, False)
        command(["nvidia-smi", "-q"], "device-after")
        status = "complete"
    except Exception:
        error = traceback.format_exc()
        (out / "failure.txt").write_text(error)
        print(error, flush=True)
    finally:
        (out / "commands.json").write_text(json.dumps(history, indent=2))
        (out / "status.json").write_text(json.dumps({"status": status, "error": error,
            "elapsed_seconds": time.monotonic()-started, "gpu": "H100!", "function_timeout_seconds": 900,
            "counter_cache_policy": "all: flush for kernel replay; separate from warm timing measurements",
            "counter_replay_policy": "kernel"}, indent=2))
        volume.commit()
    return {"run_id": run_id, "status": status, "volume": "cuda-graphs-softmax-evidence"}


@app.function(image=image, cpu=4, memory=8192, timeout=900, retries=0,
              min_containers=0, max_containers=1, scaledown_window=2, volumes={"/evidence": volume})
def package(result):
    """No GPU: export profiler tables, hash all raw files, and build both archives."""
    import hashlib
    import json
    import subprocess
    import tarfile
    import time
    volume.reload()
    root = Path("/evidence") / result["run_id"]
    exports = []
    for report in sorted(root.glob("*.nsys-rep")):
        args = ["nsys", "export", "--type=sqlite", "--force-overwrite=true", "-o", str(report.with_suffix(".sqlite")), str(report)]
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=60)
            (root/(report.stem+"-export.log")).write_text(p.stdout+p.stderr)
            exports.append({"command": args, "returncode": p.returncode})
        except subprocess.TimeoutExpired:
            exports.append({"command": args, "timeout": 60})
    for report in sorted(root.glob("*.ncu-rep")):
        args = ["ncu", "--import", str(report), "--csv", "--page", "raw"]
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=60)
            (root/(report.stem+"-metrics.csv")).write_text(p.stdout)
            (root/(report.stem+"-export.stderr")).write_text(p.stderr)
            exports.append({"command": args, "returncode": p.returncode})
        except subprocess.TimeoutExpired:
            exports.append({"command": args, "timeout": 60})
    (root/"exports.json").write_text(json.dumps(exports, indent=2))
    analyses = {str(p.parent.relative_to(root)): json.loads(p.read_text()) for p in sorted(root.rglob("analysis.json"))}
    summary = {"run": result, "analyzed_cases": len(analyses),
               "output_checks": sum(len(a["validation"]["output_checks"]) for a in analyses.values()),
               "intermediate_checks": sum(a["validation"]["intermediate_checks"] for a in analyses.values()),
               "timing_records": sum(a["timings"]["samples"] for a in analyses.values()),
               "trace_checks": sum(len(a["traces"]) for a in analyses.values()), "cases": analyses}
    (root/"summary.json").write_text(json.dumps(summary, indent=2))
    manifest = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in ("manifest.json", "review-selection.json"):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest})
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2))
    volume.commit()
    full = root.with_suffix(".tar.gz")
    with tarfile.open(full, "w:gz", compresslevel=1) as tar:
        tar.add(root, arcname=root.name)
    # Keep all sources, timings, task CSVs, graph DOTs, metadata, logs, counters, and small fixtures locally.
    omitted = [e for e in manifest if (e["path"].endswith(".f32") and e["bytes"] > 65536)
               or Path(e["path"]).suffix in (".nsys-rep", ".ncu-rep", ".sqlite") or e["path"] == "bench-executable"]
    (root/"review-selection.json").write_text(json.dumps({"omitted": omitted, "full_archive": full.name,
        "reason": "Large raw data remain on the Modal Volume; local disk space is limited."}, indent=2))
    omitted_paths = {e["path"] for e in omitted}
    review = root.parent / (root.name+"-review.tar.gz")
    with tarfile.open(review, "w:gz", compresslevel=1) as tar:
        for path in sorted(root.rglob("*")):
            if path.is_file() and str(path.relative_to(root)) not in omitted_paths:
                tar.add(path, arcname=str(Path(root.name)/path.relative_to(root)))
    volume.commit()
    result.update({"full_archive": full.name, "full_archive_bytes": full.stat().st_size,
                   "review_archive": review.name, "review_archive_bytes": review.stat().st_size,
                   "raw_files": len(manifest), "raw_bytes": sum(e["bytes"] for e in manifest),
                   "output_checks": summary["output_checks"], "trace_checks": summary["trace_checks"],
                   "timing_records": summary["timing_records"]})
    return result


@app.local_entrypoint()
def main():
    import json
    import tarfile
    result = package.remote(run.remote())
    dest = HERE / "artifacts"
    dest.mkdir(exist_ok=True)
    archive = dest / result["review_archive"]
    with archive.open("wb") as stream:
        for chunk in volume.read_file(result["review_archive"]):
            stream.write(chunk)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    (dest / "latest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "complete":
        raise RuntimeError("Required experiment check failed; partial artifacts were preserved.")
