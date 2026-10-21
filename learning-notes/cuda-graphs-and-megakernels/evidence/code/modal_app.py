"""One bounded H100 experiment; independent validation gates timing."""
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
app = modal.App("cuda-graphs-megakernel-toys")
volume = modal.Volume.from_name("cuda-graphs-megakernel-toy-evidence", create_if_missing=True)
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
         .apt_install("build-essential", "libglib2.0-0", "libnuma1", "libtinfo6")
         .pip_install("numpy==2.5.4")
         .add_local_file(HERE.parent / "cuda_case/install_profilers.py", "/workspace/install_profilers.py", copy=True)
         .run_commands("python /workspace/install_profilers.py")
         .add_local_file(HERE / "bench.cu", "/workspace/bench.cu", copy=True)
         .add_local_file(HERE / "analyze.py", "/workspace/analyze.py", copy=True)
         .add_local_file(HERE / "README.md", "/workspace/protocol.md", copy=True)
         .add_local_file(HERE / "build.py", "/workspace/build.py", copy=True)
         .run_commands("python /workspace/build.py"))


@app.function(image=image, gpu="H100!", cpu=4, memory=8192, timeout=600, retries=0,
              min_containers=0, max_containers=1, scaledown_window=2, volumes={"/evidence": volume})
def run():
    import datetime
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tarfile
    import time
    sys.path.insert(0, "/workspace")
    from analyze import analyze

    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path("/evidence") / run_id
    out.mkdir()
    for source in ("bench.cu", "analyze.py", "protocol.md", "compile.log", "bench", "bench.ptx", "bench.sass"):
        shutil.copy2(Path("/workspace") / source, out / ("bench-executable" if source == "bench" else source))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MODAL_", "AWS_"))}
    history = []
    started = time.monotonic()

    def command(args, name, timeout=60, required=True):
        before = time.monotonic()
        try:
            p = subprocess.run(args, capture_output=True, text=True, env=env, timeout=timeout)
            result = {"command": args, "returncode": p.returncode, "seconds": time.monotonic() - before}
            (out / (name + ".stdout")).write_text(p.stdout)
            (out / (name + ".stderr")).write_text(p.stderr)
        except subprocess.TimeoutExpired:
            result = {"command": args, "timeout": timeout, "seconds": time.monotonic() - before}
        history.append(result)
        (out / "commands.json").write_text(json.dumps(history, indent=2))
        volume.commit()
        if required and result.get("returncode") != 0:
            raise RuntimeError(f"failed {name}: {result}")
        return result

    status = "failed"
    try:
        command(["nvidia-smi", "-q"], "device-before")
        command(["nvcc", "--version"], "nvcc-version")
        command(["/workspace/bench", str(out / "preflight"), "validate"], "preflight", 60)
        analyze(out / "preflight")
        volume.commit()
        command(["/workspace/bench", str(out / "bench"), "bench"], "benchmark", 90)
        analyze(out / "bench")
        command(["/workspace/bench", str(out / "trace"), "trace"], "trace", 45)
        analyze(out / "trace")
        command(["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=repeat", "--force-overwrite=true", "-o", str(out / "systems"), "/workspace/bench", str(out / "systems-data"), "profile"], "nsys", 60, False)
        for report in sorted(out.glob("systems*.nsys-rep")):
            command(["nsys", "export", "--type=sqlite", "--force-overwrite=true", "-o", str(report.with_suffix(".sqlite")), str(report)], "export-" + report.stem, 30, False)
        for variant in ("separate_graph", "fused_graph"):
            command(["ncu", "--profile-from-start", "off", "--target-processes", "all", "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,launch__registers_per_thread,launch__shared_mem_per_block", "--launch-count", "3", "--force-overwrite", "-o", str(out / ("compute-" + variant)), "/workspace/bench", str(out / ("compute-data-" + variant)), "profile", variant], "ncu-" + variant, 60, False)
            reports = list(out.glob("compute-" + variant + ".ncu-rep*"))
            if reports:
                report = reports[0]
                command(["ncu", "--import", str(report), "--csv", "--page", "raw"], "metrics-" + variant, 30, False)
        command(["nvidia-smi", "-q"], "device-after")
        status = "complete"
    finally:
        (out / "status.json").write_text(json.dumps({"status": status, "elapsed_seconds": time.monotonic() - started, "gpu": "H100!", "function_timeout_seconds": 600}, indent=2))
        hashes = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest()} for p in sorted(out.rglob("*")) if p.is_file()]
        (out / "manifest.json").write_text(json.dumps(hashes, indent=2))
        volume.commit()
        archive = out.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
            tar.add(out, arcname=run_id)
        volume.commit()
    return {"run_id": run_id, "archive": archive.name, "status": status}


@app.local_entrypoint()
def main():
    import json
    import tarfile
    result = run.remote()
    dest = HERE / "artifacts"
    dest.mkdir(exist_ok=True)
    archive = dest / result["archive"]
    with archive.open("wb") as f:
        for chunk in volume.read_file(result["archive"]):
            f.write(chunk)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    (dest / "latest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
