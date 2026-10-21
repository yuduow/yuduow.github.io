"""Offline export/validation of both regular and compressed Nsight reports."""
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
volume = modal.Volume.from_name("cuda-graphs-softmax-evidence")
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
         .apt_install("build-essential", "libglib2.0-0", "libnuma1", "libtinfo6")
         .pip_install("numpy==2.5.4", "nvidia-cudnn-cu12==9.10.2.21", index_url="https://pypi.org/simple")
         .add_local_file(HERE.parent / "cuda_case/install_profilers.py", "/workspace/install_profilers.py", copy=True)
         .run_commands("python /workspace/install_profilers.py"))
app = modal.App("softmax-offline-profile-exports")


@app.function(image=image, cpu=2, memory=4096, timeout=300, retries=0,
              min_containers=0, max_containers=1, scaledown_window=2, volumes={"/evidence": volume})
def export(run_id: str):
    import csv
    import hashlib
    import io
    import json
    import sqlite3
    import subprocess
    import tarfile
    root = Path("/evidence")/run_id
    out = Path("/evidence")/(run_id+"-profile-exports")
    out.mkdir(exist_ok=True)
    summary = {"run_id": run_id, "compute": {}, "systems": {}}
    expected = {"compute-row-staged_graph": 4, "compute-row-fused_graph": 1,
                "compute-row-cudnn_graph": 1, "compute-dag-separate_graph": 3, "compute-dag-dag_graph": 2}
    reports = sorted(p for p in root.glob("*.ncu-rep*") if p.suffix in (".ncu-rep", ".ncu-repz"))
    if len(reports) != len(expected):
        raise ValueError("Missing compute reports")
    for report in reports:
        args = ["ncu", "--import", str(report), "--csv", "--page", "raw", "--print-units", "base"]
        p = subprocess.run(args, capture_output=True, text=True, timeout=45)
        (out/(report.stem+"-metrics.csv")).write_text(p.stdout)
        (out/(report.stem+"-stderr.log")).write_text(p.stderr)
        p.check_returncode()
        lines = p.stdout.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
        parsed = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
        units = next(row for row in parsed if not row["ID"])
        rows = [row for row in parsed if row["ID"].isdigit()]
        ids = sorted({row["ID"] for row in rows})
        if len(ids) != expected[report.stem]:
            raise ValueError(f"Wrong kernel count in {report.name}: {ids}")
        totals = {}
        for metric in ("dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum"):
            if units[metric] != "byte":
                raise ValueError(f"Unexpected counter unit: {metric}: {units[metric]}")
            totals[metric] = sum(float(row[metric].replace(",", "")) for row in rows)
        summary["compute"][report.name] = {"command": args, "kernels": len(ids), "metric_rows": len(rows),
            "traffic_bytes": totals, "source_sha256": hashlib.sha256(report.read_bytes()).hexdigest()}
        print(report.name, len(ids), "kernels", flush=True)
    for report in sorted(root.glob("systems-*.sqlite")):
        db = sqlite3.connect("file:"+str(report)+"?mode=ro", uri=True)
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        table = next(t for t in ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"] if t in tables)
        columns = [r[1] for r in db.execute(f'PRAGMA table_info("{table}")')]
        count = db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        graph_nodes = db.execute(f'SELECT COUNT(*) FROM "{table}" WHERE graphNodeId != 0').fetchone()[0] if "graphNodeId" in columns else None
        if count != 12 or graph_nodes != 6:
            raise ValueError(f"Expected 12 kernel intervals / 6 graph nodes in {report.name}, got {count}/{graph_nodes}")
        summary["systems"][report.name] = {"kernels": count, "graph_node_intervals": graph_nodes, "table": table,
            "source_sha256": hashlib.sha256(report.read_bytes()).hexdigest()}
        db.close()
    (out/"profile-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    manifest = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest.append({"path": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    archive = out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out, arcname=out.name)
    volume.commit()
    return archive.name


@app.local_entrypoint()
def main(run_id: str):
    import tarfile
    archive = export.remote(run_id)
    dest = HERE/"artifacts"/archive
    with dest.open("wb") as stream:
        for chunk in volume.read_file(archive):
            stream.write(chunk)
    with tarfile.open(dest) as tar:
        tar.extractall(dest.parent, filter="data")
    print(dest)
