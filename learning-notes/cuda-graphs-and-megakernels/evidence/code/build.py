"""Build on CPU and retain compiler diagnostics, including failures."""
from pathlib import Path
import subprocess

flags = ["nvcc", "-O3", "-std=c++17", "-arch=sm_90", "-lineinfo", "--fmad=false"]
for args, name in (([*flags, "-Xptxas=-v", "/workspace/bench.cu", "-o", "/workspace/bench"], "compile"),
                   ([*flags, "--ptx", "/workspace/bench.cu", "-o", "/workspace/bench.ptx"], "ptx"),
                   (["cuobjdump", "--dump-sass", "/workspace/bench"], "sass")):
    process = subprocess.run(args, text=True, capture_output=True)
    Path("/workspace/" + ("bench.sass" if name == "sass" else name + ".log")).write_text(process.stdout + process.stderr)
    if process.returncode:
        print(process.stdout, process.stderr, flush=True)
        process.check_returncode()
