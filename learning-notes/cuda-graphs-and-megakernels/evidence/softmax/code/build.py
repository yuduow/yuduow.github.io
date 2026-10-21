"""Compile and disassemble before a GPU is requested."""
import json
from pathlib import Path
import subprocess
import nvidia.cudnn

root = Path("/workspace")
config = json.loads((root/"config.json").read_text())
fixed = {"threads": 256, "tile_elements": 1024, "rounds": 21, "warmups": 10, "batch_invocations": 32,
         "balanced_length": 8192, "ragged_short_length": 4096, "ragged_long_length": 36864,
         "ragged_long_fraction": "1/8", "length_permutation_seed": 20261019}
for name, value in fixed.items():
    if config[name] != value:
        raise ValueError(f"Update the compiled CUDA constants and this guard before changing {name}")
cudnn = Path(next(iter(nvidia.cudnn.__path__)))
flags = ["nvcc", "-O3", "-std=c++17", "-arch=sm_90", "-lineinfo", "--fmad=false", "-I"+str(cudnn/"include")]
link = ["-L"+str(cudnn/"lib"), "-l:libcudnn.so.9", "-Xlinker", "-rpath", "-Xlinker", str(cudnn/"lib")]
commands = [([*flags, "-Xptxas=-v", str(root/"bench.cu"), *link, "-o", str(root/"bench")], "compile.log"),
            ([*flags, "--ptx", str(root/"bench.cu"), "-o", str(root/"bench.ptx")], "ptx.log"),
            (["cuobjdump", "--dump-sass", str(root/"bench")], "bench.sass")]
for args, name in commands:
    run = subprocess.run(args, capture_output=True, text=True)
    (root/name).write_text(run.stdout+run.stderr)
    if run.returncode:
        print(run.stdout, run.stderr, flush=True)
        run.check_returncode()
libraries = [str(p) for p in cudnn.parent.glob("*/lib")]
(root/"build.json").write_text(json.dumps({"commands": [x[0] for x in commands], "library_paths": libraries}, indent=2))
print("CPU compilation and disassembly complete", flush=True)
