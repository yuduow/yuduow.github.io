"""Compile and retain both binaries, PTX and SASS before requesting a GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import nvidia.cudnn

root = Path('/workspace')
cudnn = Path(next(iter(nvidia.cudnn.__path__)))
flags = ['nvcc', '-O3', '-std=c++17', '-arch=sm_90', '-lineinfo', '--fmad=false',
         '-Xptxas=-v', '-I' + str(cudnn/'include')]
link = ['-L'+str(cudnn/'lib'), '-l:libcudnn.so.9', '-Xlinker', '-rpath', '-Xlinker', str(cudnn/'lib')]
commands = []
for name, defines in [('bench', []), ('bench-trace', ['-DSTUDY_TRACE=1'])]:
    for args, log in [
        ([*flags, *defines, str(root/'bench.cu'), *link, '-o', str(root/name)], name+'-compile.log'),
        ([*flags, *defines, '--ptx', str(root/'bench.cu'), '-o', str(root/(name+'.ptx'))], name+'-ptx.log'),
        (['cuobjdump', '--dump-sass', str(root/name)], name+'.sass'),
    ]:
        p = subprocess.run(args, capture_output=True, text=True)
        (root/log).write_text(p.stdout+p.stderr)
        commands.append({'args': args, 'returncode': p.returncode, 'log': log})
        if p.returncode:
            print(p.stdout[-8000:]+p.stderr[-8000:], flush=True)
            p.check_returncode()
    print('CPU compiled and disassembled '+name, flush=True)
names = ('bench.cu','kernels.cuh','build.py','bench','bench-trace')
(root/'build.json').write_text(json.dumps({
    'commands': commands,
    'library_paths': [str(p) for p in cudnn.parent.glob('*/lib')],
    'hashes': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
}, indent=2)+'\n')
