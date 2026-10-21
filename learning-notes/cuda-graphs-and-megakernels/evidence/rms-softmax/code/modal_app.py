"""Separate CPU build/archive from sequential bounded H100 phases."""
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
app = modal.App('cuda-graphs-scaled-softmax-row-study')
volume = modal.Volume.from_name('cuda-graphs-scaled-softmax-evidence', create_if_missing=True)
image = (modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu24.04', add_python='3.12')
         .apt_install('build-essential','nlohmann-json3-dev','libglib2.0-0','libnuma1','libtinfo6')
         .pip_install('numpy==2.5.4','nvidia-cudnn-cu12==9.10.2.21', index_url='https://pypi.org/simple')
         .add_local_file(HERE.parent/'cuda_case/install_profilers.py','/workspace/install_profilers.py',copy=True)
         .run_commands('python /workspace/install_profilers.py'))
for name in ('bench.cu','kernels.cuh','build.py'):
    image = image.add_local_file(HERE/name,'/workspace/'+name,copy=True)
image = image.run_commands('python /workspace/build.py')
image = image.apt_install('pigz')
image = image.add_local_dir(HERE,'/workspace',copy=True,
    ignore=lambda p: len(p.parts)!=1 or p.suffix not in ('.py','.json','.md','.svg') or p.name=='build.py')


@app.function(image=image,cpu=4,memory=16384,timeout=1800,retries=0,
              min_containers=0,max_containers=1,scaledown_window=2,volumes={'/evidence':volume})
def cpu_phase(phase,run_id):
    import datetime
    import hashlib
    import json
    import shutil
    import sys
    sys.path.insert(0,'/workspace')
    volume.reload()
    if phase=='build':return {'status':'complete','build':json.loads(Path('/workspace/build.json').read_text())}
    if phase=='prepare':
        previous=run_id
        run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        root=Path('/evidence')/run_id
        (root/'sources').mkdir(parents=True)
        for p in Path('/workspace').iterdir():
            if p.is_file():shutil.copy2(p,root/'sources'/p.name)
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'sources').iterdir() if p.is_file()}
        (root/'source-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
        ledger=[]
        if previous:
            parent=Path('/evidence')/previous
            ledger=json.loads((parent/'gpu-ledger.json').read_text())
            for entry in ledger:entry.setdefault('run_id',previous)
            (root/'previous-run.json').write_text(json.dumps({'run_id':previous,
                'volume':'cuda-graphs-scaled-softmax-evidence',
                'scope':'Preserved full-grid RMS-scaled softmax study; follow-up changes only synchronization scope and retains all controls',
                'source_manifest_sha256':hashlib.sha256((parent/'source-manifest.json').read_bytes()).hexdigest()},indent=2)+'\n')
        (root/'gpu-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
        volume.commit()
        return {'status':'complete','run_id':run_id,'source_files':len(hashes)}
    if phase=='package':
        from archive import package
        result=package(Path('/evidence')/run_id)
        volume.commit()
        return result
    root=Path('/evidence')/run_id
    if phase=='inspect':
        return {'run_id':run_id,'ledger':json.loads((root/'gpu-ledger.json').read_text()),
                'statuses':{str(p.relative_to(root)):json.loads(p.read_text()) for p in root.glob('*/phase-status.json')}}
    raise ValueError(phase)


@app.function(image=image,gpu='H100!',cpu=4,memory=16384,timeout=360,retries=0,
              min_containers=0,max_containers=1,scaledown_window=2,volumes={'/evidence':volume})
def gpu_phase(phase,run_id):
    import sys
    sys.path.insert(0,'/workspace')
    volume.reload()
    from runner import run_phase
    return run_phase(phase,run_id,volume)


@app.local_entrypoint()
def main(phase: str='build',run_id: str=''):
    import hashlib
    import json
    import tarfile
    result=cpu_phase.remote(phase,run_id) if phase in ('build','prepare','package','inspect') else gpu_phase.remote(phase,run_id)
    out=HERE/'artifacts';out.mkdir(exist_ok=True)
    (out/(phase+'-receipt.json')).write_text(json.dumps(result,indent=2)+'\n')
    if phase=='package':
        archive=out/result['review_archive']
        if result['review_archive_bytes']>128*1024**2:raise RuntimeError('Review download exceeds disk bound')
        with archive.open('wb') as f:
            for chunk in volume.read_file(result['review_archive']):f.write(chunk)
        with archive.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=result['review_archive_sha256']:raise RuntimeError('Review archive checksum mismatch')
        with tarfile.open(archive) as tar:tar.extractall(out,filter='data')
        result['archive_verified']=True
        (out/'latest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result.get('status')=='failed':raise RuntimeError('Required phase failed; evidence preserved')
