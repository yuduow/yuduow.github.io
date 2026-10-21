"""Sequential GPU phases with a persistent 900-second failure-inclusive ledger."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor


def run_phase(phase,run_id,volume):
    started=time.monotonic()
    limit=360 if phase=='evidence' else 240 if phase=='pilot' else 180
    root=Path('/evidence')/run_id
    ledger_path=root/'gpu-ledger.json'
    ledger=json.loads(ledger_path.read_text())
    spent=sum(e.get('elapsed_seconds',e['limit_seconds']) for e in ledger)
    if spent+limit>900:raise RuntimeError('Insufficient remaining failure-inclusive GPU budget')
    if any(e['phase']==phase and e.get('run_id',run_id)==run_id for e in ledger):raise RuntimeError('Phase already attempted; preserve failure and explicitly review before retry')
    ledger.append({'phase':phase,'run_id':run_id,'limit_seconds':limit,'status':'running'})
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n');volume.commit()
    hashes=json.loads((root/'source-manifest.json').read_text())
    for name,digest in hashes.items():
        # Figure/review/report code can be added after execution; the frozen sources
        # and all run-time inputs still come from this checked image.
        p=Path('/workspace')/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen image/source mismatch: '+name)
    temporary=tempfile.TemporaryDirectory(prefix='scaled-softmax-')
    out=Path(temporary.name)/phase;out.mkdir()
    env=dict(os.environ)
    libs=json.loads(Path('/workspace/build.json').read_text())['library_paths']
    env['LD_LIBRARY_PATH']=':'.join(libs+[env.get('LD_LIBRARY_PATH','')])
    commands=[]

    def command(args,tag,timeout=60,required=True):
        remaining=limit-(time.monotonic()-started)-15
        if remaining<2:raise TimeoutError('Preserve GPU phase limit; no time for '+tag)
        timeout=min(timeout,remaining)
        print(f'{time.monotonic()-started:.1f}s {tag}',flush=True)
        record={'name':tag,'command':list(map(str,args)),'required':required}
        before=time.monotonic()
        try:
            with (out/(tag+'.stdout')).open('w') as stdout,(out/(tag+'.stderr')).open('w') as stderr:
                p=subprocess.run(record['command'],stdout=stdout,stderr=stderr,text=True,env=env,timeout=timeout)
            record['returncode']=p.returncode
        except subprocess.TimeoutExpired:
            record['timeout_seconds']=timeout
        record['seconds']=time.monotonic()-before
        commands.append(record)
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if required and record.get('returncode')!=0:
            print((out/(tag+'.stderr')).read_text()[-5000:],flush=True)
            raise RuntimeError('Required command failed: '+tag)

    status='failed';error=None
    try:
        command(['nvidia-smi','-q'],'device-before',10)
        command(['nvcc','--version'],'nvcc-version',10)
        command(['nsys','--version'],'nsys-version',10)
        command(['ncu','--version'],'ncu-version',10)
        command(['compute-sanitizer','--version'],'sanitizer-version',10)
        selected=root/'selected-configs.json'
        if phase=='pilot':
            command(['/workspace/bench',out/'data','pilot'],'pilot',205)
            shutil.copy2(out/'data/selected-configs.json',selected)
            freeze={'source_manifest_sha256':hashlib.sha256((root/'source-manifest.json').read_bytes()).hexdigest(),
                    'selected_configs_sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),
                    'selection':'disjoint pilot fixture before either measurement allocation'}
            (root/'frozen-manifest.json').write_text(json.dumps(freeze,indent=2)+'\n')
        elif phase.startswith('measurement-'):
            freeze=json.loads((root/'frozen-manifest.json').read_text())
            assert hashlib.sha256(selected.read_bytes()).hexdigest()==freeze['selected_configs_sha256']
            command(['/workspace/bench',out/'data','measurement',selected,phase[-1]],'measurement',140)
        elif phase=='evidence':
            if not (root/'frozen-manifest.json').exists():raise RuntimeError('Missing pilot freeze')
            for tool in ('memcheck','racecheck','synccheck'):
                for rows,n in ((2,8193),(16,131072)):
                    tag=f'{tool}-{rows}-{n}'
                    command(['compute-sanitizer','--tool',tool,'--error-exitcode','99',
                             '/workspace/bench',out/(tag+'-data'),'sanitizer',selected,rows,n],tag,60)
            command(['/workspace/bench-trace',out/'trace-data','trace',selected],'traces',45)
            configs=json.loads(selected.read_text())
            profile_specs=[(16,v) for v in ('baseline','local_materialize','local_recompute','local_materialize2','local_recompute2','retained','row_retained',
                                          'roundtrip_off','roundtrip_on','input_reload','both_reload','streaming','cudnn')]
            profile_specs += [(64,configs['64']['local_best']),(64,'retained'),(64,'row_retained')]
            for rows,variant in profile_specs:
                tag=f'nsys-{rows}-{variant}'
                command(['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none',
                         '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true',
                         '-o',out/tag,'/workspace/bench',out/(tag+'-data'),'profile',selected,rows,variant],tag,40)
            metrics=','.join(('dram__bytes_read.sum','dram__bytes_write.sum','lts__t_bytes.sum',
                'launch__registers_per_thread','launch__shared_mem_per_block',
                'sm__warps_active.avg.pct_of_peak_sustained_active','smsp__inst_executed.sum',
                'smsp__warp_issue_stalled_barrier_per_warp_active.pct',
                'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'))
            for rows,variant in profile_specs:
                tag=f'ncu-{rows}-{variant}'
                command(['ncu','--profile-from-start','off','--target-processes','all',
                         '--replay-mode','application','--cache-control','none','--clock-control','none',
                         '--metrics',metrics,'--force-overwrite','-o',out/tag,
                         '/workspace/bench',out/(tag+'-data'),'profile',selected,rows,variant],tag,45)
        else:raise ValueError(phase)
        command(['nvidia-smi','-q'],'device-after',10)
        status='complete'
    except Exception:
        error=traceback.format_exc();(out/'failure.txt').write_text(error);print(error,flush=True)
    finally:
        target=root/phase
        files=[p for p in out.rglob('*') if p.is_file()]
        def copy(p):
            destination=target/p.relative_to(out);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination)
        with ThreadPoolExecutor(max_workers=24) as pool:list(pool.map(copy,files))
        elapsed=time.monotonic()-started
        ledger[-1].update(status=status,elapsed_seconds=elapsed)
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
        result={'run_id':run_id,'phase':phase,'status':status,'elapsed_seconds':elapsed,
                'limit_seconds':limit,'commands':commands,'error':error,'source_verified':True}
        (target/'phase-status.json').write_text(json.dumps(result,indent=2)+'\n')
        volume.commit()
        temporary.cleanup()
    return result
