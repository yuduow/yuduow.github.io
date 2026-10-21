"""Independent CPU validation of saved data, graph topology and dependencies."""
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import numpy as np


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(root):
    root=Path(root)
    cfg=json.loads((root/'sources/config.json').read_text())
    phases=('pilot','measurement-1','measurement-2','evidence')
    for phase in phases:
        status=json.loads((root/phase/'phase-status.json').read_text())
        assert status['status']=='complete',phase
        assert all(x.get('returncode')==0 for x in status['commands'] if x['required']),phase
    for name,digest in json.loads((root/'source-manifest.json').read_text()).items():
        with (root/'sources'/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,name
    selected=json.loads((root/'selected-configs.json').read_text())
    freeze=json.loads((root/'frozen-manifest.json').read_text())
    assert hashlib.sha256((root/'selected-configs.json').read_bytes()).hexdigest()==freeze['selected_configs_sha256']
    array_checks=[];output_groups={};maximum_l1=0
    for validations in sorted(root.rglob('validation.jsonl')):
        # Every executable writes paths relative to its invocation output root.
        invocation=validations.parent
        entries=jsonl(validations)
        by_input={}
        for e in entries:by_input.setdefault(e['input'],[]).append(e)
        for input_name,records in by_input.items():
            first=records[0];rows,n=first['rows'],first['columns']
            input_path=invocation/input_name
            x=np.fromfile(input_path,dtype='<f4').reshape(rows,n)
            digest=hashlib.sha256(x.tobytes()).hexdigest()
            x64=x.astype(np.float64);assert np.all(np.isfinite(x64)) and np.max(np.abs(x64))<=1024;rm=x64.max(axis=1);scale=1/np.sqrt(np.mean(x64*x64,axis=1)+1e-5);e64=np.exp((x64-rm[:,None])*scale[:,None]);z=e64.sum(axis=1);reference=e64/z[:,None]
            tile_refs={}
            for item in records:
                y=np.fromfile(invocation/item['output'],dtype='<f4').reshape(rows,n)
                error=np.abs(y.astype(np.float64)-reference)
                assert np.all(np.isfinite(y)) and np.all(y>=0),item['output']
                assert np.all(error<=1e-7+2e-5*np.abs(reference)),item['output']
                row_l1=error.sum(axis=1);row_sum=np.abs(y.sum(axis=1,dtype=np.float64)-1)
                assert row_l1.max()<=2e-5 and row_sum.max()<=2e-5,item['output']
                assert item['input_unchanged'] is True
                maximum_l1=max(maximum_l1,float(row_l1.max()))
                output_groups.setdefault((digest,rows,n,item['W'],item['T']),[]).append({
                    'family':item['family'],'output':str((invocation/item['output']).relative_to(root)),
                    'sha256':hashlib.sha256(y.tobytes()).hexdigest()})
                check={'output':str((invocation/item['output']).relative_to(root)),'rows':rows,'columns':n,
                       'max_absolute_error':float(error.max()),'max_row_l1':float(row_l1.max()),'max_row_sum_error':float(row_sum.max()),
                       'intermediates':0}
                if item['intermediates']:
                    w=item['W'];k=(n+w-1)//w
                    if w not in tile_refs:
                        padded=np.zeros((rows,k*w),dtype=np.float64);padded[:,:n]=x64
                        energy=(padded.reshape(rows,k,w)**2).sum(axis=2)
                        maxima=padded.copy();maxima[:,n:]=-np.inf;tm=maxima.reshape(rows,k,w).max(axis=2)
                        epad=np.zeros((rows,k*w));epad[:,:n]=e64;ts=epad.reshape(rows,k,w).sum(axis=2)
                        tile_refs[w]={'tm':tm,'ts':ts,'energy':energy,'rm':rm,'scale':scale,'z':z}

                    for name,file in item['intermediates'].items():
                        data=np.fromfile(invocation/file,dtype='<f4')
                        if name=='e':
                            target=e64
                        elif name=='squared':target=x64*x64
                        else:target=tile_refs[w][name]
                        np.testing.assert_allclose(data.reshape(target.shape),target,atol=0 if name in ('z','energy') else 1e-7,rtol=2e-5,err_msg=file)
                        check['intermediates']+=1
                array_checks.append(check)
    # Same arithmetic at the same geometry: compare all preserved invocations,
    # including performance, trace, sanitizer and profiler builds.
    bitwise=[]
    for key,items in output_groups.items():
        matched=[i for i in items if i['family'] not in ('cudnn','streaming','local_materialize2','local_recompute2')]
        if len(matched)>1:
            assert len({i['sha256'] for i in matched})==1,matched
            bitwise.append({'input_sha256':key[0],'rows':key[1],'columns':key[2],
                            'W':key[3],'T':key[4],'outputs':len(matched),'sha256':matched[0]['sha256']})
    stress=json.loads((root/'pilot/data/structured-b16-n131072/stress.json').read_text())
    extended='local_recompute2' in selected['16']
    assert len(stress)==12 and all(e['changing_input_replays']==1000 and e['failures']==0 for e in stress)
    assert json.loads((root/'pilot/data/structured-b16-n131072/checker-selftest.json').read_text())['intentionally_zeroed_output_detected'] is True

    graphs=[]
    for path in sorted(root.rglob('metadata.jsonl')):
        for e in jsonl(path):
            if e.get('kind')!='variant':continue
            expected=9 if e['family']=='baseline' else 3 if e['family'].startswith('local_') else 1
            if e['family'] in ('local_materialize2','local_recompute2'):expected=2
            if e['family'] in ('retained','row_retained','roundtrip_off','roundtrip_on','input_reload','both_reload'):
                expected=(e['rows']+e['slab_rows']-1)//e['slab_rows']
                assert all(v==1 for v in e['cooperative'])
                resource=e['resources']['cooperative'];k=(e['columns']+e['W']-1)//e['W']
                device=json.loads((path.parent/'device.json').read_text())
                assert e['slab_rows']*k<=device['sm_count']*resource['active_blocks_per_sm']
            if e['family']=='cudnn':expected=e['kernel_nodes']
            assert e['nodes']==e['kernel_nodes']==expected,e
            graphs.append({'metadata':str(path.relative_to(root)),**e})
    traces=[]
    for path in sorted((root/'evidence/trace-data').rglob('trace-*.csv')):
        with path.open() as f:stamps=list(csv.DictReader(f))
        for row in stamps:
            for key in ('row','tile','start_ns','end_ns','sm','block','visits','slab'):row[key]=int(row[key])
            assert row['visits']==1 and row['start_ns']>0 and row['end_ns']>=row['start_ns'],path
        assert len({(s['phase'],s['row'],s['tile']) for s in stamps})==len(stamps)
        rows=sorted({s['row'] for s in stamps})
        two='local_materialize2' in path.name or 'local_recompute2' in path.name
        for r in rows:
            a=[s for s in stamps if s['phase']=='A' and s['row']==r]
            b=[s for s in stamps if s['phase']=='B' and s['row']==r]
            c=[s for s in stamps if s['phase']=='C' and s['row']==r]
            assert len(a)==len(c)>0
            if two:
                assert not b and min(s['start_ns'] for s in c)>=max(s['end_ns'] for s in a),path
            else:
                assert len(b)==len(a) and min(s['start_ns'] for s in b)>=max(s['end_ns'] for s in a),path
                assert min(s['start_ns'] for s in c)>=max(s['end_ns'] for s in b),path
        # Row-scoped barriers require only within-row ordering, checked above.
        for slab in ([] if 'row_retained' in path.name else {s['slab'] for s in stamps}):
            a=[s for s in stamps if s['phase']=='A' and s['slab']==slab]
            b=[s for s in stamps if s['phase']=='B' and s['slab']==slab]
            c=[s for s in stamps if s['phase']=='C' and s['slab']==slab]
            if two:assert min(s['start_ns'] for s in c)>=max(s['end_ns'] for s in a),path
            else:
                assert min(s['start_ns'] for s in b)>=max(s['end_ns'] for s in a),path
                assert min(s['start_ns'] for s in c)>=max(s['end_ns'] for s in b),path
        traces.append({'file':str(path.relative_to(root)),'tasks':len(stamps),'rows':len(rows),'slabs':len({s['slab'] for s in stamps}),'passed':True})
    assert len(traces)==66,len(traces)
    sanitizers=[]
    for tool in ('memcheck','racecheck','synccheck'):
        for shape in ('2-8193','16-131072'):
            path=root/'evidence'/f'{tool}-{shape}.stdout';err=path.with_suffix('.stderr')
            log=path.read_text()+err.read_text()
            assert 'ERROR SUMMARY: 0 errors' in log or (tool=='racecheck' and 'RACECHECK SUMMARY: 0 hazards' in log),str(path)
            sanitizers.append(str(path.relative_to(root)))
    profiles=[]
    specs=[(16,v) for v in ('baseline','local_materialize','local_recompute','retained','row_retained','roundtrip_off','roundtrip_on','input_reload','both_reload','streaming','cudnn')]
    if extended:specs += [(16,'local_materialize2'),(16,'local_recompute2')]
    specs += [(64,selected['64']['local_best']),(64,'retained'),(64,'row_retained')]
    for rows,family in specs:
        for tool in ('nsys','ncu'):
            stem=root/'evidence'/f'{tool}-{rows}-{family}'
            metadata=jsonl(Path(str(stem)+'-data/metadata.jsonl'))
            expected=metadata[-1]['kernel_nodes']
            if tool=='nsys':
                report=stem.with_suffix('.nsys-rep');database=stem.with_suffix('.sqlite')
                assert report.exists() and database.exists(),str(stem)
                with sqlite3.connect(database) as db:
                    count=db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
            else:
                assert stem.with_suffix('.ncu-rep').exists() or stem.with_suffix('.ncu-repz').exists(),str(stem)
                with stem.with_suffix('.metrics.csv').open() as f:records=[r for r in csv.DictReader(f) if r.get('ID','').isdigit()]
                count=len({r['ID'] for r in records})
                for metric in ('dram__bytes_read.sum','dram__bytes_write.sum','lts__t_bytes.sum','smsp__inst_executed.sum'):
                    assert all(r.get(metric) not in (None,'','n/a') for r in records),metric
            assert count==expected,(str(stem),count,expected)
            profiles.append({'tool':tool,'rows':rows,'family':family,'kernels':count,'passed':True})
    timing=[]
    for allocation in (1,2):
        for rows in (1,16,32,64):
            path=root/f'measurement-{allocation}/data/b{rows}/timings.jsonl';records=jsonl(path)
            warm=[r for r in records if r.get('cache_policy')!='external_eviction' and r['variant']!='empty_sync']
            variants={r['variant'] for r in warm}
            assert len(variants)==17,(path,len(variants))
            assert all(len([r for r in warm if r['variant']==v])==21 for v in variants)
            assert all(np.isfinite(r[m]) and r[m]>0 for r in warm for m in ('host_single_us','host_enqueue_us','host_batch_us','event_interval_us'))
            timing.append({'allocation':allocation,'rows':rows,'records':len(warm),'variants':sorted(variants)})
    devices=[json.loads((root/f'measurement-{i}/data/device.json').read_text()) for i in (1,2)]
    assert devices[0]['uuid']!=devices[1]['uuid'],'Measurements require two physical GPU allocations'
    assert any(e.get('captured_execution') is False for f in root.rglob('validation.jsonl') for e in jsonl(f)), 'Missing direct execution checks'
    ledger=json.loads((root/'gpu-ledger.json').read_text());seconds=sum(e.get('elapsed_seconds',e['limit_seconds']) for e in ledger)
    assert seconds<=900,seconds
    # Resource and instruction evidence for both sides of the runtime flag.
    sass=(root/'sources/bench.sass').read_text();sass_checks=[]
    for rows in (1,16,32,64):
        g=selected[str(rows)]['retained'];w,t=g['W'],g['T']
        parts=re.split(r'Function\s*:\s*',sass)
        bodies={flag:next(p for p in parts if f'retainedILi{w}ELi{t}ELb{flag}E' in p.splitlines()[0]) for flag in (0,1)}
        counts={flag:{op:len(re.findall(r'\b'+op+r'\b',body)) for op in ('LDG','STG')} for flag,body in bodies.items()}
        assert counts[1]['LDG']>counts[0]['LDG'] and counts[1]['STG']>counts[0]['STG'],counts
        sass_checks.append({'rows':rows,'W':w,'T':t,'static_global_access_instruction_counts':counts})
    summary={'run_id':root.name,'required_checks_passed':True,'output_checks':len(array_checks),
             'intermediate_checks':sum(c['intermediates'] for c in array_checks),'bitwise_groups':len(bitwise),
             'trace_checks':len(traces),'traced_tasks':sum(t['tasks'] for t in traces),'stress_replays':sum(s['changing_input_replays'] for s in stress),
             'sanitizer_checks':len(sanitizers),'profiler_captures':len(profiles),'gpu_seconds':seconds,
             'max_row_l1_error':maximum_l1,'timing':timing,'sass_controls':sass_checks}
    for name,value in [('independent-array-checks.json',array_checks),('bitwise-checks.json',bitwise),('graph-checks.json',graphs),('trace-checks.json',traces),('profiler-checks.json',profiles),('summary.json',summary)]:
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    return summary
