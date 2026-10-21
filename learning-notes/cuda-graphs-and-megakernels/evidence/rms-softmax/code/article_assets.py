"""Recheck collected evidence and create the single-workload article assets on CPU."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jsonl(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def audit(run):
    manifest = {e['path']: e for e in json.loads((run/'manifest.json').read_text())}
    used = []
    def checked(p):
        key = str(p.relative_to(run))
        assert key in manifest, key
        assert p.stat().st_size == manifest[key]['bytes'] and digest(p) == manifest[key]['sha256'], key
        used.append(p)
        return p
    cfg = json.loads(checked(run/'selected-configs.json').read_text())
    saved = json.loads(checked(run/'report/timing-summary.json').read_text()) if 'report/timing-summary.json' in manifest else json.loads((run/'report/timing-summary.json').read_text())
    timing = {}
    groups = 0
    for a in (1, 2):
        for b in (1, 16, 32, 64):
            timing[a, b] = jsonl(checked(run/f'measurement-{a}/data/b{b}/timings.jsonl'))
            warm = [r for r in timing[a,b] if 'cache_policy' not in r and r['variant'] != 'empty_sync']
            assert len({r['variant'] for r in warm}) == 17
            for v in {r['variant'] for r in warm}:
                rs = [r for r in warm if r['variant'] == v]
                assert len(rs) == 21 and all(r['batch'] == 32 for r in rs)
                for metric in ('host_single_us','host_enqueue_us','host_batch_us','event_interval_us'):
                    q = np.percentile([r[metric] for r in rs], [10,50,90])
                    ref = saved[f'{a}-{b}'][v][metric]
                    assert np.allclose(q, [ref['p10'],ref['median'],ref['p90']], rtol=0,atol=1e-10)
                    groups += 1
            local = cfg[str(b)]['local_best'] + '_graph'
            controls = ['local_materialize_graph','local_recompute_graph','local_materialize2_graph','local_recompute2_graph','streaming_graph']
            assert local == min(controls, key=lambda v: saved[f'{a}-{b}'][v]['event_interval_us']['median'])
    metadata = {}
    for a in (1, 2):
        entries = jsonl(checked(run/f'measurement-{a}/data/metadata.jsonl'))
        metadata[a] = entries
        for b in (1, 16, 32, 64):
            row = [e for e in entries if e.get('rows') == b]
            assert next(e for e in row if e['family']=='baseline')['kernel_nodes'] == 9
            assert next(e for e in row if e['family']=='local_recompute')['kernel_nodes'] == 3
            controls = [e for e in row if e['family'] in ('roundtrip_off','input_reload','roundtrip_on','both_reload')]
            assert len(controls) == 4 and len({json.dumps(e['resources'],sort_keys=True) for e in controls}) == 1
            assert len({(e['W'],e['T'],e['slab_rows']) for e in controls}) == 1
            for f in ('retained','row_retained'):
                e = next(e for e in row if e['family'] == f)
                assert e['kernel_nodes'] == (2 if b==64 else 1)
        checked(run/f'measurement-{a}/data/device.json')
    checks = 0
    maximum_l1 = 0
    for path in sorted(run.rglob('validation.jsonl')):
        checked(path)
        for e in jsonl(path):
            xfile, yfile = path.parent/e['input'], path.parent/e['output']
            if not xfile.exists() or not yfile.exists():
                continue
            checked(xfile); checked(yfile)
            x = np.fromfile(xfile,dtype='<f4').reshape(e['rows'],e['columns']).astype(np.float64)
            y = np.fromfile(yfile,dtype='<f4').reshape(x.shape).astype(np.float64)
            scale = 1/np.sqrt(np.sum(x*x,axis=1,keepdims=True)/x.shape[1]+1e-5)
            numer = np.exp((x-np.max(x,axis=1,keepdims=True))*scale)
            ref = numer/np.sum(numer,axis=1,keepdims=True)
            error = np.abs(y-ref)
            assert np.all(np.isfinite(y)) and np.all(y>=0)
            assert np.all(error<=1e-7+2e-5*np.abs(ref))
            l1 = float(error.sum(axis=1).max())
            assert l1<=2e-5 and np.max(np.abs(y.sum(axis=1)-1))<=2e-5
            maximum_l1=max(maximum_l1,l1); checks+=1
    assert checks>=100
    traces = 0
    for path in sorted((run/'evidence/trace-data').rglob('trace-*.csv')):
        checked(path)
        rs=list(csv.DictReader(path.open()))
        for r in rs:
            for k in ('row','tile','start_ns','end_ns','visits','slab'):r[k]=int(r[k])
            assert r['visits']==1 and 0<r['start_ns']<=r['end_ns']
        assert len({(r['phase'],r['row'],r['tile']) for r in rs})==len(rs)
        two='local_materialize2' in path.name or 'local_recompute2' in path.name
        def deps(group):
            a=[r for r in group if r['phase']=='A'];b=[r for r in group if r['phase']=='B'];c=[r for r in group if r['phase']=='C']
            assert a and c
            if two:assert not b and min(r['start_ns'] for r in c)>=max(r['end_ns'] for r in a)
            else:
                assert b and min(r['start_ns'] for r in b)>=max(r['end_ns'] for r in a)
                assert min(r['start_ns'] for r in c)>=max(r['end_ns'] for r in b)
        for row in {r['row'] for r in rs}:deps([r for r in rs if r['row']==row])
        if 'row_retained' not in path.name:
            for slab in {r['slab'] for r in rs}:deps([r for r in rs if r['slab']==slab])
        traces+=1
    assert traces==66
    summary=json.loads(checked(run/'summary.json').read_text())
    assert summary['required_checks_passed'] and summary['gpu_seconds']<=900
    for name in ('bench.cu','kernels.cuh','config.json','PROTOCOL.md'):
        checked(run/'sources'/name)
    unique=sorted(set(used))
    return timing,cfg,metadata,{'run_id':run.name,'timing_metric_groups':groups,
        'fresh_small_output_checks':checks,'fresh_max_row_l1_error':maximum_l1,'fresh_trace_checks':traces,
        'previous_full_array_audit':{k:summary[k] for k in ('output_checks','intermediate_checks','bitwise_groups','stress_replays','sanitizer_checks','profiler_captures')},
        'scope':'Fresh local checks cover available small arrays, all final-study timing variants and dependency traces. Full-size numerical checks were performed in the archived independent CPU audit; no GPU rerun.',
        'verified_sources':[{'path':str(p.relative_to(run)),'sha256':digest(p)} for p in unique]}


def build(run,post,appendix):
    timing,cfg,metadata,review=audit(run)
    dest=post/'evidence/rms-softmax';dest.mkdir(parents=True,exist_ok=True)
    assets=post/'assets';assets.mkdir(exist_ok=True)
    fontroot=Path(__file__).parent/'fonts'
    if not fontroot.exists():fontroot=Path(__file__).parent.parent/'graphs_softmax/fonts'
    fonts=list(fontroot.glob('*.ttf'));assert fonts
    for f in fonts:font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({'font.family':'Outfit Article','font.size':12,'axes.titlesize':16,'text.color':'#142536',
        'axes.labelcolor':'#526579','xtick.color':'#526579','ytick.color':'#34485f','svg.fonttype':'path','savefig.dpi':180,
        'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False})
    from article_figures import FUSION_PHASES, render_figures
    workload = json.loads((run/'sources/config.json').read_text())['workload']
    assert all(e['columns'] == workload['columns'] for entries in metadata.values()
               for e in entries if e.get('kind') == 'variant')
    points, paired, completion_summaries, retention_summaries, comparisons, layouts, outputs, graph = render_figures(timing, cfg, assets, workload)
    # Selected website evidence: never copy large arrays or raw profiler databases.
    files=[]
    for name in ('selected-configs.json','summary.json','gpu-ledger.json','frozen-manifest.json','source-manifest.json','gpu-teardown.json','archive-readback-verification.json'):
        files.append((run/name,'data/'+name))
    for name in ('bench.cu','kernels.cuh','config.json','PROTOCOL.md','build.py','analyze.py','runner.py','modal_app.py'):
        files.append((run/'sources'/name,'code/'+name))
    for pattern in ('measurement-*/data/metadata.jsonl','measurement-*/data/device.json','measurement-*/data/b*/timings.jsonl','evidence/trace-data/b*/trace-*.csv','evidence/*.metrics.csv'):
        files.extend((p,'data/'+str(p.relative_to(run))) for p in run.glob(pattern))
    for name in ('FULL_RESULTS.md','all-timing-quantiles.csv','raw-timing-samples.csv','manifest.json'):
        files.append((appendix/name,'full-results/'+name))
    for name in ('counter-summary.json','analysis.json','resource-summary.json','archive-index.json'):
        files.append((run/'report'/name,'data/'+name))
    for src,rel in files:
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
    for f in fonts+[fontroot/'OFL.txt',fontroot/'README.md']:
        if f.exists():
            target=dest/'code/fonts'/f.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    shutil.copyfile(__file__,dest/'code/article_assets.py')
    plot_script = Path(__file__).with_name('article_figures.py')
    shutil.copyfile(plot_script,dest/'code/article_figures.py')
    (dest/'validation-recheck.json').write_text(json.dumps(review,indent=2)+'\n')
    provenance={'run_id':run.name,'script_sha256':digest(Path(__file__)),'font_hashes':{f.name:digest(f) for f in fonts},'plot_script_sha256':digest(plot_script),'computation_graph':graph,
        'local_fusion_diagram':{'variant':'local_recompute','phases':FUSION_PHASES,
                               'scope':'Kernel grouping and global-memory handoffs; not a measured timeline.'},
        'graph_replay_diagram':{'kernel_order':[name for phase in FUSION_PHASES for name,_ in phase['baseline']],
                                'direct_launch_calls':9,'replay_calls':1,'kernel_nodes':9,
                                'scope':'Repeated host submission after capture and instantiation; not a timing trace.'},
        'input_shapes':{'axis_order':['rows','columns'],'columns':workload['columns'],
                        'rows':sorted({b for _,b in timing}),'primary_rows':workload['primary_rows'],
                        'source':'sources/config.json'},
        'completion_points':points,'paired_retention_points':paired,
        'aggregation':{'center':'Arithmetic mean of the two per-session medians; sessions have equal weight.',
                       'whiskers':'Minimum and maximum of those two medians; not a confidence interval or within-session quantiles.',
                       'retention':'Within each session, first take the median of round-paired changes against roundtrip_off_graph.',
                       'speedup':'Ratio of averaged per-session median latencies; not an average of per-session speedup ratios.'},
        'completion_summaries':completion_summaries,'retention_summaries':retention_summaries,
        'completion_comparisons':comparisons,
        'figure_layouts':layouts,
        'presentation':{'completion':'Compare methods within each shape. Each panel uses a zero-based linear axis fitted to its observations, with its range in the heading and a visible warning that panel scales differ. Bar lengths are not comparable across panels. Annotations give megakernel latency reduction relative to local fusion.',
                        'retention':'Dot-and-range plot compares extra latency against the fixed-binary retained control; zero-based linear axis with adjacent value labels. Controls are not additive components.'},
        'diagram_scope':'Conceptual storage lifetimes for one row group, not a measured timeline. The selected local fusion variant recomputes exponentials.',
        'figures':[{'path':str(p.relative_to(post)),'sha256':digest(p)} for p in outputs],
        'source_hashes':review['verified_sources']}
    (dest/'figure-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'run_id':run.name,'figure_files':len(outputs),'rechecked_outputs':review['fresh_small_output_checks'],'timing_metric_groups':review['timing_metric_groups'],'traces':review['fresh_trace_checks'],'evidence_files':len(files)}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--post',type=Path,required=True);ap.add_argument('--appendix',type=Path,required=True);args=ap.parse_args();build(args.run,args.post,args.appendix)
