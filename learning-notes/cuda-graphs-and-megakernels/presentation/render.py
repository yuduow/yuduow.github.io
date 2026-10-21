"""Regenerate article presentation figures from verified, archived CPU-side data."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

import figures

HERE = Path(__file__).resolve().parent
POST = HERE.parent
EVIDENCE = POST / 'evidence/rms-softmax'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = json.loads((EVIDENCE / 'figure-provenance.json').read_text())
    manifest = {f['path']: f for f in json.loads((EVIDENCE / 'bundle-manifest.json').read_text())['files']}
    inputs = []

    def checked(relative):
        path = EVIDENCE / relative
        assert digest(path) == manifest[relative]['sha256'], relative
        inputs.append({'path': relative, 'sha256': digest(path)})
        return path

    timing = {}
    metadata = {}
    cfg = json.loads(checked('data/selected-configs.json').read_text())
    workload = json.loads(checked('code/config.json').read_text())['workload']
    for name in ('bench.cu', 'kernels.cuh'):
        checked('code/' + name)
    for session in (1, 2):
        metadata[session] = [json.loads(line) for line in checked(f'data/measurement-{session}/data/metadata.jsonl').read_text().splitlines()]
        for rows in (1, 16, 32, 64):
            path = checked(f'data/measurement-{session}/data/b{rows}/timings.jsonl')
            timing[session, rows] = [json.loads(line) for line in path.read_text().splitlines()]
            variants = ['baseline_stream', 'baseline_graph', cfg[str(rows)]['local_best'] + '_graph', cfg[str(rows)]['mega_best'] + '_graph']
            for variant in variants:
                samples = [r['host_single_us'] for r in timing[session, rows] if r['variant'] == variant and 'cache_policy' not in r]
                assert len(samples) == 21
                expected = next(p for p in original['completion_summaries'] if p['rows'] == rows and p['variant'] == variant)
                assert abs(float(np.median(samples)) - expected['session_medians'][session-1]) < 1e-10
            entries = [r for r in metadata[session] if r.get('rows') == rows]
            for family, launches in [('baseline', 9), ('local_recompute', 3), (cfg[str(rows)]['mega_best'], 2 if rows == 64 else 1)]:
                entry = next(r for r in entries if r.get('family') == family)
                assert entry['kernel_nodes'] == launches
                assert entry['columns'] == workload['columns']

    for font in (EVIDENCE / 'code/fonts').glob('*.ttf'):
        assert digest(font) == original['font_hashes'][font.name]
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': 'Outfit Article', 'font.size': 12,
        'svg.fonttype': 'path', 'savefig.dpi': 180,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False})

    bounds = []
    polish = figures.polish

    def inspect(fig, mobile):
        polish(fig, mobile)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = list(fig.texts)
        for ax in fig.axes:
            texts.extend(ax.texts)
            if ax.axison:
                texts.extend(ax.get_xticklabels())
                texts.extend([ax.xaxis.label, ax.title, ax._left_title])
        items = [(t.get_text(), t.get_window_extent(renderer)) for t in texts if t.get_visible() and t.get_text()]
        outside = [s for s, box in items if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1]
        collisions = []
        for i, (s, a) in enumerate(items):
            for t, b in items[i+1:]:
                if min(a.x1,b.x1)-max(a.x0,b.x0) > 2 and min(a.y1,b.y1)-max(a.y0,b.y0) > 2:
                    collisions.append([s,t])
        bounds.append({'mobile': mobile, 'heading': items[0][0], 'outside': outside, 'font_metric_overlaps': collisions})

    figures.polish = inspect
    assets = HERE / 'assets'
    assets.mkdir(exist_ok=True)
    points, paired, completion, retention, comparisons, layouts, outputs, graph = figures.render_figures(timing, cfg, assets, workload)
    for key, actual in [('completion_points', points), ('paired_retention_points', paired),
                        ('completion_summaries', completion), ('retention_summaries', retention),
                        ('completion_comparisons', comparisons), ('figure_layouts', layouts), ('computation_graph', graph)]:
        assert json.loads(json.dumps(actual)) == original[key], key
    result = {'run_id': original['run_id'], 'archived_provenance_sha256': digest(EVIDENCE/'figure-provenance.json'),
        'scope': 'Reader-facing presentation. Archived evidence, original figures, all observations and aggregation remain unchanged.',
        'review': {'completion_session_medians_verified': 32, 'retention_session_medians_verified': 6,
                  'launch_counts_verified': 24, 'numeric_and_axis_equality': True,
                  'clarifications': ['Tile-lifetime view explicitly identifies the row-scoped variant.', 'Graph kernels are labeled by dependency order without implying internal execution-stream assignment.']},
        'style': {'font': 'Outfit Article / website Outfit', 'desktop_type_scale': 1.12, 'mobile_type_scale': 1.08,
                  'palette': figures.COLORS},
        'script_sha256': digest(Path(__file__)), 'figures_script_sha256': digest(HERE/'figures.py'),
        'inputs': inputs, 'layout_check_method': 'Font metric rectangles include Outfit leading; overlaps are diagnostics for visual review, not glyph intersection claims. Canvas bounds cover all visible labels.', 'layout_checks': bounds,
        'figures': [{'path': str(p.relative_to(POST)), 'sha256': digest(p)} for p in outputs]}
    (HERE/'provenance.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'figure_files': len(outputs), 'bounds': bounds}, indent=2))


if __name__ == '__main__':
    main()
