"""Reader-facing diagrams and measured plots; the caller verifies frozen inputs."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as DrawPath
from matplotlib.ticker import MaxNLocator
import numpy as np

INK = '#17283b'
MUTED = '#617084'
LINE = '#dce3ed'
EDGE = '#8795a8'
COLORS = ['#66758a', '#3867d6', '#8357aa', '#087f73']
METHODS = ['Unfused Baseline', '+ CUDA Graph', '+ Local Fusion', '+ Megakernel']
FUSION_PHASES = [
    {'baseline': [('baseline_square', 'Square x'),
                  ('baseline_energy', 'Tile Sum of x²'),
                  ('baseline_max', 'Tile Maximum')],
     'fused': 'local_a', 'title': '1 · Tile Statistics',
     'body': 'Read x once\nTile sum of x²; max',
     'handoff': 'Tile sums of x²\n+ tile maxima'},
    {'baseline': [('baseline_scale', 'Row Inverse RMS'),
                  ('baseline_row_max', 'Row Maximum'),
                  ('baseline_exp', 'Compute E'),
                  ('baseline_sum', 'Tile Sum of E')],
     'fused': 'local_b', 'title': '2 · Exponentials',
     'body': 'Combine stats for s, m\nReread x; compute E\nSum E per tile',
     'handoff': 'Tile sums of E\n+ row statistics'},
    {'baseline': [('baseline_denominator', 'Row Sum of E'),
                  ('baseline_normalize', 'Normalize E')],
     'fused': 'local_c', 'title': '3 · Normalize',
     'body': 'Sum tile sums for z\nReread x; recompute E\nWrite y = E / z'},
]
GRAPH = {
    'scope': 'Row-wise value dependencies on a 2D tensor; nodes are mathematical operations, not GPU kernels or CUDA Graph nodes.',
    'nodes': {
        'x': 'Input tensor with shape (B, N)',
        's': '1 / sqrt(mean(x*x) + eps)',
        'm': 'max(x)',
        'E': 'exp((x - m) * s)',
        'z': 'sum(E)',
        'y': 'E / z',
    },
    'edges': [('x', 's'), ('x', 'm'), ('x', 'E'), ('s', 'E'),
              ('m', 'E'), ('E', 'z'), ('E', 'y'), ('z', 'y')],
    'code_names': {'s': 'inv_rms', 'E': 'exp_scores'},
}


def canvas(width, height, mobile=False):
    inches = 4.5 if mobile else 11.5
    fig = plt.figure(figsize=(inches, inches * height / width))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis('off')
    return fig, ax


def title(ax, text, subtitle, mobile=False):
    ax.text(.035, .965, text, transform=ax.transAxes, va='top',
            fontsize=16 if mobile else 20, fontweight='semibold', color=INK)
    ax.text(.035, .965 - (34 / ax.get_ylim()[0]), subtitle,
            transform=ax.transAxes, va='top', fontsize=10.3 if mobile else 11.5, color=MUTED)


def chart_header(fig, heading, subtitle, note, mobile):
    height = fig.get_figheight() * 72
    for offset, text, size, weight, color in [
        (18, heading, 18.5 if mobile else 22, 'semibold', INK),
        (52, subtitle, 12 if mobile else 13, 'normal', MUTED),
        (78, note, 11.5 if mobile else 12.5, 'normal', MUTED),
    ]:
        fig.text(.045, 1-offset/height, text, ha='left', va='top',
                 fontsize=size, fontweight=weight, color=color)


def route(ax, points, label=None, label_at=None, dashed=False, color=EDGE):
    path = DrawPath(points, [DrawPath.MOVETO] + [DrawPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=9,
                 lw=1.1, color=color, linestyle=(0, (3, 3)) if dashed else '-'))
    if label:
        ax.text(*label_at, label, ha='center', va='center', color=MUTED, fontsize=10,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5})


def node(ax, rect, heading, formula, shape, accent=None, mobile=False):
    x, y, w, h = rect
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=8',
        facecolor='#eef7f4' if accent else '#f7f9fc', edgecolor=accent or LINE, lw=.9))
    ax.text(x+w/2, y+h*.2, heading, ha='center', va='center',
            fontsize=11.2 if mobile else 12, fontweight='semibold', color=accent or INK)
    ax.text(x+w/2, y+h*.51, formula, ha='center', va='center',
            fontsize=10.5 if mobile else 11, fontfamily='DejaVu Sans Mono',
            linespacing=1.4, color=INK)
    ax.text(x+w/2, y+h*.84, shape, ha='center', va='center',
            fontsize=9.2 if mobile else 10.3, color=MUTED)


def computation(mobile, columns):
    subtitle = f'Row-wise operations · input B × {columns:,}'
    if mobile:
        fig, ax = canvas(360, 730, True)
        title(ax, 'RMS Scaled Softmax', subtitle, True)
        route(ax, [(180,179),(180,201),(118,201),(118,220)])
        route(ax, [(180,179),(180,201),(290,201),(290,229)])
        route(ax, [(250,141),(352,141),(352,432),(306,432)], 'x', (352,350), True)
        route(ax, [(118,328),(118,355),(143,355),(143,386)], 's', (118,346))
        route(ax, [(290,321),(290,355),(220,355),(220,386)], 'm', (290,346))
        route(ax, [(180,478),(180,518)])
        route(ax, [(54,432),(20,432),(20,668),(54,668)], 'E', (20,604), True)
        route(ax, [(180,594),(180,630)], 'z', (180,612))
        specs = [
            ((110,103,140,76),'Input','x','B × N'),
            ((18,220,200,108),'Inverse RMS','s = 1 / sqrt(\nmean(x*x) + eps)','B × 1'),
            ((236,229,108,92),'Row Maximum','m = max(x)','B × 1'),
            ((54,386,252,92),'Exponentials','E = exp((x - m) * s)','B × N'),
            ((110,518,140,76),'Row Sum','z = sum(E)','B × 1'),
            ((54,630,252,76),'Normalize','y = E / z','B × N'),
        ]
    else:
        fig, ax = canvas(1080, 475)
        title(ax, 'RMS Scaled Softmax', subtitle)
        route(ax, [(159,282),(193,282),(193,178),(225,178)])
        route(ax, [(159,282),(193,282),(193,391),(270,391)])
        route(ax, [(159,282),(570,282)], 'x', (405,282), True)
        route(ax, [(507,178),(692,178),(692,235)], 's', (598,178))
        route(ax, [(462,391),(692,391),(692,335)], 'm', (593,391))
        route(ax, [(815,282),(841,282),(841,171),(865,171)])
        route(ax, [(815,282),(841,282),(841,394),(865,394)], dashed=True)
        route(ax, [(960,216),(960,349)], 'z', (960,282))
        specs = [
            ((24,235,135,94),'Input','x','B × N'),
            ((225,126,282,104),'Inverse RMS','s = 1 / sqrt(\nmean(x*x) + eps)','B × 1'),
            ((270,349,192,84),'Row Maximum','m = max(x)','B × 1'),
            ((570,235,245,100),'Exponentials','E = exp((x - m) * s)','B × N'),
            ((865,126,190,90),'Row Sum','z = sum(E)','B × 1'),
            ((865,349,190,90),'Normalize','y = E / z','B × N'),
        ]
    for rect, heading, formula, shape in specs:
        node(ax, rect, heading, formula, shape,
             COLORS[3] if heading=='Normalize' else None, mobile)
    return fig


def graph_replay(mobile):
    """Compare host submissions while preserving the captured nine-kernel chain."""
    fig, ax = canvas(360, 560, True) if mobile else canvas(1080, 390)
    title(ax, 'Fewer Submissions, Same Kernels', 'Repeated execution of the 9-kernel baseline', mobile)
    count = sum(len(p['baseline']) for p in FUSION_PHASES)
    panel_w = 324 if mobile else 480
    chip_w, chip_gap = (28, 8) if mobile else (38, 12)
    chip_h = 28 if mobile else 30
    chain_w = count*chip_w+(count-1)*chip_gap
    for replay, left, heading_y in [(False,18 if mobile else 36,110),
                                     (True,18 if mobile else 564,340 if mobile else 110)]:
        color = COLORS[1] if replay else COLORS[0]
        heading = 'CUDA Graph Replay' if replay else 'Direct Launches'
        ax.text(left,heading_y,heading,fontsize=12.5 if mobile else 15,
                fontweight='semibold',color=color)
        ax.text(left,heading_y+22,'After capture + instantiation' if replay else 'Each invocation',
                fontsize=9.5 if mobile else 10.5,color=MUTED)
        ax.text(left,heading_y+46,'CPU · 1 replay call' if replay else f'CPU · {count} launch calls',
                fontsize=10 if mobile else 11,color=INK)
        cpu_y = heading_y+60
        frame_y = cpu_y+(54 if mobile else 77)
        kernel_y = frame_y+(9 if mobile else 16)
        start = left+(panel_w-chain_w)/2
        frame_h = 50 if mobile else 62
        ax.add_patch(FancyBboxPatch((left,frame_y),panel_w,frame_h,
            boxstyle='round,pad=0,rounding_size=7',facecolor='#f7f9fc',edgecolor=LINE,lw=.8))
        for i in range(count):
            x = start+i*(chip_w+chip_gap)
            ax.add_patch(FancyBboxPatch((x,kernel_y),chip_w,chip_h,
                boxstyle='round,pad=0,rounding_size=4',facecolor='white',edgecolor=EDGE,lw=.8))
            ax.text(x+chip_w/2,kernel_y+chip_h/2,f'K{i+1}',ha='center',va='center',
                    fontsize=9 if mobile else 10,color=INK)
            if i+1<count:
                ax.add_patch(FancyArrowPatch((x+chip_w+1,kernel_y+chip_h/2),
                    (x+chip_w+chip_gap-1,kernel_y+chip_h/2),arrowstyle='-|>',
                    mutation_scale=5,lw=.7,color=EDGE,shrinkA=0,shrinkB=0))
            if not replay:
                ax.add_patch(FancyBboxPatch((x,cpu_y),chip_w,chip_h,
                    boxstyle='round,pad=0,rounding_size=4',facecolor='#f0f3f7',edgecolor=LINE,lw=.8))
                ax.text(x+chip_w/2,cpu_y+chip_h/2,str(i+1),ha='center',va='center',
                        fontsize=9 if mobile else 10,color=INK)
                route(ax,[(x+chip_w/2,cpu_y+chip_h+3),(x+chip_w/2,kernel_y-3)])
        if replay:
            button_w = 196 if mobile else 270
            button_x = left+(panel_w-button_w)/2
            ax.add_patch(FancyBboxPatch((button_x,cpu_y),button_w,chip_h,
                boxstyle='round,pad=0,rounding_size=5',facecolor='#edf3ff',edgecolor='#9bb4e8',lw=.9))
            ax.text(left+panel_w/2,cpu_y+chip_h/2,'cudaGraphLaunch()',ha='center',va='center',
                    fontsize=10 if mobile else 11,fontfamily='DejaVu Sans Mono',color=COLORS[1])
            route(ax,[(left+panel_w/2,cpu_y+chip_h+3),(left+panel_w/2,frame_y-3)],color=COLORS[1])
        ax.text(left,frame_y+frame_h+22,f'GPU · {count} separate kernels',
                fontsize=10 if mobile else 11,color=INK)
    return fig


def local_fusion(mobile, columns):
    """Map the baseline launch sequence to the selected recomputing implementation."""
    fig, ax = canvas(360, 840, True) if mobile else canvas(1080, 760)
    title(ax, '9 Kernels Become 3', f'Input: B × {columns:,} · same calculation', mobile)
    left, lw, right, rw = (14, 140, 180, 166) if mobile else (48, 430, 634, 398)
    ys = (160, 390, 670) if mobile else (170, 350, 570)
    chip_h, gap = (35, 10) if mobile else (32, 7)
    for x, w, label, count, color in [
        (left, lw, 'Baseline', '9 kernels', COLORS[0]),
        (right, rw, 'Local Fusion', '3 kernels', COLORS[2]),
    ]:
        ax.text(x+w/2, 107, label, ha='center', fontsize=12.5 if mobile else 16,
                fontweight='semibold', color=color)
        ax.text(x+w/2, 131, count, ha='center', fontsize=10 if mobile else 11, color=MUTED)
    number = 1
    for i, (y, phase) in enumerate(zip(ys, FUSION_PHASES)):
        n = len(phase['baseline'])
        group_h = n * chip_h + (n-1) * gap
        minimum_h = 125 if mobile else (144 if phase['body'].count('\n') == 2 else 110)
        fused_h = max(group_h, minimum_h)
        for j, (_, label) in enumerate(phase['baseline']):
            cy = y+j*(chip_h+gap)
            ax.add_patch(FancyBboxPatch((left,cy),lw,chip_h,
                boxstyle='round,pad=0,rounding_size=6',facecolor='#f7f9fc',edgecolor=LINE,lw=.9))
            ax.text(left+10 if mobile else left+18,cy+chip_h/2,f'{number}  {label}',
                    va='center',fontsize=10 if mobile else 12,color=INK)
            number += 1
            if j+1<n:
                ax.plot([left+lw/2]*2,[cy+chip_h,cy+chip_h+gap],color=EDGE,lw=.9)
        bracket_x = left+lw+7
        ax.plot([bracket_x-4,bracket_x,bracket_x,bracket_x-4],
                [y,y,y+group_h,y+group_h],color=COLORS[2],lw=.9)
        route(ax,[(bracket_x,y+group_h/2),(right-7,y+group_h/2)],color=COLORS[2])
        ax.add_patch(FancyBboxPatch((right,y),rw,fused_h,
            boxstyle='round,pad=0,rounding_size=8',facecolor='#faf8fc',edgecolor='#c3aed6',lw=1))
        ax.text(right+rw/2,y+25,phase['title'],ha='center',va='center',
                fontsize=11 if mobile else 14,fontweight='semibold',color=COLORS[2])
        ax.text(right+rw/2,y+fused_h*.64,phase['body'],ha='center',va='center',
                fontsize=10 if mobile else 12,linespacing=1.6,color=INK)
        if i<2:
            route(ax,[(left+lw/2,y+group_h+3),(left+lw/2,ys[i+1]-3)])
            mid = right+rw/2
            route(ax,[(mid,y+fused_h+3),(mid,ys[i+1]-3)],color=COLORS[2])
            ax.text(mid,(y+fused_h+ys[i+1])/2,phase['handoff'],ha='center',va='center',
                    fontsize=9.2 if mobile else 10.5,linespacing=1.3,color=COLORS[2],
                    bbox={'facecolor':'white','edgecolor':'none','pad':3})
    return fig


def storage(mobile, columns):
    width, height = (360, 730) if mobile else (1080, 640)
    fig, ax = canvas(width, height, mobile)
    title(ax, '1 Tile Across 3 Phases',
          f'Input: B × {columns:,} · one tile per block', mobile)
    left, right, w = (74, 216, 130) if mobile else (200, 650, 380)
    ys = (218, 400, 582) if mobile else (198, 358, 518)
    h = 90 if mobile else 76
    header_y = 110 if mobile else 120
    for x, label, sub, color in [
        (left, 'Local Fusion', '3 kernels', COLORS[2]),
        (right, 'Megakernel', '1 kernel /\nrow group' if mobile else '1 kernel / row group', COLORS[3]),
    ]:
        ax.text(x+w/2, header_y, label, ha='center', va='center',
                fontsize=12 if mobile else 16, fontweight='semibold', color=color)
        ax.text(x+w/2, header_y+21, sub, ha='center', va='top',
                fontsize=9.2 if mobile else 11, linespacing=1.3, color=MUTED)
    # One region represents one block's shared-memory allocation across all phases.
    buffer_top = 184 if mobile else 178
    buffer_bottom = ys[-1]+h+22
    ax.add_patch(FancyBboxPatch((right, buffer_top), w, buffer_bottom-buffer_top,
                 boxstyle='round,pad=0,rounding_size=9', facecolor='#eef7f4',
                 edgecolor='#a7cfc4', lw=1))
    ax.text(right+w/2, buffer_top+16, 'Shared Memory', ha='center', va='center',
            fontsize=9 if mobile else 10.5, color=COLORS[3])
    phase_labels = ['01\nTile\nStatistics', '02\nExponentials', '03\nNormalize'] if mobile else [
        '01 · Tile Statistics', '02 · Exponentials', '03 · Normalize']
    for i, y in enumerate(ys):
        ax.text(34 if mobile else 100, y+h/2, phase_labels[i],
                ha='center', va='center', fontsize=9 if mobile else 11.5,
                linespacing=1.35, color=MUTED)
        ax.add_patch(FancyBboxPatch((left, y), w, h,
                     boxstyle='round,pad=0,rounding_size=8',
                     facecolor='#faf8fc', edgecolor='#d9cfe3', lw=.9))
        texts = [
            ['Read tile x', 'Read x again\nCompute E', 'Read x again\nRecompute E'][i],
            ['Keep tile x', 'Replace x with E', 'Normalize\nretained E' if mobile else 'Normalize retained E'][i],
        ]
        for x, text in zip((left, right), texts):
            ax.text(x+w/2, y+h/2, text, ha='center', va='center',
                    linespacing=1.5, fontsize=10 if mobile else 14, color=INK)
        if i<2:
            # Connect phase boundaries directly; labels sit in the transition gap.
            for x, color, label, background in [
                (left, COLORS[2], 'kernel boundary', 'white'),
                (right, COLORS[3], 'row-scoped barrier', '#eef7f4'),
            ]:
                center=x+w/2
                route(ax, [(center, y+h+4), (center, ys[i+1]-4)], color=color)
                ax.text(center, (y+h+ys[i+1])/2,
                        label.replace(' ', '\n', 1) if mobile else label,
                        ha='center', va='center', fontsize=8.5 if mobile else 10,
                        linespacing=1.25, color=color,
                        bbox={'facecolor':background, 'edgecolor':'none', 'pad':3})
    return fig


def session_summary(group):
    """Give each GPU session equal weight; never pool rounds across sessions."""
    medians = [s['median'] for s in group]
    return {'session_medians': medians, 'mean': float(np.mean(medians)),
            'min': min(medians), 'max': max(medians)}


def render_figures(timing, cfg, assets: Path, workload):
    columns = workload['columns']
    points, outputs, completion_summaries, comparisons, layouts = [], [], [], [], []
    plt.rcParams.update({'svg.hashsalt': 'rms-scaled-softmax-article-v2', 'text.color': INK,
                         'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': INK})

    def save(fig, name, mobile):
        # A mobile PNG is also retained for review; the page selects the mobile SVG.
        for ext in ('svg', 'png'):
            path = assets / (name + ('-mobile' if mobile else '') + '.' + ext)
            fig.savefig(path, bbox_inches=None, pad_inches=0, facecolor='white',
                        metadata={'Date': None} if ext == 'svg' else None)
            outputs.append(path)
        plt.close(fig)

    def values(a, b, v, metric):
        return {r['round']: r[metric] for r in timing[a, b]
                if r['variant'] == v and 'cache_policy' not in r}

    def stats(a, b, v):
        p10, median, p90 = map(float, np.percentile(list(values(a, b, v, 'host_single_us').values()), [10, 50, 90]))
        point = {'allocation': a, 'rows': b, 'variant': v, 'metric': 'host_single_us',
                 'p10': p10, 'median': median, 'p90': p90}
        points.append(point)
        return point

    for mobile in (False, True):
        save(computation(mobile, columns), 'rms-computation', mobile)
        save(graph_replay(mobile), 'rms-graph-replay', mobile)
        save(local_fusion(mobile, columns), 'rms-local-fusion', mobile)
        save(storage(mobile, columns), 'rms-storage', mobile)

    for b in (1, 16, 32, 64):
        variants = ['baseline_stream', 'baseline_graph', cfg[str(b)]['local_best']+'_graph', cfg[str(b)]['mega_best']+'_graph']
        for v in variants:
            completion_summaries.append({'rows': b, 'variant': v, 'metric': 'host_single_us',
                                         **session_summary([stats(a, b, v) for a in (1, 2)])})
        local, mega = completion_summaries[-2:]
        comparisons.append({'rows': b, 'metric': 'host_single_us',
                            'local_variant': local['variant'], 'mega_variant': mega['variant'],
                            'time_saved_us': local['mean'] - mega['mean'],
                            'latency_reduction_percent': 100 * (1 - mega['mean'] / local['mean'])})

    for mobile in (False, True):
        fig, axes = plt.subplots(4 if mobile else 2, 1 if mobile else 2,
                                  figsize=(4.5,14.4) if mobile else (11.5,9.0))
        height=fig.get_figheight()*72
        fig.subplots_adjust(left=.07 if mobile else .06,right=.95,
                            top=1-(168 if mobile else 162)/height,bottom=.04,
                            hspace=.62 if mobile else .58,wspace=.28)
        chart_header(fig,'Time per Invocation','Host completion time · µs · lower is better',
                     'Rows × columns · scales differ across panels',mobile)
        for ax,b in zip(np.ravel(axes),(1,16,32,64)):
            ss=[s for s in completion_summaries if s['rows']==b]
            data_max=max(s['max'] for s in ss)
            ticks=MaxNLocator(nbins=5,steps=[1,2,2.5,4,5,10]).tick_values(0,data_max*1.05)
            maximum=float(ticks[-1])
            assert .75<=data_max/maximum<1
            for i,(item,color,method) in enumerate(zip(ss,COLORS,METHODS)):
                x=item['mean']
                ax.text(0,i-.30,method,transform=ax.get_yaxis_transform(),va='bottom',
                        fontsize=12.6 if mobile else 13.5,color=INK)
                ax.text(1,i-.30,f'{x:.1f}',transform=ax.get_yaxis_transform(),
                        ha='right',va='bottom',fontsize=12.8 if mobile else 13.5,
                        fontweight='semibold',color=INK)
                ax.barh(i,x,height=.23,color=color,zorder=2)
                ax.errorbar(x,i,xerr=[[x-item['min']],[item['max']-x]],fmt='none',
                            ecolor=INK,capsize=2.6,lw=1.15,zorder=3)
            gain=next(s for s in comparisons if s['rows']==b)['latency_reduction_percent']
            ax.text(0,4.15,f'Megakernel: {gain:.1f}% less time vs local fusion',
                    va='center',fontsize=11.6 if mobile else 12.5,color=MUTED)
            ax.set(xlim=(0,maximum),ylim=(4.55,-.95),yticks=[])
            ax.set_xticks(ticks)
            ax.grid(axis='x',color=LINE,lw=.65,alpha=.7,zorder=0)
            ax.tick_params(axis='x',length=0,labelsize=11.4 if mobile else 12.5,pad=8)
            ax.set_title(f'{b} × {columns:,}',
                         loc='left',fontsize=15 if mobile else 16.5,fontweight='semibold',pad=18)
            ax.text(1,1.08,f'0–{maximum:g} µs',transform=ax.transAxes,ha='right',
                    fontsize=11 if mobile else 12,color=MUTED)
            layouts.append({'figure':'rms-completion','mobile':mobile,'rows':b,
                            'x_min':0,'x_max':maximum,'ticks':list(map(float,ticks)),
                            'largest_plotted_value':data_max,'scale':'linear'})
        save(fig,'rms-completion',mobile)

    paired = []
    for a in (1, 2):
        baseline = values(a, 16, 'roundtrip_off_graph', 'host_single_us')
        for variant in ('input_reload_graph', 'roundtrip_on_graph', 'both_reload_graph'):
            group = values(a, 16, variant, 'host_single_us')
            p10, median, p90 = map(float, np.percentile([group[r]-baseline[r] for r in sorted(baseline)], [10, 50, 90]))
            paired.append({'allocation': a, 'variant': variant, 'metric': 'paired_host_single_delta_us',
                           'p10': p10, 'median': median, 'p90': p90})
    retention_summaries = [
        {'variant': v, 'metric': 'paired_host_single_delta_us',
         **session_summary([s for s in paired if s['variant'] == v])}
        for v in ('input_reload_graph', 'roundtrip_on_graph', 'both_reload_graph')
    ]
    for mobile in (False,True):
        fig,ax=plt.subplots(figsize=(4.5,5.35) if mobile else (11.5,5.15))
        height=fig.get_figheight()*72
        fig.subplots_adjust(left=.07 if mobile else .06,right=.95,
                            top=1-(158 if mobile else 152)/height,bottom=.17)
        chart_header(fig,'Reloading Adds Latency',f'16 × {columns:,} · grid-wide barriers',
                     'Same compiled kernel · two GPU sessions',mobile)
        for i,(item,label) in enumerate(zip(retention_summaries,
                         ['Reload input x','Write + reload E','Both'])):
            x=item['mean']
            ax.text(0,i-.32,label,transform=ax.get_yaxis_transform(),va='bottom',
                    fontsize=13 if mobile else 13.5,color=INK)
            ax.text(1,i-.32,f'+{x:.2f} µs',transform=ax.get_yaxis_transform(),
                    ha='right',va='bottom',fontsize=13 if mobile else 13.5,
                    fontweight='semibold',color=COLORS[3])
            ax.hlines(i,0,5,color=LINE,lw=.8,zorder=0)
            ax.errorbar(x,i,xerr=[[x-item['min']],[item['max']-x]],fmt='o',
                mfc=COLORS[3],mec=COLORS[3],mew=1,color=COLORS[3],
                capsize=3,markersize=6,lw=1.5,zorder=3)
        ax.set(xlim=(0,5),ylim=(2.55,-.85),yticks=[])
        ax.set_xticks([0,1,2,3,4,5])
        ax.tick_params(length=0,labelsize=12 if mobile else 12.5,pad=8)
        ax.set_xlabel('Additional host completion time (µs)',labelpad=12,
                      fontsize=11.5 if mobile else 13)
        layouts.append({'figure':'rms-retention','mobile':mobile,'rows':16,
                        'x_min':0,'x_max':5,'ticks':[0,1,2,3,4,5],
                        'largest_plotted_value':max(s['max'] for s in retention_summaries),'scale':'linear'})
        save(fig,'rms-retention',mobile)
    return points, paired, completion_summaries, retention_summaries, comparisons, layouts, outputs, GRAPH
