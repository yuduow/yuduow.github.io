/* Progressive enhancement. Original figures remain available without JS. */
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const iconPlay = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path fill="currentColor" d="M2 1.5v9l8-4.5z"/></svg>';
const kernels = [
  ['Square x', 'Read x and write the full x² tensor.'],
  ['Tile Sum of x²', 'Reduce squared values within each tile and write its sum.'],
  ['Tile Maximum', 'Find each tile’s maximum and write it to global memory.'],
  ['Row Inverse RMS', 'Combine tile sums of squares to compute s = 1 / sqrt(mean(x²) + eps).'],
  ['Row Maximum', 'Combine tile maxima into the row maximum m.'],
  ['Compute E', 'Compute E = exp((x − m) × s) and write the full E tensor.'],
  ['Tile Sum of E', 'Reduce exponentials within each tile and write its sum.'],
  ['Row Sum of E', 'Combine tile sums into the row’s softmax denominator z.'],
  ['Normalize E', 'Read E and write the output y = E / z.'],
];

const node = (tag, className, html = '') => {
  const el = document.createElement(tag);
  el.className = className;
  el.innerHTML = html;
  return el;
};
const controls = () => `<div class="viz-controls"><button type="button" class="viz-primary" data-action="play">${iconPlay} Play</button><button type="button" data-action="previous" aria-label="Previous step">← Back</button><button type="button" data-action="next" aria-label="Next step">Next →</button><button type="button" data-action="reset">Reset</button></div>`;

function mount(id, kicker, title, subtitle) {
  const figure = document.getElementById(id);
  if (!figure || figure.querySelector('.figure-explorer')) return null;
  const original = figure.querySelector('.figure-scroll');
  const caption = figure.querySelector('figcaption');
  const originalCaption = caption.innerHTML;
  let enhancedCaption = originalCaption;
  const href = original.querySelector('a').getAttribute('href');
  const root = node('div', 'figure-explorer');
  root.innerHTML = `<div class="viz-modebar"><div class="viz-view-switch" role="group" aria-label="Figure view"><button type="button" data-view="interactive" aria-pressed="true">Explore</button><button type="button" data-view="static" aria-pressed="false">Original</button></div><a href="${href}">Open SVG ↗</a></div>`;
  const panel = node('section', 'viz-panel', `<div class="viz-kicker">${kicker}</div><h3 class="viz-title" id="${id}-explorer-title">${title}</h3><p class="viz-subtitle">${subtitle}</p>`);
  panel.setAttribute('aria-labelledby', `${id}-explorer-title`);
  root.append(panel);
  let onHide = () => {};
  root.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
    const interactive = button.dataset.view === 'interactive';
    panel.hidden = !interactive;
    original.hidden = interactive;
    caption.innerHTML = interactive ? enhancedCaption : originalCaption;
    root.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    if (!interactive) onHide();
  }));
  window.addEventListener('beforeprint', () => {
    onHide();
    caption.innerHTML = originalCaption;
  });
  window.addEventListener('afterprint', () => {
    caption.innerHTML = panel.hidden ? originalCaption : enhancedCaption;
  });
  // Only hide the fallback once the component has finished initializing.
  return {root, panel, commit(stop, captionText) {
    onHide = stop || onHide;
    if (captionText) {
      enhancedCaption = `${caption.querySelector('.figure-label').outerHTML} ${captionText}`;
      caption.innerHTML = enhancedCaption;
    }
    figure.insertBefore(root, original);
    original.hidden = true;
  }};
}

function player(panel, root, count, render, noun, interval) {
  let current = 0, timer = null;
  const play = panel.querySelector('[data-action="play"]');
  const previous = panel.querySelector('[data-action="previous"]');
  const next = panel.querySelector('[data-action="next"]');
  const update = () => {
    root.dataset.step = String(current);
    root.dataset.playing = String(timer !== null);
    play.innerHTML = reducedMotion.matches ? `Next ${noun} →` : (timer !== null ? 'Pause' : `${iconPlay} ${current === count - 1 ? 'Replay' : 'Play'}`);
    play.setAttribute('aria-label', reducedMotion.matches ? `Next ${noun}` : (timer !== null ? 'Pause animation' : 'Play animation'));
    previous.disabled = current === 0;
    next.disabled = current === count - 1;
    panel.querySelectorAll('[aria-live]').forEach(el => el.setAttribute('aria-live', timer === null ? 'polite' : 'off'));
    render(current);
  };
  const stop = () => { if (timer !== null) clearInterval(timer); timer = null; update(); };
  const select = i => { if (timer !== null) clearInterval(timer); timer = null; current = Math.max(0, Math.min(count - 1, i)); update(); };
  play.addEventListener('click', () => {
    if (reducedMotion.matches) { select(current === count - 1 ? 0 : current + 1); return; }
    if (timer !== null) { stop(); return; }
    if (current === count - 1) current = 0;
    timer = setInterval(() => {
      if (current < count - 1) current += 1;
      if (current === count - 1) { clearInterval(timer); timer = null; }
      update();
    }, interval);
    update();
  });
  previous.addEventListener('click', () => select(current - 1));
  next.addEventListener('click', () => select(current + 1));
  panel.querySelector('[data-action="reset"]').addEventListener('click', () => select(0));
  document.addEventListener('visibilitychange', () => { if (document.hidden && timer !== null) stop(); });
  reducedMotion.addEventListener('change', stop);
  const observer = new IntersectionObserver(entries => { if (!entries[0].isIntersecting && timer !== null) stop(); });
  observer.observe(root);
  update();
  return {select, stop};
}

function replayExplorer() {
  const view = mount('figure-graph-replay', 'Explore · Submission', 'Same Kernels, Fewer Submissions', 'Follow CPU submissions, GPU execution, and the final completion wait.');
  if (!view) return;
  const {root, panel} = view;
  panel.insertAdjacentHTML('beforeend', `<div class="viz-toolbar"><div class="viz-segments" role="group" aria-label="Submission method"><button type="button" data-submission="direct" aria-pressed="false">Direct launches</button><button type="button" data-submission="graph" aria-pressed="true">CUDA Graph replay</button></div><span class="viz-small">Illustrative overlap · not a timing trace</span></div><div class="viz-note" data-replay-note></div>${controls()}<div class="replay-flow"><div class="viz-call-title"><strong>CPU submissions</strong><span data-call-count></span></div><div class="replay-cpu" aria-label="CPU launch calls"></div><div class="replay-handoffs" aria-hidden="true"></div><div class="replay-kernels" role="group" aria-label="Inspect a GPU kernel">${kernels.map((k, i) => `<button type="button" data-kernel="${i}" aria-label="Kernel ${i + 1}: ${k[0]}" aria-pressed="false">K${i + 1}</button>`).join('')}</div><div class="viz-call-title replay-gpu-title"><strong>GPU execution</strong><span>9 separate kernels · dependency order</span></div><div class="replay-return" aria-hidden="true"><span>↑</span></div></div><div class="replay-legend"><span>↓ Submit work</span><span>↑ Completion after K9</span></div><div class="replay-status" aria-live="polite"><div><strong>CPU</strong><span data-cpu-status></span></div><div><strong>GPU</strong><span data-gpu-status></span></div></div><div class="replay-detail" aria-live="polite"><strong></strong><span></span></div><div class="viz-foot"><span>CPU submission can overlap GPU execution. The benchmark waits once after the whole chain.</span><a href="evidence/rms-softmax/implementation.html#cuda-graph-capture-and-replay">Capture and replay code ↗</a></div>`);
  let mode = 'graph';
  const renderMode = () => {
    root.dataset.submission = mode;
    panel.querySelector('.replay-cpu').innerHTML = mode === 'graph' ? '<span class="replay-graph-call" data-call="1">cudaGraphLaunch()</span>' : Array.from({length: 9}, (_, i) => `<span data-call="${i + 1}" aria-label="Launch K${i + 1}">L${i + 1}</span>`).join('');
    panel.querySelector('.replay-handoffs').innerHTML = mode === 'graph' ? '<span class="replay-handoff replay-graph-handoff" data-handoff="1">↓</span>' : Array.from({length: 9}, (_, i) => `<span class="replay-handoff" data-handoff="${i + 1}">↓</span>`).join('');
    panel.querySelector('[data-call-count]').textContent = mode === 'graph' ? '1 replay call' : '9 launch calls';
    panel.querySelector('[data-replay-note]').textContent = mode === 'graph' ? 'After capture and instantiation, 1 replay call submits the prepared chain. The GPU follows its dependencies without another CPU launch for each kernel.' : 'L1–L9 are separate launch calls. The CPU can enqueue the next kernel before the previous one finishes; the GPU executes them in stream order.';
    panel.querySelectorAll('[data-submission]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.submission === mode)));
  };
  // One valid overlap, not a reconstruction of the measured CPU/GPU schedule.
  // Step 1 submits work; steps 2–10 execute K1–K9; step 11 observes completion.
  const render = i => {
    const submitted = mode === 'graph' ? Number(i > 0) : Math.min(i, 9);
    const active = i >= 2 && i <= 10 ? i - 2 : -1;
    const completed = Math.max(0, Math.min(9, i - 2));
    const returned = i === 11;
    Object.assign(root.dataset, {submitted: String(submitted), completed: String(completed), returned: String(returned)});
    panel.querySelectorAll('[data-call]').forEach(call => {
      const n = Number(call.dataset.call);
      call.classList.toggle('is-submitted', n <= submitted);
      call.classList.toggle('is-current', mode === 'graph' ? i === 1 : n === i);
    });
    panel.querySelectorAll('[data-handoff]').forEach(arrow => {
      const n = Number(arrow.dataset.handoff);
      arrow.classList.toggle('is-sent', n <= submitted);
      arrow.classList.toggle('is-current', mode === 'graph' ? i === 1 : n === i);
    });
    panel.querySelectorAll('[data-kernel]').forEach((button, j) => {
      button.setAttribute('aria-pressed', String(j === active));
      button.classList.toggle('is-past', j < completed);
      button.classList.toggle('is-queued', mode === 'graph' ? i > 0 : j < submitted);
    });
    let cpu = 'Ready to submit';
    if (i > 0) cpu = mode === 'direct' && i <= 9 ? `Submitting L${i} of 9` : (i === 1 ? 'Graph submitted' : 'Waiting in cudaStreamSynchronize()');
    if (returned) cpu = 'cudaStreamSynchronize() returned';
    panel.querySelector('[data-cpu-status]').textContent = cpu;
    panel.querySelector('[data-gpu-status]').textContent = returned ? '9 / 9 kernels complete' : (active >= 0 ? `Running K${active + 1} · ${completed} / 9 complete` : (i === 1 ? 'Work queued; execution can begin' : 'No work submitted yet'));
    panel.querySelector('.replay-detail strong').textContent = active >= 0 ? `K${active + 1} · ${kernels[active][0]}` : (returned ? 'Completion returns control to the CPU' : (i === 1 ? 'Submission does not wait for GPU completion' : 'One workflow, one final wait'));
    panel.querySelector('.replay-detail span').textContent = active >= 0 ? kernels[active][1] : (returned ? 'K9 has finished, so the benchmark’s stream synchronization can return and its host timer can stop.' : (i === 1 ? 'The CPU can continue submitting work while the GPU runs. There is no per-kernel completion wait in this benchmark.' : 'Play or step through submission and execution. Blue arrows submit work; the green return path becomes active only after K9.'));
  };
  renderMode();
  const p = player(panel, root, 12, render, 'step', 1100);
  panel.querySelectorAll('[data-kernel]').forEach(b => b.addEventListener('click', () => p.select(Number(b.dataset.kernel) + 2)));
  panel.querySelectorAll('[data-submission]').forEach(b => b.addEventListener('click', () => { mode = b.dataset.submission; renderMode(); p.select(0); }));
  view.commit(p.stop, 'Each blue arrow is a CPU submission: 9 direct launches or 1 graph replay. The return path represents the final stream-completion wait used by this benchmark. One possible CPU/GPU overlap is illustrated; animation steps are not durations.');
}

const phases = [
  {label: '1 · Statistics', title: 'Load x and compute tile statistics', detail: 'Each block writes its tile’s sum of squares and maximum to global memory.', local: 'Read x; compute tile statistics.', mega: 'Load x into shared memory; compute tile statistics.', value: 'x', reads: 1, exp: 0},
  {label: 'Barrier 1', title: 'Wait for the row’s initial statistics', detail: 'All tiles must publish their statistics before blocks combine them into the row’s inverse RMS and maximum.', local: 'Kernel 1 ends before kernel 2 starts.', mega: 'Keep x while this row’s blocks reach the barrier.', value: 'x', reads: 1, exp: 0, barrier: true},
  {label: '2 · Exponentials', title: 'Compute E and the tile’s exponential sum', detail: 'Each block combines row statistics, computes E = exp((x − m) × s), and publishes its tile sum.', local: 'Reread x; compute E without saving the full E tensor.', mega: 'Replace x with E in the same shared-memory buffer.', value: 'E', reads: 2, exp: 1},
  {label: 'Barrier 2', title: 'Wait for the row’s exponential sums', detail: 'All tile sums must be visible before blocks combine them into the softmax denominator z.', local: 'Kernel 2 ends before kernel 3 starts.', mega: 'Keep E while this row’s blocks reach the barrier.', value: 'E', reads: 2, exp: 1, barrier: true},
  {label: '3 · Normalize', title: 'Normalize and write y = E / z', detail: 'Each block combines tile sums into z. Local fusion recomputes E; the megakernel reads its retained E.', local: 'Reread x; recompute E, then write y = E / z.', mega: 'Read retained E and write y = E / z.', value: 'E', reads: 3, exp: 2, output: true},
];
const cells = value => Array.from({length: 6}, (_, i) => `<span>${value}${value === '—' ? '' : `<sub>${i}</sub>`}</span>`).join('');
function storageExplorer() {
  const view = mount('figure-storage', 'Explore · Tile Lifetimes', 'Follow One Tile Across 3 Phases', 'Row-scoped megakernel variant · one tile per thread block.');
  if (!view) return;
  const {root, panel} = view;
  panel.insertAdjacentHTML('beforeend', `<div class="phase-steps" role="group" aria-label="Choose a phase or barrier">${phases.map((s, i) => `<button type="button" data-phase="${i}" class="${s.barrier ? 'phase-barrier' : ''}" aria-pressed="${i === 0}">${s.label}</button>`).join('')}</div><div class="tile-lanes">${['local','mega'].map(method => `<div class="tile-lane" data-method="${method}"><h4>${method === 'local' ? 'Local Fusion' : 'Megakernel'}</h4><span class="lane-kind">${method === 'local' ? '3 kernels · graph replay' : '1 kernel / row group · graph replay'}</span><div class="lane-action"></div><div class="tile-buffer"><div class="buffer-label"></div><div class="tile-cells"></div></div><div class="lane-boundary"></div><div class="lane-output"><span class="output-label"></span><div class="output-cells"></div></div><div class="tile-counts"><span><strong data-count="reads"></strong>Reads of x</span><span><strong data-count="exp"></strong>Evaluations of E</span></div></div>`).join('')}</div><div class="phase-detail" aria-live="polite"><strong></strong><span></span></div>${controls()}<div class="viz-foot"><span>One representative tile; row metadata stays in global memory.</span><span>Schematic order · source-level counts per element</span></div>`);
  panel.insertBefore(panel.querySelector('.viz-controls'), panel.querySelector('.phase-steps'));
  const p = player(panel, root, phases.length, i => {
    const phase = phases[i];
    panel.querySelectorAll('[data-phase]').forEach((b, j) => b.setAttribute('aria-pressed', String(i === j)));
    panel.querySelector('.phase-detail strong').textContent = phase.title;
    panel.querySelector('.phase-detail span').textContent = phase.detail;
    panel.querySelectorAll('.tile-lane').forEach(lane => {
      const local = lane.dataset.method === 'local';
      const empty = local && Boolean(phase.barrier);
      lane.querySelector('.lane-action').textContent = phase[local ? 'local' : 'mega'];
      lane.querySelector('.buffer-label').textContent = local ? (empty ? 'No retained tile between kernels' : 'Current tile computation') : 'Same shared-memory buffer';
      lane.querySelector('.tile-buffer').classList.toggle('is-empty', empty);
      lane.querySelector('.tile-cells').innerHTML = cells(empty ? '—' : phase.value);
      const boundary = lane.querySelector('.lane-boundary');
      boundary.classList.toggle('is-barrier', Boolean(phase.barrier));
      boundary.textContent = phase.barrier ? (local ? 'Whole-kernel completion' : 'Row-scoped barrier') : (local ? 'Small statistics cross kernels' : 'Tile survives between phases');
      lane.querySelector('.output-label').textContent = phase.output ? 'Write global output y' : 'Global output not written yet';
      lane.querySelector('.output-cells').innerHTML = phase.output ? cells('y') : '';
      lane.querySelector('[data-count="reads"]').textContent = String(local ? phase.reads : 1);
      lane.querySelector('[data-count="exp"]').textContent = String(local ? phase.exp : Math.min(phase.exp, 1));
    });
  }, 'phase', 1800);
  panel.querySelectorAll('[data-phase]').forEach(b => b.addEventListener('click', () => p.select(Number(b.dataset.phase))));
  view.commit(p.stop, 'One block’s shared-memory buffer keeps x, then E, across the 2 row-scoped barriers. This is the variant used for 16–64 rows in the main comparison; the single-row measurement uses grid-wide barriers. Small row statistics still use global memory.');
}

const methods = [
  {label: 'Unfused Baseline', color: 'var(--viz-gray)', description: '9 kernels submitted directly. This is the deliberately unfused educational baseline.'},
  {label: '+ CUDA Graph', color: 'var(--viz-blue)', description: 'The same 9 kernels, submitted through 1 graph replay call.'},
  {label: '+ Local Fusion', color: 'var(--viz-purple)', description: '3 kernels with graph replay. The final kernel recomputes E from x.'},
  {label: '+ Megakernel', color: 'var(--viz-green)', description: 'Retained tiles with graph replay. Small row statistics still use global memory.'},
];
function latencyExplorer(data) {
  const view = mount('figure-completion', 'Explore · H100 Measurements', 'What Changes with the Row Count?', 'Warm host completion time · µs · lower is better · N = 131,072');
  if (!view) return;
  const {root, panel} = view;
  panel.insertAdjacentHTML('beforeend', `<div class="viz-toolbar"><div class="viz-segments" role="group" aria-label="Input row count">${[1,16,32,64].map(n => `<button type="button" data-rows="${n}" aria-pressed="${n === 16}">${n} ${n === 1 ? 'row' : 'rows'}</button>`).join('')}</div><span class="viz-small" data-scale-label></span></div><div class="latency-headline" aria-live="polite"><div><div class="latency-gain"></div><span class="viz-small">Megakernel saving from local fusion</span></div><div class="latency-context"></div></div><div class="latency-bars" role="group" aria-label="Inspect a measured implementation">${methods.map((m, i) => `<button type="button" class="latency-bar" data-method="${i}" aria-pressed="${i === 3}" style="--method-color:${m.color}"><span class="latency-bar-label"><span>${m.label}</span><span class="latency-value"></span></span><span class="latency-track" aria-hidden="true"><span class="latency-fill"></span><span class="latency-range"></span></span></button>`).join('')}</div><div class="latency-axis" aria-hidden="true"></div><div class="latency-options"><label class="viz-check"><input type="checkbox" data-fit>Fit axis to this row count</label><label class="viz-check"><input type="checkbox" data-sessions checked>Show session range</label></div><div class="latency-detail" aria-live="polite"><h4></h4><dl><div><dt>Mean of medians</dt><dd data-number="mean"></dd></div><div><dt>Session 1 median</dt><dd data-number="s1"></dd></div><div><dt>Session 2 median</dt><dd data-number="s2"></dd></div></dl><div class="method-description"></div></div><div class="viz-foot"><span>Ranges span the 2 session medians; not confidence intervals.</span><a href="evidence/rms-softmax/results.html#article-comparison">Measured results ↗</a></div>`);
  let rows = 16, selected = 3;
  const fit = panel.querySelector('[data-fit]');
  const sessions = panel.querySelector('[data-sessions]');
  const render = () => {
    const points = data.completion.filter(p => p.rows === rows);
    if (points.length !== 4) throw new Error('Incomplete measured row count');
    const maximum = fit.checked ? ({1:40,16:50,32:80,64:160})[rows] : 160;
    root.dataset.rows = String(rows);
    root.dataset.axisMax = String(maximum);
    root.dataset.selectedMethod = String(selected);
    panel.querySelectorAll('[data-rows]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.rows) === rows)));
    panel.querySelector('[data-scale-label]').textContent = `Axis: 0–${maximum} µs${fit.checked ? ' · fitted' : ' · fixed across rows'}`;
    const delta = points[2].mean - points[3].mean;
    panel.querySelector('.latency-gain').innerHTML = `${delta.toFixed(2)} µs <span>(${(100 * delta / points[2].mean).toFixed(1)}%)</span>`;
    panel.querySelector('.latency-context').textContent = `${rows} × 131,072 input · ${rows === 1 ? 'grid-wide' : 'row-scoped'} barriers · ${rows === 64 ? '2 cooperative launches' : '1 cooperative launch'} for the megakernel.`;
    panel.querySelectorAll('.latency-bar').forEach((button, i) => {
      const p = points[i];
      button.setAttribute('aria-pressed', String(i === selected));
      button.setAttribute('aria-label', `${methods[i].label}: ${p.mean.toFixed(1)} microseconds. Inspect both session medians.`);
      button.querySelector('.latency-value').textContent = `${p.mean.toFixed(1)} µs`;
      button.querySelector('.latency-value').dataset.value = String(p.mean);
      button.querySelector('.latency-fill').style.width = `${100 * p.mean / maximum}%`;
      const range = button.querySelector('.latency-range');
      range.style.left = `${100 * p.min / maximum}%`;
      range.style.width = `${100 * (p.max - p.min) / maximum}%`;
      range.hidden = !sessions.checked;
    });
    panel.querySelector('.latency-axis').innerHTML = Array.from({length:5}, (_, i) => `<span>${i * maximum / 4}${i === 4 ? ' µs' : ''}</span>`).join('');
    const point = points[selected];
    panel.querySelector('.latency-detail h4').textContent = `${methods[selected].label.replace(/^\+ /, '')} · ${rows} ${rows === 1 ? 'row' : 'rows'}`;
    panel.querySelector('[data-number="mean"]').textContent = `${point.mean.toFixed(4)} µs`;
    panel.querySelector('[data-number="s1"]').textContent = `${point.session_medians[0].toFixed(3)} µs`;
    panel.querySelector('[data-number="s2"]').textContent = `${point.session_medians[1].toFixed(3)} µs`;
    panel.querySelector('.method-description').textContent = methods[selected].description;
  };
  panel.querySelectorAll('[data-rows]').forEach(b => b.addEventListener('click', () => { rows = Number(b.dataset.rows); render(); }));
  panel.querySelectorAll('.latency-bar').forEach(b => b.addEventListener('click', () => { selected = Number(b.dataset.method); render(); }));
  fit.addEventListener('change', render);
  sessions.addEventListener('change', render);
  render();
  view.commit(undefined, 'Host completion time in µs; lower is better. Bars average the 2 session medians; optional ranges span them. The default axis is fixed across row counts; the Fit control changes its range. Percentages give the megakernel’s latency reduction from local fusion. Graph replay stays enabled for both fusion steps.');
}

async function init() {
  // All chart numbers come from the frozen figure provenance, without interpolation.
  const dataURL = new URL('./data.json', import.meta.url);
  dataURL.search = new URL(import.meta.url).search;
  const response = await fetch(dataURL);
  if (!response.ok) return;
  const data = await response.json();
  replayExplorer();
  storageExplorer();
  latencyExplorer(data);
  const deck = document.querySelector('.post-deck');
  if (deck) deck.insertAdjacentHTML('afterend', '<nav class="figure-jumps" aria-label="Interactive figures"><span>Explore the figures</span><a href="#figure-graph-replay">Submission</a><a href="#figure-storage">Tile lifetimes</a><a href="#figure-completion">Measured latency</a></nav>');
}
init().catch(error => console.warn('Interactive figures unavailable; original figures remain accessible.', error));
