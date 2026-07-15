// main.js — glue: read config, run the GA, animate best/median/worst previews.

import { createSim, makeBrain, clamp } from './engine.js';
import {
  randomGenome, cloneGenome, select, reproduce, GOALS, killProbability,
} from './evolution.js';
import { makeRenderer } from './render.js';

const $ = id => document.getElementById(id);

// ---- config ------------------------------------------------------------
function readConfig() {
  const cfg = {
    population: +$('population').value,
    steps: +$('steps').value,
    goal: $('goal').value,
    killFraction: +$('killFraction').value / 100,
    gradient: +$('gradient').value,
    mutationRate: +$('mutationRate').value / 100,
    mutationStep: +$('mutationStep').value,
    crossover: $('crossover').checked,
    evolveBody: $('evolveBody').checked,
    nodeChangeRate: +$('nodeChange').value / 100,
    minNodes: +$('minNodes').value,
    maxNodes: +$('maxNodes').value,
    muscleDensity: +$('muscleDensity').value,
    hidden: +$('hidden').value,
    targetX: +$('targetX').value,
    targetY: +$('targetY').value,
    world: {
      gravity: +$('gravity').value,
      maxForce: +$('maxForce').value,
      stiffness: +$('stiffness').value,
    },
  };
  return cfg;
}

// ---- population --------------------------------------------------------
let population = [];
let generation = 0;
let history = [];       // [{gen, best, avg}]
let lastScored = null;  // scored array from most recent evaluation
let customBody = null;  // optional user-defined body
let running = false;
let busy = false;

// ---- species tracking --------------------------------------------------
// A "species" is a distinct body topology: node count + muscle count +
// degree sequence (so wiring differences count, but node ordering doesn't).
let speciesHistory = [];             // [{gen, total, counts:{key:n}}]
const speciesColors = new Map();     // key -> hsl color, stable across gens
let speciesColorIdx = 0;

function speciesKey(g) {
  const N = g.nodes.length;
  const deg = new Array(N).fill(0);
  for (const m of g.muscles) { deg[m.a]++; deg[m.b]++; }
  deg.sort((a, b) => a - b);
  return `${N}|${g.muscles.length}|${deg.join(',')}`;
}
function speciesLabel(key) {
  const [n, m] = key.split('|');
  return `${n}n·${m}m`;
}
function speciesColor(key) {
  if (!speciesColors.has(key)) {
    // golden-angle hue stepping keeps successive species visually distinct
    const hue = (speciesColorIdx * 137.508) % 360;
    const light = 50 + (speciesColorIdx % 3) * 7;
    speciesColors.set(key, `hsl(${hue.toFixed(0)}, 68%, ${light}%)`);
    speciesColorIdx++;
  }
  return speciesColors.get(key);
}
function recordSpecies(scored) {
  const counts = {};
  for (const s of scored) {
    const key = speciesKey(s.genome);
    counts[key] = (counts[key] || 0) + 1;
    speciesColor(key); // assign color on first sighting
  }
  speciesHistory.push({ gen: generation + 1, total: scored.length, counts });
}

function buildPopulation(cfg) {
  const pop = [];
  for (let i = 0; i < cfg.population; i++) {
    if (customBody) {
      pop.push({
        nodes: customBody.nodes.map(n => ({ ...n })),
        muscles: customBody.muscles.map(m => ({ ...m })),
        hidden: cfg.hidden,
        brain: makeBrain(cfg.hidden, customBody.muscles.length),
      });
    } else {
      pop.push(randomGenome(cfg));
    }
  }
  return pop;
}

function resetSim() {
  const cfg = readConfig();
  population = buildPopulation(cfg);
  generation = 0;
  history = [];
  lastScored = null;
  speciesHistory = [];
  speciesColors.clear();
  speciesColorIdx = 0;
  setStatus('Fresh population of ' + cfg.population + ' creatures. Run a generation.');
  drawChart();
  drawSpeciesChart();
  updateStats(null);
}

// ---- chunked evaluation (keeps UI responsive) --------------------------
async function evaluateChunked(cfg, onProgress) {
  const goal = GOALS[cfg.goal];
  const extra = { targetX: cfg.targetX, targetY: cfg.targetY };
  const scored = [];
  const chunk = Math.max(1, Math.round(cfg.population / 20));
  for (let i = 0; i < population.length; i++) {
    const genome = population[i];
    const sim = createSim(genome, { world: cfg.world });
    const score = goal.score(sim, cfg.steps, extra);
    scored.push({ genome, score });
    if (i % chunk === 0) {
      onProgress(i / population.length);
      await new Promise(r => setTimeout(r, 0));
    }
  }
  scored.sort((a, b) => b.score - a.score);
  return scored;
}

async function runGeneration() {
  if (busy) return;
  busy = true;
  const cfg = readConfig();
  setStatus(`Simulating generation ${generation + 1}…`);
  $('progress').style.width = '0%';

  const scored = await evaluateChunked(cfg, p => {
    $('progress').style.width = Math.round(p * 100) + '%';
  });
  $('progress').style.width = '100%';

  lastScored = scored;
  const best = scored[0].score;
  const avg = scored.reduce((s, x) => s + x.score, 0) / scored.length;
  history.push({ gen: generation + 1, best, avg });
  recordSpecies(scored);

  // pick previews from THIS evaluated generation
  setPreviews(scored);

  // selection + reproduction -> next generation
  const survivors = select(scored, cfg);
  population = reproduce(survivors, cfg, cfg.population);
  generation++;

  updateStats({ best, avg, survivors: survivors.length, total: scored.length });
  drawChart();
  drawSpeciesChart();
  setStatus(`Generation ${generation} done — best ${best.toFixed(2)}${GOALS[cfg.goal].unit}, ` +
            `${survivors.length}/${scored.length} survived.`);
  busy = false;
}

async function autoRun() {
  if (running) { running = false; $('autoBtn').textContent = '▶ Auto-run'; return; }
  running = true;
  $('autoBtn').textContent = '⏸ Pause';
  while (running) {
    await runGeneration();
    await new Promise(r => setTimeout(r, 30));
  }
}

// ---- previews (best / median / worst) ----------------------------------
const previews = {
  best: { canvas: 'pvBest', label: 'lbBest', renderer: null, sim: null, genome: null },
  median: { canvas: 'pvMedian', label: 'lbMedian', renderer: null, sim: null, genome: null },
  worst: { canvas: 'pvWorst', label: 'lbWorst', renderer: null, sim: null, genome: null },
};

function initPreviews() {
  for (const k in previews) {
    const p = previews[k];
    const canvas = $(p.canvas);
    p.renderer = makeRenderer(canvas);
    p.renderer.resize();
    addZoomGestures(canvas);
  }
}

function setPreviews(scored) {
  const cfg = readConfig();
  const N = scored.length;
  const picks = {
    best: scored[0],
    median: scored[Math.floor(N / 2)],
    worst: scored[N - 1],
  };
  for (const k in previews) {
    const p = previews[k];
    p.genome = cloneGenome(picks[k].genome);
    p.score = picks[k].score;
    p.sim = createSim(p.genome, { world: cfg.world });
    p.renderer.recenter(p.sim);
    p.steps = cfg.steps;
    p.frame = 0;
    p.acc = 0;
  }
}

let previewSpeed = 1;   // physics steps advanced per animation frame (fractional ok)
let previewZoom = 1;
function previewLoop() {
  const cfg = readConfig();
  const target = cfg.goal === 'reach' ? { x: cfg.targetX, y: cfg.targetY } : null;
  for (const k in previews) {
    const p = previews[k];
    if (!p.sim) { p.renderer && idleDraw(p); continue; }
    // accumulator: at 0.1x we step once every ~10 frames; at 3x, 3 steps/frame
    p.acc = (p.acc || 0) + previewSpeed;
    while (p.acc >= 1 && p.frame < p.steps) {
      p.sim.step();
      p.frame++;
      p.acc -= 1;
    }
    p.renderer.draw(p.sim, { target });
    const unit = GOALS[cfg.goal].unit;
    $(p.label).textContent =
      `${labelFor(k)} · ${p.score != null ? p.score.toFixed(2) + unit : '—'}` +
      `  (${p.frame}/${p.steps})`;
    if (p.frame >= p.steps) {
      // loop the replay
      p.sim = createSim(p.genome, { world: cfg.world });
      p.renderer.recenter(p.sim);
      p.frame = 0;
      p.acc = 0;
    }
  }
  requestAnimationFrame(previewLoop);
}

function applyZoom(z) {
  previewZoom = clamp(z, 0.3, 4);
  for (const k in previews) previews[k].renderer && previews[k].renderer.setZoom(previewZoom);
  const zs = $('zoom'); if (zs) zs.value = previewZoom;
  const zv = $('zoomVal'); if (zv) zv.textContent = previewZoom.toFixed(1) + '×';
}

// wheel + pinch zoom directly on a preview canvas (updates the shared zoom)
function addZoomGestures(canvas) {
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    applyZoom(previewZoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12));
  }, { passive: false });
  let pinchStart = null, zoomStart = 1;
  const dist = t => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
  canvas.addEventListener('touchmove', e => {
    if (e.touches.length === 2) {
      e.preventDefault();
      const d = dist(e.touches);
      if (pinchStart == null) { pinchStart = d; zoomStart = previewZoom; }
      else applyZoom(zoomStart * d / pinchStart);
    }
  }, { passive: false });
  const end = e => { if (e.touches.length < 2) pinchStart = null; };
  canvas.addEventListener('touchend', end);
  canvas.addEventListener('touchcancel', end);
}

function idleDraw(p) {
  // nothing evaluated yet — leave canvas blank-ish
}

function labelFor(k) {
  return k === 'best' ? '🏆 Best' : k === 'median' ? '➖ Median' : '💀 Worst';
}

// ---- stats + chart -----------------------------------------------------
function updateStats(s) {
  $('genNum').textContent = generation;
  if (!s) { $('bestVal').textContent = '—'; $('avgVal').textContent = '—'; $('survVal').textContent = '—'; return; }
  const unit = GOALS[readConfig().goal].unit;
  $('bestVal').textContent = s.best.toFixed(2) + unit;
  $('avgVal').textContent = s.avg.toFixed(2) + unit;
  $('survVal').textContent = `${s.survivors}/${s.total}`;
}

function drawChart() {
  const canvas = $('chart');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = r.width, H = r.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d1526'; ctx.fillRect(0, 0, W, H);
  if (history.length < 2) {
    ctx.fillStyle = '#6b7a94'; ctx.font = '12px system-ui';
    ctx.fillText('Fitness over generations appears here', 10, H / 2);
    return;
  }
  let lo = Infinity, hi = -Infinity;
  for (const h of history) { lo = Math.min(lo, h.avg, h.best); hi = Math.max(hi, h.best, h.avg); }
  if (hi - lo < 1e-6) hi = lo + 1;
  const pad = 24;
  const xAt = i => pad + (i / (history.length - 1)) * (W - pad * 2);
  const yAt = v => H - pad - ((v - lo) / (hi - lo)) * (H - pad * 2);

  ctx.strokeStyle = '#22314e'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad, H - pad);
  ctx.moveTo(pad, pad); ctx.lineTo(pad, H - pad); ctx.stroke();

  function line(key, color) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    history.forEach((h, i) => {
      const x = xAt(i), y = yAt(h[key]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  line('avg', '#5aa9ff');
  line('best', '#4ade80');

  ctx.fillStyle = '#4ade80'; ctx.font = '11px system-ui';
  ctx.fillText('best', W - pad - 60, pad + 4);
  ctx.fillStyle = '#5aa9ff';
  ctx.fillText('average', W - pad - 60, pad + 18);
}

// ---- species dominance chart (stacked area of body topologies) ---------
function drawSpeciesChart() {
  const canvas = $('speciesChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = r.width, H = r.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d1526'; ctx.fillRect(0, 0, W, H);

  const legend = $('speciesLegend');
  if (speciesHistory.length < 1) {
    ctx.fillStyle = '#6b7a94'; ctx.font = '12px system-ui';
    ctx.fillText('Species (body plans) over generations appears here', 10, H / 2);
    if (legend) legend.innerHTML = '';
    return;
  }

  // rank species by total prevalence; show top N, lump the rest as "Other"
  const totals = {};
  for (const gen of speciesHistory)
    for (const k in gen.counts) totals[k] = (totals[k] || 0) + gen.counts[k];
  const ranked = Object.keys(totals).sort((a, b) => totals[b] - totals[a]);
  const TOP = 9;
  const shown = ranked.slice(0, TOP);
  const shownSet = new Set(shown);
  const hasOther = ranked.length > TOP;

  // stacking order: dominant at the bottom, "Other" on top
  const order = shown.slice();
  const pad = 4;
  const plotW = W - pad * 2, plotH = H - pad * 2;
  const n = speciesHistory.length;
  const xAt = i => pad + (n === 1 ? plotW / 2 : (i / (n - 1)) * plotW);

  // proportion of a species key in generation g
  const prop = (g, key) => (g.counts[key] || 0) / g.total;
  const otherProp = g => {
    let s = 0; for (const k in g.counts) if (!shownSet.has(k)) s += g.counts[k];
    return s / g.total;
  };

  const bands = order.map(k => ({ key: k, color: speciesColor(k) }));
  if (hasOther) bands.push({ key: '__other__', color: '#5b6472' });

  // cumulative baseline per generation (top = 0, grows downward)
  const base = new Array(n).fill(0);
  for (const band of bands) {
    ctx.fillStyle = band.color;
    ctx.beginPath();
    // top edge left->right
    for (let i = 0; i < n; i++) {
      const p = band.key === '__other__' ? otherProp(speciesHistory[i]) : prop(speciesHistory[i], band.key);
      const yTop = pad + base[i] * plotH;
      const x = xAt(i);
      i ? ctx.lineTo(x, yTop) : ctx.moveTo(x, yTop);
      base[i] += p;
    }
    // bottom edge right->left (new cumulative)
    for (let i = n - 1; i >= 0; i--) {
      const yBot = pad + base[i] * plotH;
      ctx.lineTo(xAt(i), yBot);
    }
    ctx.closePath();
    ctx.fill();
  }

  // for a single generation the area collapses to a line, so redraw it as a
  // centered vertical bar for clarity
  if (n === 1) {
    ctx.fillStyle = '#0d1526'; ctx.fillRect(0, 0, W, H);
    let acc = 0;
    const bw = Math.min(plotW, 120), bx = (W - bw) / 2;
    for (const band of bands) {
      const g = speciesHistory[0];
      const p = band.key === '__other__' ? otherProp(g) : prop(g, band.key);
      ctx.fillStyle = band.color;
      ctx.fillRect(bx, pad + acc * plotH, bw, p * plotH);
      acc += p;
    }
  }

  updateSpeciesLegend(shown, hasOther, totals);
}

function updateSpeciesLegend(shown, hasOther, totals) {
  const legend = $('speciesLegend');
  if (!legend) return;
  const latest = speciesHistory[speciesHistory.length - 1];
  // order legend by CURRENT share so live species lead, extinct ones trail
  const ordered = shown.slice().sort((a, b) =>
    (latest.counts[b] || 0) - (latest.counts[a] || 0));
  const items = ordered.map(key => {
    const pct = Math.round(((latest.counts[key] || 0) / latest.total) * 100);
    return `<span class="sp"><i style="background:${speciesColor(key)}"></i>${speciesLabel(key)} <b>${pct}%</b></span>`;
  });
  if (hasOther) {
    let o = 0; for (const k in latest.counts) if (!shown.includes(k)) o += latest.counts[k];
    items.push(`<span class="sp"><i style="background:#5b6472"></i>Other <b>${Math.round(o / latest.total * 100)}%</b></span>`);
  }
  const distinct = Object.keys(totals).length;
  legend.innerHTML = `<div class="sp-count">${distinct} species this run</div>` + items.join('');
}

// ---- selection-curve preview -------------------------------------------
function drawKillCurve() {
  const canvas = $('killCurve');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = r.width, H = r.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d1526'; ctx.fillRect(0, 0, W, H);
  const cfg = readConfig();
  ctx.strokeStyle = '#f87171'; ctx.lineWidth = 2; ctx.beginPath();
  for (let px = 0; px <= W; px++) {
    const rankFrac = px / W;                 // best..worst
    const p = killProbability(rankFrac, cfg.killFraction, cfg.gradient);
    const y = H - p * H;
    px ? ctx.lineTo(px, y) : ctx.moveTo(px, y);
  }
  ctx.stroke();
  ctx.fillStyle = '#6b7a94'; ctx.font = '10px system-ui';
  ctx.fillText('best', 2, H - 3);
  ctx.fillText('worst', W - 30, H - 3);
  ctx.fillText('death %', 2, 10);
}

// ---- status ------------------------------------------------------------
function setStatus(msg) { $('status').textContent = msg; }

// ---- custom body editor ------------------------------------------------
function applyCustomBody() {
  const txt = $('bodyJson').value.trim();
  if (!txt) { customBody = null; setStatus('Using random bodies.'); return; }
  try {
    const b = JSON.parse(txt);
    if (!Array.isArray(b.nodes) || !Array.isArray(b.muscles)) throw new Error('need nodes[] and muscles[]');
    for (const m of b.muscles) {
      if (b.nodes[m.a] == null || b.nodes[m.b] == null) throw new Error('muscle references missing node');
      if (m.contract == null) m.contract = 0.4;
    }
    for (const n of b.nodes) {
      if (n.x == null || n.y == null) throw new Error('node needs x and y');
      if (n.friction == null) n.friction = 0.5;
    }
    customBody = b;
    setStatus(`Custom body loaded: ${b.nodes.length} nodes, ${b.muscles.length} muscles. Reset to apply.`);
  } catch (e) {
    setStatus('Body JSON error: ' + e.message);
  }
}

const EXAMPLE_BODY = {
  nodes: [
    { x: -0.8, y: 2.2, friction: 0.2 },
    { x: 0.8, y: 2.2, friction: 0.2 },
    { x: -0.8, y: 3.6, friction: 0.9 },
    { x: 0.8, y: 3.6, friction: 0.9 },
    { x: 0.0, y: 2.9, friction: 0.5 },
  ],
  muscles: [
    { a: 0, b: 1, contract: 0.3 },
    { a: 2, b: 3, contract: 0.3 },
    { a: 0, b: 2, contract: 0.6 },
    { a: 1, b: 3, contract: 0.6 },
    { a: 0, b: 4, contract: 0.5 },
    { a: 1, b: 4, contract: 0.5 },
    { a: 2, b: 4, contract: 0.5 },
    { a: 3, b: 4, contract: 0.5 },
  ],
};

// ---- wiring ------------------------------------------------------------
function wire() {
  $('runBtn').addEventListener('click', () => runGeneration());
  $('autoBtn').addEventListener('click', () => autoRun());
  $('resetBtn').addEventListener('click', () => resetSim());
  $('speed').addEventListener('input', e => { previewSpeed = +e.target.value; $('speedVal').textContent = previewSpeed.toFixed(1) + '×'; });
  $('zoom').addEventListener('input', e => applyZoom(+e.target.value));
  $('applyBody').addEventListener('click', applyCustomBody);
  $('exampleBody').addEventListener('click', () => {
    $('bodyJson').value = JSON.stringify(EXAMPLE_BODY, null, 2);
  });
  $('clearBody').addEventListener('click', () => { $('bodyJson').value = ''; customBody = null; setStatus('Using random bodies.'); });

  // live-update dependent widgets
  ['killFraction', 'gradient'].forEach(id => $(id).addEventListener('input', drawKillCurve));
  const showVal = (id, out, suffix = '') => {
    const el = $(id), o = $(out);
    if (!el || !o) return;
    const upd = () => o.textContent = el.value + suffix;
    el.addEventListener('input', upd); upd();
  };
  showVal('killFraction', 'killFractionVal', '%');
  showVal('gradient', 'gradientVal');
  showVal('mutationRate', 'mutationRateVal', '%');
  showVal('nodeChange', 'nodeChangeVal', '%');
  showVal('goal', null); // no-op guard

  // toggle target inputs by goal
  $('goal').addEventListener('change', () => {
    $('targetRow').style.display = $('goal').value === 'reach' ? '' : 'none';
  });
  $('targetRow').style.display = $('goal').value === 'reach' ? '' : 'none';

  window.addEventListener('resize', () => {
    for (const k in previews) previews[k].renderer && previews[k].renderer.resize();
    drawChart(); drawKillCurve(); drawSpeciesChart();
  });
}

// populate goal dropdown
function fillGoals() {
  const sel = $('goal');
  for (const key in GOALS) {
    const o = document.createElement('option');
    o.value = key; o.textContent = GOALS[key].label;
    sel.appendChild(o);
  }
}

// ---- boot --------------------------------------------------------------
fillGoals();
wire();
initPreviews();
resetSim();
drawKillCurve();
requestAnimationFrame(previewLoop);

// register service worker for offline / installable PWA
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js').catch(() => {});
}
