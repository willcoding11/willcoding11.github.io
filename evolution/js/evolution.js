// evolution.js — genomes, goals, selection, reproduction, and the GA loop.

import { createSim, makeBrain, rand, randn, clamp, INPUTS } from './engine.js';

// ---- genome construction -----------------------------------------------
// A genome fully describes a creature: its body (nodes + muscles) and its
// brain (neural-net weights). Bodies within one lineage share a topology so
// their brains stay compatible for crossover.

function connectGraph(nodeCount, edges) {
  // Ensure every node is reachable; add edges to join components.
  const parent = Array.from({ length: nodeCount }, (_, i) => i);
  const find = x => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  const union = (a, b) => { parent[find(a)] = find(b); };
  for (const [a, b] of edges) union(a, b);
  for (let i = 1; i < nodeCount; i++) {
    if (find(i) !== find(0)) { edges.push([i, i - 1]); union(i, i - 1); }
  }
  return edges;
}

// Uniform integer in [lo, hi] inclusive.
function randInt(lo, hi) { return lo + Math.floor(Math.random() * (hi - lo + 1)); }

export function randomGenome(cfg) {
  // Respect the node limits exactly: clamp to >=2, tolerate min>max by swapping.
  let lo = Math.max(2, Math.round(cfg.minNodes));
  let hi = Math.max(2, Math.round(cfg.maxNodes));
  if (lo > hi) { const t = lo; lo = hi; hi = t; }
  const nCount = randInt(lo, hi);
  const nodes = [];
  for (let i = 0; i < nCount; i++) {
    nodes.push({
      x: rand(-1.2, 1.2),
      y: rand(2.0, 4.5),               // spawn above the ground plane
      friction: rand(0, 1),
    });
  }
  // Build candidate edges by proximity, keep a subset, then force-connect.
  const cand = [];
  for (let i = 0; i < nCount; i++)
    for (let j = i + 1; j < nCount; j++)
      cand.push([i, j, Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y)]);
  cand.sort((a, b) => a[2] - b[2]);
  const wanted = Math.min(cand.length, Math.round(nCount * cfg.muscleDensity));
  let edges = cand.slice(0, wanted).map(e => [e[0], e[1]]);
  edges = connectGraph(nCount, edges);
  const muscles = edges.map(([a, b]) => ({ a, b, contract: rand(0.15, 0.7) }));

  const brain = makeBrain(cfg.hidden, muscles.length);
  return { nodes, muscles, brain, hidden: cfg.hidden };
}

// Deep clone a genome (weights included).
export function cloneGenome(g) {
  return {
    nodes: g.nodes.map(n => ({ ...n })),
    muscles: g.muscles.map(m => ({ ...m })),
    hidden: g.hidden,
    brain: {
      hidden: g.brain.hidden,
      muscles: g.brain.muscles,
      w1: g.brain.w1.slice(),
      w2: g.brain.w2.slice(),
    },
  };
}

// Structural mutation: grow one new node, wired to its 2 nearest neighbours
// with fresh muscles, and extend the brain's output layer to drive them.
// Returns true if a node was added. Never exceeds cfg.maxNodes.
export function addNode(g, cfg) {
  if (g.nodes.length >= cfg.maxNodes) return false;
  const ni = g.nodes.length;
  const anchor = g.nodes[Math.floor(Math.random() * ni)];
  const nx = anchor.x + randn() * 0.5;
  const ny = clamp(anchor.y + randn() * 0.5, 1.5, 5.0);
  g.nodes.push({ x: nx, y: ny, friction: rand(0, 1) });

  // connect to the nearest existing nodes (2, or 1 if that's all there is)
  const near = [];
  for (let i = 0; i < ni; i++)
    near.push([i, Math.hypot(g.nodes[i].x - nx, g.nodes[i].y - ny)]);
  near.sort((a, b) => a[1] - b[1]);
  const conns = near.slice(0, Math.min(2, ni));
  for (const [idx] of conns)
    g.muscles.push({ a: idx, b: ni, contract: rand(0.15, 0.7) });

  // extend brain output weights (w2 is [muscles][hidden]) with rows for the
  // new muscles; w1 is unchanged since inputs are fixed.
  const h = g.brain.hidden;
  const added = conns.length;
  const old = g.brain.w2;
  const grown = new Float64Array(old.length + added * h);
  grown.set(old, 0);
  for (let i = old.length; i < grown.length; i++) grown[i] = randn() * 0.8;
  g.brain.w2 = grown;
  g.brain.muscles = g.muscles.length;
  return true;
}

// Ensure the body is a single connected graph; append muscles for any
// stragglers. Returns how many muscles were added.
function reconnect(g) {
  const n = g.nodes.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = x => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  const union = (a, b) => { parent[find(a)] = find(b); };
  for (const m of g.muscles) union(m.a, m.b);
  let added = 0;
  for (let i = 1; i < n; i++) {
    if (find(i) !== find(0)) {
      g.muscles.push({ a: i, b: i - 1, contract: rand(0.15, 0.7) });
      union(i, i - 1);
      added++;
    }
  }
  return added;
}

// Structural mutation: remove one node, its muscles, and their brain output
// rows; reindex the rest and reconnect if removal split the body. Never goes
// below cfg.minNodes (clamped to >=2). Returns true if a node was removed.
export function removeNode(g, cfg) {
  const lo = Math.max(2, Math.round(cfg.minNodes));
  if (g.nodes.length <= lo) return false;
  const ri = Math.floor(Math.random() * g.nodes.length);
  const h = g.brain.hidden;

  // keep muscles not touching the removed node; remember their old w2 rows
  const keptRows = [];
  const newMuscles = [];
  g.muscles.forEach((m, idx) => {
    if (m.a === ri || m.b === ri) return;
    newMuscles.push({
      a: m.a > ri ? m.a - 1 : m.a,
      b: m.b > ri ? m.b - 1 : m.b,
      contract: m.contract,
    });
    keptRows.push(idx);
  });
  g.nodes.splice(ri, 1);
  g.muscles = newMuscles;

  // rebuild w2 from surviving rows
  const oldW2 = g.brain.w2;
  let w2 = new Float64Array(keptRows.length * h);
  keptRows.forEach((oldIdx, newIdx) => {
    for (let j = 0; j < h; j++) w2[newIdx * h + j] = oldW2[oldIdx * h + j];
  });

  // if removal disconnected the body, add bridging muscles + weight rows
  const added = reconnect(g);
  if (added > 0) {
    const grown = new Float64Array(w2.length + added * h);
    grown.set(w2, 0);
    for (let i = w2.length; i < grown.length; i++) grown[i] = randn() * 0.8;
    w2 = grown;
  }
  g.brain.w2 = w2;
  g.brain.muscles = g.muscles.length;
  return true;
}

// ---- mutation -----------------------------------------------------------
export function mutate(g, cfg) {
  const r = cfg.mutationRate;          // 0..1 strength
  const brain = g.brain;
  for (let i = 0; i < brain.w1.length; i++)
    if (Math.random() < r) brain.w1[i] += randn() * cfg.mutationStep;
  for (let i = 0; i < brain.w2.length; i++)
    if (Math.random() < r) brain.w2[i] += randn() * cfg.mutationStep;

  if (cfg.evolveBody) {
    for (const n of g.nodes) {
      if (Math.random() < r) n.friction = clamp(n.friction + randn() * 0.15, 0, 1);
      if (Math.random() < r * 0.5) { n.x += randn() * 0.12; n.y += randn() * 0.12; }
    }
    for (const m of g.muscles) {
      if (Math.random() < r) m.contract = clamp(m.contract + randn() * 0.1, 0.05, 0.95);
    }
  }

  // rare structural change — add OR remove a node by 1, within [min,max].
  // If the coin-flip direction is blocked by a limit, try the other way.
  if (cfg.nodeChangeRate > 0 && Math.random() < cfg.nodeChangeRate) {
    if (Math.random() < 0.5) { if (!addNode(g, cfg)) removeNode(g, cfg); }
    else { if (!removeNode(g, cfg)) addNode(g, cfg); }
  }

  return g;
}

// Crossover two same-topology genomes -> child (body from A, weights mixed).
export function crossover(a, b) {
  // Only mix genomes with identical topology (same nodes AND muscles); after
  // structural mutation, two genomes can share a muscle count but differ in
  // node count, so check both before blending index-by-index.
  if (a.brain.w1.length !== b.brain.w1.length ||
      a.brain.w2.length !== b.brain.w2.length ||
      a.nodes.length !== b.nodes.length) {
    return cloneGenome(Math.random() < 0.5 ? a : b);
  }
  const child = cloneGenome(a);
  for (let i = 0; i < child.brain.w1.length; i++)
    if (Math.random() < 0.5) child.brain.w1[i] = b.brain.w1[i];
  for (let i = 0; i < child.brain.w2.length; i++)
    if (Math.random() < 0.5) child.brain.w2[i] = b.brain.w2[i];
  // blend body a touch
  for (let i = 0; i < child.nodes.length; i++) {
    if (Math.random() < 0.5) child.nodes[i].friction = b.nodes[i].friction;
  }
  return child;
}

// ---- goals (fitness functions) -----------------------------------------
// Each goal drives the sim for a fixed number of steps and scores the run.
// Higher score = fitter.
export const GOALS = {
  run_right: {
    label: 'Run right (distance →)',
    unit: 'm',
    score(sim, steps) {
      const start = sim.centroid().x;
      for (let i = 0; i < steps; i++) sim.step();
      return sim.centroid().x - start;
    },
  },
  run_far: {
    label: 'Run far (either direction)',
    unit: 'm',
    score(sim, steps) {
      const start = sim.centroid().x;
      for (let i = 0; i < steps; i++) sim.step();
      return Math.abs(sim.centroid().x - start);
    },
  },
  jump: {
    label: 'Jump high (peak height)',
    unit: 'm',
    score(sim, steps) {
      let peak = 0;
      for (let i = 0; i < steps; i++) {
        sim.step();
        const h = sim.world.groundY - sim.bounds().minY; // higher = better
        if (h > peak) peak = h;
      }
      return peak;
    },
  },
  reach: {
    label: 'Reach a target point',
    unit: 'm',
    score(sim, steps, extra) {
      const tx = extra.targetX, ty = extra.targetY;
      // Fitness = closeness on the FINAL frame, not the closest it ever got.
      // This rewards creatures that arrive AND stay, not ones that merely
      // brush past the target and drift away.
      for (let i = 0; i < steps; i++) sim.step();
      const c = sim.centroid();
      const d = Math.hypot(c.x - tx, c.y - ty);
      return -d; // closer at the end = higher score
    },
  },
};

// ---- selection with a configurable killing gradient ---------------------
// Individuals are ranked best..worst. `killFraction` sets roughly how many
// die; `gradient` sets how sharp the cutoff is. Low gradient = soft/noisy
// selection (good ones sometimes die, weak ones sometimes survive); high
// gradient = a hard guillotine at the kill line.
export function killProbability(rankFrac, killFraction, gradient) {
  // rankFrac: 0 = best, 1 = worst. Threshold sits where survival ends.
  const threshold = 1 - killFraction;
  return 1 / (1 + Math.exp(-gradient * (rankFrac - threshold)));
}

// Given scored population (desc), decide who survives -> array of survivor
// genomes. Always keeps at least the top individual (elitism).
export function select(scored, cfg) {
  const N = scored.length;
  const survivors = [];
  for (let i = 0; i < N; i++) {
    const rankFrac = N === 1 ? 0 : i / (N - 1);
    if (i === 0) { survivors.push(scored[i]); continue; } // elite
    const p = killProbability(rankFrac, cfg.killFraction, cfg.gradient);
    if (Math.random() > p) survivors.push(scored[i]);
  }
  if (survivors.length === 0) survivors.push(scored[0]);
  return survivors;
}

// Weighted pick from survivors, biased toward the top of the list.
function pickParent(survivors) {
  // rank-weighted: earlier (fitter) entries get more weight
  const n = survivors.length;
  const r = Math.random();
  // quadratic bias toward front
  const idx = Math.floor(n * r * r);
  return survivors[Math.min(idx, n - 1)];
}

export function reproduce(survivors, cfg, targetSize) {
  const next = [];
  // elitism: carry the best unchanged
  next.push(cloneGenome(survivors[0].genome));
  while (next.length < targetSize) {
    const pa = pickParent(survivors).genome;
    let child;
    if (cfg.crossover && survivors.length > 1 && Math.random() < 0.6) {
      const pb = pickParent(survivors).genome;
      child = crossover(pa, pb);
    } else {
      child = cloneGenome(pa);
    }
    mutate(child, cfg);
    next.push(child);
  }
  return next;
}

// ---- evaluate a whole population ---------------------------------------
// Returns array of { genome, score } sorted best-first. Runs headless.
export function evaluate(population, cfg) {
  const goal = GOALS[cfg.goal];
  const extra = { targetX: cfg.targetX, targetY: cfg.targetY };
  const scored = population.map(genome => {
    const sim = createSim(genome, { world: cfg.world });
    const score = goal.score(sim, cfg.steps, extra);
    return { genome, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored;
}
