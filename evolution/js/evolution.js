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

export function randomGenome(cfg) {
  const nCount = Math.round(rand(cfg.minNodes, cfg.maxNodes + 0.999));
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
  return g;
}

// Crossover two same-topology genomes -> child (body from A, weights mixed).
export function crossover(a, b) {
  if (a.brain.w1.length !== b.brain.w1.length ||
      a.brain.w2.length !== b.brain.w2.length) {
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
