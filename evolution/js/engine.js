// engine.js — 2D soft-body physics + neural brain + single-creature simulation.
//
// A creature is a set of point-mass NODES connected by spring MUSCLES.
// Each node has a ground-friction coefficient (0 = ice, 1 = grip).
// Each muscle oscillates its rest length between a contracted and an
// extended length; a small feed-forward neural net (the "brain") decides,
// every step, how contracted each muscle should be. Differential friction
// between nodes turns that wobbling into locomotion.
//
// Physics uses Verlet integration with position-based spring constraints,
// which stays stable even with stiff springs and cheap integration.

export const INPUTS = 8; // brain input vector size (fixed, body-independent)

// ---- small math helpers -------------------------------------------------
export function rand(a, b) { return a + Math.random() * (b - a); }
export function randn() {
  // Box–Muller standard normal
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
export function clamp(x, a, b) { return x < a ? a : x > b ? b : x; }
function tanh(x) { return Math.tanh(x); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// ---- world constants ----------------------------------------------------
export const WORLD = {
  gravity: 0.02,        // downward acceleration per step (y is DOWN)
  airDamping: 0.997,    // velocity retained per sub-step (air resistance)
  groundY: 6.0,         // ground plane (world units)
  stiffness: 6.0,       // muscle spring constant k (force per unit stretch)
  muscleDamping: 0.9,   // spring damping c (resists rapid length change)
  maxForce: 0.6,        // MAX actuation force a muscle can exert (user-set)
  substeps: 8,          // physics sub-steps per step (stability)
};

// ---- brain --------------------------------------------------------------
// Feed-forward net: INPUTS -> hidden (tanh) -> muscles (sigmoid).
// Weights are flat Float64Arrays so mutation/crossover is cheap.
export function makeBrain(hidden, muscles) {
  const w1 = new Float64Array(hidden * INPUTS);
  const w2 = new Float64Array(muscles * hidden);
  for (let i = 0; i < w1.length; i++) w1[i] = randn() * 0.8;
  for (let i = 0; i < w2.length; i++) w2[i] = randn() * 0.8;
  return { hidden, muscles, w1, w2 };
}

// Evaluate brain -> Float64Array of muscle activations in [0,1].
function think(brain, inputs, out, hbuf) {
  const { hidden, muscles, w1, w2 } = brain;
  for (let h = 0; h < hidden; h++) {
    let s = 0;
    const base = h * INPUTS;
    for (let i = 0; i < INPUTS; i++) s += w1[base + i] * inputs[i];
    hbuf[h] = tanh(s);
  }
  for (let m = 0; m < muscles; m++) {
    let s = 0;
    const base = m * hidden;
    for (let h = 0; h < hidden; h++) s += w2[base + h] * hbuf[h];
    out[m] = sigmoid(s);
  }
}

// ---- simulation of one creature ----------------------------------------
// genome: { nodes:[{x,y,friction}], muscles:[{a,b,rest,contract}], brain, hidden }
// Returns a live sim object you can step() and read for rendering.
export function createSim(genome, opts = {}) {
  const world = Object.assign({}, WORLD, opts.world || {});
  const nodes = genome.nodes.map(n => ({
    x: n.x, y: n.y, vx: 0, vy: 0,
    friction: n.friction, mass: 1, onGround: false,
    fx: 0, fy: 0,
  }));
  const muscles = genome.muscles.map(m => {
    const dx = genome.nodes[m.b].x - genome.nodes[m.a].x;
    const dy = genome.nodes[m.b].y - genome.nodes[m.a].y;
    const rest = m.rest != null ? m.rest : Math.hypot(dx, dy);
    const c = m.contract; // 0..1 fraction of rest length swing
    return { a: m.a, b: m.b, rest, min: rest * (1 - c), max: rest * (1 + c) };
  });
  const brain = genome.brain;
  const out = new Float64Array(muscles.length);
  const hbuf = new Float64Array(brain.hidden);
  const inputs = new Float64Array(INPUTS);
  // freqs for the three internal oscillators
  const w = [0.06, 0.13, 0.27];

  let t = 0;

  const k = world.stiffness;
  const c = world.muscleDamping;
  const maxF = world.maxForce;
  const sub = Math.max(1, world.substeps | 0);
  const h = 1 / sub;                 // sub-step dt (sub-steps sum to dt=1)

  function step() {
    // --- sensors / brain inputs ---
    let avgVX = 0, avgVY = 0, contact = 0;
    for (const n of nodes) {
      avgVX += n.vx;
      avgVY += n.vy;
      if (n.onGround) contact++;
    }
    const inv = 1 / nodes.length;
    avgVX *= inv; avgVY *= inv;
    inputs[0] = 1;                       // bias
    inputs[1] = Math.sin(t * w[0]);
    inputs[2] = Math.sin(t * w[1]);
    inputs[3] = Math.sin(t * w[2]);
    inputs[4] = contact * inv;           // fraction of nodes touching ground
    inputs[5] = clamp(avgVX * 6, -1, 1); // horizontal speed sense
    inputs[6] = clamp(avgVY * 6, -1, 1); // vertical speed sense
    inputs[7] = Math.cos(t * w[0]);      // quadrature clock

    think(brain, inputs, out, hbuf);
    for (let i = 0; i < muscles.length; i++) {
      const m = muscles[i];
      m.target = m.min + (m.max - m.min) * out[i]; // desired length this step
    }

    for (const n of nodes) n.onGround = false;

    // --- force-based integration (semi-implicit Euler, sub-stepped) ---
    for (let s = 0; s < sub; s++) {
      // reset force accumulators; gravity as a body force
      for (const n of nodes) { n.fx = 0; n.fy = world.gravity * n.mass; }

      // each muscle is a DAMPED SPRING that pulls toward its target length,
      // but the force it can exert is capped at maxForce. So a muscle can't
      // instantly snap to length — it accelerates the nodes with limited
      // strength and, under enough load, simply can't reach the target.
      for (const m of muscles) {
        const a = nodes[m.a], b = nodes[m.b];
        let dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.hypot(dx, dy) || 1e-6;
        const ux = dx / d, uy = dy / d;                 // unit axis a->b
        const stretch = d - m.target;                   // >0 too long (contract)
        const relVel = (b.vx - a.vx) * ux + (b.vy - a.vy) * uy;
        let Fmag = k * stretch + c * relVel;            // spring + damping
        if (Fmag > maxF) Fmag = maxF;                   // clamp to max force
        else if (Fmag < -maxF) Fmag = -maxF;
        const fxc = Fmag * ux, fyc = Fmag * uy;
        a.fx += fxc; a.fy += fyc;                       // pulls a toward b
        b.fx -= fxc; b.fy -= fyc;
      }

      // integrate velocity + position, then resolve the ground
      for (const n of nodes) {
        n.vx = (n.vx + (n.fx / n.mass) * h) * world.airDamping;
        n.vy = (n.vy + (n.fy / n.mass) * h) * world.airDamping;
        n.x += n.vx * h;
        n.y += n.vy * h;
        if (n.y > world.groundY) {
          n.y = world.groundY;
          n.onGround = true;
          if (n.vy > 0) n.vy = 0;          // no bounce
          n.vx *= (1 - n.friction);        // ground friction
        }
      }
    }
    t++;
  }

  function bounds() {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const n of nodes) {
      if (n.x < minX) minX = n.x;
      if (n.x > maxX) maxX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.y > maxY) maxY = n.y;
    }
    return { minX, maxX, minY, maxY };
  }

  function centroid() {
    let x = 0, y = 0;
    for (const n of nodes) { x += n.x; y += n.y; }
    return { x: x / nodes.length, y: y / nodes.length };
  }

  return { nodes, muscles, world, step, bounds, centroid, get t() { return t; } };
}
