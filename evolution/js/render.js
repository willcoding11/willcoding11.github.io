// render.js — draw a live sim into a canvas with a follow-camera.

export function makeRenderer(canvas) {
  const ctx = canvas.getContext('2d');
  let camX = 0, baseScale = 46, zoom = 1, viewY = 0;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function frictionColor(f) {
    // low friction = cyan (slippery), high friction = orange (grippy)
    const hue = 190 - f * 170;
    return `hsl(${hue}, 70%, 55%)`;
  }

  function draw(sim, opts = {}) {
    const r = canvas.getBoundingClientRect();
    const W = r.width, H = r.height;
    const c = sim.centroid();
    // smooth horizontal camera follow
    camX += (c.x - camX) * 0.1;

    const scale = baseScale * zoom;
    const groundScreenY = H * 0.72;

    // Vertical follow — engages ONLY when the creature climbs above the top
    // of the normal view. On the ground viewY eases back to 0 so the ground
    // sits in its usual place; jump high and the world scrolls down to keep
    // the creature in frame.
    const topMargin = 30;
    const b = sim.bounds();
    const topAtRest = groundScreenY + (b.minY - sim.world.groundY) * scale;
    const desiredView = Math.max(0, topMargin - topAtRest);
    viewY += (desiredView - viewY) * 0.12;
    if (viewY < 0.5) viewY = 0;
    const gy = groundScreenY + viewY; // on-screen ground line

    function sx(x) { return W / 2 + (x - camX) * scale; }
    function wy(y) { return gy + (y - sim.world.groundY) * scale; }

    ctx.clearRect(0, 0, W, H);

    // sky gradient
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, '#0b1220');
    g.addColorStop(1, '#111a2e');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    // ground (drawn at the shifted ground line gy)
    if (gy < H) {
      ctx.fillStyle = '#1c2b16';
      ctx.fillRect(0, gy, W, H - gy);
    }
    ctx.strokeStyle = '#3a5c2a';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();

    // distance grid marks (every 1 unit)
    ctx.fillStyle = 'rgba(255,255,255,0.10)';
    ctx.font = '10px system-ui, sans-serif';
    const start = Math.floor(camX - W / (2 * scale)) - 1;
    const end = Math.ceil(camX + W / (2 * scale)) + 1;
    for (let m = start; m <= end; m++) {
      const x = sx(m);
      ctx.fillRect(x, gy, 1, 8);
      if (m % 2 === 0) ctx.fillText(m + 'm', x + 2, gy + 18);
    }

    // height ruler on the left while the camera is lifted
    if (viewY > 4) {
      ctx.fillStyle = 'rgba(255,255,255,0.18)';
      ctx.font = '10px system-ui, sans-serif';
      for (let hm = 2; ; hm += 2) {
        const yy = wy(sim.world.groundY - hm);
        if (yy < 10) break;
        if (yy > gy) continue;
        ctx.fillRect(0, yy, 8, 1);
        ctx.fillText(hm + 'm', 10, yy + 3);
      }
    }

    // optional target marker
    if (opts.target) {
      const tx = sx(opts.target.x), ty = wy(opts.target.y);
      ctx.strokeStyle = '#ffd24a';
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(tx, ty, 10, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tx - 14, ty); ctx.lineTo(tx + 14, ty);
      ctx.moveTo(tx, ty - 14); ctx.lineTo(tx, ty + 14); ctx.stroke();
    }

    // muscles (line width tracks zoom so creatures look proportional)
    const muscleW = Math.max(1.5, 4 * zoom);
    const nodeR = Math.max(2.5, 7 * zoom);
    ctx.lineWidth = muscleW;
    for (const m of sim.muscles) {
      const a = sim.nodes[m.a], b = sim.nodes[m.b];
      // color by contraction state: red = contracting, green = extending
      const range = (m.max - m.min) || 1e-6;
      const t = ((m.target ?? m.rest) - m.min) / range; // 0 contracted .. 1 extended
      const hue = 0 + t * 130; // red -> green
      ctx.strokeStyle = `hsl(${hue}, 75%, 50%)`;
      ctx.beginPath();
      ctx.moveTo(sx(a.x), wy(a.y));
      ctx.lineTo(sx(b.x), wy(b.y));
      ctx.stroke();
    }

    // nodes
    for (const n of sim.nodes) {
      ctx.beginPath();
      ctx.arc(sx(n.x), wy(n.y), nodeR, 0, Math.PI * 2);
      ctx.fillStyle = frictionColor(n.friction);
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(0,0,0,0.5)';
      ctx.stroke();
    }
  }

  function recenter(sim) {
    const c = sim.centroid();
    camX = c.x; viewY = 0;
  }

  function setZoom(z) { zoom = z; }
  function getZoom() { return zoom; }

  return { draw, resize, recenter, setZoom, getZoom };
}
