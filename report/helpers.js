// ---------- a small canvas plotter
const charts = [];
function chart(canvas, draw) { const o = { canvas, draw }; charts.push(o); return o; }
function render(o) {
  const cv = o.canvas, r = cv.getBoundingClientRect(); if (!r.width) return;
  const dpr = window.devicePixelRatio || 1; cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, r.width, r.height); o.draw(g, r.width, r.height);
}
function redrawAll() { charts.forEach(render); }
new ResizeObserver(redrawAll).observe(document.body);

function niceTicks(a, b, n) { const span = b - a, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0))), f = step0 / mag;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag; const t = []; for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) t.push(+v.toFixed(10)); return t; }
function fmt(v) { const a = Math.abs(v); if (a >= 10000) return (v / 1000).toFixed(0) + 'k'; if (a >= 1000) return v.toLocaleString('en-US'); if (a >= 1 || a === 0) return String(+v.toFixed(2)); return String(+v.toPrecision(2)); }

// frame: {x0,x1,y0,y1,xlog,ylog,xrev,ml,mr,mt,mb,xticks,yticks,xfmt,yfmt,xlabel,ylabel}
function frame(g, W, H, f) {
  const ml = f.ml ?? 50, mr = f.mr ?? 10, mt = f.mt ?? 8, mb = f.mb ?? (f.xlabel ? 40 : 22);
  const tx = v => f.xlog ? Math.log10(v) : v, ty = v => f.ylog ? Math.log10(v) : v;
  let X0 = tx(f.x0), X1 = tx(f.x1); if (f.xrev) [X0, X1] = [X1, X0];
  const Y0 = ty(f.y0), Y1 = ty(f.y1);
  const sx = v => ml + (tx(v) - X0) / (X1 - X0) * (W - ml - mr);
  const sy = v => mt + (1 - (ty(v) - Y0) / (Y1 - Y0)) * (H - mt - mb);
  const P = { sx, sy, ml, mr, mt, mb, W, H, f };
  const sans = css('--sans') || 'sans-serif';
  g.font = `11.5px ${sans}`; g.lineWidth = 1;
  const xt = f.xticks || (f.xlog ? decades(f.x0, f.x1) : niceTicks(Math.min(f.x0, f.x1), Math.max(f.x0, f.x1), Math.max(3, Math.floor((W - ml) / 90))));
  const yt = f.yticks || (f.ylog ? decades(f.y0, f.y1) : niceTicks(f.y0, f.y1, Math.max(2, Math.floor((H - mt - mb) / 34))));
  g.strokeStyle = css('--axis'); g.fillStyle = css('--ink');
  g.textAlign = 'center'; g.textBaseline = 'top';
  for (const v of xt) { const x = Math.round(sx(v)) + .5; if (x < ml - 1 || x > W - mr + 1) continue; g.beginPath(); g.moveTo(x, H - mb); g.lineTo(x, H - mb - 4); g.moveTo(x, mt); g.lineTo(x, mt + 4); g.stroke(); const lab = (f.xfmt || fmt)(v), hw = g.measureText(lab).width / 2; g.fillText(lab, Math.min(Math.max(x, hw + 1), W - hw - 1), H - mb + 4); }
  g.textAlign = 'right'; g.textBaseline = 'middle';
  for (const v of yt) { const y = Math.round(sy(v)) + .5; if (y < mt - 1 || y > H - mb + 1) continue; g.beginPath(); g.moveTo(ml, y); g.lineTo(ml + 4, y); g.moveTo(W - mr, y); g.lineTo(W - mr - 4, y); g.stroke(); g.fillText((f.yfmt || fmt)(v), ml - 5, y); }
  g.strokeRect(ml + .5, mt + .5, W - ml - mr - 1, H - mt - mb - 1);
  g.font = `12.5px ${sans}`; g.fillStyle = css('--ink');
  if (f.xlabel) { g.textAlign = 'center'; g.textBaseline = 'bottom'; g.fillText(f.xlabel, ml + (W - ml - mr) / 2, H - 2); }
  if (f.ylabel) { g.save(); g.translate(12, mt + (H - mt - mb) / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(f.ylabel, 0, 0); g.restore(); }
  g.save(); g.beginPath(); g.rect(ml + 1, mt + 1, W - ml - mr - 2, H - mt - mb - 2); g.clip(); P.restore = () => g.restore();
  return P;
}
function decades(a, b) { const lo = Math.floor(Math.log10(Math.min(a, b))), hi = Math.ceil(Math.log10(Math.max(a, b))); const t = []; for (let k = lo; k <= hi; k++) t.push(Math.pow(10, k)); return t; }
function band(g, P, x0, x1) { g.fillStyle = css('--change'); g.fillRect(P.sx(x0), P.mt, P.sx(x1) - P.sx(x0), P.H - P.mt - P.mb); g.strokeStyle = css('--change-edge'); g.lineWidth = 1; g.setLineDash([3, 3]); g.beginPath(); for (const x of [x0, x1]) { const X = Math.round(P.sx(x)) + .5; if (X > P.ml + 1 && X < P.W - P.mr - 1) { g.moveTo(X, P.mt); g.lineTo(X, P.H - P.mb); } } g.stroke(); g.setLineDash([]); }
function line(g, P, xs, ys, color, width = 1.6, dash = null) {
  g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash || []); g.beginPath(); let pen = false;
  for (let i = 0; i < xs.length; i++) { const y = ys[i]; if (y == null || !isFinite(y) || !isFinite(xs[i])) { pen = false; continue; } const X = P.sx(xs[i]), Y = P.sy(y); if (!pen) { g.moveTo(X, Y); pen = true; } else g.lineTo(X, Y); }
  g.stroke(); g.setLineDash([]);
}
function hline(g, P, y, color) { line(g, P, [P.f.x0, P.f.x1], [y, y], color, 1.3, [6, 5]); }
function marker(g, P, x, y, color, label) {
  const X = P.sx(x), Y = P.sy(y); g.fillStyle = color; g.beginPath(); g.moveTo(X, Y - 7); g.lineTo(X + 6, Y + 4); g.lineTo(X - 6, Y + 4); g.closePath(); g.fill();
  if (label) { g.font = `600 11.5px ${css('--sans')}`; const right = X > P.W - 140; g.textAlign = right ? 'right' : 'left'; const top = Y - 16 < P.mt; g.textBaseline = top ? 'top' : 'bottom'; g.fillText(label, X + (right ? -9 : 9), top ? Y + 2 : Y - 2); }
}
function dot(g, P, x, y, color, r = 4.5) { g.fillStyle = color; g.beginPath(); g.arc(P.sx(x), P.sy(y), r, 0, 2 * Math.PI); g.fill(); if (r > 3) { g.strokeStyle = css('--paper'); g.lineWidth = 1.5; g.stroke(); } }
function seg(id, onChange) { const el = $(id); el.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; el.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b ? 'true' : 'false')); onChange(b.dataset.v); }); return () => el.querySelector('[aria-pressed=true]').dataset.v; }
const range = n => Array.from({ length: n }, (_, i) => i);
function firstCross(score, h, from = 0, to = score.length) { for (let i = from; i < to; i++) if (score[i] >= h) return i; return -1; }
