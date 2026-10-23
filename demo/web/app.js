"use strict";
/* The app, live: the views Run, Guess first (What happened after a run), Compare runs and Settings. The page asks /api/state
   about ten times a second (as the classic window's timer did) and draws what it gets; Guess first and Compare runs ask
   /api/guess and /api/runs when the state says that their content changed; the buttons and the settings POST to /api/*.
   The look is the approved mockup (design/app_ui/mockup.html), with the owner's three changes: the points in the file in the
   accent colour, as on the slides; the answer after the run in Guess first; a frames-lost bar for each run in Compare runs. */
const POLL_MS = 100;
const SIMULATOR_ONLY = "Only the simulator knows";
const COMPARED = 2;  // Compare runs shows where the time went for the newest two runs; the table has every run kept
const $ = (sel, root = document) => root.querySelector(sel);

let M = null;   // /api/meta: the names, colours and texts that do not change
let S = null;   // the newest state
let picture = null;  // the state whose plots and caption are on show: S, or the one held by "Hold the picture"
const UI = {view: "run", held: false, offline: false, refused: null};  // refused: {key, text, message} of a setting not taken
const DATA = {guess: {sig: null, value: null, busy: false}, runs: {sig: null, value: null, busy: false}};  // the other views' data

/* ---------- small helpers ---------- */
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const U = () => Math.min(innerWidth / 1366, innerHeight / 768);
const NB = "\u202F";  // a narrow space between a number and its unit: a full space of the mono font looks like two
const pct = x => x === null ? "–" : `${Math.round(x)}${NB}%`;
const ms = x => x === null ? "–" : `${Math.round(x)}${NB}ms`;
const narrow = text => String(text).replace(/(\d) (%|ms)/g, `$1${NB}$2`);  // the same in texts made by the server ("36 %")
const bigNum = text => { const [n, unit] = text.split(/\s/); return `${n}${unit ? `<small>${unit}</small>` : ""}`; };
const verInfo = key => M.versions.find(v => v.key === key);
/* Set the HTML of a host only when it differs from what is there: a button is never replaced between mouse down and up.
   An element with an id that had the focus has it again after the change (a settings box, after Tab). */
function setHTML(el, html) {
  if (el._html === html) return false;
  const a = document.activeElement, focused = a && a !== el && el.contains(a) && a.id ? a.id : null;
  el.innerHTML = html;
  el._html = html;
  if (focused) document.getElementById(focused)?.focus();
  return true;
}
/* Draw everything again from the server's state, for example after a refused change (a radio button that moved goes back). */
function forgetHTML() {
  for (const el of document.querySelectorAll("#view, #view *, #rail")) el._html = null;
}

const ICON = {
  run: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M7 5.5v13l11-6.5z"/></svg>',
  guess: '<svg aria-hidden="true" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M9.6 9.6a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1.1.9-1.1 1.6"/><path d="M12 16.6v.1"/></svg>',
  answer: '<svg aria-hidden="true" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M8.2 12.4l2.7 2.7 5-5.6"/></svg>',
  compare: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M6 19V11M12 19V5M18 19v-9"/></svg>',
  settings: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M5 7h8M17 7h2M5 17h2M11 17h8M5 12h4M13 12h6"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/><circle cx="11" cy="12" r="2"/></svg>',
  info: '<svg aria-hidden="true" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6.5"/><path d="M8 7.2v4M8 4.8v.1"/></svg>',
  play: '<svg aria-hidden="true" viewBox="0 0 16 16"><path d="M4 2.5v11l9.5-5.5z"/></svg>',
  bad: '<svg aria-hidden="true" viewBox="0 0 16 16" class="bad"><path d="M4 4l8 8M12 4l-8 8"/></svg>',
  ok: '<svg aria-hidden="true" viewBox="0 0 16 16" class="ok"><path d="M3 8.5l3.2 3.2L13 4.8"/></svg>',
};

/* ---------- the key (colour swatches and frame marks) ---------- */
/* Frames, as in the approved mockup: a frame in the file (once or twice) = a small dot, a lost frame = a hollow ring in the accent
   colour, the one mark that stands out; a frame not known yet = a faint dot (the newest few, no key). */
function frameMark(code, size = 14) {
  const c = size / 2, F = M.fate_codes;
  const m = code === F.lost ? `<circle cx="${c}" cy="${c}" r="4" fill="none" stroke="var(--accent)" stroke-width="1.8"/>`
                            : `<circle cx="${c}" cy="${c}" r="2.6" fill="var(--primary)"/>`;
  const px = size * U();
  return `<svg width="${px}" height="${px}" viewBox="0 0 ${size} ${size}" aria-hidden="true">${m}</svg>`;
}
function keyHTML(v, redraw) {
  const kinds = redraw && v.threaded ? v.kinds_with_redraw : v.kinds, F = M.fate_codes;
  const waits = kinds.map(k => `<span class="kg"><i class="swatch" style="background:${M.kinds[k]}"></i>${esc(k)}</span>`).join("");
  const frames = [F.once, F.lost].filter(f => v.fates.includes(f)).map(f => `<span class="kg">${frameMark(f)}${esc(M.frame_legend[f])}</span>`).join("");
  const tick = `<span class="kg"><i class="swatch" style="width:calc(2*var(--u));height:calc(12*var(--u));background:var(--primary);border-radius:0"></i>a row in the file</span>`;
  return `<div class="key"><span class="kh">Waits</span>${waits}</div>
          <div class="key"><span class="kh">Frames</span>${frames}${tick}</div>`;
}

/* ---------- charts (inline SVG, drawn at the size of their box) ---------- */
/* The label step of a time axis: the smallest round step that leaves at most 6 gaps (7 labels), and fewer when the plot is narrow. */
const TIME_STEPS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120];
function timeStep(lo, hi, width, u) {
  const gaps = Math.max(2, Math.min(6, Math.floor(width / (80 * u))));
  return TIME_STEPS.find(step => (hi - lo) / step <= gaps) ?? TIME_STEPS[TIME_STEPS.length - 1];
}
function niceTicks(lo, hi, step) {
  const out = [];
  for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + 1e-9; t += step) out.push(+t.toFixed(6));
  return out;
}
/* The band under the force plot: the share of the frames lost in each part of the window, in the accent colour; full height = all
   of them. A run that loses frames shows it here, at the same times as the force. */
function lostBand(f, X, top, u, iw, x0) {
  const labelH = 15 * u, h = 30 * u, base = top + labelH + h, parts = f.lost, step = (f.window[1] - f.window[0]) / parts.length;
  const known = parts.filter(p => p !== null), none = known.length && known.every(p => p === 0);
  let g = `<text class="bandlab" x="${x0}" y="${top + 11 * u}">Frames lost in each ${+step.toFixed(2)} s${none ? ": none" : ""}</text>
    <line x1="${x0}" x2="${x0 + iw}" y1="${top + labelH}" y2="${top + labelH}" stroke="var(--line)" stroke-dasharray="${2 * u} ${3 * u}" stroke-width="${u}"/>
    <line x1="${x0}" x2="${x0 + iw}" y1="${base}" y2="${base}" stroke="var(--line)" stroke-width="${u}"/>`;
  parts.forEach((p, i) => {
    if (!p) return;
    const x = X(f.window[0] + i * step), w = Math.max(X(f.window[0] + (i + 1) * step) - x - 1.5 * u, u);
    g += `<rect x="${x.toFixed(1)}" y="${(base - p / 100 * h).toFixed(1)}" width="${w.toFixed(1)}" height="${(p / 100 * h).toFixed(1)}" fill="var(--accent)"><title>${Math.round(p)} % lost</title></rect>`;
  });
  return g;
}
const FEW_POINTS = 100, DOT_FEW = 9, DOT_MANY = 3.4;  // dot diameters (in u): a file with few points in the window shows each one large
function drawForce(host, s) {
  const W = host.clientWidth, H = host.clientHeight, u = U(), band = 54 * u;
  const m = {l: 30 * u, r: 10 * u, t: 8 * u, b: 24 * u}, iw = W - m.l - m.r, ih = H - m.t - m.b - band;
  const f = s.force, [lo, hi] = f.window, peak = f.peak_n;
  const X = t => m.l + (t - lo) / (hi - lo) * iw, Y = v => m.t + ih - v / peak * ih;
  let g = `<clipPath id="plotarea"><rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih + 5 * u}"/></clipPath>`;
  for (const v of niceTicks(0, peak, 2)) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)" stroke-width="${v === 0 ? 1.2 * u : u}"/><text x="${m.l - 8 * u}" y="${Y(v) + 4 * u}" text-anchor="end">${v}</text>`;
  for (const t of niceTicks(lo, hi, timeStep(lo, hi, iw, u))) g += `<line x1="${X(t)}" x2="${X(t)}" y1="${m.t + ih}" y2="${m.t + ih + 4 * u}" stroke="var(--stone)" stroke-width="${u}"/><text x="${X(t)}" y="${H - 6 * u}" text-anchor="${X(t) > W - m.r - 18 * u ? "end" : "middle"}">${t}${NB}s</text>`;
  g += lostBand(f, X, m.t + ih + 6 * u, u, iw, m.l);
  if (s.phase === "before") {
    g += `<text class="msgtxt" x="${m.l + iw / 2}" y="${m.t + ih / 2}" text-anchor="middle">The force appears here when the run starts.</text>`;
  } else {
    const line = f.truth.map(([t, v], i) => `${i ? "L" : "M"}${X(t).toFixed(1)} ${Y(v).toFixed(1)}`).join("");
    const dots = f.points.map(([t, v]) => `M${X(t).toFixed(1)} ${Y(v).toFixed(1)}h0`).join("");  // a zero-length stroke with round ends is a dot
    g += `<g clip-path="url(#plotarea)"><path d="${line}" fill="none" stroke="var(--stone)" stroke-opacity=".4" stroke-width="${7 * u}" stroke-linejoin="round" stroke-linecap="round"/>
      <path d="${dots}" fill="none" stroke="var(--accent)" stroke-width="${(f.points.length < FEW_POINTS ? DOT_FEW : DOT_MANY) * u}" stroke-linecap="round"/></g>`;
  }
  setHTML(host, `<svg width="${W}" height="${H}" role="img" aria-label="Force in newtons against time: the true force and the points in the file">${g}</svg>`);
}
function drawTimeline(host, s) {
  const v = verInfo(s.version), tl = s.timeline, F = M.fate_codes;
  const W = host.clientWidth, u = U();
  const labelW = 122 * u, rStrip = 36 * u, rRow = 26 * u, axisH = 22 * u, right = 10 * u;
  const rows = v.threads, H = rStrip + rows.length * rRow + axisH;
  host.style.height = `${H}px`;
  const [lo, hi] = tl.window, iw = W - labelW - right;
  const X = t => labelW + (t - lo) / (hi - lo) * iw;
  let g = "";
  for (const t of niceTicks(lo, hi, timeStep(lo, hi, iw, u)))
    g += `<line x1="${X(t)}" x2="${X(t)}" y1="0" y2="${H - axisH}" stroke="var(--line)" stroke-opacity=".6" stroke-width="${u}"/><text x="${X(t)}" y="${H - 6 * u}" text-anchor="${X(t) > W - right - 18 * u ? "end" : "middle"}">${t} s</text>`;
  const rowTop = i => i === 0 ? 0 : rStrip + (i - 1) * rRow, rowH = i => i === 0 ? rStrip : rRow;
  const names = [{label: "Sensor frames", sub: "simulator only"}].concat(rows.map(r => ({label: r.label})));
  names.forEach((n, i) => {
    const top = rowTop(i), mid = top + rowH(i) / 2;
    g += `<line x1="${labelW}" x2="${W - right}" y1="${top + rowH(i)}" y2="${top + rowH(i)}" stroke="var(--line)" stroke-width="${u}"/>`;
    g += n.sub ? `<text class="lab" x="${labelW - 12 * u}" y="${mid - 1 * u}" text-anchor="end">${esc(n.label)}</text><text class="sub" x="${labelW - 12 * u}" y="${mid + 13 * u}" text-anchor="end">${n.sub}</text>`
               : `<text class="lab" x="${labelW - 12 * u}" y="${mid + 5 * u}" text-anchor="end">${esc(n.label)}</text>`;
  });
  if (s.phase === "before") {
    g += `<text class="msgtxt" x="${labelW + iw / 2}" y="${rStrip + rows.length * rRow / 2 + 5 * u}" text-anchor="middle">The waits of each thread appear here when the run starts.</text>`;
  } else {
    rows.forEach((r, i) => {
      const top = rowTop(i + 1), h = rRow * .62, y = top + (rRow - h) / 2;
      for (const [kind, start, width] of tl.rows[r.key])
        g += `<rect x="${X(start).toFixed(1)}" y="${y}" width="${Math.max(width / (hi - lo) * iw, 1.2 * u).toFixed(1)}" height="${h}" fill="${M.kinds[kind]}" stroke="var(--paper)" stroke-width="${.8 * u}"/>`;
    });
    const last = rows.length, ty = rowTop(last) + rRow * .12, th = rRow * .76;
    g += `<path d="${tl.ticks.map(t => `M${X(t).toFixed(1)} ${ty}v${th}`).join("")}" fill="none" stroke="var(--primary)" stroke-width="${1.3 * u}"/>`;
    const cy = rStrip / 2, tt = tl.frames.t, ff = tl.frames.fate;
    const gap = tt.length > 1 ? (tt[tt.length - 1] - tt[0]) / (tt.length - 1) / (hi - lo) * iw : 20 * u;
    const rad = Math.max(Math.min(.36 * gap, 4 * u), 1.4 * u);
    tt.forEach((t, i) => {
      const x = X(t).toFixed(1), f = ff[i];
      if (f === F.lost) g += `<circle cx="${x}" cy="${cy}" r="${rad * 1.15}" fill="none" stroke="var(--accent)" stroke-width="${1.8 * u}"/>`;
      else if (f === F.not_known) g += `<circle cx="${x}" cy="${cy}" r="${rad * .5}" fill="var(--line)"/>`;
      else g += `<circle cx="${x}" cy="${cy}" r="${rad * .5}" fill="var(--primary)"/>`;
    });
  }
  setHTML(host, `<svg width="${W}" height="${H}" role="img" aria-label="Where the time goes: one row of waits for each thread, and one mark for each frame the sensor sent">${g}</svg>`);
}
function drawCharts(s) {
  const f = $("[data-chart=force]"), t = $("[data-chart=timeline]");
  if (!s.timeline || !s.force) return;  // the page could not build the state of a run that failed
  drawTimeline(t, s);  // first: it sets its own height, and the force plot takes what is left
  drawForce(f, s);
}

/* ---------- Run view: the parts ---------- */
function controlsHTML(s) {
  const v = verInfo(s.version), locked = s.locked, threaded = v.threaded, running = s.phase === "running";
  const list = M.versions.map(x => `<label class="ver"><input type="radio" name="ver" value="${x.key}" ${x.key === s.version ? "checked" : ""} ${locked ? "disabled" : ""}>
      <span class="vb">${x.key}</span><span class="vn">${esc(x.name)}</span></label>`).join("");
  const off = !threaded || locked;
  return `
  <div class="group"><span class="label" id="vlabel">Version</span>
    <p class="first"><b>Start with V0</b>, then go down: each version fixes the one above it.</p>
    <div class="versions" role="radiogroup" aria-labelledby="vlabel">${list}</div></div>
  <div class="group">
    <label class="switch ${threaded ? "" : "off"}"><input type="checkbox" role="switch" data-act="redraw" ${s.redraw && threaded ? "checked" : ""} ${off ? "disabled" : ""}>
      <span class="track"></span>
      <span class="words"><span>Redraw inside the logging loop</span><span class="hint">${threaded ? "Chosen at Start." : "Only V2–V4."}</span></span></label>
  </div>
  <div class="group">
    <div class="buttons">
      <button class="btn primary" type="button" data-act="start" ${locked ? 'aria-disabled="true"' : ""}>${ICON.play}Start</button>
      <button class="btn" type="button" data-act="shortrun" ${locked ? 'aria-disabled="true"' : ""}>${ICON.play}Short run</button>
      <button class="btn" type="button" data-act="stop" ${running ? "" : 'aria-disabled="true"'}>Stop</button>
      <button class="btn" type="button" data-act="hold">${UI.held ? "Let it run" : "Hold the picture"}</button>
    </div>
  </div>`;
}
function statusHTML(s) {
  const offline = UI.offline ? `<div class="offline" role="alert">The app does not answer. Is it still running in the terminal?</div>` : "";
  if (s.phase === "before")  // "Start: ...\nShort run: ...": the name of each button in bold
    return offline + `<p class="plan">${s.status.split("\n").map(line => esc(line).replace(/^(.*?):/, "<b>$1</b>:")).join("<br>")}</p>`;
  const live = s.phase === "running";
  const fault = s.error ? `<div class="fault"><span>Something went wrong inside the program, so the run was stopped.</span><span class="kind">(${esc(s.error.type)})</span></div>` : "";
  return offline + `<div class="line"><span><span class="dot ${live ? "live" : ""}"></span>${esc(s.status)}</span></div>
      <div class="progress" aria-hidden="true"><i style="width:${Math.min(100, s.now_s / s.total_s * 100).toFixed(0)}%"></i></div>
      <span class="label num">${s.now_s.toFixed(0)}${NB}s of ${s.total_s.toFixed(0)}${NB}s${s.short ? " · Short run" : ""}</span>${fault}
      ${s.phase === "after" ? `<button class="btn see" type="button" data-view="guess">${ICON.answer}See what happened</button>` : ""}`;
}
const noteHTML = s => s.not_default ? `<div class="note">${ICON.info}<span>${esc(M.texts.not_default_message)}
    <button class="btn quiet inline" type="button" data-act="defaults" ${s.locked ? 'aria-disabled="true"' : ""}>Back to defaults</button></span></div>` : "";
const bannerHTML = () => UI.held ? `<div class="banner" role="status"><span class="pill">Picture held</span><span>${esc(M.texts.hold_note)}</span></div>` : "";

function heroHTML(s) {
  const c = s.counters, v = verInfo(s.version);
  let sub;
  if (!c) sub = s.phase === "before" ? "Press Start to see the numbers of a run." : "No numbers: the run stopped before they were made.";
  else if (!v.numbered) sub = "Your file cannot show this: it has no frame numbers.";
  else if (c.lost_seen_pct !== null && Math.round(c.lost_seen_pct) !== Math.round(c.lost_pct ?? 0)) sub = `Your file shows ${pct(c.lost_seen_pct)}: it has frame numbers.`;
  else sub = "Your file shows the same loss: it has frame numbers.";
  const lost = c && c.lost_pct !== null;
  return `<div class="hero"><div class="top"><span class="label">Frames lost</span><span class="tag">${SIMULATOR_ONLY}</span></div>
      <span class="big num ${lost && c.lost_pct > 0 ? "lost" : ""}">${lost ? bigNum(pct(c.lost_pct)) : "–"}</span>
      <span class="sub">${sub}</span></div>`;
}
function countsHTML(s) {
  const c = s.counters, v = verInfo(s.version);
  if (!c) return ["File", "Frames", "Gaps"].map(n => `<div><span class="label">${n}</span><dl><dt class="dim">${n === "File" ? "Rows in the file" : n === "Frames" ? "Frames the sensor sent" : "Longest gap between rows"}</dt><dd class="dim">–</dd></dl></div>`).join("");
  const seen = c.lost_seen_pct === null ? (v.numbered ? "–" : null) : pct(c.lost_seen_pct);
  const group = (name, items) => `<div><span class="label">${name}</span><dl>${items.map(([k, val, cls]) =>
    `<dt>${k}</dt><dd class="${cls || ""}">${val}</dd>`).join("")}</dl></div>`;
  const gaps = [["Longest gap between rows", ms(c.longest_gap_ms)]];
  if (c.gap_1_in_100_ms !== null) gaps.push(["1 gap in 100 is longer than", ms(c.gap_1_in_100_ms)]);
  gaps.push([`Gaps longer than ${M.texts.gap_limit_ms} ms`, c.gaps_over_limit]);
  return group("File", [["Rows in the file", c.rows], ["Sensor values equal to the one before", pct(c.repeat_pct)]]) +
    group("Frames", [["Frames the sensor sent", c.frames_sent],
      ["Frames lost, as your file shows it", seen === null ? "cannot tell" : seen, seen === null ? "na" : ""]]) +
    group("Gaps", gaps);
}
function captionHTML(p) {
  let cap = `<p class="dim">The cause of what happens in the run is told here, one sentence at a time.</p>`;
  if (p.caption) {
    const m = /^(.*?)\s*\((\d+) times so far\)$/.exec(p.caption);
    cap = `<p>${esc(m ? m[1] : p.caption)}</p>${m ? `<span class="pill times">${m[2]} times so far</span>` : ""}`;
  }
  return `<h2 class="title">${esc(M.texts.caption_title)}</h2>${cap}`;
}

const LOOK_SHOWN = 2;  // sentences of "What to look for" on show; the rest is under "More"
function lookHTML(s) {
  const sentences = verInfo(s.version).what_it_shows.split(/(?<=\.)\s+(?=[A-Z])/);
  const rest = sentences.slice(LOOK_SHOWN).join(" ");
  return `<section class="look"><h2 class="title">What to look for in ${s.version}</h2><p>${esc(sentences.slice(0, LOOK_SHOWN).join(" "))}</p>
    ${rest ? `<details><summary>More</summary><p>${esc(rest)}</p></details>` : ""}</section>`;
}

const RUN_SKELETON = `<div class="run">
  <div class="pane controls"><div class="host" id="ctl"></div><div class="group status" id="status" aria-live="polite"></div><div class="host" id="note"></div></div>
  <div class="pane plots"><div class="host" id="banner"></div>
    <section class="card force" aria-label="Force">
      <div class="cardhead"><h2 class="title">Force (N)</h2>
        <div class="legend">
          <span><svg id="legend-true" aria-hidden="true"></svg>True force</span>
          <span><svg id="legend-points" aria-hidden="true"></svg>Points in the file</span></div></div>
      <div class="chart" data-chart="force"></div>
    </section>
    <section class="card timeline" aria-label="Where the time goes">
      <div class="cardhead"><h2 class="title" id="bars-title"></h2><span class="label" id="bars-window"></span></div>
      <div class="chart" data-chart="timeline"></div>
      <div class="host" id="key"></div>
    </section></div>
  <div class="pane inspector"><div class="host" id="hero"></div><section class="caption" id="caption" aria-live="polite"></section><div class="counts" id="counts"></div><div class="host" id="look"></div></div>
</div>`;

function renderRun(s) {
  const view = $("#view");
  if (!$(".run", view)) {
    view.innerHTML = RUN_SKELETON;
    view._html = null;  // a page view set by setHTML is not on show any more
    $("#bars-title").textContent = M.texts.bars_title;
  }
  const u = U();
  setHTML($("#legend-true"), `<line x1="${2 * u}" x2="${20 * u}" y1="${4 * u}" y2="${4 * u}" stroke="var(--stone)" stroke-opacity=".4" stroke-width="${7 * u}" stroke-linecap="round"/>`);
  $("#legend-true").setAttribute("width", 22 * u), $("#legend-true").setAttribute("height", 8 * u);
  setHTML($("#legend-points"), `<circle cx="${5 * u}" cy="${5 * u}" r="${DOT_FEW / 2 * u}" fill="var(--accent)"/>`);
  $("#legend-points").setAttribute("width", 10 * u), $("#legend-points").setAttribute("height", 10 * u);
  setHTML($("#ctl"), controlsHTML(s));
  setHTML($("#status"), statusHTML(s));
  setHTML($("#note"), noteHTML(s));
  setHTML($("#banner"), bannerHTML());
  setHTML($("#key"), keyHTML(verInfo(s.version), s.redraw_in_loop));
  setHTML($("#hero"), heroHTML(s));
  setHTML($("#counts"), countsHTML(s));
  setHTML($("#caption"), captionHTML(picture));
  setHTML($("#look"), lookHTML(s));
  $("#bars-window").textContent = picture.phase === "before" ? "" : `the last ${verInfo(picture.version).window_s} s`;
  drawCharts(picture);
}
/* ---------- the data of the other views: asked for when its signature changes ---------- */
function want(name, sig) {
  const d = DATA[name];
  if (d.sig !== sig && !d.busy) {
    d.busy = true;
    fetch(`/api/${name}`, {cache: "no-store"}).then(r => r.json()).then(v => { d.value = v; d.sig = sig; })
      .catch(() => { UI.offline = true; }).finally(() => { d.busy = false; render(); });
  }
  return d.value;
}
const guessSig = s => [s.run_id, s.phase, s.version, JSON.stringify(s.values)].join("|");
const offlineHTML = () => UI.offline ? `<div class="offline" role="alert">The app does not answer. Is it still running in the terminal?</div>` : "";

/* ---------- Guess first / What happened ---------- */
function versionSeg(s) {
  return `<div class="seg" role="radiogroup" aria-label="Version">${M.versions.map(x =>
    `<label><input type="radio" name="ver" value="${x.key}" ${x.key === s.version ? "checked" : ""} ${s.locked ? "disabled" : ""}><span>${x.key}</span></label>`).join("")}</div>`;
}
function startButtons(s) {
  const off = s.locked ? 'aria-disabled="true"' : "";
  return `<button class="btn primary" type="button" data-act="start" ${off}>${ICON.play}Start ${s.version}</button>
          <button class="btn" type="button" data-act="shortrun" ${off}>${ICON.play}Short run</button>`;
}
function viewGuess(s, g) {
  if (g.answer) return viewAnswer(s, g);
  const li = g.question.choices.map(([letter, text]) => `<li><span class="ltr">${letter}</span><span>${esc(text)}</span></li>`).join("");
  const under = `<p class="instr">${esc(g.line)}</p>` + (g.phase === "running" ? "" : `<div class="acts">${startButtons(s)}</div>`);
  return `<div class="page guess">${offlineHTML()}
    <div class="pagehead"><div><h1>Guess first</h1><p class="sub">${g.version} · ${esc(verInfo(g.version).name)}</p></div>${versionSeg(s)}</div>
    <div class="cols">
      <div><p class="q">${esc(g.question.ask)}</p><ol class="choices" aria-label="Choices">${li}</ol>${under}</div>
      <aside class="side"><h2 class="title">What the timeline shows</h2><p>${esc(g.what_it_shows)}</p>
        <p class="label">Start: the whole run, about <span class="num">${Math.round(g.run_s.full)}</span>${NB}s. Short run: 1 cycle, about <span class="num">${Math.round(g.run_s.short)}</span>${NB}s.</p></aside>
    </div></div>`;
}
/* The one large number of What happened: in "This run: <result>. <why>", the first share (65 %), else the first time (58 ms),
   else the first number (30 rows) of the result. */
function keyResult(result) {
  const m = /\d+ %/.exec(result) || /\d+ ms/.exec(result) || /\d+/.exec(result);
  if (!m) return `<p class="result">${esc(result)}</p>`;
  const before = result.slice(0, m.index).trim(), after = result.slice(m.index + m[0].length).trim();
  return `<p class="result">${before ? `<span>${esc(before)}</span>` : ""}<span class="big xl num">${bigNum(m[0])}</span>${after ? `<span>${esc(after)}</span>` : ""}</p>`;
}
/* The frames of the run in one bar: those in the file, and the lost ones in the accent colour (the run view's hero number). */
function lossBar(lost) {
  if (lost === null || lost === undefined) return "";
  const w = Math.min(Math.max(lost, 0), 100);
  return `<div class="lossbar"><div class="bar" role="img" aria-label="Frames lost: ${pct(lost)}"><i class="kept" style="width:${100 - w}%"></i>${w > 0 ? `<i class="lostpart" style="width:${w}%"></i>` : ""}</div>
    <p class="label"><span class="${w > 0 ? "lostword" : ""}">${pct(lost)} of the frames lost</span> · ${SIMULATOR_ONLY.toLowerCase()}</p></div>`;
}
function viewAnswer(s, g) {
  const a = g.answer, c = a.checks;
  const m = /^This run: (.*?)\.\s+(.*)$/.exec(a.answer[0]);  // "This run: 65 % of the frames reached the file. Each ..."
  const head = m ? `${keyResult(m[1])}${lossBar(a.lost_pct)}<p>${esc(m[2])}</p>` : `${lossBar(a.lost_pct)}<p>${esc(a.answer[0])}</p>`;
  const open = a.answer.length > 1 ? `<div class="open"><h2 class="title">${esc(a.answer[1])}</h2><p>${esc(a.answer[2])}</p></div>` : "";
  const causes = a.what_happened.map(t => {
    const mm = /^(.*?)(?:\s+It happened (\d+) times in this run\.)?$/.exec(t);
    return `<li><span>${esc(mm[1])}</span>${mm[2] ? `<span class="pill">${mm[2]} times in this run</span>` : ""}</li>`;
  }).join("") || `<li><span class="dim">Nothing that the run reports on.</span></li>`;
  const checks = c.sentences.map(t => `<li>${c.failed ? ICON.bad : ICON.ok}<span>${esc(t)}</span></li>`).join("");
  return `<div class="page guess">${offlineHTML()}
    <div class="pagehead"><div><h1>What happened</h1><p class="sub">${a.version} · ${esc(a.settings)}</p></div>
      <div class="acts">${startButtons(s)}</div></div>
    <div class="cols">
      <div class="answer"><div><p class="label">The answer</p><h2 class="verdict"><span class="ltr">${a.correct}</span>${esc(a.choice)}</h2></div>${head}${open}</div>
      <aside class="side">
        <section><div class="sechead"><h2 class="title">What happened in this run</h2></div><ul class="causes">${causes}</ul></section>
        <section><div class="sechead"><h2 class="title">Checks on the file</h2><span class="pill">${c.failed} of ${c.count} fail</span></div>
          <ul class="checks">${checks}</ul></section>
      </aside></div></div>`;
}

/* ---------- Compare runs ---------- */
const DARK_KINDS = ["writing to disk", "redrawing the plot", "moving the stage"];  // a percent on these is written in the paper colour
function barsHTML(run) {
  const order = Object.keys(M.kinds).concat([M.working]);
  const rows = verInfo(run.version).threads.map(t => {
    const share = run.shares[t.key];
    const parts = order.filter(k => share[k] > 0.0005).map(k => {
      const w = share[k] * 100, empty = k === M.working;
      const label = share[k] >= M.share_label_min ? pct(w) : "";
      return `<i class="${empty ? "empty-part" : ""}" style="width:${w}%;background:${empty ? "var(--paper)" : M.kinds[k]};color:${DARK_KINDS.includes(k) ? "var(--paper)" : "var(--ink)"}" title="${esc(k)}: ${pct(w)}">${label}</i>`;
    }).join("");
    return `<span class="bl">${esc(t.label)}</span><div class="bar" role="img" aria-label="${esc(t.label)}">${parts}</div>`;
  }).join("");
  const lost = run.lost_pct, text = pct(lost), wide = lost / 100 >= M.share_label_min;
  const lostBar = `<span class="bl lostlabel">frames lost</span><div class="bar lostbar" role="img" aria-label="Frames lost: ${text}">
      ${lost > 0 ? `<i class="lostpart" style="width:${lost}%">${wide ? text : ""}</i>` : ""}<i class="empty-part rest">${wide ? "" : text}</i></div>`;
  return `<div class="bars">${rows}${lostBar}<div class="axis"><span>0${NB}%</span><span>50${NB}%</span><span>100${NB}%</span></div></div>`;
}
function viewCompare(r) {
  if (!r || !r.runs.length) return `<div class="page">${offlineHTML()}<div class="pagehead" style="min-height:0"><div><h1>Compare runs</h1></div></div>
    <div class="empty" style="flex:1"><p>No run has finished yet.<br>Each finished run is added here: where its time went, and its numbers.</p></div></div>`;
  const shown = r.runs.slice(0, COMPARED);
  const panels = shown.map((run, i) => `<section><div class="runhead"><span class="vchip">${run.version}</span><span class="dim">${esc(run.settings)}</span>${i === 0 ? '<span class="pill">newest</span>' : ""}</div>
      ${barsHTML(run)}</section>`).join("");
  const used = k => shown.some(run => verInfo(run.version).threads.some(t => (run.shares[t.key][k] || 0) >= M.share_key_min));
  const swatch = k => k === M.working ? "background:var(--paper);border:1px solid var(--line)" : `background:${M.kinds[k]}`;  // as in the bar
  const key = `<div class="key" style="margin-top:var(--sp4)">${Object.keys(M.kinds).concat([M.working]).filter(used).map(k => `<span class="kg"><i class="swatch" style="${swatch(k)}"></i>${esc(k)}</span>`).join("")}
    <span class="kg"><i class="swatch" style="background:var(--accent)"></i>frames lost (${SIMULATOR_ONLY.toLowerCase()})</span></div>`;
  const body = r.table.map((row, i) => `<tr class="${i === 0 ? "new" : ""}"><td><span class="vchip">${row[0]}</span></td><td>${esc(row[1])}</td>
    ${row.slice(2).map((cell, j) => `<td class="r ${j === 1 && parseFloat(cell) > 0 ? "lost" : ""}">${narrow(cell)}</td>`).join("")}</tr>`).join("");
  const head = M.runs_header.map((h, i) => `<th class="${i > 1 ? "r" : ""}">${esc(h)}</th>`).join("");
  const note = M.texts.bars_note[0].toUpperCase() + M.texts.bars_note.slice(1);
  return `<div class="page compare">${offlineHTML()}
    <div class="pagehead" style="min-height:0"><div><h1>Compare runs</h1></div></div>
    <section><div class="sechead" style="margin-bottom:var(--sp1)"><h2 class="title">${esc(M.texts.bars_title)}</h2></div>
      <p class="dim" style="margin-bottom:var(--sp4)">${esc(note)}${r.runs.length > COMPARED ? `, for the newest ${COMPARED} runs` : ""}.</p>
      <div class="cmp">${panels}</div>${key}</section>
    <section class="runs"><div class="sechead"><h2 class="title">${esc(M.texts.table_title)}</h2></div>
      <div class="tablebox"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>
      <p class="tnote">${esc(M.texts.table_note)}</p></section></div>`;
}

/* ---------- Settings ---------- */
const SWITCH_HINT = {busy: "Other programs use every processor core.", slow_disk: "The file writer stalls more often and longer.", sensor_time: "V3 and V4 only."};
const fmtNum = x => String(+x.toPrecision(6));
const controls = () => Object.fromEntries(M.settings.framework.concat(M.settings.simulated).map(c => [c.key, c]));
const changedDot = (s, c) => s.values[c.key] !== c.default ? '<i class="chg" title="Not the default"></i>' : "";
function numberRow(s, c) {
  const bad = UI.refused && UI.refused.key === c.key ? UI.refused : null;
  const off = s.locked && c.key !== "window" ? "disabled" : "";  // the settings are read at Start; the plot window works at once
  return `<div class="setrow"><label class="lab" for="in-${c.key}"><span>${esc(c.name)}${changedDot(s, c)}</span>
      <small class="num">default ${fmtNum(c.default)}</small></label>
    <div class="field ${bad ? "bad" : ""} ${off ? "off" : ""}"><input id="in-${c.key}" data-key="${c.key}" inputmode="decimal" autocomplete="off" value="${bad ? esc(bad.text) : fmtNum(s.values[c.key])}" ${bad ? 'aria-invalid="true" aria-describedby="setmsg"' : ""} ${off}>${c.unit ? `<span class="unit">${esc(c.unit)}</span>` : ""}</div>
    <span class="rng">${fmtNum(c.lo)}–${fmtNum(c.hi)}</span></div>`;
}
function switchHTML(s, c, hint) {
  return `<label class="switch ${s.locked ? "off" : ""}" style="align-items:center"><input type="checkbox" role="switch" id="in-${c.key}" data-key="${c.key}" ${s.values[c.key] ? "checked" : ""} ${s.locked ? "disabled" : ""}><span class="track"></span>
      <span class="words"><span>${esc(c.label)}${changedDot(s, c)}</span>${hint ? `<span class="hint">${esc(hint)}</span>` : ""}</span></label>`;
}
/* Log to: the framework's two boxes (CSV file, Database file) as one choice of the two. */
const LOG_TO = ["csv", "database"];
function logToHTML(s, C) {
  return `<div class="seg words" role="radiogroup" aria-labelledby="logto">${LOG_TO.map(k =>
    `<label><input type="radio" name="logto" value="${k}" ${s.values[k] ? "checked" : ""} ${s.locked ? "disabled" : ""}><span>${esc(C[k].label)}</span></label>`).join("")}</div>`;
}
async function setLogTo(key) {  // the chosen one on first, then the other off: the run never has nowhere to log
  await setValue(key, true, "");
  if (!UI.refused) for (const other of LOG_TO.filter(k => k !== key)) await setValue(other, false, "");
}
function viewSettings(s) {
  const C = controls(), off = s.locked ? 'aria-disabled="true"' : "";
  const msg = UI.refused ? `${ICON.info}<span>${esc(UI.refused.message)}</span>`
    : s.not_default ? `<div class="note">${ICON.info}<span>${esc(M.texts.not_default_message)} A dot <i class="chg" style="margin:0 2px"></i> marks a setting that is not its default.</span></div>` : "";
  const sub = s.locked ? "A run is going: the settings are fixed until it ends. The plot window works at once."
                       : "Read at Start and fixed for the run. The plot window works at once.";
  return `<div class="page">${offlineHTML()}
    <div class="pagehead"><div><h1>Settings</h1><p class="sub">${sub}</p></div>
      <div class="acts"><button class="btn" type="button" data-act="defaults" ${off}>Back to defaults</button></div></div>
    <div class="msg" id="setmsg" role="status">${msg}</div>
    <div class="sets">
      <section class="panel"><div class="sechead" style="margin-bottom:var(--sp1)"><h2 class="title">Framework settings</h2></div>
        <p class="simnote">The framework's own controls, with its ranges and defaults.</p>
        ${["rate", "depth", "speed", "hold", "cycles"].map(k => numberRow(s, C[k])).join("")}
        <div class="setrow sw"><span class="lab" id="logto"><span>Log to${changedDot(s, C.csv) || changedDot(s, C.database)}</span></span>${logToHTML(s, C)}</div>
        ${numberRow(s, C.window)}
      </section>
      <section class="panel"><div class="sechead"><h2 class="title">Simulated rig (not in the framework)</h2></div>
        <p class="simnote">These exist only in the simulator. They never change what the framework does.</p>
        <div class="sim">${numberRow(s, C.fps)}${["busy", "slow_disk", "sensor_time"].map(k => `<div class="setrow sw">${switchHTML(s, C[k], SWITCH_HINT[k])}</div>`).join("")}</div>
      </section>
    </div></div>`;
}

/* ---------- rail and routing ---------- */
function renderRail(s) {
  const second = s.phase === "after" ? "What happened" : "Guess first";
  const items = [["run", "Run", ICON.run], ["guess", second, s.phase === "after" ? ICON.answer : ICON.guess],
                 ["compare", "Compare runs", ICON.compare], ["settings", "Settings", ICON.settings]];
  const state = {before: "Ready", running: "Running", after: "Finished", stopped: "Stopped", error: "Failed"}[s.phase];
  const flagged = k => k === "settings" && s.not_default;
  setHTML($("#rail"), items.map(([k, label, icon]) => `<button class="nav" type="button" data-view="${k}" ${flagged(k) ? 'aria-label="Settings (not the defaults)"' : ""} ${UI.view === k ? 'aria-current="page"' : ""}>${icon}<span>${label}</span>${flagged(k) ? '<i class="flag" title="Not the defaults"></i>' : ""}</button>`).join("") +
    (UI.view === "run" ? "" : `<div class="railstatus"><span class="vchip">${s.version}</span><span><span class="dot ${s.phase === "running" ? "live" : ""}"></span>${state}</span></div>`));
}
function render() {
  if (!S) return;
  renderRail(S);
  const view = $("#view");
  if (UI.view === "run") return renderRun(S);
  if (UI.view === "guess") {
    const g = want("guess", guessSig(S));
    return setHTML(view, g ? viewGuess(S, g) : `<div class="page"><div class="pagehead"><h1>Guess first</h1></div></div>`);
  }
  if (UI.view === "compare") return setHTML(view, viewCompare(want("runs", S.runs_done)));
  setHTML(view, viewSettings(S));
}

/* ---------- talking to the server ---------- */
function apply(state) {
  S = state;
  if (!UI.held || !picture) picture = state;
  render();
}
/* POST an action; the answer is the new state. Returns the server's words when it refuses, else null. */
async function post(path, body = {}) {
  try {
    const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const data = await r.json();
    if (r.ok) { apply(data); return null; }
    forgetHTML();  // refused: draw the controls again as the server has them
    render();
    return data.error;
  } catch (e) {
    UI.offline = true;
    render();
    return null;
  }
}
async function setValue(key, value, text) {
  const refused = await post("/api/settings", {key, value});
  UI.refused = refused ? {key, text, message: refused} : null;
  forgetHTML();
  render();
}
async function poll() {
  const t0 = performance.now();
  try {
    const r = await fetch("/api/state", {cache: "no-store"});
    UI.offline = false;
    apply(await r.json());
  } catch (e) {
    UI.offline = true;
    render();
  }
  setTimeout(poll, Math.max(POLL_MS - (performance.now() - t0), 0));
}

function startRun(short) {
  if (UI.view !== "run") { UI.view = "run"; render(); }  // the run is watched in the Run view; its force plot sets the size of the redraw
  const f = $("[data-chart=force]"), r = devicePixelRatio || 1;
  return post("/api/start", {plot_px: [Math.round(f.clientWidth * r), Math.round(f.clientHeight * r)], short});
}
const ACTIONS = {
  start: () => startRun(false),
  shortrun: () => startRun(true),
  stop: () => post("/api/stop"),
  defaults() { UI.refused = null; return post("/api/defaults"); },
  hold() { UI.held = !UI.held; if (!UI.held) picture = S; render(); },
};
document.addEventListener("click", e => {
  const nav = e.target.closest("[data-view]");
  if (nav) { UI.view = nav.dataset.view; return render(); }
  const act = e.target.closest("[data-act]");
  if (act && ACTIONS[act.dataset.act] && act.getAttribute("aria-disabled") !== "true") ACTIONS[act.dataset.act]();
});
document.addEventListener("change", e => {
  const t = e.target;
  if (t.name === "ver") post("/api/version", {version: t.value});
  else if (t.name === "logto") setLogTo(t.value);
  else if (t.dataset.act === "redraw") post("/api/redraw", {on: t.checked});
  else if (t.dataset.key) setValue(t.dataset.key, t.type === "checkbox" ? t.checked : t.value, t.value);
});
addEventListener("resize", render);

async function main() {
  M = await (await fetch("/api/meta")).json();
  await document.fonts.ready;
  poll();
}
main();
