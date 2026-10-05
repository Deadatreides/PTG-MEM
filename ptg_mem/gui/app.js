/* PTG-MEM GUI — plain JS, no build step, no network beyond 127.0.0.1. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const clip = (s, n) => { s = String(s ?? "").replace(/\s+/g, " ").trim(); return s.length > n ? s.slice(0, n) + "…" : s; };

// the token arrives in the URL fragment (never sent to the server) and lives in this tab only;
// the fragment may also carry a deep link: &tab=search&q=...
const deepLink = {};
(function takeToken() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("token")) { try { sessionStorage.setItem("ptg-token", h.get("token")); } catch (e) {} deepLink.token = h.get("token"); }
  if (h.get("tab")) deepLink.tab = h.get("tab");
  if (h.get("q")) deepLink.q = h.get("q");
  if (location.hash) history.replaceState(null, "", location.pathname);
})();
function token() { try { return sessionStorage.getItem("ptg-token") || deepLink.token || ""; } catch (e) { return deepLink.token || ""; } }
async function api(path, body) {
  const opt = { method: body ? "POST" : "GET", headers: { "X-PTG-Token": token(), "Content-Type": "application/json" } };
  if (body) opt.body = JSON.stringify(body);
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (r.status === 402) { showPro(j.error); throw new Error(j.error); }
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

const state = { root: null, status: null, feedSig: "", selected: null, graph: null };

function when(t) {
  const d = new Date(t * 1000), now = Date.now() / 1000, s = now - t;
  if (s < 60) return Math.max(1, Math.round(s)) + " s ago";
  if (s < 3600) return Math.round(s / 60) + " min ago";
  return d.toLocaleString();
}

/* ---------------- status / chips ---------------- */
async function refreshStatus() {
  let s;
  try { s = await api("/api/status"); } catch (e) {
    $("#chips").innerHTML = `<span class="chip bad">${esc(token() ? "daemon unreachable" : "open this page with: ptg gui")}</span>`;
    return;
  }
  state.status = s;
  const sel = $("#project");
  const roots = s.projects.map((p) => p.root);
  if (sel.options.length !== roots.length || roots.some((r, i) => sel.options[i]?.value !== r)) {
    sel.innerHTML = s.projects.map((p) => `<option value="${esc(p.root)}">${esc(p.name || p.root)}</option>`).join("");
    if (!state.root || !roots.includes(state.root)) state.root = roots[0] || null;
    sel.value = state.root || "";
  }
  const p = s.projects.find((x) => x.root === state.root);
  const chips = [];
  if (p) {
    const cls = p.state === "ready" ? "ok" : p.state === "error" ? "bad" : "warn";
    chips.push(`<span class="chip ${cls}">${esc(p.state)}${p.error ? ": " + esc(clip(p.error, 60)) : ""}</span>`);
    if (p.atoms != null) chips.push(`<span class="chip">${p.live.toLocaleString()} memories · ${p.files.toLocaleString()} files</span>`);
    if (p.catchup && p.catchup[1]) chips.push(`<span class="chip warn">indexing ${p.catchup[0]}/${p.catchup[1]}</span>`);
    if (p.embedder) chips.push(`<span class="chip ${p.embedder.loaded ? "ok" : ""}">embedder ${p.embedder.loaded ? "in memory" : "unloaded"}</span>`);
    if (p.watch) chips.push(`<span class="chip">watching (${esc(p.watch)})</span>`);
  }
  if (s.queue) chips.push(`<span class="chip warn">queue ${s.queue}</span>`);
  $("#chips").innerHTML = chips.join("");
  const lic = s.license;
  const pill = $("#lic");
  pill.textContent = lic.registered ? `Pro · ${lic.name || ""}` : "Free · personal use";
  pill.className = "pill" + (lic.registered ? " pro" : "");
  const nagHidden = (() => { try { return Date.now() - Number(localStorage.getItem("ptg-nag") || 0) < 7 * 864e5; } catch (e) { return false; } })();
  $("#nag").hidden = !lic.nag || nagHidden;
  $("#nagDays").textContent = lic.days_used;
  renderDaemon(s);
}

/* ---------------- live feed ---------------- */
function evHTML(e) {
  const k = e.kind;
  const head = (label, extra = "") => `<div class="head"><span class="kind k-${k}">${label}</span>${extra}<span class="when">${esc(when(e.t))}</span></div>`;
  if (k === "recall") {
    const items = (e.items || []).map((i) => `<div class="item" data-id="${esc(i.id)}"><span class="score">${(i.score ?? 0).toFixed(2)}</span><div><div class="meta">${esc(i.date)} · ${esc(i.file)}${i.replaces ? '<span class="badge b-revises">current version</span>' : ""}</div>${esc(clip(i.text, 220))}${i.replaces ? `<div class="was">↳ replaces the earlier version (${esc(i.replaces.date)}): <s>${esc(clip(i.replaces.text, 150))}</s></div>` : ""}</div></div>`).join("");
    return head("context attached", `<span class="meta">${e.items?.length ? e.items.length + " memories · " + e.chars + " chars" : "nothing relevant — nothing attached"}</span>`)
      + `<div class="prompt">“${esc(clip(e.prompt, 200))}”</div><div class="items">${items}</div>`;
  }
  if (k === "ingest") return head("memory updated", `<span class="meta">${esc(shortPath(e.file))}: +${e.added} new${e.retracted ? ", " + e.retracted + " retracted" : ""}${e.kept ? ", " + e.kept + " unchanged" : ""}</span>`);
  if (k === "retract") return head("retracted", `<span class="meta">${esc(shortPath(e.file))} — ${e.retracted} memories (${esc(e.reason || "")})</span>`);
  if (k === "remember") return head("remembered", "") + `<div class="item" data-id="${esc(e.id)}"><span class="score">note</span><div>${esc(clip(e.text, 220))}</div></div>`;
  if (k === "intake") return head("intake", `<span class="meta">exit ${e.code} in ${e.seconds}s</span>`);
  if (k === "save") return head("saved", `<span class="meta">${e.nodes} memories in ${e.seconds}s</span>`);
  if (k === "secrets") return head("secret removed", `<span class="meta">${e.removed} credential(s) cut from ${esc(shortPath(e.file))} before storing</span>`);
  if (k === "error") return head("error", `<span class="meta">${esc(shortPath(e.file))}: ${esc(clip(e.error, 160))}</span>`);
  return head(k, "");
}
function shortPath(p) {
  if (!p) return "";
  const r = state.root;
  if (r && p.startsWith(r)) return p.slice(r.length + 1);
  const parts = p.split(/[\\/]/); return parts.slice(-2).join("/");
}
async function refreshFeed() {
  if (!state.root) return;
  let f;
  try { f = await api("/api/feed?limit=80&root=" + encodeURIComponent(state.root)); } catch (e) { return; }
  const sig = f.items.map((e) => e.t + e.kind).join("|");
  if (sig === state.feedSig) return;
  state.feedSig = sig;
  const box = $("#feed");
  box.innerHTML = f.items.length ? f.items.map((e) => `<div class="ev">${evHTML(e)}</div>`).join("")
    : `<div class="empty-state">Nothing yet. Start a Claude Code or Codex session in this project — every prompt will show up here with the memory attached to it.</div>`;
}

/* ---------------- search & detail ---------------- */
async function doSearch(ev) {
  ev?.preventDefault();
  const q = $("#q").value.trim();
  if (!q) return;
  $("#results").innerHTML = `<div class="muted">searching…</div>`;
  try {
    const r = await api("/api/search", { root: state.root, query: q, k: 12, all_projects: $("#allProjects").checked });
    $("#results").innerHTML = r.items.length ? r.items.map((it) => `
      <div class="res" data-id="${esc(it.id)}" data-root="${esc(it.project || state.root)}">
        <div class="meta"><span class="score">${(it.score ?? 0).toFixed(3)}</span> · ${esc(it.date)} · ${esc(it.rel_file)}
        ${it.status !== "active" ? `<span class="badge b-${esc(it.status)}">${esc(it.status)}</span>` : ""}
        ${it.current_id ? '<span class="badge b-superseded">replaced</span>' : ""}</div>
        <div class="t">${esc(clip(it.text, 400))}</div></div>`).join("")
      : `<div class="empty-state">Nothing found.</div>`;
  } catch (e) { $("#results").innerHTML = `<div class="empty-state">${esc(e.message)}</div>`; }
}
async function showNode(id, root) {
  state.selected = { id, root: root || state.root };
  $$(".res").forEach((r) => r.classList.toggle("sel", r.dataset.id === id));
  const d = $("#detail");
  d.classList.remove("empty");
  d.innerHTML = `<div class="muted">loading…</div>`;
  let n;
  try { n = await api(`/api/node?id=${encodeURIComponent(id)}&root=${encodeURIComponent(state.selected.root)}`); }
  catch (e) { d.innerHTML = esc(e.message); return; }
  const REL = { supersedes: ["replaces", "replaced by"], revises: ["edit of", "edited into"], contradicts: ["contradicts", "contradicted by"], fixes: ["fixes", "fixed by"], refines: ["refines", "refined by"] };
  const rels = (n.relations || []).map((r) => `<div class="rel" data-id="${esc(r.id)}">${(r.types || [r.type]).map((t) => `<span class="badge b-${esc(t)}">${esc((REL[t] || [t, t])[r.dir === "out" ? 0 : 1])}</span>`).join(" ")} <span class="meta">${esc(r.date)} · ${esc(r.status)}</span><div>${esc(clip(r.text, 240))}</div></div>`).join("");
  const lin = (n.lineage || []).map((l) => `<li class="${l.id === id ? "cur" : ""}" data-id="${esc(l.id)}"><span class="meta">${esc(l.date)}</span> ${esc(clip(l.text, 110))}${l.status !== "active" ? ` <span class="badge b-${esc(l.status)}">${esc(l.status)}</span>` : ""}</li>`).join("");
  d.innerHTML = `
    <div class="meta">${esc(n.date)} · ${esc(n.rel_file)} ${n.status !== "active" ? `<span class="badge b-${esc(n.status)}">${esc(n.status)}</span>` : ""}</div>
    ${n.current_id ? `<div class="rel" data-id="${esc(n.current_id)}"><span class="badge b-superseded">replaced by a newer memory</span> — open it</div>` : ""}
    <pre>${esc(n.text)}</pre>
    ${rels ? `<h4>Relations</h4>${rels}` : ""}
    ${lin ? `<h4>Line of thought</h4><ul class="lineage">${lin}</ul>` : ""}
    <button class="ghost" id="toGraph">Show neighbourhood graph</button>`;
  $("#toGraph").onclick = () => { switchTab("graph"); loadGraph(id); };
}

/* ---------------- decisions ---------------- */
const DEC = { supersedes: "replaced", contradicts: "contradicts", fixes: "fixes", revises: "edited", refines: "refines" };
async function loadDecisions() {
  if (!state.root) return;
  const box = $("#decisions");
  try {
    const r = await api("/api/decisions?limit=80&root=" + encodeURIComponent(state.root));
    box.innerHTML = r.items.length ? r.items.map((d) => `
      <div class="dec"><div class="meta">${(d.types || [d.type]).map((t) => `<span class="badge b-${esc(t)}">${esc(DEC[t] || t)}</span>`).join(" ")} ${esc(d.date)}</div>
      <div class="pair"><div data-id="${esc(d.new.id)}" class="clk"><div class="lbl">now · ${esc(d.new.file)}</div>${esc(clip(d.new.text, 260))}</div>
      <div class="old clk" data-id="${esc(d.old.id)}"><div class="lbl">before · ${esc(d.old.date)} · ${esc(d.old.file)}</div>${esc(clip(d.old.text, 260))}</div></div></div>`).join("")
      : `<div class="empty-state">No decisions recorded yet. They appear when a note replaces, contradicts or fixes an earlier one.</div>`;
  } catch (e) { box.innerHTML = `<div class="empty-state">${esc(e.message)}</div>`; }
}
async function exportDecisions() {
  const r = await api("/api/decisions/export", { root: state.root });
  const blob = new Blob([r.markdown], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "DECISIONS.md"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

/* ---------------- graph (small force layout on canvas) ---------------- */
const EDGE_COLORS = { continues: "#9aa3b2", reinforces: "#4f7cff", returns_to: "#22a699", rescue: "#c7ccd4",
  supersedes: "#e0952b", contradicts: "#d9534f", fixes: "#2e9d5b", refines: "#8a63d2", revises: "#1fb5c9" };
function legend() {
  $("#legend").innerHTML = Object.entries(EDGE_COLORS).map(([k, c]) => `<span><i style="background:${c}"></i>${k}</span>`).join("");
}
async function loadGraph(id) {
  const g = await api(`/api/graph?depth=2&id=${encodeURIComponent(id)}&root=${encodeURIComponent(state.selected?.root || state.root)}`);
  $("#graphHint").textContent = `${g.nodes.length} memories, ${g.edges.length} links around the selected one. Drag to move, click to open.`;
  const cv = $("#graph"), W = cv.clientWidth, H = cv.clientHeight;
  const idx = {};
  g.nodes.forEach((n, i) => { idx[n.id] = i; const a = (i / g.nodes.length) * Math.PI * 2; n.x = W / 2 + (n.center ? 0 : Math.cos(a) * 200); n.y = H / 2 + (n.center ? 0 : Math.sin(a) * 160); n.vx = n.vy = 0; });
  g.edges = g.edges.filter((e) => idx[e.from] != null && idx[e.to] != null);
  state.graph = { ...g, idx, drag: null, hover: null, ticks: 0 };
  runGraph();
}
function runGraph() {
  const G = state.graph; if (!G) return;
  const cv = $("#graph"), ctx = cv.getContext("2d"), dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const N = G.nodes;
  if (G.ticks < 400) {
    for (let i = 0; i < N.length; i++) for (let j = i + 1; j < N.length; j++) {
      const a = N[i], b = N[j]; let dx = a.x - b.x, dy = a.y - b.y; const d2 = dx * dx + dy * dy + 0.01, f = 2400 / d2;
      const d = Math.sqrt(d2); dx /= d; dy /= d; a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    }
    for (const e of G.edges) {
      const a = N[G.idx[e.from]], b = N[G.idx[e.to]]; const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1, f = (d - 90) * 0.02;
      a.vx += (dx / d) * f; a.vy += (dy / d) * f; b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
    }
    for (const n of N) {
      if (n === G.drag) continue;
      n.vx += (W / 2 - n.x) * 0.002; n.vy += (H / 2 - n.y) * 0.002;
      if (n.center) { n.vx *= 0.3; n.vy *= 0.3; }
      n.x += n.vx * 0.5; n.y += n.vy * 0.5; n.vx *= 0.6; n.vy *= 0.6;
      n.x = Math.max(20, Math.min(W - 20, n.x)); n.y = Math.max(20, Math.min(H - 20, n.y));
    }
    G.ticks++;
  }
  ctx.clearRect(0, 0, W, H);
  for (const e of G.edges) {
    const a = N[G.idx[e.from]], b = N[G.idx[e.to]];
    ctx.strokeStyle = EDGE_COLORS[e.type] || "#999"; ctx.lineWidth = ["supersedes", "contradicts", "fixes", "revises"].includes(e.type) ? 2.4 : 1.2;
    ctx.globalAlpha = 0.85; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  for (const n of N) {
    const r = n.center ? 11 : 7;
    ctx.fillStyle = n.status === "retracted" ? "#b9bfc9" : n.status === "superseded" ? "#e0952b" : n.center ? "#4f7cff" : "#22a699";
    ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2); ctx.fill();
    if (n === G.hover || n.center) { ctx.lineWidth = 2; ctx.strokeStyle = getComputedStyle(document.body).color; ctx.stroke(); }
  }
  requestAnimationFrame(runGraph);
}
function graphEvents() {
  const cv = $("#graph"), tip = $("#graphTip");
  const pick = (ev) => { const r = cv.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    return state.graph?.nodes.find((n) => (n.x - x) ** 2 + (n.y - y) ** 2 < 144) || null; };
  cv.addEventListener("mousemove", (ev) => {
    const G = state.graph; if (!G) return;
    if (G.drag) { const r = cv.getBoundingClientRect(); G.drag.x = ev.clientX - r.left; G.drag.y = ev.clientY - r.top; G.ticks = Math.min(G.ticks, 300); return; }
    const n = pick(ev); G.hover = n;
    if (n) { tip.hidden = false; tip.style.left = (n.x + 14) + "px"; tip.style.top = (n.y + 10) + "px";
      tip.innerHTML = `<div class="meta">${esc(n.date)} · ${esc(n.file)} · ${esc(n.status)}</div>${esc(n.label)}`; }
    else tip.hidden = true;
  });
  cv.addEventListener("mousedown", (ev) => { const n = pick(ev); if (n && state.graph) state.graph.drag = n; });
  window.addEventListener("mouseup", () => { if (state.graph) state.graph.drag = null; });
  cv.addEventListener("dblclick", (ev) => { const n = pick(ev); if (n) loadGraph(n.id); });
  cv.addEventListener("click", (ev) => { const n = pick(ev); if (n) { switchTab("search"); showNode(n.id); } });
}

/* ---------------- settings / licence / daemon ---------------- */
async function loadSettings() {
  const c = await api("/api/config");
  const i = c.inject || {};
  $("#injEnabled").checked = !!i.enabled; $("#injBrief").checked = i.session_brief !== false;
  $("#injBudget").value = i.budget_chars; $("#injItems").value = i.max_items; $("#injScore").value = i.min_score;
  $("#idle").value = Math.round((c.idle_unload_seconds || 600) / 60);
  syncLabels();
  const l = await api("/api/license");
  renderLicense(l);
}
function syncLabels() {
  $("#budgetVal").textContent = $("#injBudget").value; $("#itemsVal").textContent = $("#injItems").value;
  $("#scoreVal").textContent = Number($("#injScore").value).toFixed(2); $("#idleVal").textContent = $("#idle").value;
}
let saveTimer = null;
function saveSettings() {
  syncLabels(); clearTimeout(saveTimer);
  saveTimer = setTimeout(() => api("/api/config", {
    enabled: $("#injEnabled").checked, session_brief: $("#injBrief").checked, budget_chars: Number($("#injBudget").value),
    max_items: Number($("#injItems").value), min_score: Number($("#injScore").value), idle_unload_seconds: Number($("#idle").value) * 60,
  }).catch(() => {}), 250);
}
function renderLicense(l) {
  $("#licInfo").innerHTML = l.registered
    ? `<p><b>Pro</b> — licensed to ${esc(l.name)}${l.expires ? ", until " + esc(l.expires) : ""}.</p>`
    : `<p><b>Free</b> — personal use. ${l.days_used} days so far.</p>`;
  $("#features").innerHTML = Object.entries(l.all_features || {}).map(([k, v]) =>
    `<li class="${(l.features || []).includes(k) ? "on" : ""}">${esc(v)}${(l.features || []).includes(k) ? " ✓" : ""}</li>`).join("");
}
function renderDaemon(s) {
  const emb = (s.embedders || []).map((e) => `${e.backend} ${String(e.model).split(/[\\/]/).pop()} — ${e.loaded ? "loaded" : "unloaded"}, ${e.texts} texts embedded`).join("<br>");
  $("#daemonInfo").innerHTML = `version ${esc(s.version)} · pid ${s.pid} · up ${Math.round(s.uptime / 60)} min · queue ${s.queue}<br>${esc(emb)}`;
}

/* ---------------- tabs & wiring ---------------- */
function switchTab(name) {
  $$(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("on", t.id === "tab-" + name));
  if (name === "decisions") loadDecisions();
  if (name === "settings") loadSettings();
  if (name === "graph") legend();
  try { localStorage.setItem("ptg-tab", name); } catch (e) {}
}
function showPro(text) { $("#modalText").textContent = text || "This is a Pro feature."; $("#modal").hidden = false; }

document.addEventListener("click", (ev) => {
  const t = ev.target.closest("[data-id]");
  if (t && (t.closest("#feed") || t.closest("#results") || t.closest("#detail") || t.closest("#decisions"))) {
    if (!t.closest("#results") && !t.closest("#detail")) switchTab("search");
    showNode(t.dataset.id, t.dataset.root);
  }
});
$$(".tabs button").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));
$("#searchForm").addEventListener("submit", doSearch);
$("#project").addEventListener("change", (e) => { state.root = e.target.value; state.feedSig = ""; refreshFeed(); refreshStatus(); });
$("#exportBtn").addEventListener("click", () => exportDecisions().catch(() => {}));
$("#modalClose").addEventListener("click", () => ($("#modal").hidden = true));
$("#nagClose").addEventListener("click", () => { try { localStorage.setItem("ptg-nag", String(Date.now())); } catch (e) {} $("#nag").hidden = true; });
["#injEnabled", "#injBrief", "#injBudget", "#injItems", "#injScore", "#idle"].forEach((s) => $(s).addEventListener("input", saveSettings));
$("#licForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const r = await api("/api/license", { key: $("#licKey").value.trim() }).catch((e) => ({ ok: false, error: e.message }));
  if (!r.ok) alert("Key rejected: " + r.error); else { renderLicense(r); refreshStatus(); }
});
$("#saveBtn").addEventListener("click", () => api("/api/save", { root: state.root }).then(refreshFeed).catch((e) => alert(e.message)));
$("#unloadBtn").addEventListener("click", () => api("/api/unload", {}).then(refreshStatus));
$("#rescanBtn").addEventListener("click", () => api("/api/ingest", { root: state.root }).then(refreshStatus));

graphEvents();
refreshStatus().then(async () => {
  await refreshFeed();
  if (deepLink.tab) switchTab(deepLink.tab);
  if (deepLink.q) { $("#q").value = deepLink.q; await doSearch(); const first = $(".res"); if (first) showNode(first.dataset.id); }
});
setInterval(() => { refreshStatus(); refreshFeed(); }, 2000);
if (!deepLink.tab) { try { const t = localStorage.getItem("ptg-tab"); if (t) switchTab(t); } catch (e) {} }
