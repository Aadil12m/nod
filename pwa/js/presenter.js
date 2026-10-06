// Presenter page (design 3.1, 3.5.8): 3D view iframe, HUD, trace, STOP key.
// It can send STOP only; it cannot confirm or clear a stop.
//
// Query parameters: viewer=<url of the viser view> (else /healthz `viewer_url`,
// else http(s)://<this host>:8080/), gw=wss://host (gateway origin), noviewer=1.

import { Channel, Clock, gatewayBase, httpBase } from './net.js';
import { msg } from './proto.js';
import { EStopController, RESEND_MS } from './estop.js';
import { successText } from './ui.js';

const q = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const ms = (x) => (Number.isFinite(x) ? `${Math.round(x)} ms` : '-');

const clock = new Clock();
const S = { mode: null, run: null, proposal: null, focus: null, metrics: {}, traces: [], lastEstop: null };

let ch = null;
const estop = new EStopController({ send: (m) => ch?.send(m), onChange: () => render() });

ch = new Channel(`${gatewayBase()}/ws/presenter`, {
  onOpen: () => { conn(true); estop.tick(); ch.send(msg.ping(Date.now())); },
  onClose: () => conn(false),
  onMessage: onMessage,
});

function conn(ok) {
  const el = $('conn');
  el.dataset.ok = ok ? '1' : '0';
  el.textContent = ok ? 'Connected' : 'Offline';
}

function addTrace(kind, ref, text) {
  S.traces.unshift({ t: new Date(), kind, ref, text });
  if (S.traces.length > 200) S.traces.pop();
}

function summarize(data) {
  if (!data || typeof data !== 'object') return String(data ?? '');
  const parts = [];
  for (const [k, v] of Object.entries(data)) {
    if (parts.length >= 6) { parts.push('...'); break; }
    const s = typeof v === 'object' ? JSON.stringify(v) : String(v);
    parts.push(`${k}=${s.length > 60 ? `${s.slice(0, 57)}...` : s}`);
  }
  return parts.join(' ');
}

function onMessage(m) {
  switch (m.type) {
    case 'pong': clock.onPong(m); break;
    case 'trace':
      addTrace(m.kind, m.ref, summarize(m.data));
      // The gateway acks a stop to the peer that sent it; every other stop reaches this page as an
      // "estop" trace with the same latency split, which feeds the "E-stop (last)" tile.
      if (m.kind === 'estop' && m.data?.acked && m.data.lat_ms) S.lastEstop = { estop_seq: m.data.client_seq, lat_ms: m.data.lat_ms };
      break;
    case 'metrics': S.metrics = m; break;
    case 'cosmos.check': addTrace('cosmos', m.check_id, `${m.phase} ${m.skill_id}: ${m.question}`); break;
    case 'cosmos.result': addTrace('cosmos', m.check_id, `${m.answer} (${ms(m.latency_ms)}) ${m.decision}${m.agree === false ? ' DISAGREES with oracle' : ''}: ${m.reason}`); break;
    case 'state':
      S.mode = m.mode;
      estop.onMessage(m);
      // The phone cleared the stop: the presenter's latch follows the gateway.
      if (estop.latched && m.mode !== 'ESTOPPED' && m.estop_seq >= estop.seq) estop._unlatch();
      // The gateway already reports our stop: stop resending even if no ack reaches this page.
      if (estop.latched && !estop.acked && m.mode === 'ESTOPPED' && m.estop_seq >= estop.seq) { estop.acked = true; estop.ackedAt = Date.now(); }
      if (m.mode !== 'AWAITING_CONFIRM' && m.mode !== 'LISTENING') S.proposal = null;
      break;
    case 'run': S.run = m; break;
    case 'proposal': S.proposal = m; S.focus = m.focus; break;
    case 'estop.ack':
      S.lastEstop = m;
      estop.onMessage(m);
      addTrace('estop', `#${m.estop_seq}`, summarize(m.lat_ms));
      break;
    case 'notice': addTrace('notice', '', m.text); break;
    case 'transcript': addTrace('asr', `utt ${m.utt_id}`, `"${m.text}" (${ms(m.asr_ms)})`); break;
    default: break;
  }
  render();
}

function render() {
  const mode = $('mode');
  mode.textContent = S.mode || '-';
  mode.dataset.mode = S.mode || '';
  const r = S.run;
  $('run').textContent = r ? `${r.skill_id} ${r.phase}${r.max_steps ? ` ${r.sim_step}/${r.max_steps}` : ''}${r.plan_len > 1 ? ` (step ${r.plan_step + 1}/${r.plan_len})` : ''}` : '-';
  const mt = S.metrics;
  $('lag').textContent = ms(mt.view_lag_ms);
  $('rtt').textContent = ms(clock.rtt);
  const lat = S.lastEstop?.lat_ms;
  const total = lat ? ['phone_gw', 'gw_sim', 'sim_halt'].reduce((s, k) => s + (Number.isFinite(lat[k]) ? lat[k] : 0), 0) : null;
  $('estop-last').textContent = lat ? `${ms(total)} (${['phone_gw', 'gw_sim', 'sim_halt'].map((k) => (Number.isFinite(lat[k]) ? Math.round(lat[k]) : '?')).join('+')})` : '-';
  $('estop-p').textContent = `${ms(mt.estop_ms_p50)} / ${ms(mt.estop_ms_p95)}`;
  $('policy').textContent = ms(mt.policy_ms_p95);
  $('llm').textContent = `${ms(mt.llm_ms_p50)} / ${ms(mt.asr_ms_p50)}`;
  $('stopped').hidden = !estop.latched && S.mode !== 'ESTOPPED';
  const p = S.proposal;
  $('options').innerHTML = p ? (p.options || []).map((o) => `<li class="${o.option_id === S.focus ? 'focused' : ''}">${esc(o.label)}
      <span class="small">${esc(successText(o.success))}${o.why ? ` &middot; ${esc(o.why)}` : ''}</span></li>`).join('') : '';
  $('trace').innerHTML = S.traces.slice(0, 80).map((t) => `<li><span class="k k-${esc(t.kind)}">${esc(t.kind)}</span>
      ${t.ref ? `<b>${esc(t.ref)}</b> ` : ''}${esc(t.text)}</li>`).join('');
}

async function loadRegistry() {
  try {
    const r = await fetch(`${httpBase()}/api/registry`, { cache: 'no-store' });
    const j = await r.json();
    const skills = Array.isArray(j) ? j : j.skills || [];
    const rows = skills.filter((s) => s && s.kind !== 'home' && s.offer !== false).map((s) => {
      const a = s.stats?.alone, c = s.stats?.in_chain;
      const f = (b) => (b ? `${b.k}/${b.n}` : '-');
      return `<tr><td>${esc(s.label || s.skill_id)}</td><td class="num">${f(a)}</td><td class="num">${f(c)}</td></tr>`;
    });
    if (rows.length) {
      $('skills').querySelector('tbody').innerHTML = `<tr><td class="small">skill</td><td class="num small">alone</td><td class="num small">in chain</td></tr>${rows.join('')}`;
    }
  } catch { /* gateway without /api/registry */ }
}

async function health() {
  try {
    const r = await fetch(`${httpBase()}/healthz`, { cache: 'no-store' });
    return await r.json();
  } catch { return null; }
}

async function setupViewer() {
  const h = await health();
  $('replay').hidden = !(h && (h.mode === 'mock' || h.mode === 'replay'));
  if (q.get('noviewer') === '1') { $('viewer-empty').hidden = false; return; }
  const url = q.get('viewer') || h?.viewer_url || `${location.protocol}//${location.hostname}:8080/`;
  $('viewer').src = url;
}

function stop() {
  estop.trigger('presenter', { t_detect: Date.now() });
  render();
}

document.addEventListener('keydown', (e) => {
  if (e.key === ' ' || e.code === 'Space') { e.preventDefault(); if (!e.repeat) stop(); }
});
$('stop').addEventListener('click', stop);

setInterval(() => estop.tick(), RESEND_MS);
setInterval(() => ch.send(msg.ping(Date.now())), 2000);
ch.connect();
setupViewer();
loadRegistry();
render();
globalThis.nodPresenter = { S, estop, ch };
