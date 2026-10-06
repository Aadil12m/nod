// A scripted stand-in for the Nod gateway, for the PWA end-to-end test and for
// trying the PWA without Python. It serves pwa/ as static files, answers
// /healthz with mode "mock", and speaks the WebSocket protocol of design 3.5 on
// /ws/phone, /ws/safety and /ws/presenter with a fixed script:
//   ui.menu or utter.text -> proposal (2 options + "None of these")
//   ui.confirm (valid nonce) -> COUNTDOWN 1 s -> EXECUTING (progress) -> POSTCHECK -> done -> IDLE
//   estop -> ESTOPPED + estop.ack (optionally delayed), estop.reset -> IDLE
// It is NOT the gateway: no validator, no planner, no sim.
//
// Run standalone: node tests/stub_gateway.mjs --port 8000

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.webmanifest': 'application/manifest+json', '.svg': 'image/svg+xml',
  '.png': 'image/png', '.wasm': 'application/wasm', '.task': 'application/octet-stream',
};

const OPTIONS = [
  { option_id: 'o1', kind: 'skill', skill_ids: ['bowl_on_plate'], label: 'Put the bowl on the plate', why: 'Ready for a snack',
    success: { p_est: 0.95, basis: 'alone', n: 20, k: 19, wilson_lo: 0.76 }, est_s: 9, risk: 'low', confirm: 'single',
    checks: [{ rule: 'R4_preconditions', ok: true }], precheck: { status: 'ok' }, badges: [], locked: false },
  { option_id: 'o2', kind: 'skill', skill_ids: ['open_top_drawer'], label: 'Open the top drawer', why: 'To reach the bowl',
    success: { p_est: 0.9, basis: 'alone', n: 20, k: 18, wilson_lo: 0.70 }, est_s: 7, risk: 'low', confirm: 'single',
    checks: [], precheck: { status: 'pending' }, badges: [], locked: false },
];
const LABEL = { o1: 'Put the bowl on the plate', o2: 'Open the top drawer' };
const SKILL = { o1: 'bowl_on_plate', o2: 'open_top_drawer' };

export async function startStub({ port = 0, ackDelayMs = 0, stepMs = 200, maxSteps = 200, log = false } = {}) {
  const received = { phone: [], safety: [], presenter: [], binary: [] };
  const sockets = { phone: new Set(), safety: new Set(), presenter: new Set() };
  const g = { sess: 's_stub', mode: 'IDLE', estop_seq: 0, proposal: null, run: null, timers: [], seq: 0, acked: new Set() };

  const env = (type, payload) => ({ v: 1, type, id: `stub${++g.seq}`, ts: Date.now(), sess: g.sess, seq: g.seq, ...payload });
  const sendTo = (set, m) => { for (const ws of set) if (ws.readyState === 1) ws.send(JSON.stringify(m)); };
  const toPhone = (type, payload) => { const m = env(type, payload); sendTo(sockets.phone, m); sendTo(sockets.presenter, m); return m; };
  const trace = (kind, ref, data) => sendTo(sockets.presenter, env('trace', { kind, ref, data }));
  const later = (ms, fn) => { const t = setTimeout(fn, ms); g.timers.push(t); };
  const clearTimers = () => { g.timers.forEach(clearTimeout); g.timers = []; };
  const state = (mode) => { g.mode = mode; return toPhone('state', { mode, run_id: g.run?.run_id ?? null, scene_id: 'libero_goal_v1', estop_seq: g.estop_seq }); };

  function propose(source, transcript = null) {
    const id = `p_${Date.now().toString(36)}`;
    g.proposal = { proposal_id: id, nonce: `n_${Math.random().toString(36).slice(2, 8)}` };
    g.lastProposal = g.proposal;
    toPhone('proposal', {
      proposal_id: id, expires_at: Date.now() + 30_000, arm_delay_ms: 300, source, model: null, transcript,
      options: OPTIONS, fixed: [{ option_id: 'none', label: 'None of these' }], focus: 'o1', nonce: g.proposal.nonce,
    });
    trace('planner', id, { source, options: OPTIONS.length });
    state('AWAITING_CONFIRM');
  }

  function runUpdate(phase, extra = {}) {
    g.run = { ...g.run, phase, ...extra };
    return toPhone('run', { ...g.run });
  }

  function startRun(c) {
    clearTimers();
    g.run = { run_id: `r_${Date.now().toString(36)}`, proposal_id: c.proposal_id, option_id: c.option_id, plan_step: 0, plan_len: 1,
      skill_id: SKILL[c.option_id], phase: 'countdown', sim_step: 0, max_steps: maxSteps, elapsed_ms: 0, cosmos: null, oracle: null, fault: null };
    g.proposal = null;
    state('COUNTDOWN');
    runUpdate('countdown');
    toPhone('notice', { level: 'info', text: LABEL[c.option_id], speak: true });
    later(1000, () => {
      state('EXECUTING');
      runUpdate('executing');
      const t0 = Date.now();
      const step = () => {
        const s = Math.min(maxSteps, g.run.sim_step + 20);
        runUpdate('executing', { sim_step: s, elapsed_ms: Date.now() - t0 });
        if (s < maxSteps) later(stepMs, step);
        else {
          state('POSTCHECK');
          runUpdate('postcheck');
          later(300, () => { runUpdate('done', { oracle: true }); state('IDLE'); });
        }
      };
      later(stepMs, step);
    });
  }

  function onEstop(m, ws) {
    if (m.estop_seq > g.estop_seq) {
      g.estop_seq = m.estop_seq;
      clearTimers();
      if (g.run && !['done', 'failed', 'uncertain', 'aborted'].includes(g.run.phase)) runUpdate('aborted');
      g.proposal = null;
      state('ESTOPPED');
      trace('estop', `#${m.estop_seq}`, { source: m.source });
    }
    if (m.estop_seq === g.estop_seq) {
      // Ack every copy (the first ack may be lost); optionally late, to test resending.
      const ack = () => {
        g.acked.add(m.estop_seq);
        const a = env('estop.ack', { estop_seq: m.estop_seq, stopped_at_step: g.run?.sim_step ?? null, lat_ms: { phone_gw: Math.max(0, Date.now() - m.t_detect), gw_sim: 1, sim_halt: 12 } });
        if (ws.readyState === 1) ws.send(JSON.stringify(a));
        sendTo(sockets.presenter, a);
      };
      if (g.acked.has(m.estop_seq) || !ackDelayMs) ack();
      else setTimeout(ack, ackDelayMs);
    }
  }

  function handle(kind, m, ws) {
    received[kind].push(m);
    if (log) console.log(`[stub] ${kind} <- ${m.type}`);
    switch (m.type) {
      case 'hello':
        ws.send(JSON.stringify(env('welcome', { resume_token: 'rt_stub', lease: 'user', server_time: Date.now() })));
        ws.send(JSON.stringify(env('state', { mode: g.mode, run_id: null, scene_id: 'libero_goal_v1', estop_seq: g.estop_seq })));
        break;
      case 'ping':
        ws.send(JSON.stringify(env('pong', { t0: m.t0, t_srv: Date.now() })));
        break;
      case 'ui.menu':
        if (g.mode !== 'IDLE') break;
        state('LISTENING');
        later(100, () => propose('menu'));
        break;
      case 'utter.text':
        if (g.mode === 'ESTOPPED') break;
        toPhone('transcript', { utt_id: m.utt_id, text: m.text, asr_ms: 0 });
        state('PLANNING');
        later(200, () => propose('fake', m.text));
        break;
      case 'ui.confirm':
        if (g.mode !== 'AWAITING_CONFIRM' || !g.proposal || m.proposal_id !== g.proposal.proposal_id || m.nonce !== g.proposal.nonce || !LABEL[m.option_id]) {
          toPhone('notice', { level: 'warn', text: 'Try again', speak: true });
          break;
        }
        startRun(m);
        break;
      case 'ui.cancel':
        if (['AWAITING_CONFIRM', 'LISTENING', 'PLANNING'].includes(g.mode)) { clearTimers(); g.proposal = null; state('IDLE'); }
        break;
      case 'estop':
        onEstop(m, ws);
        break;
      case 'estop.reset':
        if (g.mode === 'ESTOPPED' && m.estop_seq === g.estop_seq) { g.run = null; state('IDLE'); }
        break;
      default:
        break;
    }
  }

  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    if (url.pathname === '/healthz') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ mode: 'mock', ok: true, stub: true }));
      return;
    }
    if (url.pathname === '/api/registry') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify(Object.entries(LABEL).map(([o, label]) => ({ skill_id: SKILL[o], kind: 'policy', label, stats: { alone: { k: 19, n: 20, wilson_lo: 0.76 } } }))));
      return;
    }
    let p = url.pathname === '/' ? '/index.html' : url.pathname === '/presenter' ? '/presenter.html' : url.pathname;
    const file = path.normalize(path.join(ROOT, decodeURIComponent(p)));
    if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); res.end('not found'); return;
    }
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream', 'cache-control': 'no-store' });
    if (req.method === 'HEAD') { res.end(); return; }
    fs.createReadStream(file).pipe(res);
  });

  const wss = new WebSocketServer({ noServer: true });
  server.on('upgrade', (req, socket, head) => {
    const kind = { '/ws/phone': 'phone', '/ws/safety': 'safety', '/ws/presenter': 'presenter' }[new URL(req.url, 'http://x').pathname];
    if (!kind) { socket.destroy(); return; }
    wss.handleUpgrade(req, socket, head, (ws) => {
      sockets[kind].add(ws);
      ws.on('close', () => sockets[kind].delete(ws));
      ws.on('message', (data, isBinary) => {
        if (isBinary) { received.binary.push(data); return; }
        let m;
        try { m = JSON.parse(String(data)); } catch { return; }
        handle(kind, m, ws);
      });
      if (kind === 'presenter') {
        ws.send(JSON.stringify(env('state', { mode: g.mode, run_id: null, scene_id: 'libero_goal_v1', estop_seq: g.estop_seq })));
        ws.send(JSON.stringify(env('metrics', { estop_ms_p50: 63, estop_ms_p95: 118, view_lag_ms: 35, policy_ms_p95: 97, asr_ms_p50: 210, llm_ms_p50: 640 })));
      }
    });
  });

  await new Promise((r) => server.listen(port, '127.0.0.1', r));
  const url = `http://127.0.0.1:${server.address().port}`;
  return {
    url, received, g, sockets,
    dropSafety: () => { for (const ws of sockets.safety) ws.terminate(); },
    close: () => new Promise((r) => { clearTimers(); for (const s of Object.values(sockets)) for (const ws of s) ws.terminate(); wss.close(); server.close(() => r()); }),
  };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const i = process.argv.indexOf('--port');
  const port = i > 0 ? Number(process.argv[i + 1]) : 8000;
  const s = await startStub({ port, log: true });
  console.log(`stub gateway on ${s.url}  (phone: ${s.url}/  presenter: ${s.url}/presenter?noviewer=1)`);
}
