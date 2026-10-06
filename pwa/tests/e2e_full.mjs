// Full mock end-to-end: real Chromium, the real PWA, the REAL gateway in mock mode.
//
// Starts `nod-gateway --mock` (MockRunner, FakePlanner, FakeASR, FakeCosmos, FakeHome) on free
// ports, then drives the phone app with the keyboard fallback (design 4.4):
//   1. B opens the menu; a typed "I'm a bit hungry" replaces it with the planner's options.
//   2. Right/Left move focus; holding Enter (long blink) confirms; the run goes to done.
//   3. A second run is started and Space (STOP) is pressed mid-run: the STOPPED latch shows,
//      estop.ack arrives, the run is aborted (never resumed).
//   4. A long blink on "Clear stop?" sends estop.reset and the gateway goes back to IDLE.
//   5. The presenter page, open the whole time, received trace messages and shows them.
//   6. A "double" option (bowl_to_user, first use) is confirmed with exactly two long blinks.
//   7. A run from the menu; the page is hidden mid-run (no face in keyboard mode): the
//      gateway pauses, offers "Resume?", and a long blink resumes the same run to done.
// All WebSocket traffic is captured through Playwright's websocket events, and the gateway's
// own view is read from GET /api/dev/state (mock mode only).
//
// Run (from pwa/):  node tests/e2e_full.mjs [--screenshots ../docs/screenshots]
// Needs gateway/.venv (make setup-gateway) and a Chromium (PLAYWRIGHT_BROWSERS_PATH or CHROME_PATH).
// NOD_GATEWAY_BIN overrides the gateway executable. NOD_E2E_RECORDINGS=1 lets the MockRunner
// replay recordings/ if present (default: its scripted timeline, as on a clean checkout).

import fs from 'node:fs';
import os from 'node:os';
import net from 'node:net';
import path from 'node:path';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const argv = process.argv.slice(2);
const shotDir = argv.includes('--screenshots') ? path.resolve(argv[argv.indexOf('--screenshots') + 1]) : null;
const verbose = argv.includes('-v') || process.env.NOD_E2E_VERBOSE === '1';

function findChromium() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const roots = [process.env.PLAYWRIGHT_BROWSERS_PATH, '/opt/pw-browsers'].filter(Boolean);
  for (const root of roots) {
    if (!fs.existsSync(root)) continue;
    const dirs = fs.readdirSync(root).filter((d) => /^chromium-\d+$/.test(d)).sort().reverse();
    for (const d of dirs) {
      const p = path.join(root, d, 'chrome-linux', 'chrome');
      if (fs.existsSync(p)) return p;
    }
  }
  return undefined; // let playwright-core resolve its own
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => { const { port } = srv.address(); srv.close(() => resolve(port)); });
  });
}

// ---------------------------------------------------------------- gateway process

async function startGateway() {
  const bin = process.env.NOD_GATEWAY_BIN || path.join(REPO, 'gateway', '.venv', 'bin', 'nod-gateway');
  if (!fs.existsSync(bin)) throw new Error(`${bin} not found: run "make setup-gateway" first`);
  const [port, ctl, estop, pub, pose] = [await freePort(), await freePort(), await freePort(), await freePort(), await freePort()];
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'nod-e2e-'));
  const recordings = process.env.NOD_E2E_RECORDINGS === '1' ? path.join(REPO, 'recordings') : path.join(tmp, 'recordings');
  fs.mkdirSync(recordings, { recursive: true });
  const env = {
    ...process.env,
    NOD_SIM_CTL: `tcp://127.0.0.1:${ctl}`,
    NOD_SIM_ESTOP: `tcp://127.0.0.1:${estop}`,
    NOD_SIM_PUB: `tcp://127.0.0.1:${pub}`,
    NOD_SIM_POSE_PUB: `tcp://127.0.0.1:${pose}`,
    NOD_RECORDINGS_DIR: recordings,
    NOD_MOCK_STEPS: '80', // 4 s per mock skill at 20 Hz: long enough to press STOP mid-run
    NOD_PWA_DIR: path.join(REPO, 'pwa'),
  };
  const log = [];
  const proc = spawn(bin, ['--mock', '--host', '127.0.0.1', '--port', String(port), '--runs-dir', path.join(tmp, 'runs'), '--log-level', 'warning'],
    { env, stdio: ['ignore', 'pipe', 'pipe'] });
  const onData = (d) => { log.push(String(d)); if (verbose) process.stderr.write(`[gw] ${d}`); };
  proc.stdout.on('data', onData);
  proc.stderr.on('data', onData);
  let exited = null;
  proc.on('exit', (code) => { exited = code; });
  const url = `http://127.0.0.1:${port}`;
  const t0 = Date.now();
  for (;;) {
    if (exited !== null) throw new Error(`gateway exited with ${exited}:\n${log.join('')}`);
    try {
      const r = await fetch(`${url}/healthz`);
      if (r.ok) {
        const j = await r.json();
        if (j.components?.sim?.scene_loaded) break;
      }
    } catch { /* not up yet */ }
    if (Date.now() - t0 > 30000) throw new Error(`gateway did not start in 30 s:\n${log.join('')}`);
    await new Promise((r) => setTimeout(r, 200));
  }
  return {
    url,
    log,
    async state() { return (await fetch(`${url}/api/dev/state`)).json(); },
    async close() {
      if (exited === null) {
        proc.kill('SIGINT');
        await new Promise((r) => { const t = setTimeout(() => { proc.kill('SIGKILL'); r(); }, 5000); proc.on('exit', () => { clearTimeout(t); r(); }); });
      }
      fs.rmSync(tmp, { recursive: true, force: true });
    },
  };
}

// ---------------------------------------------------------------- WebSocket capture

// Every JSON frame on every socket of a page: {sock: 'phone'|'safety'|'presenter', dir: 'in'|'out', t, m}
function captureSockets(page, frames) {
  page.on('websocket', (ws) => {
    const sock = new URL(ws.url()).pathname.split('/').pop();
    const add = (dir) => (f) => {
      if (typeof f.payload !== 'string') return; // binary audio or thumbnails
      let m;
      try { m = JSON.parse(f.payload); } catch { return; }
      frames.push({ sock, dir, t: Date.now(), m });
    };
    ws.on('framesent', add('out'));
    ws.on('framereceived', add('in'));
  });
}

// ---------------------------------------------------------------- test steps

const results = [];
async function step(name, fn) {
  const t0 = Date.now();
  try {
    await fn();
    results.push({ name, ok: true });
    console.log(`ok   ${name} (${Date.now() - t0} ms)`);
  } catch (e) {
    results.push({ name, ok: false });
    console.log(`FAIL ${name}\n     ${String(e?.stack || e).split('\n').slice(0, 8).join('\n     ')}`);
    throw e;
  }
}

const gw = await startGateway();
const browser = await chromium.launch({ executablePath: findChromium(), headless: true });
const phoneFrames = [];
const presFrames = [];
const consoleErrors = [];
const phoneCtx = await browser.newContext({ viewport: { width: 412, height: 915 }, deviceScaleFactor: 1 });
const page = await phoneCtx.newPage();
captureSockets(page, phoneFrames);
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(`phone: ${m.text()}`); });
page.on('pageerror', (e) => consoleErrors.push(`phone: ${e}`));
const presCtx = await browser.newContext({ viewport: { width: 1280, height: 1200 }, deviceScaleFactor: 1 });
const pres = await presCtx.newPage();
captureSockets(pres, presFrames);
pres.on('pageerror', (e) => consoleErrors.push(`presenter: ${e}`));

const sent = (sock, type) => phoneFrames.filter((f) => f.dir === 'out' && f.sock === sock && f.m.type === type).map((f) => f.m);
const got = (type) => phoneFrames.filter((f) => f.dir === 'in' && f.m.type === type).map((f) => f.m);
const presGot = (type) => presFrames.filter((f) => f.dir === 'in' && f.m.type === type).map((f) => f.m);
const screen = () => page.locator('#screen').getAttribute('data-screen');
const waitScreen = (s, timeout = 5000) =>
  page.waitForFunction((x) => document.getElementById('screen').dataset.screen === x, s, { timeout });
const holdEnter = async (ms) => { await page.keyboard.down('Enter'); await page.waitForTimeout(ms); await page.keyboard.up('Enter'); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, what, timeout = 10000) {
  const t0 = Date.now();
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() - t0 > timeout) throw new Error(`timed out after ${timeout} ms waiting for ${what}`);
    await sleep(50);
  }
}
const lastProposal = () => got('proposal').at(-1);
const runMsgs = (runId) => got('run').filter((r) => r.run_id === runId);
const shot = async (p, name) => { if (shotDir) { fs.mkdirSync(shotDir, { recursive: true }); await p.screenshot({ path: path.join(shotDir, name) }); } };

let exitCode = 0;
try {
  await step('presenter page connects to /ws/presenter and shows IDLE', async () => {
    await pres.goto(`${gw.url}/presenter?noviewer=1`);
    await pres.waitForFunction(() => document.getElementById('conn').dataset.ok === '1');
    await pres.waitForFunction(() => document.getElementById('mode').textContent === 'IDLE', null, { timeout: 5000 });
    await pres.locator('#replay').waitFor({ state: 'visible', timeout: 3000 });
  });

  await step('phone loads from the gateway, says hello on /ws/phone, gets welcome and state', async () => {
    await page.goto(`${gw.url}/?nocam=1&kws=0&nosw=1`);
    await waitScreen('idle');
    await page.waitForFunction(() => document.getElementById('conn').dataset.ok === '1');
    const hello = sent('phone', 'hello')[0];
    assert.ok(hello, 'hello sent');
    assert.equal(hello.role, 'user');
    const welcome = await until(() => got('welcome')[0], 'welcome');
    assert.match(welcome.sess, /^s_/);
    assert.equal(welcome.lease, 'user');
    await until(() => got('state').some((s) => s.mode === 'IDLE'), 'state IDLE');
    await page.locator('#replay-badge').waitFor({ state: 'visible', timeout: 3000 });
    // The safety socket heartbeats with the session id, and the gateway sees the face as present.
    await until(() => sent('safety', 'hb').some((h) => h.sess === welcome.sess), 'hb with sess');
    await until(async () => (await gw.state()).hb_fresh, 'gateway hb_fresh');
  });

  await step('B (brow raise) opens the deterministic menu', async () => {
    await page.keyboard.press('b');
    await waitScreen('choose');
    assert.equal(sent('phone', 'ui.menu')[0].source, 'brow_raise');
    const p = await until(() => got('proposal').find((x) => x.source === 'menu'), 'menu proposal');
    assert.ok(p.options.length >= 1 && p.options.length <= 3);
    assert.equal(await page.locator('.card').count(), p.options.length + p.fixed.length);
    assert.equal((await gw.state()).mode, 'AWAITING_CONFIRM');
  });

  let hungry;
  await step('typed "I\'m a bit hungry" replaces the menu with the planner\'s options', async () => {
    const n0 = got('proposal').length;
    await page.fill('#utter-text', "I'm a bit hungry");
    await page.press('#utter-text', 'Enter');
    const u = sent('phone', 'utter.text')[0];
    assert.equal(u.text, "I'm a bit hungry");
    hungry = await until(() => got('proposal').slice(n0).find((x) => x.transcript === "I'm a bit hungry" && x.source !== 'menu'), 'planner proposal');
    assert.equal(hungry.source, 'fake');
    await waitScreen('choose');
    // Wait for the planner's cards themselves: the menu shown before also has an option o1.
    await page.waitForFunction(([pid, id]) => !!document.querySelector(`.card[data-proposal="${pid}"][data-option="${id}"]`),
      [hungry.proposal_id, hungry.options[0].option_id]);
    const labels = await page.locator('.card').allInnerTexts();
    if (verbose) console.log('     options:', JSON.stringify(hungry.options.map((o) => [o.option_id, o.kind, o.skill_ids, o.label, o.confirm])));
    // Labels come from the registry (R13); the first option is the snack plan.
    assert.equal(hungry.options[0].kind, 'plan');
    assert.match(labels[0], new RegExp(hungry.options[0].label.replace(/[()]/g, '.')));
    assert.match(labels.at(-1), /None of these/);
    await page.locator('#screen').click({ position: { x: 5, y: 5 } }); // keyboard focus off the text box
    await sleep(400); // let the arm delay pass so the cards render armed
    await shot(page, '01_phone_options.png');
  });

  let run1;
  let nPropRun1 = 0;
  await step('Right then Left move focus (ui.focus); holding Enter confirms the plan option', async () => {
    const target = hungry.options[0].option_id;
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction((id) => document.querySelector('.card.focused')?.dataset.option === id, hungry.options[1].option_id);
    await sleep(700); // head-turn lockout before the next turn
    await page.keyboard.press('ArrowLeft');
    await page.waitForFunction((id) => document.querySelector('.card.focused')?.dataset.option === id, target);
    const focus = sent('phone', 'ui.focus');
    assert.deepEqual(focus.slice(-2).map((f) => f.option_id), [hungry.options[1].option_id, target]);
    await sleep(500); // focus dwell
    await holdEnter(900);
    const c = await until(() => sent('phone', 'ui.confirm')[0], 'ui.confirm');
    assert.equal(c.option_id, target);
    assert.equal(c.proposal_id, hungry.proposal_id);
    assert.equal(c.gesture, 'long_blink');
    run1 = await until(() => got('run').find((r) => r.proposal_id === hungry.proposal_id && r.phase === 'executing'), 'run executing', 8000);
    assert.equal(run1.skill_id, 'bowl_on_plate');
    assert.equal(run1.plan_len, 3);
    await waitScreen('executing', 3000);
    await page.waitForFunction(() => Number(document.querySelector('.progress')?.getAttribute('aria-valuenow')) > 0, null, { timeout: 5000 });
    await shot(page, '02_phone_executing.png');
  });

  await step('run 1 goes to done (Cosmos and oracle agree), the arm returns home', async () => {
    const done = await until(() => runMsgs(run1.run_id).find((r) => ['done', 'failed', 'uncertain', 'aborted'].includes(r.phase)), 'run 1 end', 20000);
    nPropRun1 = got('proposal').length;
    assert.equal(done.phase, 'done', JSON.stringify(done));
    const phases = [...new Set(runMsgs(run1.run_id).map((r) => r.phase))];
    if (verbose) console.log('     phases:', phases.join(' > '));
    for (const p of ['countdown', 'executing', 'postcheck', 'done']) assert.ok(phases.includes(p), `phase ${p} in ${phases}`);
    await until(async () => (await gw.state()).world.atoms['on(akita_black_bowl_1,plate_1)'] === true, 'bowl on plate in world');
  });

  let step2, run2;
  await step('the next plan step is offered; confirming it starts run 2', async () => {
    // Per-step plans (3.4): after step 1 the gateway offers step 2 as a new proposal.
    step2 = await until(() => got('proposal').slice(nPropRun1 - 1).find((p) => p.proposal_id !== hungry.proposal_id
      && p.options.some((o) => o.skill_ids?.includes('cheese_in_bowl'))), 'plan step 2 proposal', 20000);
    assert.equal(step2.options.length, 1);
    assert.match(step2.options[0].why, /Step 2 of 3/);
    await waitScreen('choose', 5000);
    const opt = step2.options.find((o) => o.skill_ids?.includes('cheese_in_bowl'));
    await page.waitForFunction((id) => document.querySelector('.card.focused')?.dataset.option === id, step2.focus);
    if (step2.focus !== opt.option_id) throw new Error(`focus ${step2.focus} is not the next step ${opt.option_id}`);
    await sleep(500);
    await holdEnter(900);
    await until(() => sent('phone', 'ui.confirm').some((c) => c.proposal_id === step2.proposal_id), 'confirm of step 2');
    // A plan step's run keeps the plan's proposal and option ids; plan_step tells the steps apart.
    run2 = await until(() => got('run').find((r) => r.plan_step === 1 && r.skill_id === 'cheese_in_bowl' && r.phase === 'executing' && r.sim_step > 5), 'run 2 executing', 8000);
    assert.notEqual(run2.run_id, run1.run_id);
    await waitScreen('executing', 3000);
  });

  let estopMsg;
  await step('Space mid-run latches STOPPED; estop goes out on /ws/safety; estop.ack comes back', async () => {
    const t0 = Date.now();
    await page.keyboard.press(' ');
    await waitScreen('stopped', 1000);
    assert.ok(Date.now() - t0 < 600, 'latched quickly');
    assert.equal(await page.locator('.stopped h1').innerText(), 'STOPPED');
    estopMsg = await until(() => sent('safety', 'estop')[0], 'estop sent');
    assert.equal(estopMsg.source, 'button');
    const ack = await until(() => got('estop.ack').find((a) => a.estop_seq === estopMsg.estop_seq), 'estop.ack', 3000);
    assert.ok(Number.isInteger(ack.stopped_at_step), 'stopped_at_step');
    assert.ok(ack.lat_ms && 'phone_gw' in ack.lat_ms, 'lat_ms');
    if (verbose) console.log('     ack:', JSON.stringify(ack));
    await page.waitForFunction(() => document.querySelector('.stopped .status')?.dataset.acked === '1', null, { timeout: 3000 });
    const n = sent('safety', 'estop').length;
    await sleep(400);
    assert.equal(sent('safety', 'estop').length, n, 'no resends after the ack');
    await shot(page, '03_phone_stopped.png');
  });

  await step('the run is aborted, the gateway is ESTOPPED, the hb says latched', async () => {
    await until(() => runMsgs(run2.run_id).some((r) => r.phase === 'aborted'), 'run 2 aborted', 3000);
    const st = await gw.state();
    assert.equal(st.mode, 'ESTOPPED');
    await until(() => got('state').some((s) => s.mode === 'ESTOPPED' && s.estop_seq === estopMsg.estop_seq), 'state ESTOPPED with the phone estop_seq');
    await until(() => sent('safety', 'hb').at(-1)?.latched === true, 'hb latched');
  });

  await step('a long blink on "Clear stop?" sends estop.reset; back to IDLE; run 2 not resumed', async () => {
    await page.locator('.stopped .clear').waitFor({ state: 'visible', timeout: 4000 });
    await sleep(300);
    await holdEnter(900);
    const r = await until(() => sent('safety', 'estop.reset')[0], 'estop.reset');
    assert.equal(r.estop_seq, estopMsg.estop_seq);
    await waitScreen('idle', 4000);
    await until(async () => (await gw.state()).mode === 'IDLE', 'gateway IDLE');
    const nRun = runMsgs(run2.run_id).length;
    await sleep(1500);
    const later = runMsgs(run2.run_id).slice(nRun);
    assert.ok(!later.some((m) => m.phase === 'executing'), 'aborted run never resumes');
    const st = await gw.state();
    assert.equal(st.mode, 'IDLE');
    assert.ok(!st.run || st.run.run_id !== run2.run_id || st.run.phase === 'aborted', 'no live run 2');
    assert.equal(await screen(), 'idle');
  });

  await step('presenter HUD received the trace and shows it', async () => {
    const kinds = new Set(presGot('trace').map((t) => t.kind));
    if (verbose) console.log('     trace kinds:', [...kinds].join(', '));
    for (const k of ['planner', 'validator', 'run', 'estop']) assert.ok(kinds.has(k), `trace kind ${k} (got ${[...kinds]})`);
    assert.ok(presGot('estop.ack').length >= 1 || presGot('state').some((s) => s.mode === 'ESTOPPED'), 'presenter saw the stop');
    await pres.waitForFunction(() => document.querySelectorAll('#trace li').length > 3, null, { timeout: 3000 });
    await pres.waitForFunction(() => document.getElementById('mode').textContent === 'IDLE', null, { timeout: 3000 });
    // The phone's stop latency split reaches the presenter through the estop trace.
    await pres.waitForFunction(() => /ms/.test(document.getElementById('estop-last').textContent), null, { timeout: 3000 });
    assert.match(await pres.locator('#run').innerText(), /cheese_in_bowl aborted/);
    await shot(pres, '04_presenter.png');
  });

  await step('a "double" option (near the user, first use) takes exactly two long blinks', async () => {
    // The double confirm lives on the gateway (validate_confirm): the phone sends every long blink;
    // the gateway answers the first with the same card and a fresh nonce.
    const n0 = got('proposal').length;
    await page.fill('#utter-text', 'bring me the bowl');
    await page.press('#utter-text', 'Enter');
    const card = await until(() => got('proposal').slice(n0).find((p) => p.options.some((o) => o.skill_ids?.includes('bowl_to_user'))), 'bowl_to_user proposal');
    const opt = card.options.find((o) => o.skill_ids?.includes('bowl_to_user'));
    assert.equal(opt.confirm, 'double', JSON.stringify(opt));
    await page.locator('#screen').click({ position: { x: 5, y: 5 } });
    await waitScreen('choose');
    await page.waitForFunction(([pid, id]) => document.querySelector(`.card.focused[data-proposal="${pid}"]`)?.dataset.option === id,
      [card.proposal_id, opt.option_id]);
    await sleep(600);
    const nConf = sent('phone', 'ui.confirm').length;
    await holdEnter(900);
    const first = await until(() => sent('phone', 'ui.confirm').slice(nConf)[0], 'first ui.confirm');
    assert.equal(first.nonce, card.nonce);
    const again = await until(() => got('proposal').find((p) => p.proposal_id === card.proposal_id && p.nonce !== card.nonce), 'second-confirm card (fresh nonce)');
    await until(() => got('notice').some((nm) => /once more/i.test(nm.text)), '"once more" notice');
    assert.equal((await gw.state()).mode, 'AWAITING_CONFIRM', 'one blink did not start anything');
    await sleep(700); // the arm delay and focus dwell start again
    await holdEnter(900);
    const second = await until(() => sent('phone', 'ui.confirm').slice(nConf)[1], 'second ui.confirm');
    assert.equal(second.nonce, again.nonce);
    const run = await until(() => got('run').find((r) => r.proposal_id === card.proposal_id && r.phase === 'executing'), 'bowl_to_user executing', 8000);
    assert.equal(run.skill_id, 'bowl_to_user');
    assert.equal(sent('phone', 'ui.confirm').length - nConf, 2, 'two long blinks in total');
    const end = await until(() => runMsgs(run.run_id).find((r) => ['done', 'failed', 'uncertain', 'aborted'].includes(r.phase)), 'bowl_to_user end', 20000);
    assert.equal(end.phase, 'done', JSON.stringify(end));
    await until(async () => (await gw.state()).mode === 'IDLE', 'gateway IDLE after bowl_to_user', 20000);
  });

  await step('face lost mid-run pauses; the gateway\'s "Resume?" card resumes the same run to done', async () => {
    // A third run from the menu. In keyboard mode the heartbeat's face_present is "page visible",
    // so hiding the page is how this test takes the face away (design 5.6, face lost 1000 ms).
    const n0 = got('proposal').length;
    await page.keyboard.press('b');
    // The deterministic menu, or the planner's prefetched suggestions (7.1 fast path) once the
    // gateway has had time to prefetch in IDLE; either is fine here.
    await until(() => got('proposal').length > n0, 'a proposal after B');
    await waitScreen('choose');
    await sleep(300);
    const menu = got('proposal').at(-1);
    const opt = menu.options.find((o) => !o.locked && o.option_id === menu.focus) || menu.options.find((o) => !o.locked);
    assert.ok(opt, `an unlocked menu option in ${JSON.stringify(menu.options)}`);
    await page.waitForFunction((id) => document.querySelector('.card.focused')?.dataset.option === id, opt.option_id);
    await sleep(600); // arm delay and focus dwell
    await holdEnter(900);
    await until(() => sent('phone', 'ui.confirm').some((c) => c.proposal_id === menu.proposal_id), 'confirm of the menu option');
    const run3 = await until(() => got('run').find((r) => r.proposal_id === menu.proposal_id && r.phase === 'executing' && r.sim_step > 5), 'run 3 executing', 8000);
    await page.evaluate(() => {
      Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => 'hidden' });
    });
    await until(() => sent('safety', 'hb').at(-1)?.face_present === false, 'hb face_present false');
    await until(() => runMsgs(run3.run_id).some((r) => r.phase === 'paused'), 'run 3 paused', 4000);
    assert.equal((await gw.state()).mode, 'PAUSED');
    const resume = await until(() => got('proposal').find((p) => p.options.some((o) => o.option_id === 'resume')), 'resume proposal');
    assert.match(resume.options[0].why, /can't see you/);
    const stepAtPause = runMsgs(run3.run_id).filter((r) => r.phase === 'paused').at(-1).sim_step;
    await sleep(600);
    assert.ok(!runMsgs(run3.run_id).some((r) => r.phase === 'executing' && r.sim_step > stepAtPause + 2), 'no motion while paused');
    await page.evaluate(() => { delete document.visibilityState; });
    await until(() => sent('safety', 'hb').at(-1)?.face_present === true, 'hb face_present true');
    await waitScreen('choose', 3000);
    await page.waitForFunction(() => document.querySelector('.card.focused')?.dataset.option === 'resume', null, { timeout: 3000 });
    await sleep(600);
    await holdEnter(900);
    const c = await until(() => sent('phone', 'ui.confirm').find((x) => x.proposal_id === resume.proposal_id), 'confirm of resume');
    assert.equal(c.option_id, 'resume');
    const nPaused = runMsgs(run3.run_id).findIndex((r) => r.phase === 'paused');
    await until(() => runMsgs(run3.run_id).slice(nPaused).some((r) => r.phase === 'executing'), 'run 3 executing again', 4000);
    const end = await until(() => runMsgs(run3.run_id).find((r) => ['done', 'failed', 'uncertain', 'aborted'].includes(r.phase)), 'run 3 end', 20000);
    assert.equal(end.phase, 'done', JSON.stringify(end));
    await until(async () => (await gw.state()).mode === 'IDLE', 'gateway IDLE after run 3', 20000);
  });

  await step('no uncaught errors in the pages', async () => {
    const real = consoleErrors.filter((e) => !/favicon|service worker/i.test(e));
    assert.deepEqual(real, []);
  });
} catch {
  exitCode = 1;
  // The last messages each way, to see what the gateway said before the failure.
  const brief = (f) => `${f.dir === 'in' ? '<-' : '->'} ${f.sock} ${JSON.stringify(f.m, (k, v) => (['id', 'v', 'ts', 'sess'].includes(k) ? undefined : v)).slice(0, 300)}`;
  console.log(phoneFrames.filter((f) => !['hb', 'ping', 'pong', 'gesture'].includes(f.m.type)).slice(-25).map(brief).join('\n'));
  if (shotDir) await page.screenshot({ path: path.join(shotDir, 'failure_phone.png') }).catch(() => {});
} finally {
  await browser.close();
  await gw.close();
}

const passed = results.filter((r) => r.ok).length;
console.log(`\ne2e_full: ${passed} passed, ${results.length - passed} failed`);
console.log(`frames: phone ${phoneFrames.length}, presenter ${presFrames.length}`);
if (exitCode && !verbose) console.log(`gateway log (tail):\n${gw.log.join('').split('\n').slice(-40).join('\n')}`);
process.exit(exitCode);
