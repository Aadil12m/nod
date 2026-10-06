// End-to-end smoke test: real Chromium, the real PWA, the stub gateway.
//
// Uses the keyboard fallback (no camera): B opens the menu, Right moves focus,
// holding Enter is a long blink that confirms, Space is STOP. Asserts the
// latched STOPPED screen, that estop was resent every ~100 ms until the
// (deliberately late) estop.ack, that resending stopped after the ack, and that
// a long blink on "Clear stop?" sends estop.reset and returns to idle. Then the
// presenter page: STOP from the presenter.
//
// Run: node tests/e2e.mjs        (exit code 0 on success)

import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { chromium } from 'playwright-core';
import { startStub } from './stub_gateway.mjs';

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

const ACK_DELAY = 450;
const results = [];
async function step(name, fn) {
  const t0 = Date.now();
  try {
    await fn();
    results.push({ name, ok: true, ms: Date.now() - t0 });
    console.log(`ok   ${name} (${Date.now() - t0} ms)`);
  } catch (e) {
    results.push({ name, ok: false, err: e });
    console.log(`FAIL ${name}\n     ${e?.message?.split('\n').join('\n     ')}`);
    throw e;
  }
}

const stub = await startStub({ ackDelayMs: ACK_DELAY, stepMs: 300, maxSteps: 400 });
const browser = await chromium.launch({ executablePath: findChromium(), headless: true });
const ctx = await browser.newContext({ viewport: { width: 412, height: 915 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();
const consoleErrors = [];
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('pageerror', (e) => consoleErrors.push(String(e)));

const types = (kind) => stub.received[kind].map((m) => m.type);
const ofType = (kind, t) => stub.received[kind].filter((m) => m.type === t);
const screen = () => page.locator('#screen').getAttribute('data-screen');
const waitScreen = (s, timeout = 5000) => page.waitForFunction((x) => document.getElementById('screen').dataset.screen === x, s, { timeout });
const holdEnter = async (ms) => { await page.keyboard.down('Enter'); await page.waitForTimeout(ms); await page.keyboard.up('Enter'); };
const sleep = (ms) => page.waitForTimeout(ms);

let exitCode = 0;
try {
  await step('page loads, connects both sockets and says hello', async () => {
    await page.goto(`${stub.url}/?nocam=1&kws=0&nosw=1`);
    await waitScreen('idle');
    await page.waitForFunction(() => document.getElementById('conn').dataset.ok === '1');
    const hello = ofType('phone', 'hello')[0];
    assert.ok(hello, 'hello sent');
    assert.equal(hello.v, 1);
    assert.equal(hello.role, 'user');
    assert.equal(hello.calib.profile_id, 'cal_default');
    assert.equal(hello.seq, 1);
  });

  await step('REPLAY badge shows in mock mode (from /healthz)', async () => {
    await page.locator('#replay-badge').waitFor({ state: 'visible', timeout: 3000 });
  });

  await step('heartbeats every 100 ms on /ws/safety with the session id', async () => {
    const n0 = ofType('safety', 'hb').length;
    await sleep(1000);
    const hbs = ofType('safety', 'hb');
    const n = hbs.length - n0;
    assert.ok(n >= 7 && n <= 13, `${n} hb in 1 s`);
    const last = hbs[hbs.length - 1];
    assert.equal(last.sess, 's_stub');
    assert.equal(last.latched, false);
    assert.equal(last.face_present, true); // keyboard mode: page visible
    const seqs = hbs.map((m) => m.seq);
    assert.ok(seqs.every((s, i) => i === 0 || s > seqs[i - 1]), 'seq increases');
  });

  await step('B (brow raise) opens the menu: 2 options + None of these', async () => {
    await page.keyboard.press('b');
    await waitScreen('choose');
    assert.equal(ofType('phone', 'ui.menu')[0].source, 'brow_raise');
    assert.equal(await page.locator('.card').count(), 3);
    assert.match(await page.locator('.card').nth(2).innerText(), /None of these/);
    assert.match(await page.locator('.card.focused').innerText(), /Put the bowl on the plate/);
    assert.match(await page.locator('.card').first().innerText(), /19\/20 in tests/);
    // Telemetry for the brow raise gesture.
    assert.ok(ofType('phone', 'gesture').some((g) => g.name === 'brow_raise' && g.phase === 'fired'));
  });

  await step('Right (head turn) moves focus and sends ui.focus', async () => {
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => document.querySelector('.card.focused')?.dataset.option === 'o2');
    const f = ofType('phone', 'ui.focus').at(-1);
    assert.equal(f.option_id, 'o2');
    assert.ok(f.proposal_id.startsWith('p_'));
  });

  await step('a short Enter (natural blink) does not confirm', async () => {
    await sleep(700); // past the turn lockout and the 300 ms dwell
    await holdEnter(250);
    await sleep(300);
    assert.equal(ofType('phone', 'ui.confirm').length, 0);
    assert.equal(await screen(), 'choose');
  });

  await step('holding Enter (long blink) fills the ring and confirms on release', async () => {
    await page.keyboard.down('Enter');
    await sleep(500);
    const ringActive = await page.locator('.card.focused .ring.active').count();
    const offset = Number(await page.locator('.card.focused .ring-fg').getAttribute('stroke-dashoffset'));
    await sleep(400);
    assert.equal(ofType('phone', 'ui.confirm').length, 0, 'no confirm while eyes are still closed');
    await page.keyboard.up('Enter');
    await page.waitForFunction(() => ['confirming', 'countdown', 'executing'].includes(document.getElementById('screen').dataset.screen));
    assert.equal(ringActive, 1, 'ring visible during the hold');
    assert.ok(offset < 2 * Math.PI * 44, 'ring progressed');
    const c = ofType('phone', 'ui.confirm')[0];
    assert.ok(c, 'ui.confirm sent');
    assert.equal(c.option_id, 'o2');
    assert.equal(c.gesture, 'long_blink');
    assert.equal(c.nonce, stub.g.lastProposal.nonce);
    assert.equal(c.proposal_id, stub.g.lastProposal.proposal_id);
    assert.ok(c.held_ms >= 700, `held ${c.held_ms}`);
    assert.ok(c.focused_at >= c.shown_at);
    assert.ok(c.ts - c.focused_at >= 300, 'dwell respected');
  });

  await step('countdown, then executing with progress and the STOP hint', async () => {
    await waitScreen('executing', 4000);
    await page.locator('.stop-hint.big').waitFor();
    assert.match(await page.locator('#screen h1').innerText(), /Open the top drawer/);
    await page.waitForFunction(() => Number(document.querySelector('.progress')?.getAttribute('aria-valuenow')) > 0, null, { timeout: 3000 });
  });

  let tStop;
  await step('Space (STOP) latches STOPPED at once and sends estop', async () => {
    tStop = Date.now();
    await page.keyboard.press(' ');
    await waitScreen('stopped', 1000);
    const latchMs = Date.now() - tStop;
    assert.ok(latchMs < 500, `latched after ${latchMs} ms`);
    await page.locator('.stopped h1').waitFor({ state: 'visible' });
    assert.equal(await page.locator('.stopped h1').innerText(), 'STOPPED');
    // Full-screen red: covers the viewport. (Polled: the screen re-renders as gateway messages arrive.)
    await page.waitForFunction(() => {
      const el = document.querySelector('.stopped');
      if (!el) return false;
      const r = el.getBoundingClientRect();
      return getComputedStyle(el).backgroundColor === 'rgb(213, 0, 0)' && r.width >= innerWidth && r.height >= innerHeight;
    }, null, { timeout: 1000 });
  });

  await step('estop resent every ~100 ms until the late estop.ack, then no more', async () => {
    await page.waitForFunction(() => document.querySelector('.stopped .status')?.dataset.acked === '1', null, { timeout: 3000 });
    const es = ofType('safety', 'estop');
    assert.ok(es.length >= 3, `${es.length} estop messages before the ack (ack delayed ${ACK_DELAY} ms)`);
    const gaps = es.slice(1).map((m, i) => m.ts - es[i].ts);
    assert.ok(gaps.every((g) => g >= 60 && g <= 250), `resend gaps ${gaps}`);
    console.log(`     ${es.length} estop copies, gaps ${gaps.join(', ')} ms`);
    assert.ok(es.every((m) => m.estop_seq === 1 && m.source === 'button'));
    const n = es.length;
    await sleep(500);
    assert.equal(ofType('safety', 'estop').length, n, 'resending stopped after the ack');
    assert.equal(stub.g.mode, 'ESTOPPED');
    assert.equal(stub.g.run.phase, 'aborted');
    const hb = ofType('safety', 'hb').at(-1);
    assert.equal(hb.latched, true);
  });

  await step('"Clear stop?" appears; a long blink sends estop.reset; back to idle, run not resumed', async () => {
    await page.locator('.stopped .clear').waitFor({ state: 'visible', timeout: 3000 });
    await holdEnter(900);
    await waitScreen('idle', 3000);
    const r = ofType('safety', 'estop.reset')[0];
    assert.ok(r, 'estop.reset sent');
    assert.equal(r.estop_seq, 1);
    assert.equal(r.gesture, 'long_blink');
    assert.equal(stub.g.mode, 'IDLE');
    assert.equal(ofType('phone', 'ui.confirm').length, 1, 'no new confirm');
  });

  await step('typed utterance sends utter.text and shows options', async () => {
    await page.fill('#utter-text', "I'm a bit hungry");
    await page.press('#utter-text', 'Enter');
    await waitScreen('choose', 3000);
    const u = ofType('phone', 'utter.text')[0];
    assert.equal(u.text, "I'm a bit hungry");
    assert.ok(Number.isInteger(u.utt_id));
  });

  await step('STOP while choosing is a cancel, not an e-stop', async () => {
    const n = ofType('safety', 'estop').length;
    await page.locator('#screen').click({ position: { x: 5, y: 5 } }); // move focus out of the text box
    await page.keyboard.press(' ');
    await waitScreen('idle', 2000);
    assert.equal(ofType('safety', 'estop').length, n);
    assert.equal(ofType('phone', 'ui.cancel').at(-1).source, 'button');
  });

  await step('a dropped safety socket reconnects and keeps heartbeating', async () => {
    stub.dropSafety();
    await sleep(1200);
    const n0 = ofType('safety', 'hb').length;
    await sleep(500);
    assert.ok(ofType('safety', 'hb').length - n0 >= 3, 'heartbeats resumed');
  });

  await step('presenter: HUD connects, shows state; Space sends a presenter estop', async () => {
    const pp = await ctx.newPage();
    pp.on('pageerror', (e) => consoleErrors.push(`presenter: ${e}`));
    await pp.goto(`${stub.url}/presenter?noviewer=1`);
    await pp.waitForFunction(() => document.getElementById('conn').dataset.ok === '1');
    await pp.waitForFunction(() => document.getElementById('mode').textContent === 'IDLE');
    await pp.locator('#replay').waitFor({ state: 'visible' });
    assert.match(await pp.locator('#estop-p').innerText(), /63 ms/);
    await pp.keyboard.press(' ');
    await pp.waitForFunction(() => document.getElementById('mode').textContent === 'ESTOPPED', null, { timeout: 3000 });
    const pe = ofType('presenter', 'estop')[0];
    assert.ok(pe, 'presenter estop sent');
    assert.equal(pe.source, 'presenter');
    assert.equal(pe.estop_seq, 2);
    // The phone follows the gateway into the latched screen.
    await waitScreen('stopped', 2000);
    await pp.close();
  });

  await step('no uncaught errors in the pages', async () => {
    const real = consoleErrors.filter((e) => !/favicon|service worker/i.test(e));
    assert.deepEqual(real, []);
  });
} catch {
  exitCode = 1;
} finally {
  await browser.close();
  await stub.close();
}

const passed = results.filter((r) => r.ok).length;
console.log(`\ne2e: ${passed} passed, ${results.length - passed} failed`);
console.log(`messages seen by the stub: phone ${types('phone').length}, safety ${types('safety').length}, presenter ${types('presenter').length}`);
process.exit(exitCode);
