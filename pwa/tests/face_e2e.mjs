// Optional check of js/face.js against the real MediaPipe Face Landmarker in
// headless Chromium, with a fake camera that shows a still portrait.
//
// Needs: vendor/ (sh scripts/vendor.sh) and network access once, to download
// MediaPipe's own test portrait (Apache-2.0) into tests/.cache/. Exits 0 with
// "skipped" when either is missing, so CI without network stays green.
//
// Run: node tests/face_e2e.mjs

import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { startStub } from './stub_gateway.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const CACHE = path.join(HERE, '.cache');
const PORTRAIT_URL = 'https://storage.googleapis.com/mediapipe-assets/portrait.jpg';

function findChromium() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  for (const root of [process.env.PLAYWRIGHT_BROWSERS_PATH, '/opt/pw-browsers'].filter(Boolean)) {
    if (!fs.existsSync(root)) continue;
    for (const d of fs.readdirSync(root).filter((x) => /^chromium-\d+$/.test(x)).sort().reverse()) {
      const p = path.join(root, d, 'chrome-linux', 'chrome');
      if (fs.existsSync(p)) return p;
    }
  }
  return undefined;
}

function skip(why) { console.log(`face e2e: skipped (${why})`); process.exit(0); }

if (!fs.existsSync(path.join(ROOT, 'vendor', 'manifest.json'))) skip('vendor/ missing: run sh scripts/vendor.sh');
fs.mkdirSync(CACHE, { recursive: true });
const jpg = path.join(CACHE, 'portrait.jpg');
if (!fs.existsSync(jpg)) {
  try {
    const r = await fetch(PORTRAIT_URL);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    fs.writeFileSync(jpg, Buffer.from(await r.arrayBuffer()));
  } catch (e) { skip(`cannot download the test portrait: ${e.message}`); }
}

const exe = findChromium();

/** Decode the JPEG in the browser and write a 640x480 I420 Y4M clip (Chrome's fake camera format). */
async function makeY4m(file, mirror) {
  if (fs.existsSync(file)) return;
  const b = await chromium.launch({ executablePath: exe });
  const p = await b.newPage();
  const b64 = fs.readFileSync(jpg).toString('base64');
  const planes = await p.evaluate(async ({ b64, mirror }) => {
    const img = new Image();
    img.src = `data:image/jpeg;base64,${b64}`;
    await img.decode();
    const W = 640, H = 480;
    const c = new OffscreenCanvas(W, H);
    const g = c.getContext('2d');
    g.fillStyle = '#282828';
    g.fillRect(0, 0, W, H);
    const s = H / img.height, w = img.width * s;
    if (mirror) { g.translate(W, 0); g.scale(-1, 1); }
    g.drawImage(img, (W - w) / 2, 0, w, H);
    const d = g.getImageData(0, 0, W, H).data;
    const Y = new Uint8Array(W * H), U = new Uint8Array(W * H / 4), V = new Uint8Array(W * H / 4);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      Y[y * W + x] = Math.max(0, Math.min(255, 0.257 * d[i] + 0.504 * d[i + 1] + 0.098 * d[i + 2] + 16));
    }
    for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2) {
      let u = 0, v = 0;
      for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
        const i = ((y + dy) * W + x + dx) * 4;
        u += -0.148 * d[i] - 0.291 * d[i + 1] + 0.439 * d[i + 2] + 128;
        v += 0.439 * d[i] - 0.368 * d[i + 1] - 0.071 * d[i + 2] + 128;
      }
      U[(y / 2) * (W / 2) + x / 2] = u / 4;
      V[(y / 2) * (W / 2) + x / 2] = v / 4;
    }
    const enc = (a) => { let s = ''; for (let i = 0; i < a.length; i += 0x8000) s += String.fromCharCode(...a.subarray(i, i + 0x8000)); return btoa(s); };
    return { Y: enc(Y), U: enc(U), V: enc(V) };
  }, { b64, mirror });
  await b.close();
  const frame = Buffer.concat([Buffer.from('FRAME\n'), ...['Y', 'U', 'V'].map((k) => Buffer.from(planes[k], 'base64'))]);
  fs.writeFileSync(file, Buffer.concat([Buffer.from('YUV4MPEG2 W640 H480 F30:1 Ip A1:1 C420jpeg\n'), ...Array(30).fill(frame)]));
}

async function track(y4m) {
  const stub = await startStub({});
  const b = await chromium.launch({
    executablePath: exe,
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${y4m}`],
  });
  try {
    const ctx = await b.newContext({ permissions: ['camera'] });
    const p = await ctx.newPage();
    await p.goto(`${stub.url}/?kws=0&nosw=1`);
    await p.click('[data-action="calib-default"]');
    await p.waitForFunction(() => globalThis.nod?.lastFrame?.face_present, null, { timeout: 90_000 });
    await p.waitForTimeout(1500);
    return await p.evaluate(() => {
      const f = nod.lastFrame;
      return { ...f, delegate: nod.tracker.delegate, vendored: nod.tracker.sources.vendored, hb: null };
    });
  } finally {
    await b.close();
    await stub.close();
  }
}

const plain = path.join(CACHE, 'face.y4m'), mirrored = path.join(CACHE, 'face_mirrored.y4m');
await makeY4m(plain, false);
await makeY4m(mirrored, true);

let failed = 0;
const t0 = Date.now();
try {
  const a = await track(plain);
  assert.equal(a.face_present, true);
  assert.equal(Object.keys(a.blendshapes).length, 52, 'all 52 blendshapes by name');
  for (const k of ['eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight']) assert.ok(Number.isFinite(a.blendshapes[k]), k);
  assert.ok(Number.isFinite(a.yaw_deg) && Math.abs(a.yaw_deg) < 20, `yaw ${a.yaw_deg}`);
  assert.ok(Number.isFinite(a.pitch_deg), 'pitch');
  assert.ok(a.track_conf >= 0.9, `track_conf ${a.track_conf}`);
  assert.equal(a.vendored, true);
  console.log(`ok   face tracked: ${a.delegate} delegate, yaw ${a.yaw_deg.toFixed(1)} deg, pitch ${a.pitch_deg.toFixed(1)} deg, ${a.fps.toFixed(1)} fps, jawOpen ${a.blendshapes.jawOpen.toFixed(2)}`);
  const m = await track(mirrored);
  // A mirrored camera frame flips the sign of yaw: why face.js never mirrors before inference.
  assert.ok(Math.sign(m.yaw_deg) === -Math.sign(a.yaw_deg) || Math.abs(a.yaw_deg) < 0.5, `yaw ${a.yaw_deg} vs mirrored ${m.yaw_deg}`);
  console.log(`ok   mirrored frame flips yaw: ${a.yaw_deg.toFixed(1)} -> ${m.yaw_deg.toFixed(1)} deg`);
} catch (e) {
  failed = 1;
  console.log(`FAIL ${e.message}`);
}
console.log(`face e2e: ${failed ? 'failed' : 'passed'} in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
process.exit(failed);
