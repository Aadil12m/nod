// node --test: calibration math and flow (design 6.3) on synthetic users.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  percentile, median, mean, sd, setupCheck, analyzeNeutral, closures, thresholdsFromReps, longBlinkMs,
  computeProfile, profileConfig, defaultProfile, rezero, BaselineTracker, scoreValidation, CalibrationFlow,
  buildSchedule, PLANS, loadProfile, saveProfile, clearProfile, PROFILE_KEY,
} from '../js/calibrate.js';
import { GestureMachine } from '../js/gestures.js';
import { Scene, rng, run, fired } from './synth.mjs';

test('statistics helpers', () => {
  assert.equal(percentile([1, 2, 3, 4, 5], 50), 3);
  assert.equal(percentile([0, 10], 99), 9.9);
  assert.equal(percentile([], 50), 0);
  assert.equal(median([5, 1, 3]), 3);
  assert.equal(mean([1, 2, 3]), 2);
  assert.ok(Math.abs(sd([2, 4, 4, 4, 5, 5, 7, 9]) - 2.138) < 0.001);
});

test('setup check needs 15 fps and 90% face presence', () => {
  const s = new Scene(3000, { seed: 1 });
  assert.equal(setupCheck([...s.frames(30)]).ok, true);
  const slow = setupCheck([...s.frames(10)]);
  assert.equal(slow.ok, false);
  assert.ok(slow.fps < 11);
  const m = new Scene(3000, { seed: 1 }).missing(0, 600);
  const r = setupCheck([...m.frames(30)]);
  assert.equal(r.ok, false);
  assert.ok(r.presence < 0.9);
});

test('long blink = p99 natural closure + 250 ms, clamped to 600-1200', () => {
  assert.equal(longBlinkMs([200, 250, 300, 350, 400]), Math.round(percentile([200, 250, 300, 350, 400], 99) + 250));
  assert.equal(longBlinkMs([100, 120, 130, 140, 150]), 600);
  assert.equal(longBlinkMs([900, 1000, 1100, 1000, 950]), 1200);
  assert.equal(longBlinkMs([300]), 700); // too few samples: default
});

test('on = baseline + 50% of the gap, off = baseline + 30%', () => {
  const t = thresholdsFromReps(0.1, 0.9);
  assert.ok(Math.abs(t.on - 0.5) < 1e-9);
  assert.ok(Math.abs(t.off - 0.34) < 1e-9);
});

test('closure measurement matches the synthetic blink durations', () => {
  const s = new Scene(5000, { seed: 2, noise: 0 }).blink(500, 300).blink(2000, 900);
  const c = closures([...s.frames(30)], 'blink', 0.55, 0.35);
  assert.equal(c.length, 2);
  assert.ok(Math.abs(c[0].dur - 270) < 40, `${c[0].dur}`);
  assert.ok(Math.abs(c[1].dur - 870) < 40, `${c[1].dur}`);
});

test('neutral analysis gives baselines and the neutral yaw', () => {
  const s = new Scene(5000, { seed: 3 }).turn(0, 6000, 6);
  const n = analyzeNeutral([...s.frames(30)].filter((f) => f.t_ms > 300 && f.t_ms < 4500));
  assert.ok(Math.abs(n.yaw.mean - 6) < 0.5);
  assert.ok(n.blink.mean > 0.02 && n.blink.mean < 0.08);
  assert.ok(n.jaw.mean < 0.08);
});

/** Run a whole CalibrationFlow with a simulated user who follows the prompts. */
function calibrateUser({ seed = 1, fps = 30, plan = 'quick', yawLeft = 28, jawPeak = 0.8, talkPeak = 0.45,
  browPeak = 0.75, longBlink = 1100, onWindow } = {}) {
  const r = rng(seed);
  const flow = new CalibrationFlow({ plan, id: `cal_test_${seed}` });
  const total = 5000 + buildSchedule(flow.plan).reduce((s, w) => s + w.dur, 0) + 3000;
  const scene = new Scene(total, { seed, noise: 0.03 });
  let lastKey = null;
  const step = 1000 / fps;
  for (let t = 0; t < total && !flow.done; t += step) {
    const v = flow.view(t);
    const key = `${v.step}:${v.gesture}:${v.i}`;
    if (key !== lastKey && flow.winStart !== null) {
      lastKey = key;
      const w = flow.schedule[flow.idx];
      const at = t + 300;
      if (onWindow) onWindow(w, at, scene);
      else if (w.phase === 'natural') {
        scene.naturalBlinks(t, t + w.dur, 16, r);
        scene.talking(t, t + w.dur, r, talkPeak);
      } else if (w.phase === 'rep' || w.phase === 'validate') {
        if (w.gesture === 'long_blink') scene.blink(at, longBlink);
        if (w.gesture === 'brow_raise') scene.browRaise(at, 900, browPeak);
        if (w.gesture === 'jaw_hold') scene.jawHold(at, 900, jawPeak);
        if (w.gesture === 'turn_left') scene.turn(at - 100, 1400, yawLeft);
        if (w.gesture === 'turn_right') scene.turn(at - 100, 1400, -yawLeft * 0.9);
      } else if (w.phase === 'rest' || w.phase === 'vrest') {
        if (r() < 0.5) scene.blink(t + 200, 150 + r() * 200); // natural blink while resting
      }
    }
    flow.feed({ t_ms: t, ...scene.sample(t) });
  }
  return flow;
}

test('full calibration flow on a typical user produces a sensible profile that validates', () => {
  const flow = calibrateUser({ seed: 4 });
  assert.equal(flow.state, 'done');
  const p = flow.profile;
  assert.equal(p.id, 'cal_test_4');
  const c = p.config;
  assert.ok(c.blink_on > 0.35 && c.blink_on < 0.7, `blink_on ${c.blink_on}`);
  assert.ok(c.blink_off < c.blink_on);
  assert.ok(c.long_blink_ms >= 600 && c.long_blink_ms <= 1000, `long ${c.long_blink_ms}`);
  // Jaw "on" sits above the natural (talking) 99th percentile.
  assert.ok(c.jaw_on > p.detail.jaw.natural_p99, `jaw_on ${c.jaw_on} p99 ${p.detail.jaw.natural_p99}`);
  assert.equal(c.yaw_sign, 1);
  assert.ok(c.yaw_on_deg >= 12 && c.yaw_off_deg === c.yaw_on_deg / 2);
  assert.deepEqual(c.disabled, []);
  assert.equal(flow.validation.pass, true, JSON.stringify(flow.validation));
  assert.equal(flow.validation.falseConfirms, 0);
  // The profile drives a GestureMachine that still ignores natural blinking.
  const s = new Scene(5 * 60_000, { seed: 5, noise: 0.03 }).naturalBlinks(0, 5 * 60_000, 18, rng(5));
  assert.equal(fired(run(new GestureMachine(profileConfig(p)), s, 30), 'long_blink').length, 0);
});

test('a camera that reports yaw with the opposite sign is detected', () => {
  const flow = calibrateUser({ seed: 6, yawLeft: -30 });
  assert.equal(flow.profile.config.yaw_sign, -1);
  assert.equal(flow.validation.pass, true, JSON.stringify(flow.validation));
});

test('weak jaw (bulbar onset): jaw stop is disabled with a warning', () => {
  const flow = calibrateUser({ seed: 7, jawPeak: 0.42, talkPeak: 0.4 });
  assert.ok(flow.profile.config.disabled.includes('jaw_hold'));
  assert.ok(flow.profile.warnings.some((w) => w.startsWith('jaw_hold')));
});

test('small comfortable head turn disables turns', () => {
  const flow = calibrateUser({ seed: 8, yawLeft: 11 });
  assert.ok(flow.profile.config.disabled.includes('turn_left'));
});

test('practice blinks that are too short give a warning', () => {
  const flow = calibrateUser({ seed: 9, longBlink: 500 });
  assert.ok(flow.profile.warnings.some((w) => w.startsWith('long_blink: only')));
});

test('setup check failure keeps the flow in setup with a message', () => {
  const flow = new CalibrationFlow({ plan: 'quick', maxSetupTries: 2 });
  const s = new Scene(10_000, { seed: 10 });
  let v;
  for (const f of s.frames(10)) { v = flow.feed(f); if (flow.done) break; }
  assert.equal(flow.state, 'failed');
  assert.match(v.message, /too slow/);
});

test('validation scoring: 90% hits, zero false confirms, at most one false menu', () => {
  const trials = [
    { intended: 'long_blink', fired: ['long_blink'] },
    { intended: 'brow_raise', fired: ['brow_raise'] },
    { intended: 'turn_left', fired: ['turn_left'] },
    { intended: 'turn_right', fired: ['turn_right'] },
    { intended: 'jaw_hold', fired: ['jaw_hold'] },
  ];
  assert.equal(scoreValidation(trials).pass, true);
  assert.equal(scoreValidation(trials, ['long_blink']).pass, false);
  assert.equal(scoreValidation(trials, ['brow_raise']).pass, true);
  assert.equal(scoreValidation(trials, ['brow_raise', 'brow_raise']).pass, false);
  assert.equal(scoreValidation([...trials.slice(1), { intended: 'long_blink', fired: [] }]).pass, false);
});

test('re-zero updates the neutral yaw; baseline drift moves thresholds slowly and boundedly', () => {
  const p = calibrateUser({ seed: 11 }).profile;
  const s = new Scene(6000, { seed: 12 }).turn(0, 7000, 4);
  const z = rezero(p, [...s.frames(30)].filter((f) => f.t_ms > 400 && f.t_ms < 5000));
  assert.ok(Math.abs(z.config.neutral_yaw_deg - 4) < 0.6);
  assert.equal(z.config.blink_on, p.config.blink_on);

  const bt = new BaselineTracker(p);
  const before = bt.config();
  // Brows resting 0.1 higher than at calibration, for 5 minutes while idle.
  for (let t = 0; t < 300_000; t += 100) bt.update({ t_ms: t, face_present: true, blendshapes: { browInnerUp: p.neutral.brow.mean + 0.1 } }, true);
  const after = bt.config();
  assert.ok(after.brow_on > before.brow_on + 0.08 && after.brow_on <= before.brow_on + 0.1 + 1e-9);
  // Not idle: no drift.
  const bt2 = new BaselineTracker(p);
  for (let t = 0; t < 300_000; t += 100) bt2.update({ t_ms: t, face_present: true, blendshapes: { browInnerUp: 0.25 } }, false);
  assert.deepEqual(bt2.config(), before);
});

test('profile storage survives missing or throwing localStorage', () => {
  const mem = new Map();
  const store = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v), removeItem: (k) => mem.delete(k) };
  const p = defaultProfile();
  assert.equal(loadProfile(store), null);
  assert.equal(saveProfile(p, store), true);
  assert.deepEqual(loadProfile(store), p);
  mem.set(PROFILE_KEY, '{not json');
  assert.equal(loadProfile(store), null);
  assert.equal(clearProfile(store), true);
  const bad = { getItem() { throw new Error('SecurityError'); }, setItem() { throw new Error('QuotaExceeded'); }, removeItem() { throw new Error('x'); } };
  assert.equal(loadProfile(bad), null);
  assert.equal(saveProfile(p, bad), false);
  assert.equal(clearProfile(bad), false);
  assert.equal(loadProfile(undefined), null);
});

test('computeProfile with no repetitions keeps defaults and warns', () => {
  const s = new Scene(4000, { seed: 13 });
  const frames = [...s.frames(30)];
  const p = computeProfile({ neutral: frames, natural: frames, reps: {} }, { id: 'x' });
  assert.equal(p.config.blink_on, undefined);
  assert.ok(p.warnings.length >= 3);
  assert.equal(PLANS.full.reps.long_blink, 5);
});
