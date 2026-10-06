// node --test: gesture state machine on synthetic traces (design 4.6, 6.1, 6.2).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GestureMachine, DEFAULTS, blinkScore, browScore, jawScore } from '../js/gestures.js';
import { Scene, run, fired, rng } from './synth.mjs';

const MIN = 60_000;
const FPS = [15, 24, 30];

test('signals are read by blendshape name', () => {
  assert.equal(blinkScore({ eyeBlinkLeft: 0.9, eyeBlinkRight: 0.6 }), 0.6);
  assert.equal(browScore({ browInnerUp: 0.2, browOuterUpLeft: 0.8, browOuterUpRight: 0.6 }), 0.7);
  assert.equal(jawScore({ jawOpen: 0.5 }), 0.5);
  assert.equal(jawScore({}), 0);
  assert.equal(blinkScore({ eyeBlinkLeft: NaN, eyeBlinkRight: 1 }), 0);
});

for (const fps of FPS) {
  test(`natural blinking for 10 minutes gives zero confirms at ${fps} fps`, () => {
    for (const seed of [1, 2, 3]) {
      const r = rng(seed);
      const s = new Scene(10 * MIN, { seed, noise: 0.04 });
      s.naturalBlinks(0, 10 * MIN, 12 + r() * 8, r);
      const ev = run(new GestureMachine(), s, fps);
      assert.ok(s.blinks.length >= 100, `generated ${s.blinks.length} blinks`);
      assert.equal(fired(ev, 'long_blink').length, 0, `seed ${seed}: false confirm`);
      assert.equal(fired(ev, 'jaw_hold').length, 0);
      assert.equal(fired(ev, 'brow_raise').length, 0);
    }
  });
}

test('worst case natural blinks (all 400 ms, 20 a minute) never confirm', () => {
  const s = new Scene(10 * MIN, { seed: 9, noise: 0.05 });
  s.naturalBlinks(0, 10 * MIN, 20, rng(9), [400, 400]);
  for (const fps of FPS) assert.equal(fired(run(new GestureMachine(), s, fps), 'long_blink').length, 0);
});

test('a 10-minute talking trace (jaw, blinks) gives zero confirms and zero stops', () => {
  const r = rng(42);
  const s = new Scene(10 * MIN, { seed: 42, noise: 0.03 });
  s.naturalBlinks(0, 10 * MIN, 18, r);
  s.talking(0, 10 * MIN, r, 0.6);
  for (const fps of FPS) {
    for (const speaking of [false, true]) {
      const gm = new GestureMachine();
      gm.setContext({ speaking });
      const ev = run(gm, s, fps);
      assert.equal(fired(ev, 'long_blink').length, 0, `${fps} fps speaking=${speaking}: false confirm`);
      assert.equal(fired(ev, 'jaw_hold').length, 0, `${fps} fps speaking=${speaking}: false stop`);
      assert.equal(fired(ev, 'brow_raise').length, 0);
    }
  }
});

for (const fps of FPS) {
  test(`deliberate 800 ms closures confirm, on reopen, at ${fps} fps`, () => {
    const s = new Scene(20_000, { seed: 5 });
    const starts = [1000, 5000, 9000, 13000];
    for (const t of starts) s.blink(t, 800);
    s.naturalBlinks(16000, 20000, 15, rng(5));
    const ev = run(new GestureMachine(), s, fps);
    const lb = fired(ev, 'long_blink');
    assert.equal(lb.length, 4);
    lb.forEach((e, i) => {
      // Fires when the eyes reopen, not while they are still closed.
      const reopen = starts[i] + 800 - 14; // signal falls below "off" here
      assert.ok(e.t_ms >= reopen - 20 && e.t_ms <= reopen + 1000 / fps + 25, `fired at ${e.t_ms}, reopen ${reopen}`);
      assert.ok(e.held_ms >= 700 && e.held_ms <= 840, `held ${e.held_ms}`);
    });
    // The ring got progress events before the fire, ending near 1.
    const prog = ev.filter((e) => e.name === 'long_blink' && (e.phase === 'progress' || e.phase === 'start'));
    assert.ok(prog.length >= 4);
    assert.ok(prog.every((e) => e.progress > 0 && e.progress <= 1));
    assert.ok(Math.max(...prog.map((e) => e.progress)) >= 0.85);
  });
}

test('closures shorter than the threshold cancel the ring instead of confirming', () => {
  const s = new Scene(4000, { seed: 6 }).blink(1000, 600);
  const ev = run(new GestureMachine(), s, 30);
  assert.equal(fired(ev, 'long_blink').length, 0);
  const c = ev.filter((e) => e.name === 'long_blink' && e.phase === 'cancelled');
  assert.equal(c.length, 1);
  assert.equal(c[0].reason, 'too_short');
});

test('open gaps under 80 ms inside a closure are ignored; longer gaps split it', () => {
  // 400 + 30 ms flicker + 400 = one closure of about 830 ms. (The gap is measured where the score
  // crosses the thresholds: with the eyelid ramps this flicker is open for about 55 ms.)
  const a = new Scene(4000, { seed: 7, noise: 0 }).blink(1000, 400).blink(1430, 400);
  const evA = run(new GestureMachine(), a, 60, { jitter: 0 });
  assert.equal(fired(evA, 'long_blink').length, 1);
  // 400 + 150 ms open + 400 = two short blinks
  const b = new Scene(4000, { seed: 7, noise: 0 }).blink(1000, 400).blink(1550, 400);
  const evB = run(new GestureMachine(), b, 60, { jitter: 0 });
  assert.equal(fired(evB, 'long_blink').length, 0);
});

test('calibrated long_blink_ms is honoured', () => {
  const s = new Scene(6000, { seed: 8 }).blink(1000, 950).blink(3500, 1150);
  assert.equal(fired(run(new GestureMachine({ long_blink_ms: 1000 }), s, 30), 'long_blink').length, 1);
  assert.equal(fired(run(new GestureMachine({ long_blink_ms: 700 }), s, 30), 'long_blink').length, 2);
});

test('eyes closed for longer than long_blink_max_ms is not a confirm', () => {
  const s = new Scene(9000, { seed: 10 }).blink(1000, 5000);
  const ev = run(new GestureMachine(), s, 30);
  assert.equal(fired(ev, 'long_blink').length, 0);
  assert.ok(ev.some((e) => e.name === 'long_blink' && e.phase === 'cancelled' && e.reason === 'too_long'));
});

test('jaw activity during the hold cancels the confirm', () => {
  const s = new Scene(4000, { seed: 11 }).blink(1000, 900).syllable(1300, 200, 0.4);
  const ev = run(new GestureMachine(), s, 30);
  assert.equal(fired(ev, 'long_blink').length, 0);
  assert.ok(ev.some((e) => e.name === 'long_blink' && e.phase === 'cancelled' && e.reason === 'jaw'));
});

test('blink with the head turned beyond 20 degrees is ignored', () => {
  const s = new Scene(5000, { seed: 12 }).turn(500, 3000, 30).blink(1500, 900);
  const ev = run(new GestureMachine(), s, 30);
  assert.equal(fired(ev, 'long_blink').length, 0);
});

test('jaw hold of 300 ms stops; a 200 ms hold does not', () => {
  for (const fps of FPS) {
    const s = new Scene(4000, { seed: 13 }).jawHold(1000, 420, 0.7).jawHold(2500, 260, 0.7);
    const ev = run(new GestureMachine(), s, fps);
    const st = fired(ev, 'jaw_hold');
    assert.equal(st.length, 1, `${fps} fps`);
    // Fires while the jaw is still open, about 300 ms after crossing "on".
    // "on" (0.45) is crossed about 35 ms into the ramp; allow two frames of sampling delay.
    assert.ok(st[0].t_ms > 1330 && st[0].t_ms < 1335 + 2000 / fps + 10, `t ${st[0].t_ms}`);
    assert.ok(st[0].held_ms >= 300);
  }
});

test('jaw hold fires once per hold, and needs 600 ms while speaking', () => {
  const s = new Scene(6000, { seed: 14 }).jawHold(500, 2000, 0.8).jawHold(3500, 500, 0.8);
  assert.equal(fired(run(new GestureMachine(), s, 30), 'jaw_hold').length, 2);
  const gm = new GestureMachine();
  gm.setContext({ speaking: true });
  const ev = run(gm, s, 30);
  assert.equal(fired(ev, 'jaw_hold').length, 1); // the 500 ms hold is too short while speaking
});

test('STOP is never blocked by a lockout', () => {
  // Brow raise fires at about 1450 ms, locking out non-stop gestures until 2450.
  const s = new Scene(4000, { seed: 15 }).browRaise(1000, 700).jawHold(1500, 400, 0.8);
  const ev = run(new GestureMachine(), s, 30);
  const brow = fired(ev, 'brow_raise')[0];
  const stop = fired(ev, 'jaw_hold')[0];
  assert.ok(brow && stop);
  assert.ok(stop.t_ms < brow.t_ms + 1000, 'stop fired inside the lockout window');
});

test('a yawn (jaw open, eyes closed) gives STOP and no confirm', () => {
  for (const fps of FPS) {
    const s = new Scene(5000, { seed: 16 }).jawHold(1000, 2200, 0.85).blink(1200, 1800);
    const ev = run(new GestureMachine(), s, fps);
    assert.equal(fired(ev, 'jaw_hold').length, 1);
    assert.equal(fired(ev, 'long_blink').length, 0);
  }
});

test('brow raise held 400 ms opens the menu, with a 1 s lockout', () => {
  const s = new Scene(6000, { seed: 17 })
    .browRaise(500, 300) // too short
    .browRaise(1500, 700) // fires
    .browRaise(2500, 600) // inside the 1 s brow lockout after ~1980: fires once the lockout ends
    .browRaise(4500, 700); // fires
  const ev = run(new GestureMachine(), s, 30);
  const br = fired(ev, 'brow_raise');
  assert.equal(br.length, 3);
  assert.ok(br[1].t_ms - br[0].t_ms >= 1000);
});

test('brow raise is suppressed while speaking and while the head is turned', () => {
  const s = new Scene(4000, { seed: 18 }).browRaise(1000, 800);
  const gm = new GestureMachine();
  gm.setContext({ speaking: true });
  assert.equal(fired(run(gm, s, 30), 'brow_raise').length, 0);
  const t = new Scene(5000, { seed: 18 }).turn(500, 3000, -28).browRaise(1500, 800);
  assert.equal(fired(run(new GestureMachine(), t, 30), 'turn_right').length, 1);
  assert.equal(fired(run(new GestureMachine(), t, 30), 'brow_raise').length, 0);
});

for (const fps of FPS) {
  test(`head turns at ${fps} fps: one step per turn, must return, brief turns ignored`, () => {
    const s = new Scene(12000, { seed: 19 })
      .turn(500, 3000, 25) // held 3 s: one step only
      .turn(4500, 800, -25) // right
      .turn(6000, 330, 25) // ~180 ms above "on": too brief
      .turn(7500, 900, 18); // left, smaller turn (above 12 deg)
    const ev = run(new GestureMachine(), s, fps);
    const names = fired(ev).filter((e) => e.name.startsWith('turn')).map((e) => e.name);
    assert.deepEqual(names, ['turn_left', 'turn_right', 'turn_left']);
  });
}

test('yaw sign and neutral come from calibration', () => {
  const s = new Scene(3000, { seed: 20 }).turn(500, 1000, 15);
  // Neutral at +5 deg: a 15 deg raw turn is only 10 deg relative, below "on".
  assert.equal(fired(run(new GestureMachine({ neutral_yaw_deg: 5 }), s, 30)).length, 0);
  // Inverted sign: raw positive yaw means the user's right.
  const ev = run(new GestureMachine({ yaw_sign: -1 }), s, 30);
  assert.deepEqual(fired(ev).map((e) => e.name), ['turn_right']);
});

test('lockout: a turn right after a confirm waits for the 500 ms lockout', () => {
  const s = new Scene(4000, { seed: 21, noise: 0 }).blink(500, 800).turn(1300, 800, 25);
  const ev = run(new GestureMachine(), s, 30);
  const lb = fired(ev, 'long_blink')[0];
  const tr = fired(ev, 'turn_left')[0];
  assert.ok(lb && tr);
  assert.ok(tr.t_ms >= lb.t_ms + 500);
});

test('double blink: two short closures, gap 100-500 ms, then 500 ms quiet', () => {
  const s = new Scene(8000, { seed: 22, noise: 0.01 })
    .blink(1000, 200).blink(1450, 200) // double
    .blink(4000, 200).blink(4400, 200).blink(4800, 200); // triple: not a double
  const ev = run(new GestureMachine(), s, 30);
  const d = fired(ev, 'double_blink');
  assert.equal(d.length, 1);
  assert.ok(d[0].t_ms >= 1450 + 200 + 400 && d[0].t_ms < 3000);
  assert.equal(fired(run(new GestureMachine({ double_blink_enabled: false }), s, 30), 'double_blink').length, 0);
});

test('face lost after 500 ms, found when it returns; a missing face cancels the confirm', () => {
  const s = new Scene(5000, { seed: 23 }).missing(1000, 900).blink(3000, 900).missing(3400, 100);
  const ev = run(new GestureMachine(), s, 30);
  const lost = fired(ev, 'face_lost'), found = fired(ev, 'face_found');
  assert.equal(lost.length, 1);
  assert.ok(lost[0].t_ms >= 1500 && lost[0].t_ms < 1600);
  assert.equal(found.length, 1);
  assert.ok(found[0].t_ms >= 1900 && found[0].t_ms < 1940);
  assert.equal(fired(ev, 'long_blink').length, 0);
});

test('15 fps and 30 fps give the same events at about the same times', () => {
  const r = rng(24);
  const s = new Scene(60_000, { seed: 24, noise: 0.03 });
  s.naturalBlinks(0, 60_000, 16, r);
  s.blink(5000, 900).blink(15000, 850).turn(20000, 900, 25).turn(25000, 900, -25);
  s.browRaise(30000, 700).jawHold(40000, 500, 0.8).blink(50000, 1000);
  const evs = [15, 30].map((fps) =>
    fired(run(new GestureMachine({ double_blink_enabled: false }), s, fps)).filter((e) => !e.name.startsWith('fps')));
  assert.deepEqual(evs[0].map((e) => e.name), evs[1].map((e) => e.name));
  assert.deepEqual(evs[0].map((e) => e.name), ['long_blink', 'long_blink', 'turn_left', 'turn_right', 'brow_raise', 'jaw_hold', 'long_blink']);
  evs[0].forEach((e, i) => assert.ok(Math.abs(e.t_ms - evs[1][i].t_ms) <= 1000 / 15 + 20, `${e.name}: ${e.t_ms} vs ${evs[1][i].t_ms}`));
});

test('low frame rate: warning after 2 s below 12 fps, holds lengthened by one frame', () => {
  // 680 ms measured closure: confirms with long_blink_ms 650 at 30 fps, but not at 10 fps (+100 ms).
  const s = new Scene(8000, { seed: 25, noise: 0 }).blink(4000, 750);
  const cfg = { long_blink_ms: 650 };
  const ev10 = run(new GestureMachine(cfg), s, 10, { jitter: 0 });
  assert.equal(fired(ev10, 'fps_low').length, 1);
  assert.ok(fired(ev10, 'fps_low')[0].t_ms >= 2000);
  assert.equal(fired(ev10, 'long_blink').length, 0);
  assert.equal(fired(run(new GestureMachine(cfg), s, 30), 'long_blink').length, 1);
  assert.equal(fired(run(new GestureMachine(cfg), s, 30), 'fps_low').length, 0);
});

test('disabled gestures never fire', () => {
  const s = new Scene(4000, { seed: 26 }).browRaise(500, 800).jawHold(2000, 600);
  const ev = run(new GestureMachine({ disabled: ['brow_raise', 'jaw_hold'] }), s, 30);
  assert.equal(fired(ev).length, 0);
});

test('non-monotonic timestamps are dropped and DEFAULTS are frozen', () => {
  const gm = new GestureMachine();
  gm.update({ t_ms: 100, face_present: true, blendshapes: {} });
  assert.deepEqual(gm.update({ t_ms: 50, face_present: true, blendshapes: {} }), []);
  assert.ok(Object.isFrozen(DEFAULTS));
});

// Two natural blinks in a row, as a step signal (eyes fully closed / fully open), sampled at a
// given frame rate and phase. Review finding: these used to merge into one long blink when the
// open gap was up to gap_ms plus two frames (132 ms at 30 fps, 198 ms at 15 fps).
function twoBlinks(cfg, fps, phase, closeMs, gapMs) {
  const score = (t) => ((t >= 0 && t < closeMs) || (t >= closeMs + gapMs && t < 2 * closeMs + gapMs) ? 0.9 : 0.05);
  const gm = new GestureMachine(cfg);
  const evs = [];
  for (let t = -2000 + phase; t < 2500; t += 1000 / fps) {
    evs.push(...gm.update({ t_ms: t + 10000, face_present: true, yaw_deg: 0, pitch_deg: 0,
      blendshapes: { eyeBlinkLeft: score(t), eyeBlinkRight: score(t) } }));
  }
  return evs.filter((e) => e.name === 'long_blink' && e.phase === 'fired');
}

test('two natural blinks with an open gap of gap_ms or more never merge into a long blink, at any fps or phase', () => {
  for (const fps of [15, 24, 30, 60]) {
    for (let phase = 0; phase < 1000 / fps; phase += 5) {
      for (const [closeMs, gapMs] of [[300, 100], [350, 80], [350, 120], [350, 198], [400, 90]]) {
        assert.equal(twoBlinks({}, fps, phase, closeMs, gapMs).length, 0, `${fps} fps, phase ${phase}, ${closeMs}+${gapMs}+${closeMs}`);
      }
      // calibrated 600 ms profile, two 250 ms blinks 100 ms apart
      assert.equal(twoBlinks({ long_blink_ms: 600 }, fps, phase, 250, 100).length, 0, `${fps} fps, 600 ms profile`);
    }
  }
});

test('one continuous long blink still fires at every fps and phase', () => {
  for (const fps of [15, 24, 30, 60]) {
    for (let phase = 0; phase < 1000 / fps; phase += 5) {
      assert.equal(twoBlinks({}, fps, phase, 850, 5000).length, 1, `${fps} fps, phase ${phase}`);
    }
  }
});
