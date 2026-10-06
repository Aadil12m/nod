// Nod gesture state machine (design section 6.1 and 6.2).
//
// PURE: no DOM, no timers, no clock reads. Feed it one frame at a time with
// `update({t_ms, blendshapes, yaw_deg, pitch_deg, face_present})` and it returns
// the events that frame produced. All timing is in milliseconds taken from
// `t_ms` (a monotonic clock such as performance.now()), never frame counts, so
// the same motion gives the same events at 15 fps and at 30 fps.
//
// Events: {name, phase, t_ms, held_ms?, progress?, yaw_deg, pitch_deg, scores, fps, reason?}
//   name  : long_blink | double_blink | brow_raise | turn_left | turn_right |
//           jaw_hold | face_lost | face_found | fps_low | fps_ok
//   phase : start | progress | fired | cancelled
// Only `fired` and `cancelled` are telemetry for the gateway (3.5.1); `start`
// and `progress` drive the confirm ring on the phone.
//
// This module only recognises gestures. Whether a gesture does anything
// (confirm only when an option is armed, menu only when idle) is decided by
// the UI state machine in ui.js.

/** Default thresholds and timings. Starting points from design 6.1; calibrate.js overrides them. */
export const DEFAULTS = Object.freeze({
  // CONFIRM: min(eyeBlinkLeft, eyeBlinkRight)
  blink_on: 0.55,
  blink_off: 0.35,
  long_blink_ms: 700, // calibrated range 600-1200
  long_blink_max_ms: 4000, // eyes closed longer than this is not a confirm (dozing); see DEVIATIONS
  gap_ms: 80, // open gaps shorter than this (real time between closed samples) inside a closure are ignored
  ring_show_ms: 250, // progress events start after this, so natural blinks do not flash the ring
  blink_jaw_cancel: 0.25, // jaw activity above this during the hold cancels the confirm
  // Double blink (never starts motion)
  double_blink_enabled: true,
  dbl_min_ms: 100,
  dbl_max_ms: 500,
  dbl_gap_min_ms: 100,
  dbl_gap_max_ms: 500,
  dbl_quiet_ms: 500,
  // MENU: max(browInnerUp, mean(browOuterUpLeft, browOuterUpRight))
  brow_on: 0.5,
  brow_off: 0.3,
  brow_hold_ms: 400,
  brow_lockout_ms: 1000,
  // STOP: jawOpen
  jaw_on: 0.45,
  jaw_off: 0.25,
  jaw_hold_ms: 300,
  jaw_hold_speaking_ms: 600,
  // NEXT / PREV: head yaw relative to neutral
  yaw_on_deg: 12, // max(12, 35% of comfortable max); calibrate.js computes it
  yaw_off_deg: 6, // half of on
  turn_hold_ms: 250,
  yaw_sign: 1, // sign of raw yaw when the user turns to THEIR left (recorded in calibration)
  neutral_yaw_deg: 0,
  yaw_tau_ms: 48, // time-based EMA; alpha is about 0.5 at 30 fps
  yaw_suppress_deg: 20, // blink and brow are ignored beyond this
  // General
  lockout_ms: 500, // after any non-stop gesture, other non-stop gestures are blocked
  face_lost_ms: 500,
  low_fps: 12,
  low_fps_window_ms: 2000,
  // Gestures turned off by the calibration separation check (6.3 step 5)
  disabled: Object.freeze([]),
});

const clamp01 = (x) => (Number.isFinite(x) ? Math.min(1, Math.max(0, x)) : 0);
const get = (bs, k) => clamp01(bs && typeof bs[k] === 'number' ? bs[k] : 0);

/** CONFIRM signal: both eyes must be closed. */
export function blinkScore(bs) {
  return Math.min(get(bs, 'eyeBlinkLeft'), get(bs, 'eyeBlinkRight'));
}
/** MENU signal. */
export function browScore(bs) {
  return Math.max(get(bs, 'browInnerUp'), (get(bs, 'browOuterUpLeft') + get(bs, 'browOuterUpRight')) / 2);
}
/** STOP signal. */
export function jawScore(bs) {
  return get(bs, 'jawOpen');
}

/**
 * Hysteresis detector with gap merging and an observable "pending release".
 * Active from the first sample >= on until the signal has been below off for
 * gapMs. `lowSince` is when the signal first dropped below off.
 *
 * The gap is measured in real time from the LAST sample that was still above
 * off (`lastHigh`), not from the first low sample: two closures are merged only
 * if the span between the closed samples on either side of the gap is shorter
 * than gapMs. That span is an upper bound on the true open time, so an open gap
 * of gapMs or more never merges at any frame rate (at low frame rates even a
 * shorter flicker may split a closure, which only costs a missed confirm).
 */
class Hold {
  constructor() { this.reset(); }
  reset() { this.active = false; this.start = 0; this.lowSince = null; this.lastHigh = 0; }
  /** Returns 'began' | 'held' | 'low' | 'ended' | 'idle'. On 'ended', `this.endedAt` is the release time. */
  step(t, x, on, off, gapMs) {
    if (!this.active) {
      if (x >= on) { this.active = true; this.start = t; this.lastHigh = t; this.lowSince = null; return 'began'; }
      return 'idle';
    }
    if (x < off) {
      if (this.lowSince === null) this.lowSince = t;
      if (t - this.lastHigh >= gapMs) return this._end();
      return 'low';
    }
    if (this.lowSince !== null && t - this.lastHigh >= gapMs) {
      // Closed again, but only after an open span of at least gapMs (sparse frames): that was a
      // separate closure. The caller may begin a new one on this same sample.
      return this._end();
    }
    this.lowSince = null; // closed again within the gap: same closure
    this.lastHigh = t;
    return 'held';
  }
  _end() {
    this.active = false;
    this.endedAt = this.lowSince;
    this.lowSince = null;
    return 'ended';
  }
}

export class GestureMachine {
  /** @param {Partial<typeof DEFAULTS>} config */
  constructor(config = {}) {
    this.setConfig(config);
    this.ctx = { speaking: false };
    this.reset();
  }

  setConfig(config = {}) {
    this.cfg = { ...DEFAULTS, ...config };
    this.cfg.disabled = new Set(config.disabled || DEFAULTS.disabled);
  }

  /** speaking: speech is being detected (lengthens the jaw hold, suppresses brow raise). */
  setContext(ctx) { this.ctx = { ...this.ctx, ...ctx }; }

  reset() {
    this.blink = new Hold();
    this.blinkState = null; // per-closure info: {suppressed, cancelled, consumed, shown}
    this.brow = new Hold();
    this.browState = null;
    this.jaw = new Hold();
    this.jawFired = false;
    this.turn = { state: 'center', dir: null, start: 0 };
    this.yawS = null; // smoothed raw yaw
    this.lastT = null;
    this.lockAll = -Infinity; // non-stop gestures blocked until this time
    this.lockBrow = -Infinity;
    this.shortBlinks = []; // [{start, end}] for double-blink detection
    this.dblCandidateAt = null;
    this.facePresent = null; // unknown until the first frame
    this.faceAbsentSince = null;
    this.faceLostSent = false;
    this.frameTimes = [];
    this.lowFps = false;
    this.lowFpsSince = null;
    this.okFpsSince = null;
    this.fps = 0;
    this.scores = { blink: 0, brow: 0, jaw: 0 };
    this.yawRel = 0;
    this.pitch = 0;
  }

  /** Effective hold time: lengthened by one frame while the frame rate is low (6.2). */
  _hold(ms) {
    return this.lowFps && this.fps > 0 ? ms + 1000 / this.fps : ms;
  }

  _enabled(name) { return !this.cfg.disabled.has(name); }

  _ev(out, t, name, phase, extra = {}) {
    out.push({
      name, phase, t_ms: t,
      yaw_deg: round1(this.yawRel), pitch_deg: round1(this.pitch),
      scores: { ...this.scores }, fps: round1(this.fps), ...extra,
    });
  }

  _lock(t, name) {
    this.lockAll = t + this.cfg.lockout_ms;
    if (name === 'brow_raise') this.lockBrow = t + this.cfg.brow_lockout_ms;
  }

  _locked(t, name) {
    if (t < this.lockAll) return true;
    return name === 'brow_raise' && t < this.lockBrow;
  }

  /**
   * Process one frame.
   * @param {{t_ms:number, blendshapes?:Object<string,number>, yaw_deg?:number, pitch_deg?:number, face_present:boolean}} f
   * @returns {Array<Object>} events
   */
  update(f) {
    const out = [];
    const t = f.t_ms;
    if (this.lastT !== null && t <= this.lastT) return out; // not monotonic: drop the frame
    const dt = this.lastT === null ? 0 : t - this.lastT;
    this.lastT = t;
    this._trackFps(t, out);

    // ---- face presence
    if (!f.face_present) {
      if (this.faceAbsentSince === null) this.faceAbsentSince = t;
      // A missing face can never complete a confirm.
      this._cancelBlink(t, out, 'face_missing');
      if (!this.faceLostSent && t - this.faceAbsentSince >= this.cfg.face_lost_ms) {
        this.faceLostSent = true;
        this.facePresent = false;
        this._resetHolds();
        this._ev(out, t, 'face_lost', 'fired', { held_ms: Math.round(t - this.faceAbsentSince) });
      }
      return out; // no observation: holds neither advance nor release
    }
    this.faceAbsentSince = null;
    if (this.faceLostSent) {
      this.faceLostSent = false;
      this._ev(out, t, 'face_found', 'fired');
    }
    this.facePresent = true;

    // ---- signals
    const bs = f.blendshapes || {};
    const blink = blinkScore(bs), brow = browScore(bs), jaw = jawScore(bs);
    this.scores = { blink: round2(blink), brow: round2(brow), jaw: round2(jaw) };
    const rawYaw = Number.isFinite(f.yaw_deg) ? f.yaw_deg : 0;
    if (this.yawS === null || dt <= 0) this.yawS = rawYaw;
    else this.yawS += (1 - Math.exp(-dt / this.cfg.yaw_tau_ms)) * (rawYaw - this.yawS);
    // Positive = the user's left.
    this.yawRel = (this.yawS - this.cfg.neutral_yaw_deg) * (this.cfg.yaw_sign < 0 ? -1 : 1);
    this.pitch = Number.isFinite(f.pitch_deg) ? f.pitch_deg : 0;
    const turned = Math.abs(this.yawRel) > this.cfg.yaw_suppress_deg;

    // STOP first: never blocked, never suppressed.
    this._stepJaw(t, jaw, out);
    this._stepBlink(t, blink, jaw, turned, out);
    this._stepBrow(t, brow, turned, out);
    this._stepTurn(t, out);
    this._stepDouble(t, out);
    return out;
  }

  _resetHolds() {
    this.blink.reset(); this.blinkState = null;
    this.brow.reset(); this.browState = null;
    this.jaw.reset(); this.jawFired = false;
    this.turn = { state: 'center', dir: null, start: 0 };
    this.shortBlinks = []; this.dblCandidateAt = null;
  }

  _trackFps(t, out) {
    const c = this.cfg;
    this.frameTimes.push(t);
    while (this.frameTimes.length > 2 && t - this.frameTimes[0] > 1000) this.frameTimes.shift();
    const span = t - this.frameTimes[0];
    if (span >= 500) this.fps = ((this.frameTimes.length - 1) * 1000) / span;
    else return; // not enough history yet
    if (this.fps < c.low_fps) {
      this.okFpsSince = null;
      if (this.lowFpsSince === null) this.lowFpsSince = t;
      if (!this.lowFps && t - this.lowFpsSince >= c.low_fps_window_ms) {
        this.lowFps = true;
        this._ev(out, t, 'fps_low', 'fired');
      }
    } else {
      this.lowFpsSince = null;
      if (this.lowFps) {
        if (this.okFpsSince === null) this.okFpsSince = t;
        if (t - this.okFpsSince >= c.low_fps_window_ms) {
          this.lowFps = false;
          this._ev(out, t, 'fps_ok', 'fired');
        }
      }
    }
  }

  _stepJaw(t, jaw, out) {
    const c = this.cfg;
    const r = this.jaw.step(t, jaw, c.jaw_on, c.jaw_off, 0);
    if (r === 'began') this.jawFired = false;
    if (r === 'ended' || r === 'idle') { this.jawFired = false; return; }
    if (!this.jaw.active || this.jawFired || !this._enabled('jaw_hold')) return;
    const need = this.ctx.speaking ? c.jaw_hold_speaking_ms : c.jaw_hold_ms;
    const held = t - this.jaw.start;
    if (held >= need) {
      this.jawFired = true;
      this._ev(out, t, 'jaw_hold', 'fired', { held_ms: Math.round(held) });
    }
  }

  _cancelBlink(t, out, reason) {
    const s = this.blinkState;
    if (!s || s.cancelled || s.consumed) return;
    s.cancelled = true;
    if (s.shown) this._ev(out, t, 'long_blink', 'cancelled', { held_ms: Math.round(t - this.blink.start), reason });
  }

  _stepBlink(t, blink, jaw, turned, out) {
    const c = this.cfg;
    let r = this.blink.step(t, blink, c.blink_on, c.blink_off, c.gap_ms);
    if (r === 'ended' && blink >= c.blink_on) {
      // A closure ended by a long open span, and this sample starts the next one.
      this._blinkEnded(t, out);
      r = this.blink.step(t, blink, c.blink_on, c.blink_off, c.gap_ms); // 'began'
    }
    if (r === 'began') {
      this.blinkState = { suppressed: turned, cancelled: false, consumed: false, shown: false };
      if (this.dblCandidateAt !== null) this.dblCandidateAt = null; // a third blink: not a double
    }
    const s = this.blinkState;
    if (!s) return;
    if (r === 'ended') {
      this._blinkEnded(t, out);
      return;
    }
    if (r === 'idle') return;
    // Closure in progress ('began', 'held' or 'low').
    if (s.suppressed || s.consumed) return;
    if (!s.cancelled) {
      if (turned) this._cancelBlink(t, out, 'head_turned');
      else if (jaw > c.blink_jaw_cancel) this._cancelBlink(t, out, 'jaw');
      else if (t - this.blink.start > c.long_blink_max_ms) this._cancelBlink(t, out, 'too_long');
      else if (!this._enabled('long_blink')) s.cancelled = true;
    }
    if (s.cancelled) return;
    const need = this._hold(c.long_blink_ms);
    if (r === 'low') {
      // Eyes have reopened (pending the gap check). Fire now if the hold qualified.
      const held = this.blink.lowSince - this.blink.start;
      if (held >= need) {
        s.consumed = true;
        if (this._locked(t, 'long_blink')) {
          this._ev(out, t, 'long_blink', 'cancelled', { held_ms: Math.round(held), reason: 'lockout' });
        } else {
          this._lock(t, 'long_blink');
          this.shortBlinks = [];
          this._ev(out, t, 'long_blink', 'fired', { held_ms: Math.round(held), progress: 1 });
        }
      }
      return;
    }
    const held = t - this.blink.start;
    if (held >= c.ring_show_ms) {
      const progress = Math.min(1, held / need);
      if (!s.shown) { s.shown = true; this._ev(out, t, 'long_blink', 'start', { held_ms: Math.round(held), progress }); }
      else this._ev(out, t, 'long_blink', 'progress', { held_ms: Math.round(held), progress });
    }
  }

  /** The closure in `this.blink` has ended (eyes open for at least gap_ms). */
  _blinkEnded(t, out) {
    const c = this.cfg;
    const s = this.blinkState;
    if (s) {
      const dur = this.blink.endedAt - this.blink.start;
      if (!s.suppressed && !s.cancelled && !s.consumed) {
        if (s.shown) this._ev(out, t, 'long_blink', 'cancelled', { held_ms: Math.round(dur), reason: 'too_short' });
        if (dur >= c.dbl_min_ms && dur <= c.dbl_max_ms) this._shortBlink(this.blink.start, this.blink.endedAt);
        else this.shortBlinks = [];
      }
    }
    this.blinkState = null;
  }

  _shortBlink(start, end) {
    const c = this.cfg;
    const prev = this.shortBlinks[this.shortBlinks.length - 1];
    if (prev) {
      const gap = start - prev.end;
      if (gap >= c.dbl_gap_min_ms && gap <= c.dbl_gap_max_ms) {
        this.shortBlinks = [];
        this.dblCandidateAt = end; // fires after the quiet period
        return;
      }
    }
    this.shortBlinks = [{ start, end }];
  }

  _stepDouble(t, out) {
    const c = this.cfg;
    if (this.dblCandidateAt === null) return;
    if (this.blink.active) { this.dblCandidateAt = null; return; }
    if (t - this.dblCandidateAt < c.dbl_quiet_ms) return;
    this.dblCandidateAt = null;
    if (!c.double_blink_enabled || !this._enabled('double_blink') || this._locked(t, 'double_blink')) return;
    this._lock(t, 'double_blink');
    this._ev(out, t, 'double_blink', 'fired');
  }

  _stepBrow(t, brow, turned, out) {
    const c = this.cfg;
    const r = this.brow.step(t, brow, c.brow_on, c.brow_off, 0);
    if (r === 'began') this.browState = { suppressed: turned || this.ctx.speaking, fired: false };
    if (r === 'ended' || r === 'idle') { this.browState = null; return; }
    const s = this.browState;
    if (!s || s.suppressed || s.fired) return;
    if (turned || this.ctx.speaking) { s.suppressed = true; return; }
    if (!this._enabled('brow_raise')) return;
    const held = t - this.brow.start;
    // A completed hold waits out a lockout as long as the brows stay up.
    if (held >= this._hold(c.brow_hold_ms) && !this._locked(t, 'brow_raise')) {
      s.fired = true;
      this._lock(t, 'brow_raise');
      this._ev(out, t, 'brow_raise', 'fired', { held_ms: Math.round(held) });
    }
  }

  _stepTurn(t, out) {
    const c = this.cfg;
    const y = this.yawRel;
    const tr = this.turn;
    const dir = y >= c.yaw_on_deg ? 'turn_left' : y <= -c.yaw_on_deg ? 'turn_right' : null;
    if (tr.state === 'center') {
      if (dir) { tr.state = 'holding'; tr.dir = dir; tr.start = t; }
    } else if (tr.state === 'holding') {
      const still = tr.dir === 'turn_left' ? y > c.yaw_off_deg : y < -c.yaw_off_deg;
      if (!still) { tr.state = 'center'; tr.dir = null; return; }
    } else if (tr.state === 'fired') {
      // The head must come back inside "off" before the next step.
      if (Math.abs(y) <= c.yaw_off_deg) { tr.state = 'center'; tr.dir = null; }
      return;
    }
    if (tr.state !== 'holding' || !this._enabled(tr.dir)) return;
    const held = t - tr.start;
    if (held >= this._hold(c.turn_hold_ms) && !this._locked(t, tr.dir)) {
      tr.state = 'fired';
      this._lock(t, tr.dir);
      this._ev(out, t, tr.dir, 'fired', { held_ms: Math.round(held) });
    }
  }
}

function round1(x) { return Math.round(x * 10) / 10; }
function round2(x) { return Math.round(x * 100) / 100; }
