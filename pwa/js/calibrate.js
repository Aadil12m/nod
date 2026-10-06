// Nod calibration (design 6.3).
//
// Everything except `loadProfile` / `saveProfile` is pure: the math works on
// recorded frames ({t_ms, blendshapes, yaw_deg, pitch_deg, face_present}) and
// the CalibrationFlow sequencer advances only on the frames it is fed, so the
// whole flow can be tested in node with synthetic traces.

import { DEFAULTS, GestureMachine, blinkScore, browScore, jawScore } from './gestures.js';

export const PROFILE_KEY = 'nod.profile.v1';
export const PROFILE_VERSION = 1;

// ------------------------------------------------------------------ small statistics

export function mean(a) { return a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0; }
export function sd(a) {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1));
}
/** Percentile with linear interpolation, p in [0, 100]. */
export function percentile(a, p) {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  const i = (p / 100) * (s.length - 1);
  const lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}
export const median = (a) => percentile(a, 50);
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const r3 = (x) => Math.round(x * 1000) / 1000;

const SIGNALS = { blink: blinkScore, brow: browScore, jaw: jawScore };
const present = (frames) => frames.filter((f) => f.face_present);
const series = (frames, k) => present(frames).map((f) => SIGNALS[k](f.blendshapes || {}));

// ------------------------------------------------------------------ steps 1-4: measurements

/** Step 1: proceed only if fps >= 15 and face presence >= 0.9. */
export function setupCheck(frames, { minFps = 15, minPresence = 0.9 } = {}) {
  if (frames.length < 2) return { ok: false, fps: 0, presence: 0 };
  const span = frames[frames.length - 1].t_ms - frames[0].t_ms;
  const fps = span > 0 ? ((frames.length - 1) * 1000) / span : 0;
  const presence = present(frames).length / frames.length;
  return { ok: fps >= minFps && presence >= minPresence, fps: r3(fps), presence: r3(presence) };
}

/** Step 2: baseline mean and spread of each signal, and the neutral head pose. */
export function analyzeNeutral(frames) {
  const out = {};
  for (const k of Object.keys(SIGNALS)) {
    const s = series(frames, k);
    out[k] = { mean: r3(mean(s)), sd: r3(sd(s)) };
  }
  const yaws = present(frames).map((f) => f.yaw_deg || 0);
  out.yaw = { mean: r3(mean(yaws)), sd: r3(sd(yaws)) };
  return out;
}

/**
 * Closures of a signal using the same hysteresis and gap rule as gestures.js.
 * Returns [{start, end, dur, peak}].
 */
export function closures(frames, key, on, off, gapMs = DEFAULTS.gap_ms) {
  const out = [];
  let cur = null, lowSince = null;
  for (const f of present(frames)) {
    const x = SIGNALS[key](f.blendshapes || {});
    const t = f.t_ms;
    if (!cur) {
      if (x >= on) { cur = { start: t, peak: x }; lowSince = null; }
      continue;
    }
    cur.peak = Math.max(cur.peak, x);
    if (x < off) {
      if (lowSince === null) lowSince = t;
      if (t - lowSince >= gapMs) { out.push({ start: cur.start, end: lowSince, dur: lowSince - cur.start, peak: cur.peak }); cur = null; }
    } else lowSince = null;
  }
  if (cur && lowSince !== null) out.push({ start: cur.start, end: lowSince, dur: lowSince - cur.start, peak: cur.peak });
  return out;
}

/** Peak of a signal inside one repetition window. */
export function repPeak(frames, key) {
  const s = series(frames, key);
  return s.length ? Math.max(...s) : 0;
}

/** Signed yaw (raw degrees minus neutral) furthest from neutral inside one turn window. */
export function turnPeak(frames, neutralYaw) {
  let best = 0;
  for (const f of present(frames)) {
    const d = (f.yaw_deg || 0) - neutralYaw;
    if (Math.abs(d) > Math.abs(best)) best = d;
  }
  return best;
}

/** on = baseline + 50% of the gap to the median peak, off = baseline + 30% (step 4). */
export function thresholdsFromReps(base, medianPeak) {
  const gap = Math.max(0, medianPeak - base);
  return { on: base + 0.5 * gap, off: base + 0.3 * gap };
}

/** Long blink = 99th percentile natural closure + 250 ms, clamped to 600-1200 ms (step 3). */
export function longBlinkMs(naturalDurations) {
  if (naturalDurations.length < 5) return DEFAULTS.long_blink_ms;
  return Math.round(clamp(percentile(naturalDurations, 99) + 250, 600, 1200));
}

// ------------------------------------------------------------------ step 4-5: the profile

/**
 * Build a profile from recorded phases.
 * @param {{setup?:Array, neutral:Array, natural:Array, reps:Object<string, Array<Array>>}} rec
 *   reps keys: long_blink, brow_raise, jaw_hold, turn_left, turn_right; each an array of frame windows.
 */
export function computeProfile(rec, { id, now = 0 } = {}) {
  const warnings = [];
  const disabled = [];
  const neutral = analyzeNeutral(rec.neutral);
  const reps = rec.reps || {};
  const cfg = {};
  const detail = {};

  // Amplitude thresholds for the three blendshape signals.
  const sig = { blink: 'long_blink', brow: 'brow_raise', jaw: 'jaw_hold' };
  for (const [k, gesture] of Object.entries(sig)) {
    const peaks = (reps[gesture] || []).map((w) => repPeak(w, k));
    const base = neutral[k].mean;
    const natP99 = percentile(series(rec.natural || [], k), 99);
    const d = { base, natural_p99: r3(natP99), median_peak: r3(median(peaks)), n_reps: peaks.length };
    detail[k] = d;
    if (!peaks.length) { warnings.push(`${gesture}: no repetitions recorded, defaults kept`); continue; }
    let { on, off } = thresholdsFromReps(base, d.median_peak);
    if (k !== 'blink') {
      // Brow and jaw: "on" must sit above the natural 99th percentile (step 3).
      on = Math.max(on, natP99 + 0.02);
      // Separation check (step 5).
      if (d.median_peak - natP99 < 0.1) {
        disabled.push(gesture);
        warnings.push(`${gesture}: the deliberate movement is too close to natural movement; disabled`);
      }
    }
    on = clamp(on, 0.15, 0.95);
    off = clamp(Math.min(off, on - 0.05), 0.08, on - 0.05);
    cfg[`${k}_on`] = r3(on);
    cfg[`${k}_off`] = r3(off);
  }

  // Long blink length from natural closures, measured with the new thresholds.
  const bOn = cfg.blink_on ?? DEFAULTS.blink_on, bOff = cfg.blink_off ?? DEFAULTS.blink_off;
  const nat = closures(rec.natural || [], 'blink', bOn, bOff).map((c) => c.dur);
  cfg.long_blink_ms = longBlinkMs(nat);
  detail.blink.natural_closures = nat.length;
  detail.blink.natural_p99_ms = Math.round(percentile(nat, 99));
  if (nat.length < 5) warnings.push('long_blink: too few natural blinks recorded; using 700 ms');
  const held = (reps.long_blink || []).map((w) => Math.max(0, ...closures(w, 'blink', bOn, bOff).map((c) => c.dur)));
  const reached = held.filter((h) => h >= cfg.long_blink_ms).length;
  detail.blink.reps_long_enough = `${reached}/${held.length}`;
  if (held.length && reached / held.length < 0.8) {
    warnings.push(`long_blink: only ${reached} of ${held.length} practice blinks were long enough; hold your eyes closed a little longer`);
  }

  // Head turns: neutral, sign of "left", comfortable range.
  cfg.neutral_yaw_deg = neutral.yaw.mean;
  const left = (reps.turn_left || []).map((w) => turnPeak(w, neutral.yaw.mean));
  const right = (reps.turn_right || []).map((w) => turnPeak(w, neutral.yaw.mean));
  if (left.length && right.length) {
    const ml = median(left), mr = median(right);
    if (Math.sign(ml) === Math.sign(mr) || ml === 0) {
      warnings.push('turns: left and right turns gave the same direction; check the camera; sign kept at +1');
      cfg.yaw_sign = 1;
    } else cfg.yaw_sign = ml > 0 ? 1 : -1;
    const comfy = Math.min(Math.abs(ml), Math.abs(mr));
    const on = Math.max(DEFAULTS.yaw_on_deg, 0.35 * comfy);
    cfg.yaw_on_deg = r3(on);
    cfg.yaw_off_deg = r3(on / 2);
    detail.yaw = { left: r3(ml), right: r3(mr), comfortable_max: r3(comfy) };
    if (comfy < on + 3) {
      disabled.push('turn_left', 'turn_right');
      warnings.push('turns: the comfortable head turn is too small to step between options; disabled');
    }
  } else warnings.push('turns: no repetitions recorded, defaults kept');

  return {
    v: PROFILE_VERSION,
    id: id || `cal_${Math.round(now).toString(36)}`,
    created_at: now,
    config: { ...cfg, disabled },
    neutral,
    detail,
    warnings,
    setup: rec.setup ? setupCheck(rec.setup) : null,
  };
}

export function defaultProfile() {
  return { v: PROFILE_VERSION, id: 'cal_default', created_at: 0, config: { disabled: [] }, neutral: null, detail: {}, warnings: [], setup: null };
}

/** GestureMachine config for a profile (DEFAULTS filled in by the machine itself). */
export function profileConfig(profile) {
  return { ...(profile?.config || {}) };
}

/** Step 7: re-zero the head pose (and record fresh baselines) at the start of each session. */
export function rezero(profile, frames) {
  const n = analyzeNeutral(frames);
  if (!present(frames).length) return profile;
  return { ...profile, neutral: { ...(profile.neutral || {}), ...n }, config: { ...profile.config, neutral_yaw_deg: n.yaw.mean } };
}

/**
 * Step 7: slow baseline drift while idle (time constant about 60 s). Thresholds
 * move with the baseline by at most `maxShift`. Pure apart from its own state.
 */
export class BaselineTracker {
  constructor(profile, { tauMs = 60_000, maxShift = 0.1 } = {}) {
    this.base0 = {};
    this.ema = {};
    for (const k of Object.keys(SIGNALS)) {
      const b = profile?.neutral?.[k]?.mean;
      this.base0[k] = Number.isFinite(b) ? b : null;
      this.ema[k] = this.base0[k];
    }
    this.cfg = { ...DEFAULTS, ...profileConfig(profile) };
    this.tau = tauMs;
    this.maxShift = maxShift;
    this.lastT = null;
  }

  /** Feed a frame; only frames while idle and outside any gesture move the baseline. */
  update(frame, idle) {
    const dt = this.lastT === null ? 0 : frame.t_ms - this.lastT;
    this.lastT = frame.t_ms;
    if (!idle || !frame.face_present || dt <= 0) return;
    const a = 1 - Math.exp(-dt / this.tau);
    for (const k of Object.keys(SIGNALS)) {
      if (this.ema[k] === null) continue;
      const x = SIGNALS[k](frame.blendshapes || {});
      if (x >= this.cfg[`${k}_off`]) continue; // inside a gesture: not baseline
      this.ema[k] += a * (x - this.ema[k]);
    }
  }

  /** Threshold overrides shifted by the drift since calibration. */
  config() {
    const out = {};
    for (const k of Object.keys(SIGNALS)) {
      if (this.base0[k] === null) continue;
      const shift = clamp(this.ema[k] - this.base0[k], -this.maxShift, this.maxShift);
      out[`${k}_on`] = r3(clamp(this.cfg[`${k}_on`] + shift, 0.1, 0.97));
      out[`${k}_off`] = r3(clamp(this.cfg[`${k}_off`] + shift, 0.05, out[`${k}_on`] - 0.05));
    }
    return out;
  }
}

// ------------------------------------------------------------------ step 6: validation game

/**
 * Score the validation game. trials: [{intended, fired: [names]}]; rest: [names fired
 * in rest windows]. Pass: >= 90% registered, zero false confirms, at most one false menu.
 */
export function scoreValidation(trials, rest = []) {
  let hits = 0, falseConfirms = 0, falseMenus = 0;
  for (const tr of trials) {
    if (tr.fired.includes(tr.intended)) hits++;
    if (tr.intended !== 'long_blink') falseConfirms += tr.fired.filter((n) => n === 'long_blink').length;
    if (tr.intended !== 'brow_raise') falseMenus += tr.fired.filter((n) => n === 'brow_raise').length;
  }
  falseConfirms += rest.filter((n) => n === 'long_blink').length;
  falseMenus += rest.filter((n) => n === 'brow_raise').length;
  const rate = trials.length ? hits / trials.length : 0;
  return { hits, n: trials.length, rate: r3(rate), falseConfirms, falseMenus, pass: rate >= 0.9 && falseConfirms === 0 && falseMenus <= 1 };
}

// ------------------------------------------------------------------ the flow

const PROMPTS = {
  long_blink: 'Close your eyes and hold them closed for about one second, then open',
  brow_raise: 'Raise your eyebrows and hold',
  jaw_hold: 'Open your mouth wide and hold',
  turn_left: 'Turn your head to your left, as far as is comfortable, then back',
  turn_right: 'Turn your head to your right, as far as is comfortable, then back',
};

/** Durations for the full flow (design: about 2 minutes) and a quick demo version. */
export const PLANS = {
  full: { setup_ms: 3000, neutral_ms: 10_000, natural_ms: 60_000, rep_ms: 3000, rest_ms: 1200,
    reps: { long_blink: 5, brow_raise: 5, jaw_hold: 3, turn_left: 3, turn_right: 3 }, validate: true },
  quick: { setup_ms: 2000, neutral_ms: 5000, natural_ms: 20_000, rep_ms: 2500, rest_ms: 1000,
    reps: { long_blink: 3, brow_raise: 3, jaw_hold: 2, turn_left: 2, turn_right: 2 }, validate: true },
};

/**
 * Build the timeline of windows after the setup check:
 * [{phase, gesture?, i?, n?, dur, prompt}]
 */
export function buildSchedule(plan) {
  const s = [];
  s.push({ phase: 'neutral', dur: plan.neutral_ms, prompt: 'Look at the screen and relax your face' });
  s.push({ phase: 'natural', dur: plan.natural_ms, prompt: 'Chat or look around naturally. Blink as usual' });
  for (const [g, n] of Object.entries(plan.reps)) {
    for (let i = 0; i < n; i++) {
      s.push({ phase: 'rest', dur: plan.rest_ms, prompt: 'Relax' });
      s.push({ phase: 'rep', gesture: g, i, n, dur: plan.rep_ms, prompt: PROMPTS[g] });
    }
  }
  if (plan.validate) {
    for (const g of ['long_blink', 'brow_raise', 'turn_left', 'turn_right', 'jaw_hold']) {
      s.push({ phase: 'vrest', dur: plan.rest_ms * 2, prompt: 'Relax' });
      s.push({ phase: 'validate', gesture: g, dur: plan.rep_ms, prompt: PROMPTS[g] });
    }
    s.push({ phase: 'vrest', dur: plan.rest_ms * 2, prompt: 'Relax' });
  }
  return s;
}

/**
 * Pure calibration sequencer. Feed every camera frame to `feed(frame)`; it returns
 * a view model {step, prompt, gesture, i, n, progress, remaining_ms, done, message}.
 * After `done`, read `profile` and `validation`.
 */
export class CalibrationFlow {
  constructor({ plan = 'full', id, maxSetupTries = 20 } = {}) {
    this.plan = typeof plan === 'string' ? PLANS[plan] : plan;
    this.schedule = buildSchedule(this.plan);
    this.id = id;
    this.maxSetupTries = maxSetupTries;
    this.state = 'setup';
    this.setupFrames = [];
    this.setupStart = null;
    this.setupTries = 0;
    this.idx = 0;
    this.winStart = null;
    this.rec = { setup: null, neutral: [], natural: [], reps: {} };
    this.trials = [];
    this.rest = [];
    this.profile = null;
    this.validation = null;
    this.message = '';
    this.gm = null;
    this.cur = [];
    this.curFired = [];
  }

  get done() { return this.state === 'done' || this.state === 'failed'; }

  feed(f) {
    if (this.done) return this.view(f.t_ms);
    if (this.state === 'setup') return this._setup(f);
    if (this.winStart === null) this.winStart = f.t_ms;
    const w = this.schedule[this.idx];
    // Frames belong to the current window.
    this.cur.push(f);
    if (w.phase === 'validate' || w.phase === 'vrest') {
      for (const e of this.gm.update(f)) if (e.phase === 'fired') this.curFired.push(e.name);
    }
    if (f.t_ms - this.winStart >= w.dur) this._closeWindow(w, f.t_ms);
    return this.view(f.t_ms);
  }

  _setup(f) {
    if (this.setupStart === null) this.setupStart = f.t_ms;
    this.setupFrames.push(f);
    if (f.t_ms - this.setupStart >= this.plan.setup_ms) {
      const chk = setupCheck(this.setupFrames);
      this.setupTries++;
      if (chk.ok) {
        this.rec.setup = this.setupFrames;
        this.state = 'run';
        this.message = '';
      } else {
        this.message = chk.fps < 15
          ? `The camera is too slow (${chk.fps.toFixed(0)} fps). Close other apps or add light.`
          : 'I cannot see your face well. Hold the phone at eye level, 35 to 60 cm away.';
        if (this.setupTries >= this.maxSetupTries) this.state = 'failed';
      }
      this.setupFrames = [];
      this.setupStart = f.t_ms;
    }
    return this.view(f.t_ms);
  }

  _closeWindow(w, t) {
    const frames = this.cur;
    this.cur = [];
    if (w.phase === 'neutral') this.rec.neutral = frames;
    else if (w.phase === 'natural') this.rec.natural = frames;
    else if (w.phase === 'rep') (this.rec.reps[w.gesture] ||= []).push(frames);
    else if (w.phase === 'validate') this.trials.push({ intended: w.gesture, fired: this.curFired });
    else if (w.phase === 'vrest') this.rest.push(...this.curFired);
    this.curFired = [];
    this.idx++;
    this.winStart = t;
    const next = this.schedule[this.idx];
    // When the last repetition is done, compute the profile and start validating with it.
    if (!next || (next.phase === 'vrest' && w.phase === 'rep')) this._compute(t);
    if (!next) this._finish();
  }

  _compute(t) {
    if (this.profile) return;
    this.profile = computeProfile(this.rec, { id: this.id, now: t });
    this.gm = new GestureMachine(profileConfig(this.profile));
    this.curFired = [];
  }

  _finish() {
    if (this.plan.validate) this.validation = scoreValidation(this.trials, this.rest);
    this.state = 'done';
  }

  view(t) {
    if (this.state === 'setup') {
      return { step: 'setup', prompt: 'Hold the phone at eye level, 35 to 60 cm from your face', message: this.message,
        progress: this.setupStart === null ? 0 : Math.min(1, (t - this.setupStart) / this.plan.setup_ms), done: false };
    }
    if (this.done) return { step: this.state, done: true, profile: this.profile, validation: this.validation, message: this.message };
    const w = this.schedule[this.idx];
    const el = this.winStart === null ? 0 : t - this.winStart;
    const total = this.schedule.reduce((s, x) => s + x.dur, 0);
    const before = this.schedule.slice(0, this.idx).reduce((s, x) => s + x.dur, 0);
    return {
      step: w.phase, gesture: w.gesture || null, i: w.i ?? null, n: w.n ?? null, prompt: w.prompt,
      progress: Math.min(1, el / w.dur), remaining_ms: Math.max(0, w.dur - el),
      overall: Math.min(1, (before + el) / total), done: false, message: this.message,
    };
  }
}

// ------------------------------------------------------------------ storage (browser only)

/** Load the stored profile, or null. Never throws (private mode, blocked storage). */
export function loadProfile(storage = globalThis.localStorage) {
  try {
    const raw = storage?.getItem(PROFILE_KEY);
    if (!raw) return null;
    const p = JSON.parse(raw);
    return p && p.v === PROFILE_VERSION && p.config ? p : null;
  } catch {
    return null;
  }
}

/** Store a profile. Returns false when storage is unavailable. */
export function saveProfile(profile, storage = globalThis.localStorage) {
  try {
    storage.setItem(PROFILE_KEY, JSON.stringify(profile));
    return true;
  } catch {
    return false;
  }
}

export function clearProfile(storage = globalThis.localStorage) {
  try { storage.removeItem(PROFILE_KEY); return true; } catch { return false; }
}
