// Synthetic face traces for the gesture tests.
//
// A Scene describes what the face does over continuous time (blinks, jaw
// movement, brow raises, head turns, face missing). `sample(t)` returns what
// MediaPipe would report at time t, and `frames(fps)` samples it at a given
// frame rate with timing jitter, the way a phone camera delivers frames.

export function rng(seed) {
  // mulberry32: small, deterministic PRNG so failures are reproducible.
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Envelope of a held gesture: ramps up, holds, ramps down. Returns 0..1. */
function trapezoid(t, start, dur, rampIn, rampOut) {
  if (t < start || t > start + dur) return 0;
  const x = t - start;
  if (x < rampIn) return x / rampIn;
  if (x > dur - rampOut) return Math.max(0, (dur - x) / rampOut);
  return 1;
}

/** Smooth bump (half sine) for syllables. */
function bump(t, start, dur) {
  if (t < start || t > start + dur) return 0;
  return Math.sin((Math.PI * (t - start)) / dur);
}

export class Scene {
  constructor(durationMs, { seed = 1, noise = 0.02 } = {}) {
    this.duration = durationMs;
    this.blinks = []; // {start, dur}  (dur = time below the open level, ramp included)
    this.jaw = []; // {start, dur, peak, shape: 'hold'|'bump'}
    this.brow = []; // {start, dur, peak}
    this.yaw = []; // {start, dur, deg}
    this.absent = []; // {start, dur}
    this.noise = noise;
    this.r = rng(seed);
  }

  blink(start, dur) { this.blinks.push({ start, dur }); return this; }
  jawHold(start, dur, peak = 0.7) { this.jaw.push({ start, dur, peak, shape: 'hold' }); return this; }
  syllable(start, dur, peak) { this.jaw.push({ start, dur, peak, shape: 'bump' }); return this; }
  browRaise(start, dur, peak = 0.75) { this.brow.push({ start, dur, peak }); return this; }
  turn(start, dur, deg) { this.yaw.push({ start, dur, deg }); return this; }
  missing(start, dur) { this.absent.push({ start, dur }); return this; }

  /** Natural blinking: `perMin` blinks a minute, each 150-400 ms, never closer than 200 ms apart. */
  naturalBlinks(from, to, perMin, r = this.r, durRange = [150, 400]) {
    let t = from + r() * 2000;
    const mean = 60000 / perMin;
    while (t < to) {
      const dur = durRange[0] + r() * (durRange[1] - durRange[0]);
      this.blink(t, dur);
      // Exponential intervals (random arrival), with a physiological refractory gap.
      const gap = -Math.log(1 - r()) * mean;
      t += Math.max(dur + 200, gap);
    }
    return this;
  }

  /** Talking: jaw syllables 120-280 ms with peaks up to `maxPeak`, pauses between sentences. */
  talking(from, to, r = this.r, maxPeak = 0.6) {
    let t = from;
    while (t < to) {
      const words = 4 + Math.floor(r() * 12);
      for (let i = 0; i < words && t < to; i++) {
        const dur = 120 + r() * 160;
        this.syllable(t, dur, 0.15 + r() * (maxPeak - 0.15));
        t += dur + 100 + r() * 200;
      }
      t += 400 + r() * 1500; // pause between sentences
    }
    return this;
  }

  sample(t) {
    const present = !this.absent.some((a) => t >= a.start && t < a.start + a.dur);
    if (!present) return { face_present: false, blendshapes: {}, yaw_deg: 0, pitch_deg: 0 };
    const n = () => (this.r() - 0.5) * 2 * this.noise;
    let eye = 0;
    for (const b of this.blinks) eye = Math.max(eye, trapezoid(t, b.start, b.dur, 30, 40));
    let jaw = 0;
    for (const j of this.jaw) {
      const v = j.shape === 'hold' ? trapezoid(t, j.start, j.dur, 60, 60) : bump(t, j.start, j.dur);
      jaw = Math.max(jaw, v * j.peak);
    }
    let brow = 0;
    for (const b of this.brow) brow = Math.max(brow, trapezoid(t, b.start, b.dur, 80, 80) * b.peak);
    let yaw = 0;
    for (const y of this.yaw) yaw += trapezoid(t, y.start, y.dur, 150, 150) * y.deg;
    const closed = 0.05 + 0.85 * eye;
    return {
      face_present: true,
      blendshapes: {
        eyeBlinkLeft: clamp(closed + n()),
        eyeBlinkRight: clamp(closed - 0.02 + n()),
        jawOpen: clamp(0.04 + jaw + n()),
        browInnerUp: clamp(0.08 + brow + n()),
        browOuterUpLeft: clamp(0.05 + brow * 0.9 + n()),
        browOuterUpRight: clamp(0.05 + brow * 0.85 + n()),
      },
      yaw_deg: yaw + n() * 20, // about +-0.4 deg of tracking noise
      pitch_deg: n() * 20,
    };
  }

  /** Sample at `fps` with +-jitter ms of timing noise, starting at `offset`. */
  *frames(fps, { jitter = 3, offset = 0 } = {}) {
    const step = 1000 / fps;
    const jr = rng(Math.round(fps * 1000) + 7);
    let last = -Infinity;
    for (let k = 0; ; k++) {
      let t = offset + k * step + (jr() - 0.5) * 2 * jitter;
      if (t > this.duration) return;
      if (t <= last) t = last + 0.5;
      last = t;
      yield { t_ms: t, ...this.sample(t) };
    }
  }
}

function clamp(x) { return Math.min(1, Math.max(0, x)); }

/** Run a GestureMachine over a scene; returns all events. `onFrame(t, gm)` may change context. */
export function run(gm, scene, fps, opts = {}) {
  const events = [];
  for (const f of scene.frames(fps, opts)) {
    if (opts.onFrame) opts.onFrame(f.t_ms, gm);
    for (const e of gm.update(f)) events.push(e);
  }
  return events;
}

export const fired = (events, name) => events.filter((e) => e.phase === 'fired' && (!name || e.name === name));
