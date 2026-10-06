// Microphone capture for utterances (design 3.5.1).
//
// The mic is open only while the UI is LISTENING. Audio is resampled to 16 kHz
// mono PCM16 and sent as binary frames [u32 LE utt_id][pcm16le samples] between
// `utter.start` and `utter.end`. A simple energy voice-activity detector finds
// the start and end of speech; it also tells the gesture machine when the user
// is speaking (jaw hold needs 600 ms then, and brow raise is suppressed).
//
// The resampler, VAD and segmenter are pure and tested in node; MicCapture is
// the browser part.

export const TARGET_SR = 16000;

export function rms(f32) {
  if (!f32.length) return 0;
  let s = 0;
  for (let i = 0; i < f32.length; i++) s += f32[i] * f32[i];
  return Math.sqrt(s / f32.length);
}

export function floatTo16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const x = Math.max(-1, Math.min(1, f32[i]));
    out[i] = x < 0 ? Math.round(x * 0x8000) : Math.round(x * 0x7fff);
  }
  return out;
}

/** Streaming linear-interpolation resampler (keeps state between chunks). */
export class Resampler {
  constructor(fromRate, toRate = TARGET_SR) {
    this.ratio = fromRate / toRate;
    this.pos = 1; // fractional read position in the virtual stream (see push); starts at the first sample
    this.prev = 0; // last sample of the previous chunk
  }
  push(f32) {
    if (this.ratio === 1) return Float32Array.from(f32);
    const out = [];
    // Virtual stream: index 0 = last sample of the previous chunk, index k = f32[k - 1].
    const n = f32.length;
    let p = this.pos;
    while (p < n) {
      const i = Math.floor(p);
      const a = i === 0 ? this.prev : f32[i - 1];
      out.push(a + (f32[i] - a) * (p - i));
      p += this.ratio;
    }
    this.pos = p - n;
    if (n) this.prev = f32[n - 1];
    return Float32Array.from(out);
  }
}

/**
 * Energy VAD, time-based. Speech starts after `onMs` above threshold and ends
 * after `offMs` below it. The threshold adapts to the noise floor.
 */
export class Vad {
  constructor({ minRms = 0.012, factor = 3, onMs = 120, offMs = 700, floorTauMs = 2000 } = {}) {
    Object.assign(this, { minRms, factor, onMs, offMs, floorTauMs });
    this.floor = null;
    this.speaking = false;
    this.aboveSince = null;
    this.belowSince = null;
    this.lastT = null;
  }
  threshold() { return Math.max(this.minRms, (this.floor ?? 0) * this.factor); }
  /** Returns 'start' | 'end' | null. */
  update(level, t) {
    const dt = this.lastT === null ? 0 : t - this.lastT;
    this.lastT = t;
    // Start by assuming a quiet room, so speech that begins the moment the mic opens is still found.
    if (this.floor === null) this.floor = this.minRms / this.factor;
    if (!this.speaking && level < this.threshold()) {
      this.floor += (1 - Math.exp(-dt / this.floorTauMs)) * (level - this.floor);
    }
    const loud = level >= this.threshold();
    if (!this.speaking) {
      if (loud) {
        if (this.aboveSince === null) this.aboveSince = t;
        if (t - this.aboveSince >= this.onMs) { this.speaking = true; this.belowSince = null; return 'start'; }
      } else this.aboveSince = null;
    } else if (!loud) {
      if (this.belowSince === null) this.belowSince = t;
      if (t - this.belowSince >= this.offMs) { this.speaking = false; this.aboveSince = null; return 'end'; }
    } else this.belowSince = null;
    return null;
  }
}

/**
 * Turns a 16 kHz PCM16 stream into utterances. push() returns actions:
 *   {kind:'start', utt} {kind:'audio', utt, pcm} {kind:'end', utt, dur_ms} {kind:'timeout'}
 * Keeps `prerollMs` of audio before the detected start so the first word is not cut.
 */
export class Segmenter {
  constructor({ nextUttId, vad = new Vad(), prerollMs = 300, maxMs = 10_000, noSpeechMs = 8000 } = {}) {
    this.nextUttId = nextUttId;
    this.vad = vad;
    this.prerollMs = prerollMs;
    this.maxMs = maxMs;
    this.noSpeechMs = noSpeechMs;
    this.pre = []; // [{pcm, t}]
    this.utt = null;
    this.startT = null;
    this.openedAt = null;
    this.finished = false;
  }
  push(pcm16, t) {
    const out = [];
    if (this.finished) return out;
    if (this.openedAt === null) this.openedAt = t;
    const f = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) f[i] = pcm16[i] / 32768;
    const ev = this.vad.update(rms(f), t);
    if (this.utt === null) {
      this.pre.push({ pcm: pcm16, t });
      while (this.pre.length && t - this.pre[0].t > this.prerollMs) this.pre.shift();
      if (ev === 'start') {
        this.utt = this.nextUttId();
        this.startT = this.pre[0]?.t ?? t;
        out.push({ kind: 'start', utt: this.utt });
        for (const p of this.pre) out.push({ kind: 'audio', utt: this.utt, pcm: p.pcm });
        this.pre = [];
      } else if (t - this.openedAt >= this.noSpeechMs) {
        this.finished = true;
        out.push({ kind: 'timeout' });
      }
      return out;
    }
    out.push({ kind: 'audio', utt: this.utt, pcm: pcm16 });
    const dur = t - this.startT;
    if (ev === 'end' || dur >= this.maxMs) {
      out.push({ kind: 'end', utt: this.utt, dur_ms: Math.round(dur) });
      this.finished = true;
    }
    return out;
  }
  /** Close an utterance in progress (e.g. the user stopped listening). */
  flush(t) {
    if (this.finished || this.utt === null) { this.finished = true; return []; }
    this.finished = true;
    return [{ kind: 'end', utt: this.utt, dur_ms: Math.round(t - this.startT) }];
  }
}

// ---------------------------------------------------------------- browser capture

const WORKLET = `
class NodCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      this.buf.push(new Float32Array(ch));
      this.n += ch.length;
      if (this.n >= sampleRate / 20) { // about 50 ms per message
        const out = new Float32Array(this.n);
        let o = 0;
        for (const b of this.buf) { out.set(b, o); o += b.length; }
        this.port.postMessage(out, [out.buffer]);
        this.buf = []; this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('nod-capture', NodCapture);
`;

/**
 * Opens the microphone and delivers 16 kHz PCM16 chunks to onPcm(int16, t_ms).
 * Throws if the mic is unavailable; callers fall back to the text box.
 */
export class MicCapture {
  constructor({ onPcm }) { this.onPcm = onPcm; this.ctx = null; this.stream = null; this.node = null; }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const AC = globalThis.AudioContext || globalThis.webkitAudioContext;
    try { this.ctx = new AC({ sampleRate: TARGET_SR }); } catch { this.ctx = new AC(); }
    const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
    try { await this.ctx.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'nod-capture');
    const rs = new Resampler(this.ctx.sampleRate, TARGET_SR);
    this.node.port.onmessage = (ev) => this.onPcm(floatTo16(rs.push(ev.data)), performance.now());
    src.connect(this.node);
    // Keep the graph pulling without making a sound.
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    this.node.connect(mute).connect(this.ctx.destination);
    if (this.ctx.state === 'suspended') await this.ctx.resume();
  }

  stop() {
    try { this.stream?.getTracks().forEach((t) => t.stop()); } catch { /* ignore */ }
    try { this.ctx?.close(); } catch { /* ignore */ }
    this.stream = null; this.ctx = null; this.node = null;
  }
}
