// On-device keyword "stop" (design 5.5, 6.1) with TF.js speech-commands.
//
// Optional and failure tolerant: if the scripts, the model or the microphone
// are unavailable, start() resolves to false and the other stop channels (jaw
// hold, STOP button) still work. Not the Web Speech API, which is not on device.
//
// Loads vendor/ copies first (scripts/vendor.sh), then falls back to the CDN.
// speech-commands 0.5.4 declares tfjs ^3 as its peer, so tfjs is pinned to 3.21.0.

const TFJS_VER = '3.21.0';
const SC_VER = '0.5.4';
const SOURCES = {
  tf: ['vendor/tfjs/tf.min.js', `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@${TFJS_VER}/dist/tf.min.js`],
  sc: ['vendor/speech-commands/speech-commands.min.js', `https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands@${SC_VER}/dist/speech-commands.min.js`],
};
const VENDOR_MODEL = 'vendor/speech-commands/18w/model.json';
const VENDOR_META = 'vendor/speech-commands/18w/metadata.json';

export const STOP_THRESHOLD = 0.75;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.onload = () => resolve(src);
    s.onerror = () => { s.remove(); reject(new Error(`failed to load ${src}`)); };
    document.head.appendChild(s);
  });
}

async function loadFirst(list) {
  let last;
  for (const src of list) {
    try { return await loadScript(src); } catch (e) { last = e; }
  }
  throw last;
}

async function vendored() {
  try {
    const r = await fetch(new URL('vendor/manifest.json', document.baseURI).href);
    return r.ok ? (await r.json()) || {} : {};
  } catch {
    return {};
  }
}

/** Pick the score of "stop" from a recognizer result. Pure; tested in node. */
export function stopScore(scores, labels) {
  const i = labels.indexOf('stop');
  return i >= 0 && scores && scores.length > i ? scores[i] : 0;
}

export class StopKeyword {
  /** @param {{onStop:(p:number)=>void, onStatus?:(s:string)=>void, threshold?:number}} opts */
  constructor({ onStop, onStatus, threshold = STOP_THRESHOLD }) {
    this.onStop = onStop;
    this.onStatus = onStatus || (() => {});
    this.threshold = threshold;
    this.rec = null;
    this.status = 'off';
  }

  _set(s) { this.status = s; this.onStatus(s); }

  /** Resolves true when listening, false when unavailable (never throws). */
  async start() {
    if (this.status === 'listening') return true;
    this._set('loading');
    try {
      const man = await vendored();
      // Vendored copies first when the manifest lists them; the CDN otherwise.
      if (!globalThis.tf) await loadFirst(man.tfjs ? SOURCES.tf : SOURCES.tf.slice(1));
      if (!globalThis.speechCommands) await loadFirst(man.speech_commands ? SOURCES.sc : SOURCES.sc.slice(1));
      const sc = globalThis.speechCommands;
      const absolute = (p) => new URL(p, document.baseURI).href;
      this.rec = man.speech_commands_model
        ? sc.create('BROWSER_FFT', undefined, absolute(VENDOR_MODEL), absolute(VENDOR_META))
        : sc.create('BROWSER_FFT');
      await this.rec.ensureModelLoaded();
      const labels = this.rec.wordLabels();
      if (!labels.includes('stop')) throw new Error('model has no "stop" word');
      await this.rec.listen(async (result) => {
        const p = stopScore(result.scores, labels);
        if (p >= this.threshold) this.onStop(p);
      }, { probabilityThreshold: this.threshold, overlapFactor: 0.5, includeSpectrogram: false, invokeCallbackOnNoiseAndUnknown: false });
      this._set('listening');
      return true;
    } catch (e) {
      console.warn('[kws] unavailable:', e?.message || e);
      this._set('unavailable');
      return false;
    }
  }

  async stop() {
    try { if (this.rec?.isListening()) await this.rec.stopListening(); } catch { /* ignore */ }
    this._set('off');
  }
}
