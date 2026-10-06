// Face tracking with MediaPipe Face Landmarker (design 6.2).
//
// @mediapipe/tasks-vision 1.0.1, VIDEO mode, one face, blendshapes and facial
// transformation matrixes, GPU delegate with CPU fallback. The camera frame is
// never mirrored before inference (mirroring swaps the face's Left and Right
// blendshapes); only the preview <video> is mirrored, with CSS.
// Camera frames never leave the phone; only gesture events are sent.
//
// Files come from vendor/ (scripts/vendor.sh) when present, else from the CDN.

export const TASKS_VISION_VER = '1.0.1';
export const CDN_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VER}`;
export const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
export const VENDOR = {
  bundle: 'vendor/mediapipe/vision_bundle.mjs',
  wasm: 'vendor/mediapipe/wasm',
  model: 'vendor/models/face_landmarker.task',
};

const DEG = 180 / Math.PI;

/**
 * Head yaw and pitch in degrees from the 4x4 facial transformation matrix,
 * stored column-major: R(r, c) = data[c * 4 + r]. Uniform scale cancels.
 * The sign of yaw is not assumed; calibration records which sign "left" gives.
 */
export function yawPitchFromMatrix(data) {
  if (!data || data.length < 16) return { yaw: 0, pitch: 0 };
  const R = (r, c) => data[c * 4 + r];
  const yaw = Math.atan2(-R(2, 0), Math.hypot(R(0, 0), R(1, 0))) * DEG;
  const pitch = Math.atan2(R(2, 1), R(2, 2)) * DEG;
  return { yaw, pitch };
}

/** Blendshape categories to {name: score}; look up by name, never by index. */
export function blendshapeMap(categories) {
  const out = {};
  for (const c of categories || []) out[c.categoryName] = c.score;
  return out;
}

/** Convert one FaceLandmarkerResult to the frame format gestures.js expects. */
export function frameFromResult(res, t_ms) {
  const bs = res?.faceBlendshapes?.[0]?.categories;
  const m = res?.facialTransformationMatrixes?.[0]?.data;
  if (!bs || !bs.length) return { t_ms, face_present: false, blendshapes: {}, yaw_deg: 0, pitch_deg: 0 };
  const { yaw, pitch } = yawPitchFromMatrix(m);
  return { t_ms, face_present: true, blendshapes: blendshapeMap(bs), yaw_deg: yaw, pitch_deg: pitch };
}

/**
 * vendor/manifest.json is written by scripts/vendor.sh and lists what was vendored.
 * Reading it (a GET, cached by the service worker) works offline, unlike a HEAD probe.
 */
export async function vendorManifest(base = document.baseURI) {
  try {
    const r = await fetch(new URL('vendor/manifest.json', base).href);
    if (!r.ok) return null;
    const m = await r.json();
    return m && typeof m === 'object' ? m : null;
  } catch {
    return null;
  }
}

/** Resolve where the library, wasm and model come from: vendor/ first, CDN second. */
export async function resolveSources(base = document.baseURI) {
  const abs = (p) => new URL(p, base).href;
  const man = await vendorManifest(base);
  const lib = !!man?.mediapipe;
  return {
    bundle: lib ? abs(VENDOR.bundle) : `${CDN_BASE}/vision_bundle.mjs`,
    wasm: lib ? abs(VENDOR.wasm) : `${CDN_BASE}/wasm`,
    model: man?.face_model ? abs(VENDOR.model) : MODEL_URL,
    vendored: lib,
  };
}

/**
 * Camera plus Face Landmarker loop. onFrame receives every processed frame:
 * {t_ms, face_present, blendshapes, yaw_deg, pitch_deg, track_conf, fps}
 * where t_ms is performance.now() and track_conf is the fraction of frames in
 * the last second that had a face.
 */
export class FaceTracker {
  constructor({ video, onFrame, onStatus } = {}) {
    this.video = video;
    this.onFrame = onFrame || (() => {});
    this.onStatus = onStatus || (() => {});
    this.landmarker = null;
    this.delegate = null;
    this.stream = null;
    this.running = false;
    this.lastTs = 0;
    this.lastVideoTime = -1;
    this.recent = []; // [{t, present}] for track_conf and fps
    this.sources = null;
  }

  async _create(vision, delegate) {
    const fileset = await vision.FilesetResolver.forVisionTasks(this.sources.wasm);
    return vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: this.sources.model, delegate },
      runningMode: 'VIDEO',
      numFaces: 1, // temporal smoothing only applies with one face
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
  }

  /** Opens the camera and loads the model. Throws if either fails (the app then offers the keyboard). */
  async start() {
    this.onStatus('loading');
    this.sources = await resolveSources();
    this.vision = await import(/* @vite-ignore */ this.sources.bundle);
    try {
      this.landmarker = await this._create(this.vision, 'GPU');
      this.delegate = 'GPU';
    } catch (e) {
      console.warn('[face] GPU delegate failed, using CPU:', e?.message || e);
      this.landmarker = await this._create(this.vision, 'CPU');
      this.delegate = 'CPU';
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
    });
    this.video.srcObject = this.stream;
    this.video.muted = true;
    this.video.playsInline = true;
    await this.video.play();
    this.running = true;
    this.onStatus(`running (${this.delegate}${this.sources.vendored ? ', vendored' : ', CDN'})`);
    this._schedule();
  }

  _schedule() {
    if (!this.running) return;
    if (this.video.requestVideoFrameCallback) this.video.requestVideoFrameCallback(() => this._tick());
    else requestAnimationFrame(() => this._tick());
  }

  async _tick() {
    if (!this.running) return;
    try {
      if (this.video.readyState >= 2 && this.video.currentTime !== this.lastVideoTime) {
        this.lastVideoTime = this.video.currentTime;
        let ts = performance.now();
        if (ts <= this.lastTs) ts = this.lastTs + 0.001; // detectForVideo needs increasing timestamps
        this.lastTs = ts;
        let res;
        try {
          res = this.landmarker.detectForVideo(this.video, ts);
        } catch (e) {
          if (this.delegate !== 'GPU') throw e;
          console.warn('[face] GPU inference failed, switching to CPU:', e?.message || e);
          this.landmarker.close();
          this.landmarker = await this._create(this.vision, 'CPU');
          this.delegate = 'CPU';
          this.onStatus('running (CPU)');
          this._schedule();
          return;
        }
        const f = frameFromResult(res, ts);
        this.recent.push({ t: ts, present: f.face_present });
        while (this.recent.length > 1 && ts - this.recent[0].t > 1000) this.recent.shift();
        f.track_conf = this.recent.filter((r) => r.present).length / this.recent.length;
        const span = ts - this.recent[0].t;
        f.fps = span > 0 ? ((this.recent.length - 1) * 1000) / span : 0;
        this.onFrame(f);
      }
    } catch (e) {
      console.error('[face] tracking error:', e);
      this.onStatus(`error: ${e?.message || e}`);
    }
    this._schedule();
  }

  stop() {
    this.running = false;
    try { this.stream?.getTracks().forEach((t) => t.stop()); } catch { /* ignore */ }
    try { this.landmarker?.close(); } catch { /* ignore */ }
    this.landmarker = null;
  }
}
