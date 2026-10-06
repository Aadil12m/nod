// Phone side of the Nod wire protocol (design 3.5.1-3.5.3).
//
// The source of truth is the Python package `nod_protocol`; its generated JSON
// Schema (protocol/schema/*.json) lists every field as required, so each
// builder here fills every field, using null where a value is absent. The node
// tests validate every builder's output against those schema files.
//
// Pure module: no DOM, no sockets.

export const PROTOCOL_VERSION = 1;
export const APP_VERSION = '0.3.0';

// ---------------------------------------------------------------- ids and clocks

const CROCKFORD = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
let lastMs = -1;
let lastRand = null;

function randomBytes(n) {
  const b = new Uint8Array(n);
  if (globalThis.crypto?.getRandomValues) globalThis.crypto.getRandomValues(b);
  else for (let i = 0; i < n; i++) b[i] = Math.floor(Math.random() * 256);
  return b;
}

/** 26-character ULID; monotonic within one page (same-ms ids increment the random part). */
export function ulid(now = Date.now()) {
  let rand;
  if (now === lastMs && lastRand) {
    rand = lastRand.slice();
    for (let i = rand.length - 1; i >= 0; i--) { if (rand[i] === 255) { rand[i] = 0; } else { rand[i]++; break; } }
  } else rand = randomBytes(10);
  lastMs = now;
  lastRand = rand;
  let time = '';
  let t = Math.floor(now);
  for (let i = 0; i < 10; i++) { time = CROCKFORD[t % 32] + time; t = Math.floor(t / 32); }
  // 80 random bits -> 16 base32 characters.
  let bits = 0, acc = 0, out = '';
  for (const byte of rand) {
    acc = (acc << 8) | byte; bits += 8;
    while (bits >= 5) { bits -= 5; out += CROCKFORD[(acc >> bits) & 31]; }
    acc &= (1 << bits) - 1;
  }
  return time + out;
}

export function newId(prefix) { return `${prefix}_${ulid()}`; }

// ---------------------------------------------------------------- envelope

/**
 * Wrap a payload in the common envelope {v, type, id, ts, sess, seq}.
 * `seq` must increase per sender per socket (net.js keeps the counter).
 */
export function envelope(type, payload, { sess = null, seq = 0, now = Date.now() } = {}) {
  return { v: PROTOCOL_VERSION, type, id: ulid(now), ts: Math.round(now), sess, seq, ...payload };
}

const int = (x) => (x === null || x === undefined ? null : Math.round(x));
const num = (x) => (x === null || x === undefined || !Number.isFinite(x) ? null : Math.round(x * 100) / 100);

// ---------------------------------------------------------------- /ws/phone (3.5.1)

export const msg = {
  hello: ({ device, role = 'user', resume_token = null, calib = null }) => ['hello', {
    device: String(device || 'web'), app_ver: APP_VERSION, role, resume_token,
    calib: calib ? { profile_id: calib.profile_id, long_blink_ms: int(calib.long_blink_ms), fps: num(calib.fps) ?? 0 } : null,
  }],
  /** Telemetry for a fired or cancelled gesture event from gestures.js. */
  gesture: (ev) => ['gesture', {
    name: ev.name, phase: ev.phase, held_ms: int(ev.held_ms), yaw_deg: num(ev.yaw_deg), pitch_deg: num(ev.pitch_deg),
    scores: { ...(ev.scores || {}) }, fps: num(ev.fps),
  }],
  uiMenu: (source) => ['ui.menu', { source }],
  uiFocus: (proposal_id, option_id) => ['ui.focus', { proposal_id, option_id }],
  uiConfirm: ({ proposal_id, option_id, nonce, gesture = 'long_blink', held_ms, shown_at, focused_at }) => ['ui.confirm', {
    proposal_id, option_id, nonce, gesture, held_ms: int(held_ms), shown_at: int(shown_at), focused_at: int(focused_at),
  }],
  uiCancel: (proposal_id, source) => ['ui.cancel', { proposal_id: proposal_id ?? null, source }],
  utterStart: (utt_id) => ['utter.start', { utt_id, sr: 16000, fmt: 'pcm16le' }],
  utterEnd: (utt_id, dur_ms) => ['utter.end', { utt_id, dur_ms: int(dur_ms) }],
  utterText: (utt_id, text) => ['utter.text', { utt_id, text: String(text) }],
  ping: (t0 = Date.now()) => ['ping', { t0: int(t0) }],
  // ------------------------------------------------------------ /ws/safety (3.5.2)
  hb: ({ face_present, track_conf, fps, latched }) => ['hb', {
    face_present: !!face_present, track_conf: num(track_conf) ?? 0, fps: num(fps) ?? 0, latched: !!latched,
  }],
  estop: ({ estop_seq, source, t_detect, held_ms = null }) => ['estop', { estop_seq, source, t_detect: int(t_detect), held_ms: int(held_ms) }],
  estopReset: ({ estop_seq, gesture, held_ms = null }) => ['estop.reset', { estop_seq, gesture, held_ms: int(held_ms) }],
};

export const ESTOP_SOURCES = ['jaw_hold', 'voice_stop', 'button', 'presenter'];
export const GESTURE_NAMES = ['long_blink', 'double_blink', 'brow_raise', 'turn_left', 'turn_right', 'jaw_hold', 'face_lost', 'face_found'];

// ---------------------------------------------------------------- binary audio frames

/** [u32 little-endian utt_id][pcm16le samples] (3.5.1). */
export function audioFrame(uttId, pcm16) {
  const buf = new ArrayBuffer(4 + pcm16.length * 2);
  const dv = new DataView(buf);
  dv.setUint32(0, uttId >>> 0, true);
  for (let i = 0; i < pcm16.length; i++) dv.setInt16(4 + 2 * i, pcm16[i], true);
  return buf;
}

export function parseAudioFrame(buf) {
  const dv = new DataView(buf);
  const uttId = dv.getUint32(0, true);
  const n = (buf.byteLength - 4) >> 1;
  const pcm = new Int16Array(n);
  for (let i = 0; i < n; i++) pcm[i] = dv.getInt16(4 + 2 * i, true);
  return { uttId, pcm };
}

// ---------------------------------------------------------------- gateway to phone (3.5.3)

export const GATEWAY_TYPES = new Set(['welcome', 'pong', 'state', 'transcript', 'proposal', 'run', 'estop.ack', 'notice', 'thumb']);
export const PRESENTER_TYPES = new Set(['trace', 'metrics', 'cosmos.check', 'cosmos.result']);
export const TERMINAL_RUN_PHASES = new Set(['done', 'failed', 'uncertain', 'aborted']);
export const SESSION_MODES = ['IDLE', 'LISTENING', 'PLANNING', 'AWAITING_CONFIRM', 'COUNTDOWN', 'EXECUTING', 'POSTCHECK', 'PAUSED', 'ESTOPPED'];

/** Parse a text frame; returns the object or null if it is not a typed protocol message. */
export function parseMessage(text) {
  try {
    const m = JSON.parse(text);
    return m && typeof m === 'object' && typeof m.type === 'string' ? m : null;
  } catch {
    return null;
  }
}
