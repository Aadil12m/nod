// Keyboard fallback for judges without a camera (design 4.4).
//
//   Left / Right  head turn        Hold Enter  long blink
//   B             brow raise       Space       STOP
//
// Keys become synthetic face frames fed to their own GestureMachine (default
// thresholds), so the keyboard goes through the same timing rules as the
// camera: Enter must be held for the long-blink time, and the ring fills while
// it is held. Space is a direct STOP (no 300 ms hold), because it is a button.
//
// Pure apart from the optional DOM binding in `bindKeyboard`.

import { GestureMachine } from './gestures.js';

export const TURN_PULSE_MS = 400;
export const BROW_PULSE_MS = 550;
export const TURN_DEG = 25;

export class KeyboardGestures {
  constructor(config = {}) {
    this.gm = new GestureMachine({ double_blink_enabled: false, ...config });
    this.enter = false;
    this.yawUntil = -Infinity;
    this.yaw = 0;
    this.browUntil = -Infinity;
  }

  /** Returns 'stop' for Space, true if the key was used, false otherwise. */
  keydown(key, t) {
    switch (key) {
      case 'ArrowLeft': this.yaw = TURN_DEG; this.yawUntil = t + TURN_PULSE_MS; return true; // the user's left
      case 'ArrowRight': this.yaw = -TURN_DEG; this.yawUntil = t + TURN_PULSE_MS; return true;
      case 'Enter': this.enter = true; return true;
      case 'b': case 'B': this.browUntil = t + BROW_PULSE_MS; return true;
      case ' ': case 'Spacebar': return 'stop';
      default: return false;
    }
  }

  keyup(key) {
    if (key === 'Enter') { this.enter = false; return true; }
    return false;
  }

  frame(t) {
    return {
      t_ms: t, face_present: true,
      blendshapes: {
        eyeBlinkLeft: this.enter ? 1 : 0, eyeBlinkRight: this.enter ? 1 : 0,
        browInnerUp: t < this.browUntil ? 0.9 : 0, jawOpen: 0,
      },
      yaw_deg: t < this.yawUntil ? this.yaw : 0, pitch_deg: 0,
    };
  }

  /** Advance to time t; returns gesture events. Call at about 30 Hz. */
  tick(t) { return this.gm.update(this.frame(t)); }
}

/**
 * Bind to a document. onEvents(events) gets gesture events, onStop() is Space.
 * Keys typed into text fields are left alone.
 */
export function bindKeyboard(doc, { onEvents, onStop, now = () => performance.now(), hz = 30 }) {
  const kb = new KeyboardGestures();
  const typing = (e) => {
    const el = e.target;
    return el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
  };
  doc.addEventListener('keydown', (e) => {
    if (typing(e) || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.repeat) { if (['Enter', ' ', 'ArrowLeft', 'ArrowRight'].includes(e.key)) e.preventDefault(); return; }
    const r = kb.keydown(e.key, now());
    if (r === 'stop') { e.preventDefault(); onStop(); } else if (r) e.preventDefault();
  });
  doc.addEventListener('keyup', (e) => { if (kb.keyup(e.key)) e.preventDefault(); });
  // Losing focus while Enter is held must not leave the "eyes" closed.
  globalThis.addEventListener?.('blur', () => kb.keyup('Enter'));
  const timer = setInterval(() => { const ev = kb.tick(now()); if (ev.length) onEvents(ev); }, 1000 / hz);
  return { kb, stop: () => clearInterval(timer) };
}
