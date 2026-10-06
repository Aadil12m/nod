// Phone e-stop latch (design 5.5).
//
// On detection the phone latches at once (full-screen red STOPPED, vibration,
// spoken "Stopped"), then sends `estop` on /ws/safety every 100 ms until an
// `estop.ack` with the same estop_seq arrives. Only `estop.reset` clears the
// latch, and only after the gateway reports it has left ESTOPPED.
//
// No DOM and no timers here: the app calls `tick()` every 100 ms (and right
// after the safety socket reconnects), so the logic is testable with a fake clock.

import { msg } from './proto.js';

export const RESEND_MS = 100;

export class EStopController {
  /**
   * @param {{send:(m:any)=>any, now?:()=>number, onChange?:(c:EStopController)=>void,
   *          vibrate?:(p:number[])=>void, speak?:(t:string)=>void}} deps
   *   send: sends a proto.msg tuple on /ws/safety; returns a falsy value if it could not be sent.
   */
  constructor({ send, now = () => Date.now(), onChange, vibrate, speak, getToken } = {}) {
    this.sendFn = send;
    this.getToken = getToken || (() => null); // resume_token from `welcome`: authorises estop.reset
    this.now = now;
    this.onChange = onChange || (() => {});
    this.vibrate = vibrate || (() => {});
    this.speak = speak || (() => {});
    this.latched = false;
    this.seq = 0; // estop_seq of the current (or last) stop
    this.knownSeq = 0; // highest estop_seq the gateway has reported
    this.acked = false;
    this.ack = null;
    this.source = null;
    this.tDetect = null;
    this.heldMs = null;
    this.lastSent = -Infinity;
    this.sent = 0; // estop messages sent for the current stop
    this.resetPending = false;
    this.resetSeq = null; // estop_seq the pending reset asked to clear
    this.latchedAt = null;
    this.ackedAt = null;
  }

  /** Latch and start sending. Returns false if already latched (one stop at a time). */
  trigger(source, { held_ms = null, t_detect } = {}) {
    if (this.latched) return false;
    const now = this.now();
    this.latched = true;
    this.latchedAt = now;
    this.seq = Math.max(this.seq, this.knownSeq) + 1;
    this.acked = false;
    this.ackedAt = null;
    this.ack = null;
    this.resetPending = false;
    this.source = source;
    this.tDetect = Math.round(t_detect ?? now);
    this.heldMs = held_ms;
    this.sent = 0;
    this.lastSent = -Infinity;
    // Local effects first; they must not wait on the network.
    this.onChange(this);
    try { this.vibrate([200, 100, 200]); } catch { /* not supported */ }
    try { this.speak('Stopped'); } catch { /* not supported */ }
    this._send(now);
    return true;
  }

  _send(now) {
    const ok = this.sendFn(msg.estop({ estop_seq: this.seq, source: this.source, t_detect: this.tDetect, held_ms: this.heldMs }));
    if (ok) { this.sent++; this.lastSent = now; }
  }

  /** Call every 100 ms: resends until acked. */
  tick() {
    if (!this.latched || this.acked) return;
    const now = this.now();
    if (now - this.lastSent >= RESEND_MS - 5) this._send(now);
  }

  /** Feed every gateway message (from either socket). */
  onMessage(m) {
    if (m.type === 'estop.ack') {
      if (Number.isFinite(m.estop_seq)) this.knownSeq = Math.max(this.knownSeq, m.estop_seq);
      if (this.latched && m.estop_seq === this.seq && !this.acked) {
        this.acked = true;
        this.ackedAt = this.now();
        this.ack = m;
        this.onChange(this);
      }
    } else if (m.type === 'state') {
      const s = Number.isFinite(m.estop_seq) ? m.estop_seq : 0;
      this.knownSeq = Math.max(this.knownSeq, s);
      if (m.mode === 'ESTOPPED' && !this.latched) {
        // Stopped from elsewhere (presenter): latch here too, already known to the gateway.
        this.latched = true;
        this.latchedAt = this.now();
        this.seq = s;
        this.acked = true;
        this.ackedAt = this.latchedAt;
        this.source = 'presenter';
        this.resetPending = false;
        this.onChange(this);
        try { this.vibrate([200, 100, 200]); } catch { /* ignore */ }
        try { this.speak('Stopped'); } catch { /* ignore */ }
      } else if (this.latched && m.mode === 'ESTOPPED' && this.resetPending && s > this.resetSeq) {
        // A newer stop (presenter, caregiver) arrived while our reset was in flight: the gateway
        // refused it. Stay latched and offer "Clear stop?" again for the newest stop.
        this.resetPending = false;
        this.seq = Math.max(this.seq, s);
        this.onChange(this);
      } else if (this.latched && m.mode !== 'ESTOPPED') {
        if (s >= this.seq && (this.resetPending || this.acked)) {
          // Cleared: by our own reset, or by an authorised reset from another device (caregiver).
          this._unlatch();
        } else if (!this.resetPending && s < this.seq && this.acked) {
          // The gateway does not know about our stop (it restarted?): send it again.
          this.acked = false;
          this.lastSent = -Infinity;
          this.tick();
        }
      }
    }
  }

  /** Clearing is offered only once the gateway has acknowledged the stop. */
  canReset() { return this.latched && this.acked; }

  /** Ask the gateway to clear the stop (long blink on "Clear stop?", or caregiver touch). */
  requestReset(gesture, held_ms = null) {
    if (!this.canReset()) return false;
    // Clear the newest stop the gateway has told us about (ours, or a newer one from elsewhere);
    // the gateway refuses a reset for an older stop.
    const seq = Math.max(this.seq, this.knownSeq);
    const ok = this.sendFn(msg.estopReset({ estop_seq: seq, gesture, held_ms, resume_token: this.getToken() }));
    if (ok) { this.resetPending = true; this.resetSeq = seq; this.onChange(this); }
    return !!ok;
  }

  _unlatch() {
    this.latched = false;
    this.resetPending = false;
    this.acked = false;
    this.onChange(this);
  }
}
