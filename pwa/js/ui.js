// Phone UI: state machine (pure) and screens (DOM). Design 3.4, 5.5, 5.7, 6.
//
// UiMachine is pure: it takes gesture events, gateway messages and clock
// ticks, decides what they mean in the current screen, and emits messages
// through injected `send` / `speak` functions. It never touches the DOM, so it
// is tested in node. UiView renders UiMachine.view() into the page.
//
// Screens: connecting, idle, listening, planning, choose, confirming,
// countdown, executing, postcheck, paused, result, stopped. Overlays: "I can't
// see you", low frame rate, REPLAY badge.

import { msg, TERMINAL_RUN_PHASES } from './proto.js';

export const DWELL_MS = 300; // focus must rest this long before an option is armed (5.7)
export const COUNTDOWN_MS = 1000;
export const RESULT_MS = 6000; // how long done / failed / uncertain stays up
export const CONFIRM_WAIT_MS = 3000; // no answer to ui.confirm within this: "Try again"
export const DOUBLE_CONFIRM_MS = 8000; // second blink of a double confirm must come within this
export const CLEAR_STOP_DELAY_MS = 1500; // "Clear stop?" is offered this long after the latch
export const LOCAL_STATE_MS = 4000; // optimistic listening/planning without a gateway answer

const MOTION_MODES = new Set(['COUNTDOWN', 'EXECUTING', 'POSTCHECK', 'PAUSED']);
// PAUSED keeps a proposal: the gateway offers "Resume?" as a one-option proposal while paused.
const CLEARS_PROPOSAL = new Set(['IDLE', 'COUNTDOWN', 'EXECUTING', 'POSTCHECK', 'ESTOPPED']);
const TELEMETRY = new Set(['long_blink', 'double_blink', 'brow_raise', 'turn_left', 'turn_right', 'jaw_hold', 'face_lost', 'face_found']);

/** "19/20 in tests" when k is known (Success.k), else a percentage. */
export function successText(s) {
  if (!s) return '';
  if (Number.isInteger(s.k) && s.n) return `${s.k}/${s.n} in tests${s.basis === 'in_chain' ? ' (in a chain)' : ''}`;
  if (Number.isFinite(s.p_est)) return `${Math.round(s.p_est * 100)}% in ${s.n} tests`;
  return '';
}

const BADGE_TEXT = {
  less_reliable: 'Less reliable',
  needs_caregiver: 'Needs caregiver',
  unsure_from_camera: 'Unsure from camera',
  near_user: 'Comes near you',
  scripted: 'Scripted move',
  first_use: 'First use',
};
export const badgeText = (b) => BADGE_TEXT[b] || b;

const PRECHECK_TEXT = { pending: 'Checking camera...', ok: 'Camera check ok', unsure: 'Camera unsure', block: 'Camera says no', error: 'Camera check failed' };

export class UiMachine {
  /**
   * @param {{now?:()=>number, send:(ch:'phone'|'safety', m:any)=>any, speak?:(t:string, o?:any)=>void,
   *          estop:import('./estop.js').EStopController, clock?:{toPhone:(t:number)=>number},
   *          doubleBlinkBack?:boolean}} deps
   */
  constructor({ now = () => Date.now(), send, speak, estop, clock, doubleBlinkBack = false } = {}) {
    this.now = now;
    this.sendFn = send;
    this.speak = speak || (() => {});
    this.estop = estop;
    this.clock = clock || { toPhone: (t) => t };
    this.doubleBlinkBack = doubleBlinkBack;
    this.mode = null; // gateway session mode (3.4); null until the first `state`
    this.sceneId = null;
    this.proposal = null;
    this.run = null;
    this.result = null;
    this.pendingConfirm = null;
    this.lastConfirmed = null; // {option_id, label, plan_len}
    this.listeningLocal = null;
    this.planningLocal = null;
    this.transcript = '';
    this.toast = null;
    this.hint = null;
    this.ring = null;
    this.faceLost = false;
    this.fpsLow = false;
    this.connected = { phone: false, safety: false };
    this.replay = false;
    this.countdownAt = null;
    this.uttSeq = 0;
    this.lastStopAt = -Infinity;
    this.localIdle = null; // cancelled here; show idle until the gateway catches up
  }

  send(ch, m) { return this.sendFn(ch, m); }
  nextUttId() { return ++this.uttSeq; }

  // ------------------------------------------------------------ gateway messages

  onServer(m) {
    const now = this.now();
    switch (m.type) {
      case 'welcome':
        if (m.mode === 'mock' || m.mode === 'replay') this.replay = true; // forward compatible
        break;
      case 'state': {
        this.estop?.onMessage(m);
        const prev = this.mode;
        this.mode = m.mode;
        if (m.mode !== prev || m.mode === 'IDLE') this.localIdle = null;
        if (m.scene_id) this.sceneId = m.scene_id;
        if (CLEARS_PROPOSAL.has(m.mode)) { this.proposal = null; this.ring = null; }
        if (m.mode !== 'IDLE') { this.listeningLocal = null; }
        if (m.mode !== 'IDLE' && m.mode !== 'LISTENING') this.planningLocal = null;
        if (m.mode !== prev && (MOTION_MODES.has(m.mode) || m.mode === 'AWAITING_CONFIRM' || m.mode === 'IDLE')) this.pendingConfirm = null;
        if (m.mode === 'COUNTDOWN' && prev !== 'COUNTDOWN') this.countdownAt = now;
        if (MOTION_MODES.has(m.mode) && m.mode !== 'POSTCHECK') this.result = null;
        break;
      }
      case 'proposal':
        this._setProposal(m, now);
        break;
      case 'run':
        this.run = { ...m, at: now };
        if (m.phase === 'countdown' && this.countdownAt === null) this.countdownAt = now;
        if (TERMINAL_RUN_PHASES.has(m.phase)) {
          // An aborted run follows a stop or cancel; the stopped screen already says so.
          if (m.phase !== 'aborted') this.result = { phase: m.phase, at: now, run: m, reason: m.cosmos?.reason || '', fault: m.fault || null };
          this.countdownAt = null;
        }
        break;
      case 'transcript':
        this.transcript = m.text || '';
        break;
      case 'notice':
        this.toast = { text: m.text, level: m.level || 'info', at: now };
        if (m.speak) this.speak(m.text);
        break;
      case 'estop.ack':
        this.estop?.onMessage(m);
        break;
      default:
        break;
    }
  }

  _setProposal(m, now) {
    const fixed = Array.isArray(m.fixed) && m.fixed.length ? m.fixed : [{ option_id: 'none', label: 'None of these' }];
    const items = [...(m.options || []).map((o) => ({ ...o, fixed: false })), ...fixed.map((f) => ({ ...f, fixed: true }))];
    let idx = items.findIndex((it) => it.option_id === m.focus);
    if (idx < 0) idx = 0;
    this.proposal = { msg: m, items, focusIdx: idx, shownAt: now, focusedAt: now, stage: 0, stageAt: null };
    this.localIdle = null;
    this.listeningLocal = null;
    this.planningLocal = null;
    this.pendingConfirm = null;
    this.result = null;
    this.ring = null;
    this.hint = null;
    if (m.transcript) this.transcript = m.transcript;
    const n = m.options?.length || 0;
    this.speak(`${n} option${n === 1 ? '' : 's'}. ${items[idx].label}`);
  }

  /** Time (phone clock) at which the focused option becomes armed. */
  armedAt() {
    const p = this.proposal;
    if (!p) return Infinity;
    const arm = Number.isFinite(p.msg.arm_delay_ms) ? p.msg.arm_delay_ms : 300;
    return Math.max(p.shownAt + arm, p.focusedAt + DWELL_MS);
  }

  _proposalExpired(now) {
    const p = this.proposal;
    return !!p && Number.isFinite(p.msg.expires_at) && now > this.clock.toPhone(p.msg.expires_at);
  }

  // ------------------------------------------------------------ gesture events

  onGesture(ev) {
    const now = this.now();
    if ((ev.phase === 'fired' || ev.phase === 'cancelled') && TELEMETRY.has(ev.name)) this.send('phone', msg.gesture(ev));
    switch (ev.name) {
      case 'jaw_hold':
        if (ev.phase === 'fired') this.stop('jaw_hold', { held_ms: ev.held_ms });
        break;
      case 'face_lost': this.faceLost = true; this.ring = null; break;
      case 'face_found': this.faceLost = false; break;
      case 'fps_low': this.fpsLow = true; break;
      case 'fps_ok': this.fpsLow = false; break;
      case 'long_blink':
        if (ev.phase === 'start' || ev.phase === 'progress') {
          const start = now - (ev.held_ms || 0);
          this.ring = { progress: ev.progress, armed: this._ringArmed(start), at: now };
        } else if (ev.phase === 'cancelled') this.ring = null;
        else if (ev.phase === 'fired') { this.ring = null; this._longBlink(ev, now); }
        break;
      case 'brow_raise':
        if (ev.phase === 'fired' && (this.screen() === 'idle' || this.screen() === 'result')) this.menu('brow_raise');
        break;
      case 'turn_left':
      case 'turn_right':
        if (ev.phase === 'fired' && this.screen() === 'choose') this.moveFocus(ev.name === 'turn_right' ? 1 : -1);
        break;
      case 'double_blink':
        if (this.doubleBlinkBack && ev.phase === 'fired' && this.screen() === 'choose') this.cancelProposal('double_blink');
        break;
      default:
        break;
    }
  }

  /** Would a long blink that started at `start` count on the current screen? */
  _ringArmed(start) {
    const sc = this.screen();
    if (sc === 'choose') {
      const p = this.proposal;
      const it = p.items[p.focusIdx];
      if (it.locked) return false;
      if (p.stage === 1 && start < p.stageAt) return false;
      return start >= this.armedAt();
    }
    if (sc === 'stopped') return this._clearOfferedAt() !== null && start >= this._clearOfferedAt();
    return false;
  }

  _longBlink(ev, now) {
    const start = now - (ev.held_ms || 0);
    const sc = this.screen();
    if (sc === 'stopped') {
      const at = this._clearOfferedAt();
      if (at !== null && start >= at) this.estop.requestReset('long_blink', ev.held_ms);
      return;
    }
    if (sc !== 'choose') return;
    const p = this.proposal;
    const it = p.items[p.focusIdx];
    if (start < this.armedAt()) {
      this.hint = 'Hold your gaze on one card for a moment, then close your eyes';
      return;
    }
    if (it.fixed) { this.cancelProposal('none_of_these'); return; }
    if (it.locked) {
      this.hint = 'That needs a caregiver';
      this.speak('That needs a caregiver');
      return;
    }
    if (it.confirm === 'double' && p.stage === 0) {
      p.stage = 1;
      p.stageAt = now;
      this.hint = 'Blink long once more to confirm';
      this.speak('Once more to confirm');
      return;
    }
    if (p.stage === 1 && start < p.stageAt) return;
    this.send('phone', msg.uiConfirm({
      proposal_id: p.msg.proposal_id, option_id: it.option_id, nonce: p.msg.nonce, gesture: 'long_blink',
      held_ms: ev.held_ms, shown_at: p.shownAt, focused_at: p.focusedAt,
    }));
    this.pendingConfirm = { at: now, option_id: it.option_id, label: it.label };
    this.lastConfirmed = { option_id: it.option_id, label: it.label, plan_len: it.skill_ids?.length || 1 };
    this.hint = null;
  }

  _clearOfferedAt() {
    const e = this.estop;
    if (!e?.canReset() || e.resetPending) return null;
    return Math.max((e.latchedAt ?? 0) + CLEAR_STOP_DELAY_MS, e.ackedAt ?? 0);
  }

  // ------------------------------------------------------------ actions (also used by buttons and keys)

  moveFocus(delta) {
    const p = this.proposal;
    if (!p) return;
    const idx = Math.max(0, Math.min(p.items.length - 1, p.focusIdx + delta)); // no wrap-around
    if (idx === p.focusIdx) return;
    p.focusIdx = idx;
    p.focusedAt = this.now();
    p.stage = 0;
    p.stageAt = null;
    this.ring = null;
    this.hint = null;
    const it = p.items[idx];
    if (!it.fixed) this.send('phone', msg.uiFocus(p.msg.proposal_id, it.option_id));
    this.speak(it.label);
  }

  menu(source) {
    if (this.estop?.latched) return;
    this.send('phone', msg.uiMenu(source));
    this.listeningLocal = { at: this.now() };
    this.localIdle = null;
    this.result = null;
    this.transcript = '';
  }

  cancelProposal(source) {
    const p = this.proposal;
    this.send('phone', msg.uiCancel(p?.msg.proposal_id ?? null, source));
    this.proposal = null;
    this.listeningLocal = null;
    this.planningLocal = null;
    this.ring = null;
    this.hint = null;
    this.localIdle = { at: this.now() };
    this.speak('Cancelled');
  }

  submitText(text) {
    const t = String(text || '').trim();
    if (!t || this.estop?.latched) return null;
    const id = this.nextUttId();
    this.send('phone', msg.utterText(id, t));
    this.transcript = t;
    this.planningLocal = { at: this.now() };
    this.listeningLocal = null;
    this.localIdle = null;
    this.result = null;
    return id;
  }

  /** Audio utterance finished (audio.js); the gateway now transcribes and plans. */
  utteranceEnded() {
    this.planningLocal = { at: this.now() };
    this.listeningLocal = null;
  }

  /** Nothing was said while listening. */
  listenTimeout() {
    if (this.screen() === 'listening') this.cancelProposal('timeout');
  }

  /** Is the arm moving, or about to? Then STOP must be an e-stop, not a cancel. */
  motionPossible() {
    if (MOTION_MODES.has(this.mode)) return true;
    if (this.pendingConfirm) return true; // a confirm is in flight
    if (this.run && !TERMINAL_RUN_PHASES.has(this.run.phase)) return true;
    return false;
  }

  /**
   * Any STOP channel: jaw hold, voice "stop", the STOP button, the Space key.
   * Moving (or about to move): latch the e-stop. Otherwise: cancel or back.
   */
  stop(source, { held_ms = null } = {}) {
    const now = this.now();
    if (this.estop?.latched) return 'latched';
    if (this.motionPossible()) {
      this.estop.trigger(source, { held_ms, t_detect: now });
      this.ring = null;
      return 'estop';
    }
    this.lastStopAt = now;
    const sc = this.screen();
    if (sc === 'choose' || sc === 'listening' || sc === 'planning') {
      this.cancelProposal(source);
      return 'cancel';
    }
    if (sc === 'result') { this.result = null; return 'back'; }
    return 'none';
  }

  /** Caregiver touch on the stopped screen. */
  clearByTouch() {
    return this.estop?.requestReset('touch', null) || false;
  }

  // ------------------------------------------------------------ time

  tick() {
    const now = this.now();
    if (this.proposal && this._proposalExpired(now)) {
      this.proposal = null;
      this.toast = { text: 'Options timed out', level: 'info', at: now };
    }
    if (this.pendingConfirm && now - this.pendingConfirm.at > CONFIRM_WAIT_MS) {
      this.pendingConfirm = null;
      this.hint = 'Try again';
    }
    if (this.proposal?.stage === 1 && now - this.proposal.stageAt > DOUBLE_CONFIRM_MS) {
      this.proposal.stage = 0;
      this.hint = null;
    }
    if (this.listeningLocal && now - this.listeningLocal.at > LOCAL_STATE_MS && this.mode !== 'LISTENING') this.listeningLocal = null;
    if (this.planningLocal && now - this.planningLocal.at > 4 * LOCAL_STATE_MS && this.mode !== 'PLANNING') this.planningLocal = null;
    if (this.result && now - this.result.at > RESULT_MS) this.result = null;
    if (this.localIdle && now - this.localIdle.at > LOCAL_STATE_MS) this.localIdle = null;
    if (this.toast && now - this.toast.at > 5000) this.toast = null;
    if (this.ring && now - this.ring.at > 500) this.ring = null; // stale (eyes reopened, no event)
  }

  /** Microphone wanted: gateway LISTENING, or the optimistic local state after a brow raise. */
  wantsMic() {
    return !this.estop?.latched && !this.localIdle && (this.mode === 'LISTENING' || !!this.listeningLocal);
  }

  // ------------------------------------------------------------ view model

  screen() {
    const now = this.now();
    if (this.estop?.latched) return 'stopped';
    if (this.mode === null && !this.connected.phone) return 'connecting';
    if (this.mode === 'PAUSED') {
      // The gateway offers "Resume?" (and "Cancel") as a proposal while paused (5.6).
      if (this.pendingConfirm) return 'confirming';
      return this.proposal && !this._proposalExpired(now) ? 'choose' : 'paused';
    }
    if (this.mode === 'COUNTDOWN') return 'countdown';
    if (this.mode === 'EXECUTING') return 'executing';
    if (this.mode === 'POSTCHECK') return 'postcheck';
    if (this.pendingConfirm) return 'confirming';
    if (this.proposal && !this._proposalExpired(now)) return 'choose';
    if (this.result) return 'result';
    if (this.localIdle) return 'idle';
    if (this.planningLocal || this.mode === 'PLANNING') return 'planning';
    if (this.listeningLocal || this.mode === 'LISTENING') return 'listening';
    if (this.mode === 'AWAITING_CONFIRM') return 'planning';
    return 'idle';
  }

  view() {
    const now = this.now();
    const screen = this.screen();
    const v = {
      screen, mode: this.mode, replay: this.replay, connected: { ...this.connected }, faceLost: this.faceLost,
      fpsLow: this.fpsLow, hint: this.hint, toast: this.toast?.text || null, toastLevel: this.toast?.level || null,
      transcript: this.transcript, mic: this.wantsMic(), ring: this.ring ? { ...this.ring } : null,
    };
    if (screen === 'choose') {
      const p = this.proposal;
      const armed = now >= this.armedAt();
      v.proposal_id = p.msg.proposal_id;
      v.source = p.msg.source;
      v.double = p.stage === 1;
      v.cards = p.items.map((it, i) => ({
        option_id: it.option_id, label: it.label, why: it.fixed ? '' : (it.why || ''), fixed: it.fixed,
        focused: i === p.focusIdx, armed: i === p.focusIdx && armed && !it.locked, locked: !!it.locked,
        success: successText(it.success), est: Number.isFinite(it.est_s) ? `about ${Math.round(it.est_s)} s` : '',
        risk: it.risk || null, kind: it.kind || null, steps: it.kind === 'plan' ? (it.skill_ids || []).length : 0,
        badges: (it.badges || []).map(badgeText), precheck: it.precheck ? PRECHECK_TEXT[it.precheck.status] || '' : '',
        confirm: it.confirm || 'single',
      }));
      v.focusIdx = p.focusIdx;
    }
    if (screen === 'countdown' || screen === 'executing' || screen === 'postcheck' || screen === 'paused' || screen === 'confirming') {
      const r = this.run;
      v.label = this.pendingConfirm?.label || this.lastConfirmed?.label || r?.skill_id || '';
      if (r) {
        v.progress = r.max_steps ? Math.min(1, (r.sim_step || 0) / r.max_steps) : 0;
        v.step = (r.plan_step ?? 0) + 1;
        v.of = r.plan_len ?? 1;
        v.elapsed_s = Math.round((r.elapsed_ms || 0) / 100) / 10;
        v.phase = r.phase;
      }
      if (screen === 'countdown') v.countdown = Math.max(0, 1 - (now - (this.countdownAt ?? now)) / COUNTDOWN_MS);
    }
    if (screen === 'result') {
      const r = this.result;
      v.result = {
        phase: r.phase,
        title: { done: 'Done', failed: "That didn't work", uncertain: "I'm not sure it worked", aborted: 'Stopped' }[r.phase] || r.phase,
        reason: r.reason || (r.fault ? `Fault: ${r.fault.replace(/_/g, ' ')}` : ''),
        label: this.lastConfirmed?.label || r.run?.skill_id || '',
      };
    }
    if (screen === 'stopped') {
      const e = this.estop;
      const at = this._clearOfferedAt();
      v.stopped = {
        acked: !!e.acked, resetPending: !!e.resetPending, source: e.source, seq: e.seq,
        canClear: at !== null && now >= at, lat_ms: e.ack?.lat_ms || null,
      };
    }
    return v;
  }
}

// =====================================================================================
// DOM view
// =====================================================================================

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const RING_R = 44;
const RING_C = 2 * Math.PI * RING_R;
const ringSvg = (cls = '') => `<svg class="ring ${cls}" viewBox="0 0 100 100" aria-hidden="true">
  <circle class="ring-bg" cx="50" cy="50" r="${RING_R}"></circle>
  <circle class="ring-fg" cx="50" cy="50" r="${RING_R}" stroke-dasharray="${RING_C}" stroke-dashoffset="${RING_C}"></circle></svg>`;

export class UiView {
  /** @param {{root: HTMLElement, live: HTMLElement}} els */
  constructor({ root, live, doc = document }) {
    this.root = root;
    this.live = live;
    this.doc = doc;
    this.lastKey = null;
    this.lastAnnounce = '';
  }

  render(v) {
    const { ring, countdown, progress, ...rest } = v;
    const key = JSON.stringify({ ...rest, cd: countdown === undefined ? null : Math.ceil(countdown * 10), pr: progress });
    if (key !== this.lastKey) {
      this.lastKey = key;
      this.root.dataset.screen = v.screen;
      this.doc.body.dataset.screen = v.screen;
      this.root.innerHTML = this._screen(v);
      this._announce(v);
    }
    this._chrome(v);
    this._ring(ring);
    const bar = this.root.querySelector('.countdown-bar > span');
    if (bar && countdown !== undefined) bar.style.width = `${Math.round(countdown * 100)}%`;
  }

  _announce(v) {
    let text = '';
    if (v.screen === 'choose') {
      const c = v.cards[v.focusIdx];
      text = `${c.label}. ${c.armed ? 'Ready.' : ''}`;
    } else if (v.screen === 'stopped') text = 'Stopped';
    else if (v.screen === 'result') text = v.result.title;
    else text = { listening: 'Listening', planning: 'Thinking', executing: `Moving: ${v.label}`, countdown: v.label }[v.screen] || '';
    if (text && text !== this.lastAnnounce) { this.live.textContent = text; this.lastAnnounce = text; }
  }

  _chrome(v) {
    const d = this.doc;
    const set = (id, on) => { const el = d.getElementById(id); if (el) el.hidden = !on; };
    set('replay-badge', v.replay);
    set('noface', v.faceLost && v.screen !== 'stopped');
    set('fps-warn', v.fpsLow);
    set('mic-dot', v.mic);
    const conn = d.getElementById('conn');
    if (conn) {
      const ok = v.connected.phone && v.connected.safety;
      conn.dataset.ok = ok ? '1' : '0';
      conn.textContent = ok ? 'Connected' : v.connected.safety ? 'Reconnecting...' : 'Offline';
    }
    const stopBtn = d.getElementById('stop-btn');
    if (stopBtn) stopBtn.hidden = v.screen === 'stopped';
    const toast = d.getElementById('toast');
    if (toast) {
      toast.hidden = !v.toast;
      toast.textContent = v.toast || '';
      toast.dataset.level = v.toastLevel || 'info';
    }
  }

  _ring(ring) {
    const el = this.root.querySelector('.ring-host .ring');
    if (!el) return;
    const fg = el.querySelector('.ring-fg');
    const p = ring ? Math.max(0, Math.min(1, ring.progress)) : 0;
    fg.setAttribute('stroke-dashoffset', String(RING_C * (1 - p)));
    el.classList.toggle('active', !!ring);
    el.classList.toggle('armed', !!ring?.armed);
    el.classList.toggle('full', p >= 1);
  }

  _screen(v) {
    switch (v.screen) {
      case 'connecting':
        return `<section class="center"><h1>Connecting...</h1><p class="sub">Looking for the Nod gateway</p></section>`;
      case 'idle':
        return `<section class="center"><h1>Ready</h1>
          <p class="sub">Raise your eyebrows for options, or say what you need.</p>
          ${v.hint ? `<p class="hint">${esc(v.hint)}</p>` : ''}</section>`;
      case 'listening':
        return `<section class="center"><div class="spinner" aria-hidden="true"></div><h1>Listening...</h1>
          <p class="sub">Say what you need</p>${v.transcript ? `<p class="transcript">"${esc(v.transcript)}"</p>` : ''}</section>`;
      case 'planning':
        return `<section class="center"><div class="spinner" aria-hidden="true"></div><h1>Thinking...</h1>
          ${v.transcript ? `<p class="transcript">"${esc(v.transcript)}"</p>` : ''}</section>`;
      case 'choose':
        return this._choose(v);
      case 'confirming':
        return `<section class="center"><div class="spinner" aria-hidden="true"></div><h1>Starting</h1><p class="sub">${esc(v.label)}</p></section>`;
      case 'countdown':
        return `<section class="center run"><p class="kicker">Starting in a moment</p><h1>${esc(v.label)}</h1>
          <div class="countdown-bar"><span style="width:100%"></span></div>
          <p class="stop-hint">Open your mouth wide or say STOP to stop</p></section>`;
      case 'executing':
      case 'postcheck':
        return `<section class="center run"><p class="kicker">${v.screen === 'postcheck' ? 'Checking the result' : 'Moving'}${v.of > 1 ? ` &middot; step ${v.step} of ${v.of}` : ''}</p>
          <h1>${esc(v.label)}</h1>
          <div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round((v.progress || 0) * 100)}">
            <span style="width:${Math.round((v.progress || 0) * 100)}%"></span></div>
          <p class="elapsed">${v.elapsed_s ?? 0} s</p>
          <p class="stop-hint big">To STOP: open your mouth wide, or say "stop"</p></section>`;
      case 'paused':
        return `<section class="center"><h1>Paused</h1><p class="sub">${esc(v.label)}</p>
          <p class="stop-hint">To STOP: open your mouth wide, or say "stop"</p>${v.hint ? `<p class="hint">${esc(v.hint)}</p>` : ''}</section>`;
      case 'result': {
        const r = v.result;
        return `<section class="center result result-${esc(r.phase)}"><h1>${esc(r.title)}</h1>
          ${r.label ? `<p class="sub">${esc(r.label)}</p>` : ''}${r.reason ? `<p class="reason">${esc(r.reason)}</p>` : ''}
          <p class="hint">Raise your eyebrows for more options</p></section>`;
      }
      case 'stopped':
        return this._stopped(v);
      default:
        return '';
    }
  }

  _choose(v) {
    const cards = v.cards.map((c, i) => {
      const meta = [c.steps ? `${c.steps} steps` : '', c.success, c.est, c.risk && c.risk !== 'none' ? `${c.risk} risk` : '', c.precheck]
        .filter(Boolean).map((m) => `<span>${esc(m)}</span>`).join('');
      const badges = [...c.badges, c.confirm === 'double' ? 'Two blinks' : ''].filter(Boolean)
        .map((b) => `<span class="badge">${esc(b)}</span>`).join('');
      return `<li class="card${c.focused ? ' focused' : ''}${c.armed ? ' armed' : ''}${c.locked ? ' locked' : ''}${c.fixed ? ' fixed' : ''}"
          role="option" aria-selected="${c.focused}" data-option="${esc(c.option_id)}" id="card-${i}">
        <div class="card-text"><div class="label">${esc(c.label)}</div>
          ${c.why ? `<div class="why">${esc(c.why)}</div>` : ''}
          ${meta ? `<div class="meta">${meta}</div>` : ''}${badges ? `<div class="badges">${badges}</div>` : ''}</div>
        ${c.focused ? `<div class="ring-host">${ringSvg()}</div>` : ''}</li>`;
    }).join('');
    const focused = v.cards[v.focusIdx];
    const help = v.double ? 'Long blink once more to confirm'
      : focused.locked ? 'This option needs a caregiver'
        : focused.armed ? (focused.fixed ? 'Long blink to close the options' : 'Long blink to choose')
          : 'Turn your head to move';
    return `<section class="choose">
      ${v.transcript ? `<p class="transcript small">"${esc(v.transcript)}"</p>` : ''}
      <ol class="cards" role="listbox" aria-label="Options" aria-activedescendant="card-${v.focusIdx}">${cards}</ol>
      <p class="help">${esc(help)}</p>${v.hint ? `<p class="hint">${esc(v.hint)}</p>` : ''}</section>`;
  }

  _stopped(v) {
    const s = v.stopped;
    const lat = s.lat_ms ? Object.entries(s.lat_ms).filter(([, x]) => x !== null && x !== undefined).map(([k, x]) => `${k} ${Math.round(x)} ms`).join(' &middot; ') : '';
    const status = s.resetPending ? 'Clearing...' : s.acked ? 'Robot stopped' : 'Sending stop...';
    return `<section class="stopped" role="alertdialog" aria-label="Stopped">
      <h1>STOPPED</h1><p class="status" data-acked="${s.acked ? 1 : 0}">${status}</p>
      ${lat ? `<p class="lat">${lat}</p>` : ''}
      ${s.canClear && !s.resetPending ? `<div class="clear"><div class="ring-host">${ringSvg()}</div>
        <h2>Clear stop?</h2><p>Long blink to clear. The robot will not resume.</p>
        <button type="button" class="clear-touch" data-action="clear-touch">Caregiver: clear stop</button></div>` : ''}
    </section>`;
  }

  /** Calibration screens, from CalibrationFlow.view() plus the app's own start/finish screens. */
  renderCalibration(cv) {
    this.lastKey = null;
    this.root.dataset.screen = 'calibrate';
    this.doc.body.dataset.screen = 'calibrate';
    if (cv.step === 'intro') {
      this.root.innerHTML = `<section class="center calib">
        <h1>Set up Nod</h1>
        <p class="sub">Nod learns how your face moves, so it can tell a deliberate gesture from a natural one. Hold the phone at eye level, 35 to 60 cm away.</p>
        ${cv.message ? `<p class="hint">${esc(cv.message)}</p>` : ''}
        <div class="buttons">
          <button type="button" data-action="calib-full">Calibrate (about 2 minutes)</button>
          <button type="button" data-action="calib-quick">Quick calibration (about 1 minute)</button>
          ${cv.hasProfile ? '<button type="button" data-action="calib-keep">Use my saved profile</button>' : ''}
          <button type="button" data-action="calib-default">Use default settings</button>
          <button type="button" data-action="calib-keyboard">No camera: keyboard only</button>
        </div></section>`;
      return;
    }
    if (cv.step === 'done' || cv.step === 'failed') {
      const val = cv.validation;
      const warn = (cv.profile?.warnings || []).map((w) => `<li>${esc(w)}</li>`).join('');
      this.root.innerHTML = `<section class="center calib"><h1>${cv.step === 'done' ? 'Calibration finished' : 'Calibration stopped'}</h1>
        ${cv.message ? `<p class="hint">${esc(cv.message)}</p>` : ''}
        ${val ? `<p class="sub">Practice round: ${val.hits} of ${val.n} gestures recognised, ${val.falseConfirms} false confirms. ${val.pass ? 'Passed.' : 'Not passed yet.'}</p>` : ''}
        ${warn ? `<ul class="warnings">${warn}</ul>` : ''}
        <div class="buttons"><button type="button" data-action="calib-accept">Start using Nod</button>
        <button type="button" data-action="calib-full">Calibrate again</button></div></section>`;
      return;
    }
    const titles = { setup: 'Getting ready', neutral: 'Relax', natural: 'Be natural', rest: 'Relax', rep: 'Practice', validate: 'Try it', vrest: 'Relax' };
    this.root.innerHTML = `<section class="center calib"><p class="kicker">${esc(titles[cv.step] || cv.step)}${cv.n ? ` &middot; ${cv.i + 1} of ${cv.n}` : ''}</p>
      <h1 class="prompt">${esc(cv.prompt)}</h1>
      <div class="progress"><span style="width:${Math.round((cv.progress || 0) * 100)}%"></span></div>
      ${cv.message ? `<p class="hint">${esc(cv.message)}</p>` : ''}
      <p class="overall">${cv.overall !== undefined ? `${Math.round(cv.overall * 100)}% done` : ''}</p>
      <button type="button" data-action="calib-cancel" class="secondary">Cancel</button></section>`;
  }
}
