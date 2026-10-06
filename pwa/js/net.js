// WebSocket links to the gateway (design 3.3, 3.5).
//
// Two sockets: /ws/phone (events, audio, proposals) and /ws/safety
// (heartbeat every 100 ms, estop). Safety has its own socket so a large
// proposal or an audio upload never delays a stop behind it (TCP
// head-of-line blocking). Each socket keeps its own `seq` counter, which keeps
// increasing across reconnects so the gateway can de-duplicate.

import { envelope, msg, parseMessage } from './proto.js';

/** Reconnect delay: 250 ms doubling to 5 s, with +-20% jitter. */
export function backoffMs(attempt, rand = Math.random) {
  const base = Math.min(5000, 250 * 2 ** Math.max(0, attempt));
  return Math.round(base * (0.8 + 0.4 * rand()));
}

/**
 * Phone clock vs gateway clock from ping/pong. offset = server - phone.
 * Keeps the sample with the smallest round trip among the last 8.
 */
export class Clock {
  constructor() { this.samples = []; this.offset = 0; this.rtt = null; }
  onPong(pong, now = Date.now()) {
    const rtt = now - pong.t0;
    if (!(rtt >= 0) || rtt > 30_000) return;
    this.samples.push({ rtt, offset: pong.t_srv - (pong.t0 + rtt / 2) });
    if (this.samples.length > 8) this.samples.shift();
    const best = this.samples.reduce((a, b) => (b.rtt < a.rtt ? b : a));
    this.offset = best.offset;
    this.rtt = rtt;
  }
  toServer(tPhone) { return tPhone + this.offset; }
  toPhone(tServer) { return tServer - this.offset; }
}

/** One auto-reconnecting WebSocket that sends enveloped JSON. */
export class Channel {
  /**
   * @param {string} url
   * @param {{onMessage?:Function, onBinary?:Function, onOpen?:Function, onClose?:Function,
   *          getSess?:()=>string|null, WebSocketImpl?:any, setTimeout?:Function, clearTimeout?:Function}} opts
   */
  constructor(url, opts = {}) {
    this.url = url;
    this.opts = opts;
    this.WS = opts.WebSocketImpl || globalThis.WebSocket;
    this.st = opts.setTimeout || globalThis.setTimeout.bind(globalThis);
    this.ct = opts.clearTimeout || globalThis.clearTimeout.bind(globalThis);
    this.seq = 0;
    this.attempt = 0;
    this.ws = null;
    this.closed = false;
    this.timer = null;
    this.openedAt = null;
  }

  get connected() { return !!this.ws && this.ws.readyState === 1; }

  connect() {
    if (this.closed) return;
    let ws;
    try {
      ws = new this.WS(this.url);
    } catch {
      this._schedule();
      return;
    }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => {
      this.attempt = 0;
      this.openedAt = Date.now();
      this.opts.onOpen?.(this);
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data === 'string') {
        const m = parseMessage(ev.data);
        if (m) this.opts.onMessage?.(m, this);
      } else this.opts.onBinary?.(ev.data, this);
    };
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      this.opts.onClose?.(this);
      this._schedule();
    };
    ws.onerror = () => { /* onclose follows */ };
  }

  _schedule() {
    if (this.closed || this.timer) return;
    const d = backoffMs(this.attempt++);
    this.timer = this.st(() => { this.timer = null; this.connect(); }, d);
  }

  /** Send a message built by proto.msg.* ([type, payload]). Returns the envelope, or null if not connected. */
  send([type, payload], now = Date.now()) {
    if (!this.connected) return null;
    const m = envelope(type, payload, { sess: this.opts.getSess?.() ?? null, seq: ++this.seq, now });
    try {
      this.ws.send(JSON.stringify(m));
      return m;
    } catch {
      return null;
    }
  }

  sendBinary(buf) {
    if (!this.connected) return false;
    try { this.ws.send(buf); return true; } catch { return false; }
  }

  close() {
    this.closed = true;
    if (this.timer) this.ct(this.timer);
    try { this.ws?.close(); } catch { /* ignore */ }
  }
}

/** Base URL for gateway sockets: ?gw=wss://host overrides the page's own origin. */
export function gatewayBase(loc = globalThis.location) {
  const q = new URLSearchParams(loc.search);
  const gw = q.get('gw');
  if (gw) return gw.replace(/\/$/, '');
  return `${loc.protocol === 'https:' ? 'wss:' : 'ws:'}//${loc.host}`;
}

/** HTTP base matching gatewayBase. */
export function httpBase(loc = globalThis.location) {
  const q = new URLSearchParams(loc.search);
  const gw = q.get('gw');
  if (gw) return gw.replace(/^ws/, 'http').replace(/\/$/, '');
  return '';
}

/**
 * The phone's pair of sockets plus session bookkeeping.
 * Messages from either socket go to `onMessage` (the gateway may ack an estop on either).
 */
export class Link {
  constructor({ base, onMessage, onBinary, onStatus, hello, WebSocketImpl, pingMs = 2000 } = {}) {
    this.sess = null;
    this.resumeToken = null;
    this.clock = new Clock();
    this.hello = hello; // () => hello payload args
    this.onMessage = onMessage;
    this.onStatus = onStatus;
    this.pingMs = pingMs;
    const common = { getSess: () => this.sess, WebSocketImpl };
    this.phone = new Channel(`${base}/ws/phone`, {
      ...common,
      onOpen: (ch) => { ch.send(msg.hello(this.hello?.() || {})); this._status(); },
      onClose: () => this._status(),
      onMessage: (m) => this._in(m, 'phone'),
      onBinary,
    });
    this.safety = new Channel(`${base}/ws/safety`, {
      ...common,
      onOpen: () => this._status(),
      onClose: () => this._status(),
      onMessage: (m) => this._in(m, 'safety'),
    });
    this.pingTimer = null;
  }

  start() {
    this.phone.connect();
    this.safety.connect();
    this.pingTimer = setInterval(() => this.phone.send(msg.ping(Date.now())), this.pingMs);
  }

  stop() {
    clearInterval(this.pingTimer);
    this.phone.close();
    this.safety.close();
  }

  _in(m, from) {
    if (m.type === 'welcome') {
      this.sess = m.sess ?? this.sess;
      this.resumeToken = m.resume_token ?? this.resumeToken;
      // First rough offset until ping/pong samples arrive.
      if (Number.isFinite(m.server_time) && !this.clock.samples.length) this.clock.offset = m.server_time - Date.now();
    } else if (m.type === 'pong') {
      this.clock.onPong(m);
    }
    this.onMessage?.(m, from);
  }

  _status() {
    this.onStatus?.({ phone: this.phone.connected, safety: this.safety.connected });
  }
}
