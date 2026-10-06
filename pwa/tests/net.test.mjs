// node --test: sockets (envelope, seq, reconnect with backoff), clock offset, e-stop controller.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Channel, Clock, Link, backoffMs, gatewayBase, httpBase } from '../js/net.js';
import { EStopController, RESEND_MS } from '../js/estop.js';
import { msg } from '../js/proto.js';

/** Minimal fake WebSocket and timers. */
function fakeEnv() {
  const sockets = [];
  const timers = [];
  class FakeWS {
    constructor(url) { this.url = url; this.readyState = 0; this.sent = []; sockets.push(this); }
    send(d) { if (this.readyState !== 1) throw new Error('not open'); this.sent.push(d); }
    close() { this.readyState = 3; this.onclose?.(); }
    open() { this.readyState = 1; this.onopen?.(); }
    recv(obj) { this.onmessage?.({ data: typeof obj === 'string' ? obj : JSON.stringify(obj) }); }
    drop() { this.readyState = 3; this.onclose?.(); }
  }
  const setT = (fn, ms) => { const t = { fn, ms }; timers.push(t); return t; };
  const clearT = (t) => { const i = timers.indexOf(t); if (i >= 0) timers.splice(i, 1); };
  const runTimers = () => { const ts = timers.splice(0); ts.forEach((t) => t.fn()); return ts.map((t) => t.ms); };
  return { FakeWS, sockets, timers, setT, clearT, runTimers };
}

test('backoff doubles from 250 ms to a 5 s cap, with jitter', () => {
  assert.equal(backoffMs(0, () => 0.5), 250);
  assert.equal(backoffMs(1, () => 0.5), 500);
  assert.equal(backoffMs(3, () => 0.5), 2000);
  assert.equal(backoffMs(10, () => 0.5), 5000);
  assert.equal(backoffMs(0, () => 0), 200);
  assert.equal(backoffMs(0, () => 1), 300);
});

test('clock offset from ping/pong keeps the lowest round trip', () => {
  const c = new Clock();
  c.onPong({ t0: 1000, t_srv: 1600 }, 1100); // rtt 100, offset 550
  assert.equal(c.offset, 550);
  c.onPong({ t0: 2000, t_srv: 2520 }, 2040); // rtt 40, offset 500
  assert.equal(c.offset, 500);
  c.onPong({ t0: 3000, t_srv: 3900 }, 3400); // rtt 400: worse, ignored for the offset
  assert.equal(c.offset, 500);
  assert.equal(c.rtt, 400);
  assert.equal(c.toServer(10), 510);
  assert.equal(c.toPhone(510), 10);
  c.onPong({ t0: 5000, t_srv: 1 }, 4000); // negative rtt: ignored
  assert.equal(c.offset, 500);
});

test('channel: envelope with increasing seq across reconnects, parsed messages, binary', () => {
  const env = fakeEnv();
  const got = [];
  const bin = [];
  const ch = new Channel('ws://x/ws/phone', {
    WebSocketImpl: env.FakeWS, setTimeout: env.setT, clearTimeout: env.clearT, getSess: () => 's_1',
    onMessage: (m) => got.push(m), onBinary: (b) => bin.push(b),
  });
  assert.equal(ch.send(msg.ping(1)), null, 'not connected yet');
  ch.connect();
  env.sockets[0].open();
  const m1 = ch.send(msg.ping(5));
  const m2 = ch.send(msg.uiMenu('brow_raise'));
  assert.deepEqual([m1.seq, m2.seq, m1.sess, m1.v], [1, 2, 's_1', 1]);
  assert.equal(JSON.parse(env.sockets[0].sent[1]).type, 'ui.menu');
  env.sockets[0].recv({ type: 'pong', t0: 5, t_srv: 9 });
  env.sockets[0].recv('garbage');
  env.sockets[0].onmessage({ data: new ArrayBuffer(4) });
  assert.equal(got.length, 1);
  assert.equal(bin.length, 1);
  // Drop: reconnect after backoff; seq continues.
  env.sockets[0].drop();
  assert.equal(ch.connected, false);
  const delays = env.runTimers();
  assert.ok(delays[0] >= 200 && delays[0] <= 300);
  env.sockets[1].open();
  assert.equal(ch.send(msg.ping(6)).seq, 3);
  // Repeated failures back off further.
  env.sockets[1].drop();
  env.runTimers();
  env.sockets[2].drop();
  const d3 = env.runTimers()[0];
  assert.ok(d3 >= 400 && d3 <= 600, `${d3}`);
  ch.close();
  assert.equal(env.timers.length, 0);
});

test('link: hello on open, session from welcome, messages from both sockets', () => {
  const env = fakeEnv();
  const got = [];
  globalThis.setInterval = globalThis.setInterval || (() => 0);
  const link = new Link({ base: 'ws://h', WebSocketImpl: env.FakeWS, onMessage: (m, from) => got.push([m.type, from]),
    hello: () => ({ device: 'test', calib: { profile_id: 'cal_x', long_blink_ms: 700, fps: 30 } }) });
  link.phone.connect();
  link.safety.connect();
  const [ph, sf] = env.sockets;
  assert.equal(ph.url, 'ws://h/ws/phone');
  assert.equal(sf.url, 'ws://h/ws/safety');
  ph.open();
  sf.open();
  const hello = JSON.parse(ph.sent[0]);
  assert.equal(hello.type, 'hello');
  assert.equal(hello.sess, null);
  ph.recv({ type: 'welcome', sess: 's_7f3a', resume_token: 'rt_1', lease: 'user', server_time: Date.now() + 1000 });
  assert.equal(link.sess, 's_7f3a');
  assert.equal(link.resumeToken, 'rt_1');
  assert.ok(Math.abs(link.clock.offset - 1000) < 50);
  link.safety.send(msg.hb({ face_present: true, track_conf: 1, fps: 30, latched: false }));
  assert.equal(JSON.parse(sf.sent[0]).sess, 's_7f3a');
  sf.recv({ type: 'estop.ack', estop_seq: 1, lat_ms: {} });
  assert.deepEqual(got, [['welcome', 'phone'], ['estop.ack', 'safety']]);
  link.phone.close();
  link.safety.close();
});

test('gateway base URL from the page, or ?gw=', () => {
  assert.equal(gatewayBase({ protocol: 'https:', host: 'nod.example', search: '' }), 'wss://nod.example');
  assert.equal(gatewayBase({ protocol: 'http:', host: 'localhost:8000', search: '' }), 'ws://localhost:8000');
  assert.equal(gatewayBase({ protocol: 'http:', host: 'x', search: '?gw=wss://vm.example/' }), 'wss://vm.example');
  assert.equal(httpBase({ protocol: 'http:', host: 'x', search: '?gw=wss://vm.example/' }), 'https://vm.example');
  assert.equal(httpBase({ protocol: 'http:', host: 'x', search: '' }), '');
});

test('e-stop: a stop with the safety socket down is sent the moment it is back', () => {
  let up = false;
  const sent = [];
  let t = 0;
  const e = new EStopController({ send: (m) => (up ? (sent.push(m), true) : null), now: () => t });
  e.trigger('jaw_hold', { held_ms: 300 });
  assert.equal(e.latched, true, 'latched locally without the network');
  for (let i = 0; i < 5; i++) { t += RESEND_MS; e.tick(); }
  assert.equal(sent.length, 0);
  up = true;
  e.tick();
  assert.equal(sent.length, 1);
  assert.equal(sent[0][1].estop_seq, 1);
  t += 50; e.tick();
  assert.equal(sent.length, 1, 'not faster than every 100 ms');
  t += 60; e.tick();
  assert.equal(sent.length, 2);
});

test('e-stop: reset only after the ack; unlatch only when the gateway leaves ESTOPPED', () => {
  const sent = [];
  let t = 0;
  const changes = [];
  const e = new EStopController({ send: (m) => (sent.push(m), true), now: () => t, onChange: (c) => changes.push(c.latched) });
  e.trigger('button');
  assert.equal(e.requestReset('long_blink', 800), false);
  e.onMessage({ type: 'estop.ack', estop_seq: 1, lat_ms: {} });
  assert.equal(e.requestReset('long_blink', 800), true);
  assert.equal(sent.at(-1)[0], 'estop.reset');
  e.onMessage({ type: 'state', mode: 'ESTOPPED', estop_seq: 1 });
  assert.equal(e.latched, true);
  e.onMessage({ type: 'state', mode: 'IDLE', estop_seq: 1 });
  assert.equal(e.latched, false);
  assert.ok(changes.includes(false));
  // Next stop gets the next seq.
  e.trigger('button');
  assert.equal(sent.at(-1)[1].estop_seq, 2);
});
