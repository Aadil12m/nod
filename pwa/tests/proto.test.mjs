// node --test: every message the phone or presenter sends validates against the
// JSON Schema generated from nod_protocol (protocol/schema/*.json).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Ajv2020 from 'ajv/dist/2020.js';
import { msg, envelope, ulid, newId, audioFrame, parseAudioFrame, parseMessage, ESTOP_SOURCES, GESTURE_NAMES } from '../js/proto.js';
import { GestureMachine } from '../js/gestures.js';
import { Scene, run } from './synth.mjs';

const SCHEMA_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../protocol/schema');
const haveSchema = fs.existsSync(path.join(SCHEMA_DIR, 'index.json'));
const ajv = new Ajv2020({ strict: false, allErrors: true });
const validators = {};
if (haveSchema) {
  const index = JSON.parse(fs.readFileSync(path.join(SCHEMA_DIR, 'index.json'), 'utf8'));
  for (const [type, info] of Object.entries(index.messages)) {
    validators[type] = ajv.compile(JSON.parse(fs.readFileSync(path.join(SCHEMA_DIR, info.file), 'utf8')));
  }
}

function valid(m) {
  const v = validators[m.type];
  assert.ok(v, `no schema for ${m.type}`);
  const ok = v(m);
  assert.ok(ok, `${m.type}: ${ajv.errorsText(v.errors)}\n${JSON.stringify(m)}`);
}
const wrap = ([type, payload], seq = 1) => envelope(type, payload, { sess: 's_7f3a', seq });

test('ULIDs: 26 Crockford characters, time-ordered, monotonic within a millisecond', () => {
  const a = ulid(1791205200123), b = ulid(1791205200123), c = ulid(1791205200124);
  for (const x of [a, b, c]) assert.match(x, /^[0-9A-HJKMNP-TV-Z]{26}$/);
  assert.ok(a < b && b < c);
  assert.equal(a.slice(0, 10), b.slice(0, 10));
  assert.match(newId('p'), /^p_[0-9A-Z]{26}$/);
});

test('envelope carries v, type, id, ts, sess, seq', () => {
  const m = envelope('ping', { t0: 5 }, { sess: 's_1', seq: 42, now: 1791205200123 });
  assert.deepEqual(Object.keys(m).slice(0, 6), ['v', 'type', 'id', 'ts', 'sess', 'seq']);
  assert.equal(m.ts, 1791205200123);
  assert.equal(m.seq, 42);
});

test('schema files are present (protocol/schema)', { skip: !haveSchema && 'protocol/schema not found' }, () => {
  for (const t of ['hello', 'gesture', 'ui.menu', 'ui.focus', 'ui.confirm', 'ui.cancel', 'utter.start', 'utter.end', 'utter.text', 'ping', 'hb', 'estop', 'estop.reset']) {
    assert.ok(validators[t], t);
  }
});

test('/ws/phone messages validate (3.5.1)', { skip: !haveSchema }, () => {
  valid(wrap(msg.hello({ device: 'pixel-7', calib: { profile_id: 'cal_01J9', long_blink_ms: 700, fps: 24 } })));
  valid(wrap(msg.hello({ device: 'web' })));
  valid(wrap(msg.hello({ device: 'web', role: 'caregiver', resume_token: 'rt_x', calib: { profile_id: 'cal_default', long_blink_ms: 742.4, fps: 0 } })));
  valid(wrap(msg.uiMenu('brow_raise')));
  valid(wrap(msg.uiFocus('p_01J9', 'o2')));
  valid(wrap(msg.uiConfirm({ proposal_id: 'p_01J9', option_id: 'o2', nonce: 'n_5c1e', held_ms: 742.6, shown_at: 1791205201000, focused_at: 1791205202100.4 })));
  valid(wrap(msg.uiCancel('p_01J9', 'jaw_hold')));
  valid(wrap(msg.uiCancel(null, 'timeout')));
  valid(wrap(msg.utterStart(7)));
  valid(wrap(msg.utterEnd(7, 2140.2)));
  valid(wrap(msg.utterText(8, "I'm a bit hungry")));
  valid(wrap(msg.ping(1791205200123)));
});

test('every gesture event the machine can emit validates as telemetry', { skip: !haveSchema }, () => {
  const s = new Scene(12000, { seed: 3 })
    .blink(500, 900).blink(2500, 500).browRaise(4000, 700).turn(6000, 800, 25).turn(7500, 800, -25)
    .jawHold(9000, 500).missing(10500, 800).blink(3500, 150).blink(3800, 150);
  const ev = run(new GestureMachine(), s, 24).filter((e) => (e.phase === 'fired' || e.phase === 'cancelled') && GESTURE_NAMES.includes(e.name));
  const names = new Set(ev.map((e) => e.name));
  for (const n of ['long_blink', 'brow_raise', 'turn_left', 'turn_right', 'jaw_hold', 'face_lost', 'face_found']) assert.ok(names.has(n), n);
  for (const e of ev) valid(wrap(msg.gesture(e)));
});

test('/ws/safety messages validate (3.5.2)', { skip: !haveSchema }, () => {
  valid(wrap(msg.hb({ face_present: true, track_conf: 0.97, fps: 24, latched: false })));
  valid(wrap(msg.hb({ face_present: false, track_conf: undefined, fps: NaN, latched: true })));
  for (const source of ESTOP_SOURCES) valid(wrap(msg.estop({ estop_seq: 3, source, t_detect: 1791205203456.7, held_ms: 310 })));
  valid(wrap(msg.estop({ estop_seq: 4, source: 'button', t_detect: 1791205203456 })));
  valid(wrap(msg.estopReset({ estop_seq: 3, gesture: 'long_blink', held_ms: 820 })));
  valid(wrap(msg.estopReset({ estop_seq: 3, gesture: 'touch' })));
});

test('an invalid message is caught by the schema (the check has teeth)', { skip: !haveSchema }, () => {
  const m = wrap(msg.estop({ estop_seq: 3, source: 'button', t_detect: 1 }));
  m.source = 'elbow';
  assert.equal(validators.estop(m), false);
  const h = wrap(msg.hb({ face_present: true, track_conf: 1, fps: 30, latched: false }));
  delete h.sess;
  assert.equal(validators.hb(h), false);
});

test('design 3.5.3 gateway examples parse as messages', () => {
  const welcome = parseMessage('{"v":1,"type":"welcome","sess":"s_7f3a","resume_token":"rt_","lease":"user","server_time":1791205200150}');
  assert.equal(welcome.type, 'welcome');
  assert.equal(parseMessage('not json'), null);
  assert.equal(parseMessage('{"no":"type"}'), null);
  assert.equal(parseMessage('[1,2]'), null);
});

test('binary audio frames: [u32 LE utt_id][pcm16le]', () => {
  const pcm = Int16Array.from([0, 1, -1, 32767, -32768]);
  const buf = audioFrame(7, pcm);
  assert.equal(buf.byteLength, 4 + 10);
  const dv = new DataView(buf);
  assert.equal(dv.getUint32(0, true), 7);
  assert.equal(dv.getInt16(4 + 6, true), 32767);
  const back = parseAudioFrame(buf);
  assert.equal(back.uttId, 7);
  assert.deepEqual([...back.pcm], [...pcm]);
});
