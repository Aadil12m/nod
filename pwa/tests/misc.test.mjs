// node --test: audio helpers, keyboard fallback, face math, keyword helper.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Resampler, floatTo16, rms, Vad, Segmenter } from '../js/audio.js';
import { KeyboardGestures, TURN_PULSE_MS } from '../js/keyboard.js';
import { yawPitchFromMatrix, blendshapeMap, frameFromResult } from '../js/face.js';
import { stopScore } from '../js/kws.js';

// ---------------------------------------------------------------- audio

test('float to PCM16 clamps and scales', () => {
  assert.deepEqual([...floatTo16(Float32Array.from([0, 1, -1, 2, -2, 0.5]))], [0, 32767, -32768, 32767, -32768, 16384]);
  assert.equal(rms(Float32Array.from([1, -1, 1, -1])), 1);
  assert.equal(rms(new Float32Array(0)), 0);
});

test('resampler 48 kHz -> 16 kHz keeps a 440 Hz tone and is continuous across chunks', () => {
  const sr = 48000, f = 440, n = 4800;
  const src = Float32Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * f * i) / sr));
  const rs = new Resampler(sr, 16000);
  const parts = [];
  for (let i = 0; i < n; i += 128) parts.push(...rs.push(src.subarray(i, i + 128)));
  assert.ok(Math.abs(parts.length - n / 3) <= 1, `${parts.length}`);
  // Sample k of the output is source sample 3k.
  for (let k = 0; k < parts.length - 1; k += 97) assert.ok(Math.abs(parts[k] - src[3 * k]) < 1e-6, `k ${k}`);
  // 44.1 kHz: fractional ratio, still about the right count and values.
  const rs2 = new Resampler(44100, 16000);
  const src2 = Float32Array.from({ length: 4410 }, (_, i) => Math.sin((2 * Math.PI * f * i) / 44100));
  const out2 = [];
  for (let i = 0; i < 4410; i += 100) out2.push(...rs2.push(src2.subarray(i, i + 100)));
  assert.ok(Math.abs(out2.length - 1600) <= 1, `${out2.length}`);
  for (let k = 0; k < out2.length; k += 50) assert.ok(Math.abs(out2[k] - Math.sin((2 * Math.PI * f * k) / 16000)) < 0.01);
});

function tone(ms, amp, sr = 16000) {
  return Int16Array.from({ length: (sr * ms) / 1000 }, (_, i) => Math.round(amp * 32767 * Math.sin(i / 3)));
}

test('VAD and segmenter: pre-roll, start, audio, end after silence', () => {
  let id = 6;
  const seg = new Segmenter({ nextUttId: () => ++id });
  const acts = [];
  let t = 0;
  const feed = (amp, ms) => { for (let i = 0; i < ms; i += 50) { acts.push(...seg.push(tone(50, amp), t)); t += 50; } };
  feed(0.002, 1000); // room noise
  feed(0.3, 1500); // speech
  feed(0.002, 1000); // silence
  const kinds = acts.map((a) => a.kind);
  assert.equal(kinds.filter((k) => k === 'start').length, 1);
  assert.equal(acts.find((a) => a.kind === 'start').utt, 7);
  const end = acts.find((a) => a.kind === 'end');
  assert.ok(end, 'utterance ended');
  assert.ok(end.dur_ms >= 1500 && end.dur_ms <= 2600, `${end.dur_ms}`);
  // Pre-roll: audio from before the detected start is included.
  const audio = acts.filter((a) => a.kind === 'audio');
  assert.ok(audio.length * 50 >= 1500 + 250);
  assert.equal(seg.push(tone(50, 0.3), t).length, 0, 'finished');
});

test('segmenter: nothing said -> timeout; max length -> end', () => {
  const s1 = new Segmenter({ nextUttId: () => 1, noSpeechMs: 2000 });
  const a1 = [];
  for (let t = 0; t <= 2100; t += 50) a1.push(...s1.push(tone(50, 0.001), t));
  assert.deepEqual(a1.map((a) => a.kind), ['timeout']);
  const s2 = new Segmenter({ nextUttId: () => 1, maxMs: 1000 });
  const a2 = [];
  for (let t = 0; t <= 3000; t += 50) a2.push(...s2.push(tone(50, 0.3), t));
  assert.equal(a2.filter((a) => a.kind === 'end').length, 1);
  assert.ok(new Vad().threshold() > 0);
});

// ---------------------------------------------------------------- keyboard

test('keyboard: Enter held 900 ms is a long blink (fires on release); 300 ms is not', () => {
  const kb = new KeyboardGestures();
  const ev = [];
  let t = 0;
  const run = (ms) => { for (const end = t + ms; t < end; t += 33) ev.push(...kb.tick(t)); };
  run(200);
  kb.keydown('Enter', t); run(300); kb.keyup('Enter'); run(300);
  assert.equal(ev.filter((e) => e.phase === 'fired').length, 0);
  kb.keydown('Enter', t); run(900);
  assert.equal(ev.filter((e) => e.name === 'long_blink' && e.phase === 'fired').length, 0, 'not while held');
  assert.ok(ev.some((e) => e.name === 'long_blink' && e.phase === 'progress'));
  kb.keyup('Enter'); run(200);
  assert.equal(ev.filter((e) => e.name === 'long_blink' && e.phase === 'fired').length, 1);
});

test('keyboard: arrows are one head-turn step each, B is a brow raise, Space is STOP', () => {
  const kb = new KeyboardGestures();
  const ev = [];
  let t = 0;
  const run = (ms) => { for (const end = t + ms; t < end; t += 33) ev.push(...kb.tick(t)); };
  run(100);
  kb.keydown('ArrowRight', t); run(TURN_PULSE_MS + 400);
  kb.keydown('ArrowRight', t); run(TURN_PULSE_MS + 400);
  kb.keydown('ArrowLeft', t); run(TURN_PULSE_MS + 400);
  kb.keydown('b', t); run(1200);
  const names = ev.filter((e) => e.phase === 'fired').map((e) => e.name);
  assert.deepEqual(names, ['turn_right', 'turn_right', 'turn_left', 'brow_raise']);
  assert.equal(kb.keydown(' ', t), 'stop');
  assert.equal(kb.keydown('x', t), false);
});

// ---------------------------------------------------------------- face math

function matrixFromYawPitch(yawDeg, pitchDeg) {
  // R = Ry(yaw) * Rx(pitch), stored column-major in a 4x4 with a translation.
  const y = (yawDeg * Math.PI) / 180, p = (pitchDeg * Math.PI) / 180;
  const Ry = [[Math.cos(y), 0, Math.sin(y)], [0, 1, 0], [-Math.sin(y), 0, Math.cos(y)]];
  const Rx = [[1, 0, 0], [0, Math.cos(p), -Math.sin(p)], [0, Math.sin(p), Math.cos(p)]];
  const R = Ry.map((row, i) => [0, 1, 2].map((j) => row.reduce((s, _, k) => s + Ry[i][k] * Rx[k][j], 0)));
  const s = 1.3; // uniform scale cancels
  const d = new Array(16).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) d[c * 4 + r] = s * R[r][c];
  d[12] = 1; d[13] = -2; d[14] = -40; d[15] = 1;
  return d;
}

test('yaw and pitch from the column-major facial transformation matrix', () => {
  for (const [yaw, pitch] of [[0, 0], [25, 0], [-30, 0], [15, 10], [-10, -12]]) {
    const r = yawPitchFromMatrix(matrixFromYawPitch(yaw, pitch));
    assert.ok(Math.abs(r.yaw - yaw) < 1e-6, `yaw ${yaw} -> ${r.yaw}`);
    assert.ok(Math.abs(r.pitch - pitch) < 1e-6, `pitch ${pitch} -> ${r.pitch}`);
  }
  assert.deepEqual(yawPitchFromMatrix(null), { yaw: 0, pitch: 0 });
});

test('blendshapes are read by name and a result becomes a frame', () => {
  const cats = [{ categoryName: 'jawOpen', score: 0.7, index: 25 }, { categoryName: '_neutral', score: 0.1, index: 0 }];
  assert.deepEqual(blendshapeMap(cats), { jawOpen: 0.7, _neutral: 0.1 });
  const f = frameFromResult({ faceBlendshapes: [{ categories: cats }], facialTransformationMatrixes: [{ rows: 4, columns: 4, data: matrixFromYawPitch(20, 0) }] }, 123);
  assert.equal(f.face_present, true);
  assert.equal(f.t_ms, 123);
  assert.ok(Math.abs(f.yaw_deg - 20) < 1e-6);
  const none = frameFromResult({ faceBlendshapes: [], facialTransformationMatrixes: [] }, 5);
  assert.equal(none.face_present, false);
});

test('keyword helper reads the "stop" score by label', () => {
  const labels = ['_background_noise_', '_unknown_', 'down', 'stop', 'up'];
  assert.equal(stopScore([0.1, 0.1, 0, 0.8, 0], labels), 0.8);
  assert.equal(stopScore([0.1], ['go']), 0);
});
