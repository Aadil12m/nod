// Nod phone app: wires camera, gestures, calibration, sockets, e-stop, audio and UI.
//
// Query parameters (all optional):
//   nocam=1     keyboard only, no camera (also used by the end-to-end test)
//   gw=wss://h  gateway origin, when the page is not served by the gateway
//   role=caregiver   hello.role
//   kws=0       do not load the on-device "stop" keyword model
//   nosw=1      do not register the service worker
//   preview=0   hide the mirrored camera preview

import { GestureMachine } from './gestures.js';
import {
  CalibrationFlow, BaselineTracker, defaultProfile, loadProfile, saveProfile, profileConfig, rezero,
} from './calibrate.js';
import { msg, audioFrame } from './proto.js';
import { Link, gatewayBase, httpBase } from './net.js';
import { EStopController, RESEND_MS } from './estop.js';
import { UiMachine, UiView } from './ui.js';
import { bindKeyboard } from './keyboard.js';
import { FaceTracker } from './face.js';
import { MicCapture, Segmenter } from './audio.js';
import { StopKeyword } from './kws.js';
import * as tts from './tts.js';

const q = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const els = { root: $('screen'), live: $('live'), video: $('preview'), form: $('utter-form'), text: $('utter-text') };

// ---------------------------------------------------------------- state

let profile = loadProfile() || defaultProfile();
let gm = new GestureMachine(profileConfig(profile));
let drift = new BaselineTracker(profile);
let cameraOn = false;
let tracker = null;
let lastFrame = null; // newest camera frame
let calib = null; // {flow} while calibrating, or {step:'intro'} on the start screen
let rezeroFrames = null; // frames collected for the 5 s session re-zero
let link = null;

// ---------------------------------------------------------------- core objects

const speak = (t, o) => tts.speak(t, o);
const estop = new EStopController({
  send: (m) => link?.safety.send(m),
  getToken: () => link?.resumeToken ?? null,
  vibrate: (p) => navigator.vibrate?.(p),
  speak: (t) => tts.speak(t, { urgent: true }),
  onChange: () => render(),
});
const ui = new UiMachine({
  send: (ch, m) => (ch === 'safety' ? link?.safety : link?.phone)?.send(m),
  speak,
  estop,
  clock: { toPhone: (t) => (link ? link.clock.toPhone(t) : t) },
});
const view = new UiView({ root: els.root, live: els.live });

function device() {
  const ua = navigator.userAgentData;
  return (ua?.platform || navigator.platform || 'web').toLowerCase().slice(0, 40);
}

link = new Link({
  base: gatewayBase(),
  hello: () => ({
    device: device(),
    role: q.get('role') === 'caregiver' ? 'caregiver' : 'user',
    resume_token: link?.resumeToken ?? null,
    calib: { profile_id: profile.id, long_blink_ms: profile.config.long_blink_ms ?? 700, fps: lastFrame?.fps ?? 0 },
  }),
  onMessage: (m) => {
    if (m.type === 'welcome') checkMode();
    ui.onServer(m);
    render();
  },
  onStatus: (s) => {
    ui.connected = s;
    if (s.safety) estop.tick(); // a pending stop goes out the moment the socket is back
    render();
  },
});

// ---------------------------------------------------------------- rendering

let renderQueued = false;
function render() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    if (calib) return;
    view.render(ui.view());
    syncMic();
  });
}

function renderCalibration(cv) {
  view.renderCalibration(cv);
}

// ---------------------------------------------------------------- gestures

function onGestureEvents(events) {
  for (const ev of events) ui.onGesture(ev);
  if (events.length) render();
}

function onCameraFrame(f) {
  lastFrame = f;
  if (calib?.flow) {
    const cv = calib.flow.feed(f);
    if (calib.flow.done) finishCalibration();
    else renderCalibration(cv);
    return;
  }
  if (rezeroFrames) {
    rezeroFrames.push(f);
    if (f.t_ms - rezeroFrames[0].t_ms >= 5000) {
      profile = rezero(profile, rezeroFrames);
      saveProfile(profile);
      applyProfile();
      rezeroFrames = null;
      ui.toast = { text: 'Ready', level: 'info', at: Date.now() };
    }
  }
  drift.update(f, ui.screen() === 'idle');
  onGestureEvents(gm.update(f));
}

function applyProfile() {
  drift = new BaselineTracker(profile);
  gm.setConfig(profileConfig(profile));
}

// Every 30 s, fold the slow baseline drift (6.3 step 7) into the thresholds.
setInterval(() => { if (!calib) gm.setConfig({ ...profileConfig(profile), ...drift.config() }); }, 30_000);

// ---------------------------------------------------------------- camera and calibration

async function startCamera() {
  if (cameraOn) return true;
  tracker = new FaceTracker({
    video: els.video,
    onFrame: onCameraFrame,
    onStatus: (s) => { const el = $('face-status'); if (el) el.textContent = `Camera: ${s}`; },
  });
  try {
    await tracker.start();
    cameraOn = true;
    if (q.get('preview') !== '0') els.video.hidden = false;
    return true;
  } catch (e) {
    console.warn('[app] camera unavailable:', e);
    tracker?.stop();
    tracker = null;
    ui.toast = { text: 'Camera unavailable. Use the keyboard: arrows, hold Enter, B, Space.', level: 'warn', at: Date.now() };
    return false;
  }
}

function showIntro(message = '') {
  calib = { step: 'intro' };
  renderCalibration({ step: 'intro', hasProfile: profile.id !== 'cal_default', message });
}

async function beginCalibration(plan) {
  const ok = await startCamera();
  if (!ok) { showIntro('The camera could not start. You can use default settings or the keyboard.'); return; }
  calib = { flow: new CalibrationFlow({ plan, id: `cal_${Date.now().toString(36)}` }) };
  speak('Hold the phone at eye level');
}

function finishCalibration() {
  const flow = calib.flow;
  calib = { step: 'result', flow };
  if (flow.state === 'done') {
    profile = flow.profile;
    saveProfile(profile);
    gm = new GestureMachine(profileConfig(profile));
    drift = new BaselineTracker(profile);
  }
  renderCalibration({ step: flow.state, profile: flow.profile, validation: flow.validation, message: flow.message });
}

function leaveCalibration() {
  calib = null;
  if (profile.config.disabled?.includes('jaw_hold')) {
    ui.toast = { text: 'Mouth stop is off for you. Say "stop" or use the STOP button.', level: 'warn', at: Date.now() };
  }
  render();
}

async function useProfile(p, { withCamera = true } = {}) {
  profile = p;
  gm = new GestureMachine(profileConfig(profile));
  drift = new BaselineTracker(profile);
  leaveCalibration();
  if (withCamera && (await startCamera()) && p.neutral) {
    rezeroFrames = []; // 5 s re-zero at the start of each session (6.3 step 7)
    ui.toast = { text: 'Look at the screen for a moment', level: 'info', at: Date.now() };
  }
  render();
}

els.root.addEventListener('click', async (e) => {
  const b = e.target.closest('[data-action]');
  const card = e.target.closest('.card');
  startKws(); // first user gesture: audio may start now
  if (card && !b) {
    const idx = [...els.root.querySelectorAll('.card')].indexOf(card);
    const cur = ui.proposal?.focusIdx;
    if (cur !== undefined && idx >= 0 && idx !== cur) { ui.moveFocus(idx - cur); render(); }
    return;
  }
  if (!b) return;
  switch (b.dataset.action) {
    case 'calib-full': await beginCalibration('full'); break;
    case 'calib-quick': await beginCalibration('quick'); break;
    case 'calib-keep': await useProfile(profile); break;
    case 'calib-default': await useProfile(defaultProfile()); break;
    case 'calib-keyboard': await useProfile(profile, { withCamera: false }); break;
    case 'calib-accept': leaveCalibration(); break;
    case 'calib-cancel': showIntro(); break;
    case 'clear-touch': ui.clearByTouch(); render(); break;
    default: break;
  }
});

// ---------------------------------------------------------------- buttons, text, keyboard

$('stop-btn').addEventListener('click', () => { ui.stop('button'); render(); });
$('menu-btn').addEventListener('click', () => {
  startKws();
  const sc = ui.screen();
  if (sc === 'idle' || sc === 'result') ui.menu('button');
  render();
});
els.form.addEventListener('submit', (e) => {
  e.preventDefault();
  if (ui.submitText(els.text.value)) els.text.value = '';
  render();
});

bindKeyboard(document, {
  onEvents: (ev) => { if (!calib) onGestureEvents(ev); },
  onStop: () => { ui.stop('button'); render(); },
});

// ---------------------------------------------------------------- voice: keyword "stop" and utterances

let kws = null;
function startKws() {
  if (kws || q.get('kws') === '0') return;
  kws = new StopKeyword({
    onStop: () => { ui.stop('voice_stop'); render(); },
    onStatus: (s) => { const el = $('kws-status'); if (el) el.textContent = `Voice stop: ${s}`; },
  });
  kws.start();
}

let mic = null;
let seg = null;
let micBroken = false;
async function syncMic() {
  const want = ui.wantsMic() && !micBroken;
  if (want && !mic) {
    seg = new Segmenter({ nextUttId: () => ui.nextUttId() });
    mic = new MicCapture({ onPcm: onPcm });
    const m = mic;
    try { await m.start(); } catch (e) {
      console.warn('[app] microphone unavailable:', e);
      micBroken = true;
      mic = null;
      ui.toast = { text: 'Microphone unavailable. Type below instead.', level: 'warn', at: Date.now() };
      render();
    }
  } else if (!want && mic) {
    for (const a of seg?.flush(performance.now()) || []) onSegment(a);
    mic.stop();
    mic = null;
    seg = null;
    gm.setContext({ speaking: false });
  }
}

function onPcm(pcm, t) {
  if (!seg) return;
  for (const a of seg.push(pcm, t)) onSegment(a);
  gm.setContext({ speaking: seg?.vad.speaking ?? false });
}

function onSegment(a) {
  if (a.kind === 'start') link.phone.send(msg.utterStart(a.utt));
  else if (a.kind === 'audio') link.phone.sendBinary(audioFrame(a.utt, a.pcm));
  else if (a.kind === 'end') { link.phone.send(msg.utterEnd(a.utt, a.dur_ms)); ui.utteranceEnded(); render(); }
  else if (a.kind === 'timeout') { ui.listenTimeout(); render(); }
}

// ---------------------------------------------------------------- safety heartbeat (100 ms)

function facePresent() {
  if (cameraOn) return !!lastFrame && lastFrame.face_present && performance.now() - lastFrame.t_ms < 500;
  // Keyboard mode has no camera: presence means the page is visible. See docs/DEVIATIONS.md (pwa).
  return document.visibilityState === 'visible';
}

setInterval(() => {
  estop.tick();
  link.safety.send(msg.hb({
    face_present: facePresent(),
    track_conf: cameraOn ? lastFrame?.track_conf ?? 0 : 0,
    fps: cameraOn ? lastFrame?.fps ?? 0 : 0,
    latched: estop.latched,
    resume_token: link.resumeToken,
  }));
}, RESEND_MS);

setInterval(() => { ui.tick(); render(); }, 100);

// ---------------------------------------------------------------- REPLAY badge

async function checkMode() {
  try {
    const r = await fetch(`${httpBase()}/healthz`, { cache: 'no-store' });
    const j = await r.json();
    if (j && (j.mode === 'mock' || j.mode === 'replay')) { ui.replay = true; render(); }
    else if (j && j.mode === 'live') { ui.replay = false; render(); }
  } catch { /* gateway without /healthz: no badge */ }
}

// ---------------------------------------------------------------- start

if ('serviceWorker' in navigator && q.get('nosw') !== '1') {
  navigator.serviceWorker.register('sw.js').catch((e) => console.warn('[app] service worker:', e));
}

link.start();
checkMode();
if (q.get('nocam') === '1') {
  calib = null;
  render();
} else if (profile.id !== 'cal_default') {
  useProfile(profile);
} else {
  showIntro();
}

// Debug handle for the console and the end-to-end test.
globalThis.nod = { ui, estop, link, get profile() { return profile; }, get gm() { return gm; }, get lastFrame() { return lastFrame; }, get tracker() { return tracker; } };
