// node --test: the phone UI state machine (design 3.4, 5.5, 5.7) with a fake clock.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { UiMachine, successText, CLEAR_STOP_DELAY_MS } from '../js/ui.js';
import { EStopController } from '../js/estop.js';

function rig({ doubleBlinkBack = false } = {}) {
  const clock = { t: 1_000_000 };
  const sent = { phone: [], safety: [] };
  const spoken = [];
  const now = () => clock.t;
  const estop = new EStopController({ send: (m) => { sent.safety.push({ type: m[0], ...m[1] }); return true; }, now, speak: (t) => spoken.push(t) });
  const ui = new UiMachine({
    now, estop, doubleBlinkBack,
    send: (ch, m) => { sent[ch].push({ type: m[0], ...m[1] }); return true; },
    speak: (t) => spoken.push(t),
  });
  ui.connected = { phone: true, safety: true };
  const adv = (ms) => { clock.t += ms; ui.tick(); estop.tick(); };
  const of = (ch, type) => sent[ch].filter((m) => m.type === type);
  return { ui, estop, clock, sent, spoken, adv, of };
}

const OPTS = [
  { option_id: 'o1', kind: 'plan', skill_ids: ['bowl_on_plate', 'cheese_in_bowl', 'bowl_to_user'], label: 'Set up a snack (3 steps)', why: 'You said you are hungry',
    success: { p_est: 0.62, basis: 'in_chain', n: 20, wilson_lo: 0.41 }, est_s: 30, risk: 'low', confirm: 'per_step', checks: [], precheck: { status: 'pending' } },
  { option_id: 'o2', kind: 'skill', skill_ids: ['cheese_in_bowl'], label: 'Cream cheese into the bowl', why: 'Quick snack prep',
    success: { p_est: 0.95, basis: 'alone', n: 40, k: 38, wilson_lo: 0.83 }, est_s: 8, risk: 'low', confirm: 'single', checks: [], precheck: { status: 'ok' } },
  { option_id: 'o3', kind: 'skill', skill_ids: ['stove_on'], label: 'Turn on the stove', why: '', success: null, est_s: 5, risk: 'high',
    confirm: 'double', checks: [], precheck: null, badges: ['needs_caregiver'], locked: true },
];

function propose(r, extra = {}) {
  r.ui.onServer({ type: 'state', mode: 'AWAITING_CONFIRM', run_id: null, scene_id: 'libero_goal_v1', estop_seq: 0 });
  r.ui.onServer({ type: 'proposal', proposal_id: 'p_01J9', expires_at: r.clock.t + 30_000, arm_delay_ms: 300, source: 'nemotron',
    options: OPTS, fixed: [{ option_id: 'none', label: 'None of these' }], focus: 'o1', nonce: 'n_5c1e', ...extra });
}
const blink = (r, held_ms) => r.ui.onGesture({ name: 'long_blink', phase: 'fired', held_ms, t_ms: 0, scores: {}, yaw_deg: 0, pitch_deg: 0, fps: 30 });
const g = (r, name, phase = 'fired', extra = {}) => r.ui.onGesture({ name, phase, t_ms: 0, scores: {}, yaw_deg: 0, pitch_deg: 0, fps: 30, ...extra });

test('success text: "19/20 in tests" when k is known', () => {
  assert.equal(successText({ k: 19, n: 20, p_est: 0.95, basis: 'alone' }), '19/20 in tests');
  assert.equal(successText({ p_est: 0.62, n: 20, basis: 'in_chain' }), '62% in 20 tests');
  assert.equal(successText(null), '');
});

test('connecting until the first state; then idle', () => {
  const r = rig();
  r.ui.connected = { phone: false, safety: false };
  assert.equal(r.ui.screen(), 'connecting');
  r.ui.onServer({ type: 'state', mode: 'IDLE', estop_seq: 0, scene_id: 'x' });
  assert.equal(r.ui.screen(), 'idle');
});

test('brow raise in idle opens the menu: ui.menu, listening at once, mic on', () => {
  const r = rig();
  r.ui.onServer({ type: 'state', mode: 'IDLE', estop_seq: 0 });
  g(r, 'brow_raise');
  assert.equal(r.of('phone', 'ui.menu')[0].source, 'brow_raise');
  assert.equal(r.ui.screen(), 'listening');
  assert.equal(r.ui.wantsMic(), true);
  // Telemetry for the gesture.
  assert.equal(r.of('phone', 'gesture')[0].name, 'brow_raise');
  // Brow raise while choosing does nothing.
  propose(r);
  g(r, 'brow_raise');
  assert.equal(r.of('phone', 'ui.menu').length, 1);
});

test('proposal shows 3 cards + "None of these"; labels, why, success, badges', () => {
  const r = rig();
  propose(r);
  const v = r.ui.view();
  assert.equal(v.screen, 'choose');
  assert.equal(v.cards.length, 4);
  assert.equal(v.cards[3].label, 'None of these');
  assert.equal(v.cards[0].label, 'Set up a snack (3 steps)');
  assert.equal(v.cards[0].why, 'You said you are hungry');
  assert.equal(v.cards[0].steps, 3);
  assert.equal(v.cards[1].success, '38/40 in tests');
  assert.deepEqual(v.cards[2].badges, ['Needs caregiver']);
  assert.equal(v.cards[0].focused, true);
  assert.equal(v.cards[0].armed, false, 'not armed at once');
  r.adv(300);
  assert.equal(r.ui.view().cards[0].armed, true, 'armed after 300 ms');
  assert.match(r.spoken.at(-1), /3 options\. Set up a snack/);
});

test('head turns move focus without wrap-around and send ui.focus', () => {
  const r = rig();
  propose(r);
  g(r, 'turn_left'); // already first: nothing
  assert.equal(r.of('phone', 'ui.focus').length, 0);
  g(r, 'turn_right');
  g(r, 'turn_right');
  g(r, 'turn_right'); // to "None of these" (fixed: no ui.focus for it)
  g(r, 'turn_right'); // at the end: nothing
  assert.equal(r.ui.proposal.focusIdx, 3);
  assert.deepEqual(r.of('phone', 'ui.focus').map((m) => m.option_id), ['o2', 'o3']);
  g(r, 'turn_left');
  assert.equal(r.ui.proposal.focusIdx, 2);
});

test('long blink confirms only when armed; ui.confirm carries the binding fields', () => {
  const r = rig();
  propose(r);
  const shown = r.clock.t;
  r.adv(100);
  g(r, 'turn_right'); // focus o2 at shown+100
  const focused = r.clock.t;
  r.adv(500);
  blink(r, 700); // closure started at focused-200: before the focus landed
  assert.equal(r.of('phone', 'ui.confirm').length, 0);
  assert.match(r.ui.hint, /Hold your gaze/);
  r.adv(1000);
  blink(r, 750); // closure started 750 ms ago, focus rested since well before
  const c = r.of('phone', 'ui.confirm');
  assert.equal(c.length, 1);
  assert.deepEqual({ ...c[0], type: undefined }, {
    type: undefined, proposal_id: 'p_01J9', option_id: 'o2', nonce: 'n_5c1e', gesture: 'long_blink', held_ms: 750, shown_at: shown, focused_at: focused,
  });
  assert.equal(r.ui.screen(), 'confirming');
  // Gateway answers: countdown.
  r.ui.onServer({ type: 'state', mode: 'COUNTDOWN', run_id: 'r_1', scene_id: 'x', estop_seq: 0 });
  assert.equal(r.ui.screen(), 'countdown');
  assert.equal(r.ui.view().label, 'Cream cheese into the bowl');
});

test('a hold that started before the option was armed does not confirm', () => {
  const r = rig();
  propose(r);
  r.adv(800);
  blink(r, 700); // started at +100, armed at +300
  assert.equal(r.of('phone', 'ui.confirm').length, 0);
});

test('"None of these" cancels; a locked option never confirms', () => {
  const r = rig();
  propose(r);
  g(r, 'turn_right'); g(r, 'turn_right');
  r.adv(1500);
  blink(r, 800); // o3 is locked (needs caregiver)
  assert.equal(r.of('phone', 'ui.confirm').length, 0);
  assert.match(r.ui.hint, /caregiver/);
  g(r, 'turn_right');
  r.adv(1500);
  blink(r, 800);
  assert.deepEqual(r.of('phone', 'ui.cancel').map((m) => [m.proposal_id, m.source]), [['p_01J9', 'none_of_these']]);
  assert.equal(r.ui.screen(), 'idle');
});

test('confirm "double" needs a second long blink', () => {
  const r = rig();
  propose(r, { options: [{ ...OPTS[1], confirm: 'double', badges: ['less_reliable'] }] });
  r.adv(1500);
  blink(r, 800);
  assert.equal(r.of('phone', 'ui.confirm').length, 0);
  assert.equal(r.ui.view().double, true);
  r.adv(1200);
  blink(r, 800);
  assert.equal(r.of('phone', 'ui.confirm').length, 1);
});

test('second blink of a double confirm expires after 8 s', () => {
  const r = rig();
  propose(r, { options: [{ ...OPTS[1], confirm: 'double' }], expires_at: r.clock.t + 60_000 });
  r.adv(1500);
  blink(r, 800);
  r.adv(9000);
  blink(r, 800);
  assert.equal(r.of('phone', 'ui.confirm').length, 0, 'the second blink restarted the double confirm');
});

test('no answer to a confirm within 3 s: "Try again"', () => {
  const r = rig();
  propose(r);
  r.adv(1500);
  blink(r, 800);
  r.adv(3100);
  assert.equal(r.ui.hint, 'Try again');
  assert.equal(r.ui.screen(), 'choose');
});

test('proposal expires on the gateway clock', () => {
  const r = rig();
  propose(r, { expires_at: r.clock.t + 5000 });
  r.adv(5100);
  assert.equal(r.ui.screen(), 'planning'); // gateway still AWAITING_CONFIRM: waiting for it
  r.ui.onServer({ type: 'state', mode: 'IDLE', estop_seq: 0 });
  assert.equal(r.ui.screen(), 'idle');
});

test('jaw hold while executing latches the e-stop; estop resent every 100 ms until the ack', () => {
  const r = rig();
  r.ui.onServer({ type: 'state', mode: 'EXECUTING', run_id: 'r_1', estop_seq: 2 });
  g(r, 'jaw_hold', 'fired', { held_ms: 310 });
  assert.equal(r.ui.screen(), 'stopped');
  assert.ok(r.spoken.includes('Stopped'));
  const first = r.of('safety', 'estop');
  assert.equal(first.length, 1);
  assert.deepEqual([first[0].estop_seq, first[0].source, first[0].held_ms, first[0].t_detect], [3, 'jaw_hold', 310, r.clock.t]);
  for (let i = 0; i < 5; i++) r.adv(100);
  assert.equal(r.of('safety', 'estop').length, 6);
  r.ui.onServer({ type: 'estop.ack', estop_seq: 2, lat_ms: {} }); // wrong seq: ignored
  r.adv(100);
  assert.equal(r.of('safety', 'estop').length, 7);
  r.ui.onServer({ type: 'estop.ack', estop_seq: 3, stopped_at_step: 143, lat_ms: { phone_gw: 41, gw_sim: 2, sim_halt: 18 } });
  for (let i = 0; i < 5; i++) r.adv(100);
  assert.equal(r.of('safety', 'estop').length, 7, 'no resend after the ack');
  assert.deepEqual(r.ui.view().stopped.lat_ms, { phone_gw: 41, gw_sim: 2, sim_halt: 18 });
  // A second stop while latched does nothing.
  assert.equal(r.ui.stop('button'), 'latched');
});

test('STOP in each screen: e-stop when motion is possible, else cancel or nothing', () => {
  for (const [mode, expect] of [['COUNTDOWN', 'estop'], ['EXECUTING', 'estop'], ['POSTCHECK', 'estop'], ['PAUSED', 'estop'], ['IDLE', 'none']]) {
    const r = rig();
    r.ui.onServer({ type: 'state', mode, estop_seq: 0 });
    assert.equal(r.ui.stop('voice_stop'), expect, mode);
  }
  const r = rig();
  propose(r);
  assert.equal(r.ui.stop('jaw_hold'), 'cancel');
  assert.equal(r.of('phone', 'ui.cancel')[0].source, 'jaw_hold');
  assert.equal(r.of('safety', 'estop').length, 0);
  // A confirm in flight counts as motion.
  const r2 = rig();
  propose(r2);
  r2.adv(1500);
  blink(r2, 800);
  assert.equal(r2.ui.stop('jaw_hold'), 'estop');
  // A run that has not finished counts as motion even if the state message is late.
  const r3 = rig();
  r3.ui.onServer({ type: 'state', mode: 'IDLE', estop_seq: 0 });
  r3.ui.onServer({ type: 'run', run_id: 'r_9', skill_id: 'x', phase: 'returning_home', max_steps: 160 });
  assert.equal(r3.ui.stop('button'), 'estop');
});

test('"Clear stop?" after ack + 1.5 s; long blink sends estop.reset; unlatch when the gateway leaves ESTOPPED', () => {
  const r = rig();
  r.ui.onServer({ type: 'state', mode: 'EXECUTING', estop_seq: 0 });
  r.ui.stop('button');
  r.ui.onServer({ type: 'estop.ack', estop_seq: 1, lat_ms: {} });
  r.ui.onServer({ type: 'state', mode: 'ESTOPPED', estop_seq: 1 });
  assert.equal(r.ui.view().stopped.canClear, false);
  r.adv(CLEAR_STOP_DELAY_MS - 200);
  blink(r, 800); // started before the prompt was offered
  assert.equal(r.of('safety', 'estop.reset').length, 0);
  r.adv(1500);
  assert.equal(r.ui.view().stopped.canClear, true);
  blink(r, 820);
  const reset = r.of('safety', 'estop.reset');
  assert.deepEqual(reset.map((m) => [m.estop_seq, m.gesture, m.held_ms]), [[1, 'long_blink', 820]]);
  assert.equal(r.ui.screen(), 'stopped', 'still latched until the gateway confirms');
  r.ui.onServer({ type: 'run', run_id: 'r_1', skill_id: 'x', phase: 'aborted', max_steps: 400 });
  r.ui.onServer({ type: 'state', mode: 'IDLE', estop_seq: 1 });
  assert.equal(r.ui.screen(), 'idle', 'aborted run: no result screen, straight to idle');
});

test('caregiver touch clears too', () => {
  const r = rig();
  r.ui.onServer({ type: 'state', mode: 'EXECUTING', estop_seq: 0 });
  r.ui.stop('button');
  assert.equal(r.ui.clearByTouch(), false, 'not before the ack');
  r.ui.onServer({ type: 'estop.ack', estop_seq: 1, lat_ms: {} });
  assert.equal(r.ui.clearByTouch(), true);
  assert.equal(r.of('safety', 'estop.reset')[0].gesture, 'touch');
});

test('a stop from the presenter latches the phone too', () => {
  const r = rig();
  r.ui.onServer({ type: 'state', mode: 'EXECUTING', estop_seq: 4 });
  r.ui.onServer({ type: 'state', mode: 'ESTOPPED', estop_seq: 5 });
  assert.equal(r.ui.screen(), 'stopped');
  assert.equal(r.estop.seq, 5);
  r.adv(2500);
  blink(r, 800);
  assert.equal(r.of('safety', 'estop.reset')[0].estop_seq, 5);
  // The next stop from the phone uses the next sequence number.
  r.ui.onServer({ type: 'state', mode: 'EXECUTING', estop_seq: 5 });
  r.ui.stop('jaw_hold');
  assert.equal(r.of('safety', 'estop').at(-1).estop_seq, 6);
});

test('if the gateway forgets our stop (restart), the phone sends it again', () => {
  const r = rig();
  r.ui.onServer({ type: 'state', mode: 'EXECUTING', estop_seq: 0 });
  r.ui.stop('jaw_hold');
  r.ui.onServer({ type: 'estop.ack', estop_seq: 1, lat_ms: {} });
  const n = r.of('safety', 'estop').length;
  r.ui.onServer({ type: 'state', mode: 'IDLE', estop_seq: 0 }); // a fresh gateway
  assert.equal(r.of('safety', 'estop').length, n + 1);
  assert.equal(r.ui.screen(), 'stopped');
});

test('run results: done / failed / uncertain screens, then idle', () => {
  const r = rig();
  r.ui.onServer({ type: 'state', mode: 'EXECUTING', estop_seq: 0 });
  r.ui.onServer({ type: 'run', run_id: 'r_1', skill_id: 'bowl_on_plate', phase: 'executing', sim_step: 100, max_steps: 400, plan_step: 1, plan_len: 3, elapsed_ms: 3200 });
  const v = r.ui.view();
  assert.equal(v.screen, 'executing');
  assert.equal(v.progress, 0.25);
  assert.deepEqual([v.step, v.of, v.elapsed_s], [2, 3, 3.2]);
  r.ui.onServer({ type: 'run', run_id: 'r_1', skill_id: 'bowl_on_plate', phase: 'failed', max_steps: 400,
    cosmos: { check_id: 'k', phase: 'post', answer: 'no', reason: 'The bowl is still in the drawer.', decision: 'failed' } });
  r.ui.onServer({ type: 'state', mode: 'IDLE', estop_seq: 0 });
  const res = r.ui.view();
  assert.equal(res.screen, 'result');
  assert.equal(res.result.title, "That didn't work");
  assert.match(res.result.reason, /still in the drawer/);
  r.adv(6100);
  assert.equal(r.ui.screen(), 'idle');
});

test('paused: the gateway offers "Resume?" as a proposal; long blink confirms it; STOP stays an e-stop', () => {
  const r = rig();
  r.ui.onServer({ type: 'state', mode: 'EXECUTING', run_id: 'r_1', estop_seq: 0 });
  r.ui.onServer({ type: 'run', run_id: 'r_1', proposal_id: 'p_1', option_id: 'o1', skill_id: 'bowl_on_plate', phase: 'paused', max_steps: 400 });
  r.ui.onServer({ type: 'state', mode: 'PAUSED', run_id: 'r_1', estop_seq: 0 });
  assert.equal(r.ui.screen(), 'paused');
  r.ui.onServer({ type: 'proposal', proposal_id: 'p_res', expires_at: r.clock.t + 86_400_000, arm_delay_ms: 300, source: 'menu',
    options: [{ option_id: 'resume', kind: 'skill', skill_ids: ['bowl_on_plate'], label: 'Put the bowl on the plate', why: "I can't see you. Resume?",
      success: null, est_s: null, risk: 'low', confirm: 'single', checks: [], precheck: null }],
    fixed: [{ option_id: 'none', label: 'Cancel' }], focus: 'resume', nonce: 'n_res' });
  const v = r.ui.view();
  assert.equal(v.screen, 'choose');
  assert.deepEqual(v.cards.map((c) => c.label), ['Put the bowl on the plate', 'Cancel']);
  r.adv(1500);
  blink(r, 800);
  const c = r.of('phone', 'ui.confirm')[0];
  assert.deepEqual([c.proposal_id, c.option_id, c.nonce], ['p_res', 'resume', 'n_res']);
  assert.equal(r.ui.screen(), 'confirming');
  assert.equal(r.ui.stop('jaw_hold'), 'estop');
  // "Cancel" on the resume proposal is a ui.cancel.
  const r2 = rig();
  r2.ui.onServer({ type: 'state', mode: 'PAUSED', run_id: 'r_1', estop_seq: 0 });
  r2.ui.onServer({ type: 'proposal', proposal_id: 'p_res', expires_at: r2.clock.t + 60_000, arm_delay_ms: 300, source: 'menu',
    options: [{ option_id: 'resume', kind: 'skill', skill_ids: ['x'], label: 'X', risk: 'low', confirm: 'single' }],
    fixed: [{ option_id: 'none', label: 'Cancel' }], focus: 'resume', nonce: 'n' });
  g(r2, 'turn_right');
  r2.adv(1500);
  blink(r2, 800);
  assert.deepEqual(r2.of('phone', 'ui.cancel').map((m) => [m.proposal_id, m.source]), [['p_res', 'none_of_these']]);
});

test('telemetry: fired and cancelled gestures are sent, progress and fps events are not', () => {
  const r = rig();
  g(r, 'long_blink', 'start', { progress: 0.3, held_ms: 250 });
  g(r, 'long_blink', 'progress', { progress: 0.5, held_ms: 350 });
  g(r, 'long_blink', 'cancelled', { held_ms: 400, reason: 'too_short' });
  g(r, 'fps_low');
  g(r, 'face_lost');
  assert.deepEqual(r.of('phone', 'gesture').map((m) => [m.name, m.phase]), [['long_blink', 'cancelled'], ['face_lost', 'fired']]);
  assert.equal(r.ui.view().faceLost, true);
  assert.equal(r.ui.view().fpsLow, true);
  g(r, 'face_found');
  assert.equal(r.ui.view().faceLost, false);
});

test('ring: progress events drive the ring; armed only for a hold that started after arming', () => {
  const r = rig();
  propose(r);
  r.adv(200);
  g(r, 'long_blink', 'progress', { progress: 0.4, held_ms: 280 });
  assert.equal(r.ui.view().ring.armed, false);
  r.adv(600);
  g(r, 'long_blink', 'progress', { progress: 0.4, held_ms: 280 });
  assert.equal(r.ui.view().ring.armed, true);
  r.adv(600); // stale ring disappears
  assert.equal(r.ui.view().ring, null);
});

test('typed utterance sends utter.text with increasing utt_id and shows planning', () => {
  const r = rig();
  r.ui.onServer({ type: 'state', mode: 'IDLE', estop_seq: 0 });
  r.ui.submitText("  I'm a bit hungry ");
  r.ui.submitText('lights on');
  r.ui.submitText('   ');
  assert.deepEqual(r.of('phone', 'utter.text').map((m) => [m.utt_id, m.text]), [[1, "I'm a bit hungry"], [2, 'lights on']]);
  assert.equal(r.ui.screen(), 'planning');
});

test('notices are shown and spoken when asked', () => {
  const r = rig();
  r.ui.onServer({ type: 'notice', level: 'info', text: 'Putting the bowl on the plate', speak: true });
  r.ui.onServer({ type: 'notice', level: 'info', text: 'quiet', speak: false });
  assert.ok(r.spoken.includes('Putting the bowl on the plate'));
  assert.ok(!r.spoken.includes('quiet'));
  assert.equal(r.ui.view().toast, 'quiet');
});

test('double blink is "back" only when enabled, and never confirms', () => {
  const r = rig({ doubleBlinkBack: true });
  propose(r);
  r.adv(1000);
  g(r, 'double_blink');
  assert.equal(r.of('phone', 'ui.cancel')[0].source, 'double_blink');
  assert.equal(r.of('phone', 'ui.confirm').length, 0);
  const r2 = rig();
  propose(r2);
  g(r2, 'double_blink');
  assert.equal(r2.of('phone', 'ui.cancel').length, 0);
});
