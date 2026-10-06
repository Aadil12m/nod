"""Every message round-trips through JSON and msgpack; discriminated parsing works."""

from __future__ import annotations

import json

import msgpack
import numpy as np
import pytest
from pydantic import ValidationError

import nod_protocol as p
from conftest import make_samples

SAMPLES = make_samples()
IDS = [m.type for m in SAMPLES]
BACKENDS = ["msgpack", "ormsgpack"]


def test_samples_cover_every_message_type():
    assert sorted(IDS) == sorted(p.TYPE_TO_MODEL)
    assert len(set(IDS)) == len(IDS) == 43


@pytest.mark.parametrize("msg", SAMPLES, ids=IDS)
def test_json_round_trip(msg):
    text = p.to_ws(msg)
    back = p.parse_ws(text, p.ANY)
    assert type(back) is type(msg)
    assert back == msg
    assert p.to_ws(back) == text  # stable re-serialisation
    assert json.loads(text)["type"] == msg.type


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("msg", SAMPLES, ids=IDS)
def test_msgpack_round_trip(msg, backend):
    data = p.dumps(msg, backend=backend)
    assert isinstance(data, bytes)
    back = p.loads(data, backend=backend)
    assert type(back) is type(msg)
    assert back == msg


@pytest.mark.parametrize("msg", SAMPLES, ids=IDS)
def test_msgpack_backends_interoperate(msg):
    assert p.loads(p.dumps(msg, backend="msgpack"), backend="ormsgpack") == msg
    assert p.loads(p.dumps(msg, backend="ormsgpack"), backend="msgpack") == msg


@pytest.mark.parametrize("msg", [m for m in SAMPLES if isinstance(m, p.Envelope)], ids=lambda m: m.type)
def test_ws_messages_parse_with_default_adapter_and_carry_envelope(msg):
    back = p.parse_ws(p.to_ws(msg))
    assert back == msg
    d = json.loads(p.to_ws(msg))
    for key in ("v", "type", "id", "ts", "sess", "seq"):
        assert key in d
    assert p.is_ulid(d["id"])


def test_zmq_types_are_not_ws_and_vice_versa():
    with pytest.raises(ValidationError):
        p.parse_ws(json.dumps({"type": "ctl.estop", "estop_seq": 1, "t_gw": 1}))  # WS adapter only
    with pytest.raises(ValidationError):
        p.loads(p.packb({"type": "estop", "estop_seq": 1, "source": "button", "t_detect": 1}), p.CTL)


def test_socket_adapters_reject_foreign_types():
    hb = p.to_ws(p.Heartbeat(face_present=True, track_conf=0.9, fps=24, latched=False))
    assert isinstance(p.parse_ws(hb, p.SAFETY_IN), p.Heartbeat)
    with pytest.raises(ValidationError):
        p.parse_ws(hb, p.PHONE_IN)
    confirm = p.UiConfirm(proposal_id="p", option_id="o1", nonce="n", gesture="long_blink", held_ms=700,
                          shown_at=1, focused_at=2)
    with pytest.raises(ValidationError):  # confirm never travels on the presenter socket
        p.parse_ws(p.to_ws(confirm), p.PRESENTER_IN)
    estop = p.EStop(estop_seq=1, source="presenter", t_detect=5)
    assert isinstance(p.parse_ws(p.to_ws(estop), p.PRESENTER_IN), p.EStop)


def test_unknown_type_and_bad_values_fail():
    with pytest.raises(ValidationError):
        p.parse_ws('{"type": "nope"}')
    with pytest.raises(ValidationError):
        p.parse_ws('{"type": "gesture", "name": "wink", "phase": "fired"}')
    with pytest.raises(ValidationError):
        p.parse_ws('{"type": "estop", "estop_seq": 1, "source": "llm", "t_detect": 1}')
    with pytest.raises(ValidationError):
        p.parse_ws("not json")


def test_unknown_fields_are_ignored_for_forward_compatibility():
    m = p.parse_ws('{"type": "ping", "t0": 5, "new_field": 1}')
    assert isinstance(m, p.Ping) and m.t0 == 5


# Examples copied verbatim from design 3.5 (envelope fields are defaulted when absent).
DESIGN_EXAMPLES = [
    '{"type": "hello", "device": "pixel-7", "app_ver": "0.3.0", "role": "user", "resume_token": null,'
    ' "calib": {"profile_id": "cal_01J9", "long_blink_ms": 700, "fps": 24}}',
    '{"type": "gesture", "name": "long_blink", "phase": "fired", "held_ms": 742,'
    ' "yaw_deg": 3.1, "pitch_deg": -2.0, "scores": {"blink": 0.81, "brow": 0.05, "jaw": 0.02}, "fps": 24}',
    '{"type": "ui.menu", "source": "brow_raise"}',
    '{"type": "ui.focus", "proposal_id": "p_01J9", "option_id": "o2"}',
    '{"type": "ui.confirm", "proposal_id": "p_01J9", "option_id": "o2", "nonce": "n_5c1e",'
    ' "gesture": "long_blink", "held_ms": 742, "shown_at": 1791205201000, "focused_at": 1791205202100}',
    '{"type": "ui.cancel", "proposal_id": "p_01J9", "source": "jaw_hold"}',
    '{"type": "utter.start", "utt_id": 7, "sr": 16000, "fmt": "pcm16le"}',
    '{"type": "utter.end", "utt_id": 7, "dur_ms": 2140}',
    '{"type": "utter.text", "utt_id": 8, "text": "I\'m a bit hungry"}',
    '{"type": "ping", "t0": 1791205200123}',
    '{"type": "hb", "face_present": true, "track_conf": 0.97, "fps": 24, "latched": false}',
    '{"type": "estop", "estop_seq": 3, "source": "jaw_hold", "t_detect": 1791205203456, "held_ms": 310}',
    '{"type": "estop.reset", "estop_seq": 3, "gesture": "long_blink", "held_ms": 820}',
    '{"type": "welcome", "sess": "s_7f3a", "resume_token": "rt_...", "lease": "user", "server_time": 1791205200150}',
    '{"type": "pong", "t0": 1791205200123, "t_srv": 1791205200160}',
    '{"type": "state", "mode": "AWAITING_CONFIRM", "run_id": null, "scene_id": "libero_goal_v1", "estop_seq": 2}',
    '{"type": "transcript", "utt_id": 7, "text": "I\'m a bit hungry", "asr_ms": 212}',
    """{"type": "proposal", "proposal_id": "p_01J9", "expires_at": 1791205231000, "arm_delay_ms": 300,
 "source": "nemotron", "model": "nvidia/Nemotron-3_5-Lightning", "transcript": "I'm a bit hungry",
 "options": [
  {"option_id": "o1", "kind": "plan", "skill_ids": ["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"],
   "label": "Set up a snack (3 steps)", "why": "You said you are hungry",
   "success": {"p_est": 0.62, "basis": "in_chain", "n": 20, "wilson_lo": 0.41},
   "est_s": 30, "risk": "low", "confirm": "per_step",
   "checks": [{"rule": "R4_preconditions", "ok": true}, {"rule": "R6_reliability", "ok": true}],
   "precheck": {"status": "pending"}},
  {"option_id": "o2", "kind": "skill", "skill_ids": ["cheese_in_bowl"], "label": "Cream cheese into the bowl",
   "why": "Quick snack prep", "success": {"p_est": 0.95, "basis": "alone", "n": 40, "wilson_lo": 0.83},
   "est_s": 8, "risk": "low", "confirm": "single", "checks": [], "precheck": {"status": "ok"}},
  {"option_id": "o3", "kind": "home", "home_action": "call_caregiver", "label": "Call my caregiver",
   "why": "If you would rather have help", "success": null, "est_s": 2, "risk": "none", "confirm": "single",
   "checks": [], "precheck": null}],
 "fixed": [{"option_id": "none", "label": "None of these"}],
 "focus": "o1", "nonce": "n_5c1e"}""",
    '{"type": "run", "run_id": "r_01J9", "proposal_id": "p_01J9", "option_id": "o1", "plan_step": 0, "plan_len": 3,'
    ' "skill_id": "bowl_on_plate", "phase": "executing", "sim_step": 64, "max_steps": 400, "elapsed_ms": 3200,'
    ' "cosmos": null, "oracle": null}',
    '{"type": "estop.ack", "estop_seq": 3, "stopped_at_step": 143, "lat_ms": {"phone_gw": 41, "gw_sim": 2, "sim_halt": 18}}',
    '{"type": "notice", "level": "info", "text": "Putting the bowl on the plate", "speak": true}',
    '{"type": "thumb", "camera": "agentview", "w": 256, "h": 256, "step": 64}',
    '{"type": "ctl.load_scene", "req_id": "q_1", "scene_id": "libero_goal_v1", "init_index": 0, "epoch": 12}',
    """{"type": "ctl.run_skill", "req_id": "q_2", "run_id": "r_01J9", "epoch": 12, "skill_id": "bowl_on_plate",
 "kind": "policy", "instruction": "put the bowl on the plate", "goal_ref": "libero_goal/put_the_bowl_on_the_plate",
 "n_action_steps": 8, "max_steps": 400,
 "limits": {"eef_box": [[-0.40, -0.40, 0.85], [0.20, 0.40, 1.35]], "max_abs_action": 1.0}}""",
    '{"type": "ctl.run_scripted", "req_id": "q_3", "run_id": "r_01JA", "epoch": 12, "script": "return_home", "max_steps": 160}',
    '{"type": "ctl.pause", "req_id": "q_4", "epoch": 12}',
    '{"type": "ctl.resume", "req_id": "q_5", "epoch": 12}',
    '{"type": "ctl.abort", "req_id": "q_6", "epoch": 12}',
    '{"type": "ctl.capture", "req_id": "q_7", "cameras": ["frontview", "agentview"], "w": 640, "h": 480}',
    '{"type": "ctl.world", "req_id": "q_8"}',
    '{"type": "ctl.hb", "t": 1791205200150, "phone_ok": true}',
    '{"type": "ctl.estop", "estop_seq": 3, "t_gw": 1791205203497}',
    '{"type": "sim.ack", "req_id": "q_2", "ok": true, "error": null}',
    '{"type": "sim.status", "run_id": "r_01J9", "epoch": 12, "phase": "executing", "step": 64, "success": false,'
    ' "fault": null, "policy_ms": 88.1, "step_ms": 14.2}',
    '{"type": "sim.estop_ack", "estop_seq": 3, "epoch": 13, "stopped_at_step": 143, "t_rx": 1791205203499, "t_halt": 1791205203517}',
    """{"type": "sim.world", "scene_id": "libero_goal_v1", "step": 247, "t_sim": 12.35,
 "atoms": {"on(akita_black_bowl_1,plate_1)": false, "on(cream_cheese_1,akita_black_bowl_1)": false,
           "on(wine_bottle_1,wine_rack_1_top_region)": true, "open(wooden_cabinet_1_middle_region)": true,
           "turnon(flat_stove_1)": false, "on(akita_black_bowl_1,main_table_handover_region)": false},
 "objects": {"akita_black_bowl_1": {"pos": [-0.09, 0.0, 0.90]}},
 "eef": {"pos": [-0.21, 0.02, 1.12], "gripper": "open"}}""",
    '{"type": "sim.episode_header", "scene_id": "libero_goal_v1", "model_hash": "sha256:...", "mj_ver": "3.3.1",'
    ' "body_names": ["world", "..."], "dyn_body_ids": [12, 13, 14], "model_patch": {"body_pos": {}, "body_quat": {}}}',
    """{"type": "cosmos.check", "check_id": "k_01J9", "run_id": "r_01J9", "phase": "post", "skill_id": "bowl_on_plate",
 "question": "Is the black bowl resting on the plate? Reply with JSON {\\"answer\\": \\"yes\\"|\\"no\\"|\\"unsure\\", \\"reason\\": \\"...\\"}",
 "frames": [{"camera": "frontview", "w": 640, "h": 480, "step": 312}], "model": "nvidia/Cosmos-Reason2-8B"}""",
    """{"type": "cosmos.result", "check_id": "k_01J9", "answer": "yes", "reason": "The bowl sits on the white plate.",
 "latency_ms": 840, "oracle": {"atom": "on(akita_black_bowl_1,plate_1)", "value": true}, "agree": true,
 "decision": "done"}""",
    '{"type": "trace", "kind": "planner", "ref": "p_01J9", "data": {}}',
    '{"type": "metrics", "estop_ms_p50": 63, "estop_ms_p95": 118, "view_lag_ms": 35, "policy_ms_p95": 97, "asr_ms_p50": 210, "llm_ms_p50": 640}',
]


@pytest.mark.parametrize("text", DESIGN_EXAMPLES, ids=lambda t: json.loads(t)["type"])
def test_design_examples_parse(text):
    msg = p.parse_ws(text, p.ANY)
    assert msg.type == json.loads(text)["type"]
    # and survive both encodings afterwards
    assert p.loads(p.dumps(msg)) == msg
    assert p.parse_ws(p.to_ws(msg), p.ANY) == msg


def test_design_examples_cover_all_but_pose_and_frames():
    seen = {json.loads(t)["type"] for t in DESIGN_EXAMPLES}
    assert set(p.TYPE_TO_MODEL) - seen == {"pose", "sim.frames"}


def test_proposal_defaults_fixed_card_and_caps_options():
    opt = p.Option(option_id="o1", kind="skill", skill_ids=["bowl_on_plate"], label="x", risk="low", confirm="single")
    prop = p.Proposal(proposal_id="p", expires_at=1, source="menu", options=[opt], nonce="n")
    assert prop.fixed == [p.FixedOption(option_id="none", label="None of these")]
    with pytest.raises(ValidationError):
        p.Proposal(proposal_id="p", expires_at=1, source="menu", options=[opt] * 4, nonce="n")
    with pytest.raises(ValidationError):
        p.Option(option_id="o1", kind="skill", label="x", why="x" * 81, risk="low", confirm="single")


def test_sim_world_converts_to_world_state(samples):
    sw = next(m for m in samples if isinstance(m, p.SimWorld))
    w = sw.to_world()
    assert type(w) is p.WorldState and w.atoms == sw.atoms
    assert p.SimWorld.from_world(w) == sw


# ---------------------------------------------------------------- pose arrays


def _pose(**kw):
    base = dict(epoch=1, step=2, t_sim=0.1, t_wall=3.0, xpos=np.zeros((2, 3)), xquat=np.zeros((2, 4)), qpos=np.zeros(5))
    base.update(kw)
    return p.Pose(**base)


def test_pose_wire_format_decodes_with_plain_msgpack_and_numpy():
    xpos = np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32)
    pose = _pose(xpos=xpos, xquat=np.ones((5, 4), np.float32), qpos=np.arange(9, dtype=np.float32))
    raw = msgpack.unpackb(p.dumps(pose), raw=False)  # no nod_protocol on the reading side
    m = raw["xpos"]
    assert m["dtype"] == "<f4" and m["shape"] == [5, 3] and isinstance(m["data"], bytes)
    assert len(m["data"]) == 5 * 3 * 4
    np.testing.assert_array_equal(np.frombuffer(m["data"], dtype="<f4").reshape(m["shape"]), xpos)
    # and the arrays come back as float32 ndarrays
    back = p.loads(p.dumps(pose))
    assert back.xpos.dtype == np.float32 and back.xpos.flags.writeable


def test_pose_accepts_lists_and_casts_to_float32():
    pose = _pose(xpos=[[1, 2, 3], [4, 5, 6]], qpos=[0.5])
    assert pose.xpos.dtype == np.float32 and pose.xpos.shape == (2, 3)


@pytest.mark.parametrize(
    "kw",
    [
        dict(xpos=np.zeros((2, 4))),  # wrong width
        dict(xquat=np.zeros((3, 4))),  # body count mismatch
        dict(qpos=np.zeros((2, 2))),  # qpos must be 1-D
        dict(xpos=np.array([[np.nan, 0, 0], [0, 0, 0]])),  # non-finite
        dict(xpos={"dtype": "<f8", "shape": [2, 3], "data": b"\0" * 48}),  # wrong dtype
        dict(xpos={"dtype": "<f4", "shape": [2, 3], "data": b"\0" * 20}),  # wrong byte count
    ],
)
def test_pose_rejects_bad_arrays(kw):
    with pytest.raises(ValidationError):
        _pose(**kw)


def test_frames_jpeg_is_standard_base64_in_json():
    f = p.SimFrames(req_id="q", step=1, frames=[p.Frame(camera="c", w=1, h=1, jpeg=b"\xff\xd8\xff\xfe")])
    d = json.loads(p.to_ws(f))
    assert d["frames"][0]["jpeg"] == "/9j//g=="  # standard alphabet, as JS atob() expects
    assert isinstance(msgpack.unpackb(p.dumps(f), raw=False)["frames"][0]["jpeg"], bytes)


def test_core_works_without_numpy_or_ormsgpack():
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['numpy'] = None; sys.modules['ormsgpack'] = None\n"
        "import nod_protocol as p\n"
        "assert p.loads(p.dumps(p.Ping(t0=3))).t0 == 3\n"
        "try:\n"
        "    p.Pose(epoch=1, step=1, t_sim=0, t_wall=0, xpos=[[0, 0, 0]], xquat=[[1, 0, 0, 0]], qpos=[0])\n"
        "except ImportError as e:\n"
        "    assert 'numpy' in str(e)\n"
        "else:\n"
        "    raise SystemExit('Pose should need numpy')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
