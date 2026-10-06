"""Shared fixtures: one fully populated sample per message type."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import nod_protocol as p

PROTOCOL_DIR = Path(__file__).resolve().parents[1]
REPO = PROTOCOL_DIR.parent
SCHEMA_DIR = PROTOCOL_DIR / "schema"
SKILLS_YAML = REPO / "registry" / "skills.yaml"
STATS_JSON = REPO / "registry" / "stats" / "latest.json"


def _world_fields() -> dict:
    return dict(
        scene_id="libero_goal_v1", step=247, t_sim=12.35,
        atoms={"on(akita_black_bowl_1,plate_1)": False, "open(wooden_cabinet_1_middle_region)": True},
        objects={"akita_black_bowl_1": p.ObjPose(pos=(-0.09, 0.0, 0.90), quat=(1.0, 0.0, 0.0, 0.0))},
        eef=p.Eef(pos=(-0.21, 0.02, 1.12), gripper="open"),
    )


def make_samples() -> list:
    env = dict(sess="s_7f3a", seq=42)
    success = p.Success(p_est=0.62, basis="in_chain", n=20, wilson_lo=0.41, k=12)
    return [
        # 3.5.1
        p.Hello(device="pixel-7", app_ver="0.3.0", role="user", resume_token=None,
                calib=p.Calib(profile_id="cal_01J9", long_blink_ms=700, fps=24), **env),
        p.Gesture(name="long_blink", phase="fired", held_ms=742, yaw_deg=3.1, pitch_deg=-2.0,
                  scores={"blink": 0.81, "brow": 0.05, "jaw": 0.02}, fps=24, **env),
        p.UiMenu(source="brow_raise", **env),
        p.UiFocus(proposal_id="p_01J9", option_id="o2", **env),
        p.UiConfirm(proposal_id="p_01J9", option_id="o2", nonce="n_5c1e", gesture="long_blink", held_ms=742,
                    shown_at=1791205201000, focused_at=1791205202100, **env),
        p.UiCancel(proposal_id="p_01J9", source="jaw_hold", **env),
        p.UtterStart(utt_id=7, **env),
        p.UtterEnd(utt_id=7, dur_ms=2140, **env),
        p.UtterText(utt_id=8, text="I'm a bit hungry", **env),
        p.Ping(t0=1791205200123, **env),
        # 3.5.2
        p.Heartbeat(face_present=True, track_conf=0.97, fps=24, latched=False, **env),
        p.EStop(estop_seq=3, source="jaw_hold", t_detect=1791205203456, held_ms=310, **env),
        p.EStopReset(estop_seq=3, gesture="long_blink", held_ms=820, **env),
        # 3.5.3
        p.Welcome(resume_token="rt_x", lease="user", server_time=1791205200150, **env),
        p.Pong(t0=1791205200123, t_srv=1791205200160, **env),
        p.State(mode="AWAITING_CONFIRM", run_id=None, scene_id="libero_goal_v1", estop_seq=2, **env),
        p.Transcript(utt_id=7, text="I'm a bit hungry", asr_ms=212, **env),
        p.Proposal(
            proposal_id="p_01J9", expires_at=1791205231000, arm_delay_ms=300, source="nemotron",
            model="nvidia/Nemotron-3_5-Lightning", transcript="I'm a bit hungry",
            options=[
                p.Option(option_id="o1", kind="plan", skill_ids=["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"],
                         label="Set up a snack (3 steps)", why="You said you are hungry", success=success, est_s=30,
                         risk="low", confirm="per_step",
                         checks=[p.Check(rule="R4_preconditions", ok=True), p.Check(rule="R6_reliability", ok=True)],
                         precheck=p.Precheck(status="pending"), badges=["near_user"]),
                p.Option(option_id="o3", kind="home", home_action="call_caregiver", label="Call my caregiver",
                         why="If you would rather have help", success=None, est_s=2, risk="none", confirm="single"),
            ],
            focus="o1", nonce="n_5c1e", **env),
        p.RunUpdate(run_id="r_01J9", proposal_id="p_01J9", option_id="o1", plan_step=0, plan_len=3,
                    skill_id="bowl_on_plate", phase="postcheck", sim_step=64, max_steps=400, elapsed_ms=3200,
                    cosmos=p.CosmosVerdict(check_id="k_1", phase="post", answer="yes", reason="ok", decision="done"),
                    oracle=True, fault=None, **env),
        p.EStopAck(estop_seq=3, stopped_at_step=143, lat_ms=p.LatMs(phone_gw=41, gw_sim=2, sim_halt=18), **env),
        p.Notice(level="info", text="Putting the bowl on the plate", speak=True, **env),
        p.Thumb(camera="agentview", w=256, h=256, step=64, **env),
        # 3.5.4
        p.LoadScene(req_id="q_1", scene_id="libero_goal_v1", init_index=0, epoch=12),
        p.RunSkill(req_id="q_2", run_id="r_01J9", epoch=12, skill_id="bowl_on_plate",
                   instruction="put the bowl on the plate", goal_ref="libero_goal/put_the_bowl_on_the_plate",
                   n_action_steps=8, max_steps=400, action_scale=0.6),
        p.RunScripted(req_id="q_3", run_id="r_01JA", epoch=12, script="return_home", max_steps=160),
        p.Pause(req_id="q_4", epoch=12),
        p.Resume(req_id="q_5", epoch=12),
        p.Abort(req_id="q_6", epoch=12),
        p.Capture(req_id="q_7", cameras=["frontview", "agentview"], w=640, h=480),
        p.WorldRequest(req_id="q_8"),
        p.CtlHeartbeat(t=1791205200150, phone_ok=True),
        p.CtlEStop(estop_seq=3, t_gw=1791205203497),
        # 3.5.5
        p.SimAck(req_id="q_2", ok=False, error="busy"),
        p.SimStatus(run_id="r_01J9", epoch=12, phase="executing", step=64, success=False, fault="budget",
                    policy_ms=88.1, step_ms=14.2),
        p.SimEStopAck(estop_seq=3, epoch=13, stopped_at_step=143, t_rx=1791205203499, t_halt=1791205203517),
        p.SimWorld(**_world_fields()),
        p.SimFrames(req_id="q_7", step=247, frames=[p.Frame(camera="frontview", w=640, h=480,
                                                            jpeg=bytes(range(256)) + b"\xff\xd8", step=247)]),
        p.SimEpisodeHeader(scene_id="libero_goal_v1", model_hash="sha256:abc", mj_ver="3.3.1",
                           body_names=["world", "robot0_base"], dyn_body_ids=[12, 13, 14],
                           model_patch=p.ModelPatch(body_pos={"wooden_cabinet_1_main": (0.1, 0.2, 0.9)},
                                                    body_quat={"wooden_cabinet_1_main": (1.0, 0.0, 0.0, 0.0)})),
        # 3.5.6
        p.Pose(epoch=13, step=247, t_sim=12.35, t_wall=1791205204.512,
               xpos=np.arange(12, dtype=np.float32).reshape(4, 3) / 7,
               xquat=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (4, 1)),
               qpos=np.linspace(-1, 1, 9, dtype=np.float32)),
        # 3.5.7
        p.CosmosCheck(check_id="k_01J9", run_id="r_01J9", phase="post", skill_id="bowl_on_plate",
                      question="Is the black bowl resting on the plate?",
                      frames=[p.FrameRef(camera="frontview", w=640, h=480, step=312)], **env),
        p.CosmosResult(check_id="k_01J9", answer="yes", reason="The bowl sits on the white plate.", latency_ms=840,
                       oracle=p.OracleAtom(atom="on(akita_black_bowl_1,plate_1)", value=True), agree=True,
                       decision="done", **env),
        # 3.5.8
        p.Trace(kind="planner", ref="p_01J9", data={"latency_ms": 640, "options": ["o1"]}, **env),
        p.Metrics(estop_ms_p50=63, estop_ms_p95=118, view_lag_ms=35, policy_ms_p95=97, asr_ms_p50=210,
                  llm_ms_p50=640, **env),
    ]


@pytest.fixture(scope="session")
def samples() -> list:
    return make_samples()


@pytest.fixture(scope="session")
def specs() -> list[p.SkillSpec]:
    return p.load_skills(SKILLS_YAML, STATS_JSON)
