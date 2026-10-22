# Nod: technical design and build spec

Version 1, 2026-10-05. Audience: the Nod team and the coding agents working on the repo. Deadline: Fri Oct 30 2026, 10:00 AM PT (submit Thu Oct 29). Kill gate: Sat Oct 10.

Labels used in this document:
- **[checked]** means someone on the team read the source code or page cited, at the commit or date given.
- **[unverified]** means an estimate, a design parameter, or a claim nobody has confirmed yet. Section 10 lists them all with a plan to check each one.

## Key decisions at a glance

1. **The actor is GR00T N1.7 on a simulated Franka Panda in LIBERO-Goal**, checkpoint `nvidia/GR00T-N1.7-LIBERO` (subfolder `libero_goal`, 97.5% published success). All ten LIBERO-Goal tasks share one scene, so skills can chain in one continuous session with no reset. Whether success holds up when chained is the main unknown, and the Oct 10 kill gate measures it.
2. **Fallback actor:** scripted waypoint skills in the same scene. They are the same scripts we write anyway to generate demos for our custom skill.
3. **Confirm is a long blink (about 700 ms), not a double blink.** Natural blinking produces a few accidental double blinks every minute. Stop is a held jaw-open or the spoken word "stop", detected on the phone.
4. **The robot view is a pose stream, not video.** The sim publishes body poses, and the browser draws the scene at native resolution with `mjviser`/`viser`. This was measured at about 37 KB/s.
5. **Nemotron proposes; a pure-function validator decides; the user confirms.** The e-stop path never touches an LLM, the network to Token Factory, or Cosmos.
6. **Everything that touches the GPU runs on one Nebius RTX PRO 6000 VM in uk-south2.** That covers the GR00T server, the sim, Cosmos-Reason2-8B (vLLM) and the gateway. A nightly Nebius Serverless Job measures each skill's reliability, and those numbers drive both the planner and the UI.
7. **Mock mode** runs the full product on a laptop CPU with no GPU and no API keys. It replays recorded real runs and uses canned planner output.

---

## 1. Verdict on feasibility

### 1.1 Verdict

**The GR00T path is viable, with one unmeasured risk: success when skills are chained.** Build on:

| Item | Choice | Evidence |
|---|---|---|
| Policy | GR00T N1.7, 3B | [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T) at commit `51d4c89` (2026-08-20) **[checked]** |
| Checkpoint | `nvidia/GR00T-N1.7-LIBERO`, subfolder `libero_goal` (about 6.9 GB of safetensors; download with `--include`) | HF model page exists, NVIDIA Open Model License ([HF](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO)) **[checked 2026-10-05]** |
| Published success | Goal suite 195/200 = 97.5% (20K steps, batch 640) | `examples/LIBERO/README.md` **[checked]** |
| Simulator | LIBERO on robosuite 1.4.0, MuJoCo pinned to 3.3.1, headless EGL | `gr00t/eval/sim/LIBERO/setup_libero.sh` **[checked]** |
| Embodiment tag | `LIBERO_PANDA` (modality config `libero_sim`: 2 cameras at 256 px, 8-D state, 16-step action chunk) | `gr00t/configs/data/embodiment_configs.py` **[checked]** |
| Gated dependency | Every N1.7 checkpoint loads `nvidia/Cosmos-Reason2-2B` on first use; without HF access it fails with `GatedRepoError` | Isaac-GR00T README **[checked]**. **Request access today.** |

Backup loader: the LeRobot port `nvidia/gr00t17-lerobot-libero_goal-640` (98% reported) runs in a single venv through `lerobot-eval` (groot-sim report; not tested by us).

### 1.2 Why LIBERO-Goal: one scene, ten skills

I hashed the ten LIBERO-Goal BDDL task files after removing the `:language`, `:obj_of_interest` and `:goal` blocks. All ten hashes are identical, so regions, fixtures, objects and initial placement are the same in every task **[checked, LIBERO `8f1084e`, the commit Isaac-GR00T pins]**. The scene is a table holding a wooden cabinet (three drawers plus its top surface), a flat stove, a wine rack, a black bowl, a plate, cream cheese and a wine bottle.

The policy is conditioned on the exact instruction string. LIBERO derives that string from the file name (`grab_language_from_filename`) **[checked]**.

| Registry id (ours) | Exact instruction | Goal predicate (BDDL) | Assistive reading |
|---|---|---|---|
| `bowl_on_plate` | put the bowl on the plate | `On akita_black_bowl_1 plate_1` | set up a meal |
| `cheese_in_bowl` | put the cream cheese in the bowl | `On cream_cheese_1 akita_black_bowl_1` | handle an object |
| `bottle_on_rack` | put the wine bottle on the rack | `On wine_bottle_1 wine_rack_1_top_region` | put things away |
| `bottle_on_cabinet` | put the wine bottle on top of the cabinet | `On wine_bottle_1 wooden_cabinet_1_top_side` | put things away |
| `bowl_on_cabinet` | put the bowl on top of the cabinet | `On akita_black_bowl_1 wooden_cabinet_1_top_side` | put things away |
| `bowl_in_top_drawer` | open the top drawer and put the bowl inside | `In akita_black_bowl_1 wooden_cabinet_1_top_region` | storage |
| `open_middle_drawer` | open the middle drawer of the cabinet | `Open wooden_cabinet_1_middle_region` | open cabinets and drawers |
| `plate_to_stove_front` | push the plate to the front of the stove | `On plate_1 main_table_stove_front_region` | move a dish |
| `bowl_on_stove` | put the bowl on the stove | `On akita_black_bowl_1 flat_stove_1_cook_region` | hazardous: gated |
| `stove_on` | turn on the stove | `Turnon flat_stove_1` | hazardous: gated |

No published per-task breakdown exists for the Goal suite. The nightly job produces ours.

### 1.3 How chaining works in code [checked]

- `LiberoEnv` (in `gr00t/eval/sim/LIBERO/libero_env.py`) builds `OffScreenRenderEnv(..., ignore_done=True)`, so robosuite never ends the episode by itself.
- LIBERO's own "done" is just the result of `_check_success()`. Nothing stops on it.
- `_check_success()` reads `self.parsed_problem["goal_state"]` (`envs/problems/libero_tabletop_manipulation.py:135-143`). To switch the active goal, assign it the output of `robosuite_parse_problem(bddl)["goal_state"]` for the next skill. This works because all ten BDDLs share the same object and region names.
- The instruction comes from `env._task_description`, which `_process_observation` copies into every observation. To switch skills, set that attribute.
- The policy server keeps no state between calls, and the language is sent on every call (`Gr00tSimPolicyWrapper`). Changing the instruction mid-session is therefore safe.
- Resets come only from the eval harnesses (Isaac-GR00T's `MultiStepWrapper` and vector env, and LeRobot's autoreset). Our own loop never resets.
- Fixed initial states: `env._env.reset()`, then `env._env.set_init_state(state)`, then 10 settling steps.
  - Load the states with `torch.load(path, weights_only=False)`. LIBERO calls plain `torch.load`, which in torch 2.9 defaults to `weights_only=True` and will probably refuse these pickled arrays **[unverified, likely]**.
- Action units, from `normalize_gripper_action` and `invert_gripper_action`:
  - Policy gripper value 1.0 means open.
  - Translation is an OSC delta of at most 0.05 m per step at 20 Hz (robosuite `osc_pose.json`: `output_max` 0.05, `control_freq` 20).
  - `position_limits` defaults to `null`, so **robosuite does not clip the workspace for us. We clip in our own loop (section 5.3).** This corrects the streaming report's claim that robosuite already clips.

### 1.4 What is not proven

**Success when chained is unknown.** After skill A, skill B starts with the arm somewhere else and the scene changed. Neither condition appears in the training demos, which always start from the initial layout. Evidence that this can hurt:
- [LiLo-VLA](https://yy-gx.github.io/LiLo-VLA/) reports that pi0.5 fell from 83% to 0% when the same tasks were reordered.
- [LIBERO-PRO](https://arxiv.org/abs/2510.03827) finds that policies often memorise trajectories rather than follow the language.

None of these papers tested GR00T. Two mitigations:
1. **Return home between skills.** A scripted move lifts the arm, returns it to the start pose and opens the gripper, so each skill begins close to its training start.
2. **Chain A** touches only objects still in their initial state, so the only drift is the arm pose plus visual distractors.

The kill gate (section 8) measures this directly.

| Chain | Steps | Why |
|---|---|---|
| **A, independent (primary)** | `bottle_on_rack` → `open_middle_drawer` → `cheese_in_bowl` → `plate_to_stove_front` | Every skill acts on objects untouched by earlier steps. If each step matched the 97.5% suite average, 4 steps would succeed about 90% of the time **[unverified]** |
| **B, dependent ("snack")** | `bowl_on_plate` → `cheese_in_bowl` → `bowl_to_user` (custom) | The bowl is no longer where training put it. Measured, not gated |

### 1.5 Alternatives considered and rejected for the core

| Option | Why not (source: groot-sim report unless noted) |
|---|---|
| RoboCasa kitchen tasks | Better story, but each reset builds a new kitchen layout and places the robot base at the task's fixture. No public N1.7 RoboCasa checkpoint (N1.7 needs a 60k-step, 8-GPU fine-tune). N1.6 pick-and-place runs 19-63%, and a "make coffee" chain is about 19%. Use only for B-roll. The interface report's RoboCasa N1.7 numbers (e.g. CoffeeServeMug 85%) come from NVIDIA's non-public checkpoint. |
| Other LIBERO suites | Spatial: ten different layouts. Object: the target sits in a fixed region, so the layout gives the task away. LIBERO-10: six scenes. |
| RoboLab with the DROID checkpoint | 8.58% average zero-shot. |
| SimplerEnv, GR1 tabletop, BEHAVIOR-1K | One scene per task or low success. |

### 1.6 Fallback actor and how the demo changes

Decide at the kill gate on Oct 10 (criteria in section 8.8).

| Tier | Condition | Actor | What the demo says |
|---|---|---|---|
| **T1 (target)** | G0 and G1 pass | GR00T for all policy skills, chained in one session; scripted return-home and undo moves labelled "scripted" | "One continuous session, ten learned skills plus one we trained" |
| **T2** | G0 passes, G1 partial (full chain 50-70%) | GR00T, offering in-chain only those skills that measured at least 85% in chain; a visible "reset table" action | Same as T1, but some skills are marked "best from a fresh table" |
| **T3** | G0 passes, G1 below 50% | GR00T one skill per scene reset, with the reset shown honestly as a scripted "tidy the table" step | "Learned skills, reset between tasks"; chaining becomes future work |
| **T4 (fallback)** | G0 fails, or Cosmos-Reason2-2B access is not granted by Oct 10 | Scripted waypoint skills using privileged object poses and the OSC controller, in the same scene | "Scripted skills today; GR00T is a drop-in actor," showing the GR00T eval numbers we did get |

Gestures, Nemotron, the validator, Cosmos, the nightly Serverless Job, Home Assistant and mock mode stay the same in every tier. The NVIDIA open-model requirement is met by Nemotron alone, plus Parakeet and Cosmos.

---

## 2. Revised scope that answers the five critiques

### 2.1 Critique by critique

| # | Critique | Our answer | How we show it |
|---|---|---|---|
| 1 | LIBERO checkpoints are tied to fixed scenes; tasks from different scenes cannot chain | We use only LIBERO-Goal, whose ten tasks share one byte-identical scene. Chaining is done in our own loop with `ignore_done=True`, goal swapping and return-home | Chain A runs live in one unbroken shot. The nightly job reports each skill's success alone and in-chain |
| 2 | The registry is a benchmark's canned instruction list: "a face-gesture menu bolted onto a benchmark replay" | Each registry entry is a capability model, not a string. It holds preconditions and effects (BDDL predicates evaluated on live sim state), measured success alone and in-chain with a Wilson lower bound, risk class, confirm policy, Cosmos questions and an undo. Nemotron composes multi-step plans that a symbolic checker verifies (STRIPS-style rollout of effects). We add at least one skill that is not in the benchmark (`bowl_to_user`), trained on our own demos on Nebius. Runs are closed-loop GR00T inference, never trajectory replay; mock mode, which does replay, is labelled REPLAY on screen | The HUD shows the planner trace, validator checks, measured stats and nightly job id. The video includes the custom skill and a rejected unsafe proposal |
| 3 | LIBERO has none of the tasks this user needs | We say plainly that LIBERO objects are stand-ins. Skills are chosen and named from published need studies: handling objects (79% actual use in ALS), opening cabinets and drawers (64%), pressing buttons and switches (71%), serving food (64%) ([Spittel 2024](https://link.springer.com/article/10.1186/s42466-024-00342-3)). The custom skill brings an object to a handover zone in front of the user (fetching). Lights and caregiver calls go to Home Assistant, not the arm. Drinking, feeding, near-face tasks and fetching from the floor are out of scope; they need taught fixed trajectories (as Obi does) and a human model, so they are future work | Section 2.3 mapping table goes into the README and the video |
| 4 | Cloud sim video adds lag to the user's only feedback channel | (a) Every gesture is acknowledged on the phone within 100 ms, with no network involved. (b) The e-stop latches on the phone instantly, and the sim honours it within one control step (50 ms) of receipt. (c) The robot view is a pose stream drawn locally at native resolution: about 37 KB/s, no codec, and newest-pose-wins so frames never queue. (d) The user picks goals rather than steering, so a few tens of ms of view lag does not affect control. (e) The HUD shows measured view lag and e-stop latency | Latency numbers on screen in the video. E-stop shown mid-motion |
| 5 | Low-res renders and per-task failures on camera | The robot view renders in the browser at screen resolution. The 256 px policy cameras appear only in a small "what the robot sees" inset. For the video, logged sim states are re-rendered offline at 1080p (same physics, so it is honest). Failures are part of the product: the measured success rate is shown before the user confirms, Cosmos plus the oracle detect failure, and the planner offers retry or alternatives | One real failure, caught and recovered, appears in the video |

### 2.2 In scope and out of scope

**In scope (Oct 29):**
- Phone PWA: calibration, the gesture state machine, on-device stop (jaw-open and keyword), 1-3 option cards, spoken feedback.
- Voice input through Parakeet; intent and planning through Nemotron on Token Factory.
- Deterministic validator; confirm flow; e-stop with latency telemetry.
- GR00T N1.7 executing LIBERO-Goal skills in one continuous session.
- One custom GR00T skill, `bowl_to_user`. Stretch: `close_middle_drawer` as an undo.
- Cosmos-Reason2-8B checks before and after each skill.
- Nightly Nebius Serverless Job that measures reliability and refreshes the mock library.
- Home Assistant: demo light entities and a caregiver notification.
- Presenter page: 3D view, HUD and trace.
- Mock mode.

**Out of scope:**
- A physical arm.
- Feeding, drinking or anything near the face.
- Fetching from the floor.
- RoboCasa in the live product (B-roll only).
- Multiple users; production auth.
- Any claim of clinical validation.

### 2.3 Need-to-skill mapping (for README and video)

| User need (source) | Nod today | Honest gap |
|---|---|---|
| Handling objects, reach and grasp: top need in tetraplegia ([Hutmacher 2025](https://link.springer.com/article/10.1186/s12984-025-01642-8)), 79% actual use in ALS | `cheese_in_bowl`, `bowl_on_plate`, `bottle_on_rack`, `bowl_to_user` (custom) | Toy objects; one scene |
| Opening cabinets and drawers: 64% actual, 85% expected; manual joystick success only 39% ([Sensors 2025](https://www.mdpi.com/1424-8220/25/15/4768)) | `open_middle_drawer`, `bowl_in_top_drawer` | No close skill until the custom undo exists |
| Pressing buttons and switches: 71% actual | `stove_on` (gated, caregiver-only), Home Assistant lights | The stove is a hazard demo, not a feature |
| Serving drinks: 79% actual; JACO has a drinking mode | Not in scope | Needs a mug in the scene and a taught approach to the face |
| Mode switching costs 17.4% of task time on JACO ([Herlant 2016](https://personalrobotics.cs.washington.edu/publications//herlant2016modeswitching.pdf)) | Goal-level choices, with no mode switching | Users also want fine control ([Styler 2024](https://www.mdpi.com/2076-0825/13/3/104)). Stretch: "nudge" options |

---

## 3. Architecture

### 3.1 Components and where they run

| Component | Runs on | Tech | Notes |
|---|---|---|---|
| **Phone PWA** | User's phone (Android Chrome first) | Plain JS ES modules, MediaPipe Face Landmarker `@mediapipe/tasks-vision@1.0.1`, TF.js speech-commands for "stop", Web Speech synthesis for spoken output | Camera frames never leave the phone; only events are sent. Needs a secure context (HTTPS) |
| **Gateway** | GPU VM (live) or laptop (mock) | Python 3.11, FastAPI, asyncio, pyzmq, httpx | Session state machine, e-stop fast path, watchdog relay, planner client, validator, registry, ASR, Cosmos client, Home Assistant client, recorder |
| **ASR** | Inside the gateway process (VM CPU) | Parakeet TDT 0.6B v3 via `onnx-asr` int8 (MIT; model CC-BY-4.0) | Fake ASR in mock mode (text input plus canned transcripts) |
| **Sim runner** | GPU VM | Python 3.12 LIBERO venv, robosuite 1.4.0, MuJoCo 3.3.1, EGL | Interruptible 20 Hz loop, goal swapping, scripted moves, oracle world state, frame capture, pose publisher, recorder |
| **GR00T policy server** | GPU VM | Isaac-GR00T `run_gr00t_server.py` (Python 3.12, torch 2.9 cu128) | ZMQ REQ/REP on port 5555, never exposed outside the VM |
| **Cosmos verifier** | GPU VM (co-hosted) | vLLM serving `nvidia/Cosmos-Reason2-8B` (32 GB minimum per the [cosmos-reason2 README](https://github.com/nvidia-cosmos/cosmos-reason2)) | OpenAI-compatible on `127.0.0.1:8001`. Alternative: Cosmos3-Nano from the Nebius Serverless template (section 7.6) |
| **Viewer** | GPU VM (live, CPU only) or laptop (mock) | Python, MuJoCo 3.14 and `mjviser`/`viser` | Separate process, because mjviser needs MuJoCo 3.6 or newer while the sim is pinned to 3.3.1 |
| **Presenter page** | Laptop or TV browser | Static page served by the gateway: viser iframe, HUD, trace, stop key | Can send STOP only; cannot confirm |
| **Home Assistant** | Docker on the VM (live) or laptop (mock, optional) | HA `demo` integration lights plus a notify target | Gateway calls the REST API. FakeHome in mock mode |
| **Nemotron** | Nebius Token Factory | `nvidia/Nemotron-3_5-Lightning` (fast path), `nvidia/nemotron-3-super-120b-a12b` (plans and recovery) | Catalog confirmed 2026-10-05 ([catalog](https://tokenfactory.nebius.com/model-catalog.md)) **[checked]** |
| **Nightly eval** | Nebius Serverless Job (RTX PRO 6000, uk-south2) | Our image: GR00T plus the LIBERO venv plus `nod_eval` | Triggered by a GitHub Actions cron. Writes stats and recordings to Object Storage |
| **TLS front** | GPU VM | Caddy with Let's Encrypt on `<ip-dashed>.sslip.io` hostnames; Nebius Tunnels as a fallback | Ports 443 and 22 open only |

**Where the gateway runs.** The brief placed the gateway on the laptop. In live mode we run it on the VM, for three reasons:
1. The e-stop then crosses one internet hop (phone to VM) instead of two processes and two links (phone to laptop to VM).
2. The phone needs a publicly trusted HTTPS origin anyway, which is easy on the VM and awkward on a LAN.
3. The sim watchdog can trust a gateway that sits next to it.

The same gateway container runs on the laptop in mock mode. If the team still wants a laptop gateway in live mode, the only change is that `SimLink` connects over an SSH tunnel. The cost is one more process in the stop path.

**Region: uk-south2, platform `gpu-rtx6000-a`, preset `1gpu-24vcpu-218gb`.** Default RTX PRO 6000 quota is 32 there, 32 in eu-south1 and 0 in us-central1, and Serverless AI is not offered in eu-south1 (nebius-ops report, Nebius quota and region docs). Do not use H100, H200 or B200 for anything that renders. Nebius's own Physical AI repo says rendering "require[s] RT cores", and the H100 has no NVENC ([NVIDIA matrix](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)).

### 3.2 Diagram

```
 USER'S PHONE (PWA, HTTPS)                                   LAPTOP / TV (presenter page)
 +-------------------------------------------+               +-------------------------------+
 | camera -> MediaPipe FaceLandmarker (GPU)  |               | viser 3D view (iframe)        |
 |   -> gesture FSM -> UI FSM (1-3 cards)    |               | HUD: state, lag, e-stop ms,   |
 | STOP detectors: jaw-hold, KWS "stop",     |               |   success stats, trace        |
 |   on-screen button -> local red latch     |               | STOP key (space)              |
 | mic PCM16 (only while LISTENING)          |               +---------------+---------------+
 +------+-------------------------+----------+                               |
        | wss /ws/phone           | wss /ws/safety                            | wss (viser) + /ws/presenter
        | (events, audio)         | (hb 100 ms, estop)                        |
 =======|=========================|=========================================|============ internet
        v                         v                                          v
 NEBIUS GPU VM, uk-south2, RTX PRO 6000 96 GB  (docker compose)
 +---------------------------------------------------------------------------------------------+
 | caddy :443  nod.<ip>.sslip.io -> gateway:8000     view.<ip>.sslip.io -> viewer:8080          |
 |                                                                                             |
 | GATEWAY (py3.11)                                                                            |
 |  session FSM | SAFETY FAST PATH (no LLM, no awaits on other work) | watchdog relay           |
 |  ASR Parakeet (onnx-asr, CPU) | Planner client --HTTPS--> Token Factory (Nemotron)          |
 |  Validator (pure) <- Registry (skills.yaml + nightly stats) | Cosmos client -> vLLM :8001    |
 |  Home Assistant client -> homeassistant:8123 | Recorder (runs/, traces)                     |
 |      | ZMQ ctl (DEALER) :5560    | ZMQ estop (PUSH) :5561      ^ ZMQ status/world PUB :5562   |
 |      v                           v                             |                             |
 | SIM RUNNER (py3.12, LIBERO, mujoco 3.3.1, EGL) ------------------+                            |
 |  20 Hz interruptible loop, goal swap, scripted moves, oracle, frame capture                 |
 |      | ZMQ REQ :5555 (localhost only)          | ZMQ PUB poses :5563                         |
 |      v                                         v                                             |
 | GR00T N1.7 server (libero_goal)          VIEWER (mujoco 3.14 + mjviser) :8080                |
 | vLLM Cosmos-Reason2-8B :8001             homeassistant :8123 (demo lights, notify)           |
 +---------------------------------------------------------------------------------------------+
 NIGHTLY: GitHub Actions cron -> nebius ai job create (same sim+policy image)
          -> per-skill and per-chain eval -> s3://nod-results/<date>/{stats.json, runs/*}
          -> gateway pulls stats at start; recordings feed the mock library
 MOCK (any laptop, CPU): gateway(NOD_MODE=mock: FakePlanner, FakeASR, FakeCosmos, FakeHome,
          MockRunner replaying poses.npz) + viewer(scene.mjb) + PWA on http://localhost
```

### 3.3 Transports and ports

| Link | Transport | Port | Notes |
|---|---|---|---|
| Phone to gateway, events and audio | WSS `/ws/phone` | 443 | JSON text frames plus binary audio frames |
| Phone to gateway, safety | WSS `/ws/safety` | 443 | **A separate socket**, so audio uploads and large proposals never delay a stop (TCP head-of-line blocking) |
| Presenter to gateway | WSS `/ws/presenter` | 443 | Read-only trace, plus `estop` |
| Browsers to viewer | WSS (viser protocol) | 443 to 8080 | viser coalesces updates in 1/60 s windows, newest wins |
| Gateway to sim, control | ZMQ DEALER to ROUTER, msgpack | 5560 | Requests carry `epoch` |
| Gateway to sim, e-stop | ZMQ PUSH to PULL, msgpack | 5561 | Dedicated socket, polled every control step |
| Sim to gateway, status and world | ZMQ PUB to SUB | 5562 | |
| Sim to viewer, poses | ZMQ PUB to SUB, msgpack with float32 | 5563 | |
| Sim to GR00T | Isaac-GR00T `PolicyClient` (ZMQ REQ/REP, msgpack-numpy) | 5555 | `timeout_ms=2000`. Not published outside the compose network, because `run_gr00t_server.py` exposes no `api_token` flag **[checked]** |
| Gateway to Cosmos | HTTP, OpenAI-compatible | 8001 | localhost only |
| Gateway to Token Factory | HTTPS, OpenAI SDK | 443 out | |

### 3.4 Session state machine (gateway)

```
IDLE --menu|utterance--> LISTENING --utter.end--> PLANNING --proposal--> AWAITING_CONFIRM
AWAITING_CONFIRM --confirm ok--> COUNTDOWN(1.0 s) --> EXECUTING --> POSTCHECK --> IDLE (or next plan step)
AWAITING_CONFIRM --cancel|expire--> IDLE
EXECUTING --hb lost|face lost--> PAUSED --confirm resume--> EXECUTING ; --cancel--> IDLE
any --estop--> ESTOPPED --estop.reset (long blink or caregiver touch)--> IDLE   (never resumes the run)
```

Only one run can be in flight. A plan with several steps runs as consecutive runs. The default is to confirm each step. The optional "run whole plan" mode starts each next step after a 3 s spoken countdown that can be cancelled.

### 3.5 Message schemas

The source of truth is the `nod_protocol` Python package (pydantic v2), which supports Python 3.11 and 3.12. JSON Schema is generated from it into `protocol/schema/*.json`, and the PWA validates against those files in development.

**Common envelope.** Every WebSocket message carries these fields. The payload fields sit at the top level next to them, with `type` as the discriminator:

```json
{"v": 1, "type": "<type>", "id": "01J9ZK3M8Q4T6W1Y2R5B7C9D0E", "ts": 1791205200123, "sess": "s_7f3a", "seq": 42}
```

- `id` is a ULID. `ts` is the sender's wall clock in ms. `seq` increases per sender per socket and is used for de-duplication.
- The gateway estimates the phone clock offset from `ping`/`pong`.

#### 3.5.1 Phone to gateway, `/ws/phone`

```json
{"type": "hello", "device": "pixel-7", "app_ver": "0.3.0", "role": "user", "resume_token": null,
 "calib": {"profile_id": "cal_01J9", "long_blink_ms": 700, "fps": 24}}

{"type": "gesture", "name": "long_blink", "phase": "fired", "held_ms": 742,
 "yaw_deg": 3.1, "pitch_deg": -2.0, "scores": {"blink": 0.81, "brow": 0.05, "jaw": 0.02}, "fps": 24}
```

- `gesture.name` is one of `long_blink | double_blink | brow_raise | turn_left | turn_right | jaw_hold | face_lost | face_found`.
- `gesture.phase` is one of `fired | cancelled`. Start and progress phases stay on the phone. These messages are telemetry; the phone's UI state machine has already acted on them.

```json
{"type": "ui.menu", "source": "brow_raise"}
{"type": "ui.focus", "proposal_id": "p_01J9", "option_id": "o2"}
{"type": "ui.confirm", "proposal_id": "p_01J9", "option_id": "o2", "nonce": "n_5c1e",
 "gesture": "long_blink", "held_ms": 742, "shown_at": 1791205201000, "focused_at": 1791205202100}
{"type": "ui.cancel", "proposal_id": "p_01J9", "source": "jaw_hold"}

{"type": "utter.start", "utt_id": 7, "sr": 16000, "fmt": "pcm16le"}
// binary frame: [u32 little-endian utt_id][pcm16le samples]
{"type": "utter.end", "utt_id": 7, "dur_ms": 2140}
{"type": "utter.text", "utt_id": 8, "text": "I'm a bit hungry"}   // typed by a caregiver, or mock mode
{"type": "ping", "t0": 1791205200123}
```

#### 3.5.2 Phone to gateway, `/ws/safety`

```json
{"type": "hb", "face_present": true, "track_conf": 0.97, "fps": 24, "latched": false}
{"type": "estop", "estop_seq": 3, "source": "jaw_hold", "t_detect": 1791205203456, "held_ms": 310}
{"type": "estop.reset", "estop_seq": 3, "gesture": "long_blink", "held_ms": 820}
```

- `hb` is sent every 100 ms.
- `estop.source` is one of `jaw_hold | voice_stop | button | presenter`.
- The phone resends `estop` every 100 ms until it receives the matching `estop.ack`.

#### 3.5.3 Gateway to phone

```json
{"type": "welcome", "sess": "s_7f3a", "resume_token": "rt_...", "lease": "user", "server_time": 1791205200150}
{"type": "pong", "t0": 1791205200123, "t_srv": 1791205200160}

{"type": "state", "mode": "AWAITING_CONFIRM", "run_id": null, "scene_id": "libero_goal_v1", "estop_seq": 2}

{"type": "transcript", "utt_id": 7, "text": "I'm a bit hungry", "asr_ms": 212}

{"type": "proposal", "proposal_id": "p_01J9", "expires_at": 1791205231000, "arm_delay_ms": 300,
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
 "focus": "o1", "nonce": "n_5c1e"}

{"type": "run", "run_id": "r_01J9", "proposal_id": "p_01J9", "option_id": "o1", "plan_step": 0, "plan_len": 3,
 "skill_id": "bowl_on_plate", "phase": "executing", "sim_step": 64, "max_steps": 400, "elapsed_ms": 3200,
 "cosmos": null, "oracle": null}

{"type": "estop.ack", "estop_seq": 3, "stopped_at_step": 143, "lat_ms": {"phone_gw": 41, "gw_sim": 2, "sim_halt": 18}}

{"type": "notice", "level": "info", "text": "Putting the bowl on the plate", "speak": true}

{"type": "thumb", "camera": "agentview", "w": 256, "h": 256, "step": 64}
// thumb is followed by a binary JPEG frame; at most 5 fps; optional inset
```

- `run.phase` is one of `countdown | precheck | executing | returning_home | postcheck | done | failed | uncertain | aborted | paused`.
- `label` always comes from the registry, never from the LLM (rule R13). Nemotron's `why` text is shown smaller, under the label.

#### 3.5.4 Gateway to sim, ZMQ msgpack

```json
{"type": "ctl.load_scene", "req_id": "q_1", "scene_id": "libero_goal_v1", "init_index": 0, "epoch": 12}

{"type": "ctl.run_skill", "req_id": "q_2", "run_id": "r_01J9", "epoch": 12, "skill_id": "bowl_on_plate",
 "kind": "policy", "instruction": "put the bowl on the plate", "goal_ref": "libero_goal/put_the_bowl_on_the_plate",
 "n_action_steps": 8, "max_steps": 400,
 "limits": {"eef_box": [[-0.40, -0.40, 0.85], [0.20, 0.40, 1.35]], "max_abs_action": 1.0}}

{"type": "ctl.run_scripted", "req_id": "q_3", "run_id": "r_01JA", "epoch": 12,
 "script": "return_home", "max_steps": 160}
// script is one of return_home | reset_scene | close_middle_drawer | handover_retreat

{"type": "ctl.pause", "req_id": "q_4", "epoch": 12}
{"type": "ctl.resume", "req_id": "q_5", "epoch": 12}
{"type": "ctl.abort", "req_id": "q_6", "epoch": 12}
{"type": "ctl.capture", "req_id": "q_7", "cameras": ["frontview", "agentview"], "w": 640, "h": 480}
{"type": "ctl.world", "req_id": "q_8"}
{"type": "ctl.hb", "t": 1791205200150, "phone_ok": true}

// e-stop socket (PUSH -> PULL):
{"type": "ctl.estop", "estop_seq": 3, "t_gw": 1791205203497}
```

- `goal_ref` names the BDDL whose goal the sim swaps in. For custom skills it is `nod/bring_the_bowl_to_me`.
- `limits.eef_box` holds placeholder values. Fill them on Day 1 from the extremes of the demo trajectories **[unverified]**.
- `ctl.hb` is sent every 100 ms, and only while the phone's heartbeat is fresh.

#### 3.5.5 Sim to gateway, PUB

```json
{"type": "sim.ack", "req_id": "q_2", "ok": true, "error": null}
{"type": "sim.status", "run_id": "r_01J9", "epoch": 12, "phase": "executing", "step": 64, "success": false,
 "fault": null, "policy_ms": 88.1, "step_ms": 14.2}
{"type": "sim.estop_ack", "estop_seq": 3, "epoch": 13, "stopped_at_step": 143, "t_rx": 1791205203499, "t_halt": 1791205203517}
{"type": "sim.world", "scene_id": "libero_goal_v1", "step": 247, "t_sim": 12.35,
 "atoms": {"on(akita_black_bowl_1,plate_1)": false, "on(cream_cheese_1,akita_black_bowl_1)": false,
           "on(wine_bottle_1,wine_rack_1_top_region)": true, "open(wooden_cabinet_1_middle_region)": true,
           "turnon(flat_stove_1)": false, "on(akita_black_bowl_1,main_table_handover_region)": false},
 "objects": {"akita_black_bowl_1": {"pos": [-0.09, 0.0, 0.90]}},
 "eef": {"pos": [-0.21, 0.02, 1.12], "gripper": "open"}}
{"type": "sim.frames", "req_id": "q_7", "step": 247,
 "frames": [{"camera": "frontview", "w": 640, "h": 480, "jpeg": "<bytes>"}]}
{"type": "sim.episode_header", "scene_id": "libero_goal_v1", "model_hash": "sha256:...", "mj_ver": "3.3.1",
 "body_names": ["world", "..."], "dyn_body_ids": [12, 13, 14], "model_patch": {"body_pos": {}, "body_quat": {}}}
```

- `sim.status.fault` is one of `eef_out_of_box | policy_timeout | nan_action | hb_lost | budget`.
- In `sim.world`, `atoms` is the oracle: the ground atoms used by the registry, evaluated with LIBERO's `_eval_predicate`.

#### 3.5.6 Sim to viewer, PUB (msgpack, arrays as float32)

```
{"type": "pose", "epoch": 13, "step": 247, "t_sim": 12.35, "t_wall": 1791205204.512,
 "xpos": f32[nb,3], "xquat": f32[nb,4], "qpos": f32[nq]}
```

The viewer applies `model_patch` from `sim.episode_header` before it builds the scene. LIBERO moves fixtures by writing `model.body_pos` at reset; the streaming report measured 0.9 m errors without this patch.

#### 3.5.7 Cosmos check (gateway internal; logged; mirrored to the presenter)

```json
{"type": "cosmos.check", "check_id": "k_01J9", "run_id": "r_01J9", "phase": "post", "skill_id": "bowl_on_plate",
 "question": "Is the black bowl resting on the plate? Reply with JSON {\"answer\": \"yes\"|\"no\"|\"unsure\", \"reason\": \"...\"}",
 "frames": [{"camera": "frontview", "w": 640, "h": 480, "step": 312}], "model": "nvidia/Cosmos-Reason2-8B"}

{"type": "cosmos.result", "check_id": "k_01J9", "answer": "yes", "reason": "The bowl sits on the white plate.",
 "latency_ms": 840, "oracle": {"atom": "on(akita_black_bowl_1,plate_1)", "value": true}, "agree": true,
 "decision": "done"}
```

`decision` is one of `done | failed | uncertain | proceed | block`; see the table in section 5.4. The request to vLLM is a standard chat completion with one `image_url` data URI per frame, `temperature` 0, `max_tokens` 128, and `response_format` set to a JSON schema. That vLLM's guided decoding works with Cosmos-Reason2's reasoning parser is **[unverified]**. The fallback is to parse a leading yes/no.

#### 3.5.8 Gateway to presenter, `/ws/presenter`

```json
{"type": "trace", "kind": "planner|validator|cosmos|run|estop|asr|home", "ref": "p_01J9", "data": {}}
{"type": "metrics", "estop_ms_p50": 63, "estop_ms_p95": 118, "view_lag_ms": 35, "policy_ms_p95": 97, "asr_ms_p50": 210, "llm_ms_p50": 640}
```

---

## 4. Repo layout and interfaces

License: Apache-2.0 (Devpost requires an open-source license). No model weights go in the repo.

### 4.1 Tree

```
nod/
  README.md                  setup, mock-mode quickstart, honest-scope section, need-to-skill table
  LICENSE  Makefile  .env.example  compose.yaml
  docs/design.md             this file
  docs/runbooks/             day1-spike.md, demo-day.md, nightly.md
  protocol/
    nod_protocol/            pydantic models for every message in 3.5; msgpack helpers
    schema/                  generated JSON Schema (make schema)
  registry/
    skills.yaml              skill entries (4.2.3)
    stats/latest.json        written by the nightly job (pulled, not hand-edited)
  gateway/                   Python 3.11, FastAPI
    pyproject.toml
    nod_gateway/
      app.py  config.py  session.py  safety.py  simlink.py  registry.py  validator.py
      planner/{base.py, nemotron.py, fake.py, prompts/}  asr/{base.py, parakeet.py, fake.py}
      cosmos/{base.py, vllm.py, fake.py}  home/{base.py, hass.py, fake.py}
      recorder.py  presenter.py  mock_runner.py
    tests/
  pwa/                       plain JS, no build step
    index.html  presenter.html  manifest.webmanifest  sw.js
    js/{face.js, gestures.js, calibrate.js, ui.js, net.js, estop.js, kws.js, audio.js, tts.js}
    vendor/                  pinned MediaPipe wasm + face_landmarker.task + tfjs (make vendor)
    tests/                   node --test on gestures.js with recorded traces
  sim/                       Python 3.12, runs inside the LIBERO venv / image
    nod_sim/{runner.py, scene.py, skills.py, scripted.py, oracle.py, capture.py, recorder.py, rerender.py}
    bddl/nod/bring_the_bowl_to_me.bddl
    tests/
  viewer/                    mujoco 3.14 + mjviser
    nod_viewer/{main.py, export_scene.py}
  training/                  custom skill
    gen_demos.py  to_lerobot.py  finetune.sh  eval_custom.sh
  eval/
    nod_eval/{run.py, chains.py, cosmos_agree.py, publish.py}
  docker/
    policy.Dockerfile  sim.Dockerfile  gateway.Dockerfile  viewer.Dockerfile  eval.Dockerfile
  infra/
    nebius/{create_vm.sh, cloud-init.yaml, security_group.sh, job_nightly.sh}
    caddy/Caddyfile
    hass/configuration.yaml
    github/workflows/{ci.yml, nightly.yml}
  recordings/                mock library (git LFS) + scene_libero_goal_v1.mjb (make fetch-mock)
```

### 4.2 Module responsibilities and public interfaces

#### 4.2.1 `protocol/nod_protocol`

```python
class Envelope(BaseModel): v: int = 1; type: str; id: str; ts: int; sess: str | None; seq: int
# One model per message type in 3.5, plus:
class WorldState(BaseModel): scene_id: str; step: int; t_sim: float; atoms: dict[str, bool]; objects: dict[str, ObjPose]; eef: Eef
class SkillSpec(BaseModel): ...   # mirrors skills.yaml
def dumps(msg: BaseModel) -> bytes            # msgpack for ZMQ
def loads(data: bytes) -> BaseModel           # discriminated on "type"
```

#### 4.2.2 `gateway/nod_gateway`

```python
# app.py: FastAPI factory
def create_app(cfg: Config) -> FastAPI

# config.py: NOD_MODE in {"live", "mock"}; each dependency is chosen by mode and can be overridden by env
class Config(BaseSettings):
    mode: Literal["live", "mock"]; tf_api_key: str | None; tf_fast_url: str; tf_plan_url: str
    fast_model: str = "nvidia/Nemotron-3_5-Lightning"; plan_model: str = "nvidia/nemotron-3-super-120b-a12b"
    sim_ctl: str = "tcp://sim:5560"; sim_estop: str = "tcp://sim:5561"; sim_pub: str = "tcp://sim:5562"
    cosmos_url: str | None; hass_url: str | None; hass_token: str | None
    hb_timeout_ms: int = 500; face_lost_ms: int = 1000; proposal_ttl_s: int = 30

# safety.py: the fast path; must not await anything except a socket send
class SafetyChannel:
    def on_estop(self, msg: EStop) -> None        # sync: forward to sim, latch session, cancel tasks
    def on_hb(self, msg: Heartbeat) -> None
    def watchdog_tick(self) -> None               # every 50 ms: relay ctl.hb only while phone hb is fresh

# session.py
class Session:
    async def on_menu(self, src: str) -> None
    async def on_utterance(self, utt: Utterance) -> None
    async def on_confirm(self, c: UiConfirm) -> None
    async def on_cancel(self, c: UiCancel) -> None
    async def on_sim_status(self, s: SimStatus) -> None
    def snapshot(self) -> SessionView              # immutable view handed to the validator

# validator.py: PURE, no I/O, no clock reads (now_ms is passed in)
def feasible_skills(world: WorldState, reg: Registry, pol: SafetyPolicy) -> list[Feasibility]
def check_plan(skill_ids: list[str], world: WorldState, reg: Registry) -> PlanCheck
def validate_proposal(out: PlannerOutput, world: WorldState, reg: Registry, sess: SessionView,
                      pol: SafetyPolicy, now_ms: int) -> ValidationResult
def validate_confirm(c: UiConfirm, p: Proposal, sess: SessionView, now_ms: int) -> ConfirmDecision
def post_decision(oracle: bool | None, cosmos: CosmosResult | None) -> Literal["done", "failed", "uncertain"]

# registry.py
def load_registry(path: Path, stats: Path | None) -> Registry
def wilson_lower(k: int, n: int, z: float = 1.96) -> float

# planner/base.py
class Planner(Protocol):
    async def propose(self, req: PlanRequest) -> PlannerOutput      # raises PlannerError on timeout or schema failure
# planner/nemotron.py
class NemotronPlanner(Planner):
    def __init__(self, cfg: Config, http: httpx.AsyncClient) -> None
# planner/fake.py
class FakePlanner(Planner):
    def __init__(self, canned: Path) -> None   # regex on transcript -> recorded PlannerOutput; deterministic

# asr/base.py, cosmos/base.py, home/base.py
class ASR(Protocol):      async def transcribe(self, pcm16: bytes, sr: int = 16000) -> Transcript
class Verifier(Protocol): async def check(self, req: CosmosCheck) -> CosmosResult
class Home(Protocol):     async def call(self, action: Literal["lights_on", "lights_off", "call_caregiver"]) -> HomeResult

# simlink.py
class SimLink:
    async def load_scene(self, scene_id: str, init_index: int) -> None
    async def run(self, cmd: RunSkill | RunScripted) -> None
    def estop(self, seq: int) -> None                     # non-blocking send on the dedicated socket
    async def capture(self, cams: list[str], w: int, h: int) -> list[Frame]
    async def world(self) -> WorldState
    def status(self) -> AsyncIterator[SimStatus | SimEStopAck | WorldState]

# mock_runner.py: speaks the SAME ZMQ protocol as nod_sim.runner; needs only numpy + pyzmq
class MockRunner:
    def __init__(self, library: Path, ctl: str, estop: str, pub: str, pose_pub: str, hz: int = 20) -> None
    def serve_forever(self) -> None
```

#### 4.2.3 `registry/skills.yaml` entry

```yaml
- skill_id: bowl_on_plate
  ver: 1
  kind: policy                     # policy | scripted | home
  label: "Put the bowl on the plate"
  scene_id: libero_goal_v1
  policy: {checkpoint: "GR00T-N1.7-LIBERO/libero_goal", embodiment: LIBERO_PANDA,
           instruction: "put the bowl on the plate", n_action_steps: 8, max_steps: 400}
  goal_ref: libero_goal/put_the_bowl_on_the_plate
  pre:     ["not on(akita_black_bowl_1,plate_1)", "not in(akita_black_bowl_1,wooden_cabinet_1_top_region)",
            "not on(cream_cheese_1,akita_black_bowl_1)"]
  effects: ["on(akita_black_bowl_1,plate_1)"]
  risk: low                        # none | low | high (high = caregiver-gated)
  confirm: single                  # single | double
  near_user: false
  cosmos:
    pre:  "Is there a black bowl on the table and is the plate empty?"
    post: "Is the black bowl resting on the plate?"
  undo: null
  stats: {}                        # filled from registry/stats/latest.json at load
```

The third precondition on `bowl_on_plate` is a design choice: don't move a bowl that has cheese in it, which keeps us closer to the training distribution.

#### 4.2.4 `sim/nod_sim`

```python
# scene.py
def make_session_env(scene_id: str = "libero_goal_v1", init_index: int = 0) -> LiberoEnv
def load_goals() -> dict[str, list]            # goal_ref -> parsed goal_state (all 10 + custom BDDLs)

# skills.py
def swap_skill(env: LiberoEnv, instruction: str, goal: list) -> None   # sets _task_description + parsed_problem
# runner.py
class SimRunner:
    def __init__(self, env: LiberoEnv, policy: PolicyClient, addrs: SimAddrs, hz: int = 20) -> None
    def serve_forever(self) -> None
    def run_policy_skill(self, cmd: RunSkill) -> RunResult      # interruptible loop, 5.5
    def run_scripted(self, cmd: RunScripted) -> RunResult
    def controlled_stop(self) -> int                            # returns stopped_at_step
# oracle.py
def world_state(env: LiberoEnv, atoms: list[str]) -> WorldState     # uses problem env _eval_predicate
# scripted.py: waypoint controllers on privileged poses; also used by training/gen_demos.py and fallback T4
def return_home(env, obs, home_xyz) -> Iterator[Action]
def pick_place(env, obj: str, target_xyz: np.ndarray) -> Iterator[Action]
# capture.py
def render(env, camera: str, w: int, h: int) -> np.ndarray        # env._env.sim.render(..., camera_name=...)[::-1]
# recorder.py: per run -> manifest.json, poses.npz, model_patch.json, policy_cam.mp4, events.jsonl
# rerender.py: replay saved sim states with set_state + forward, render 1920x1080 frontview for the video
```

#### 4.2.5 Other modules

- **`viewer/nod_viewer`**
  - `export_scene.py --out recordings/scene_libero_goal_v1.mjb` runs inside the sim image: `env.sim.model.get_xml()`, then compile with MuJoCo 3.14, then `mj_saveModel`.
  - `main.py --mjb ... --sub tcp://sim:5563 --port 8080` builds a `ViserMujocoScene` and calls `update_from_arrays(xpos, xmat)` on every pose. It also draws a translucent handover-zone box and a seated-user marker. These are viewer-only overlays, so the policy cameras never see them.
- **`eval/nod_eval`**
  - `run.py --registry registry/skills.yaml --episodes 20 --chains A,B --out /mnt/results/<date>` reuses `SimRunner` without the gateway and records every episode.
  - `publish.py` writes `stats.json` with `{skill_id: {alone: {k, n, wilson_lo, median_s}, in_chain: {...}}, chains: {...}, job_id, git_sha, at}`.
  - `cosmos_agree.py` scores Cosmos verdicts against oracle atoms on the recorded frames.
- **`training`**
  - `gen_demos.py --n 80 --keep-success` writes HDF5 episodes from the scripted controller.
  - `to_lerobot.py` converts them to LeRobot v2.1 with `meta/modality.json` copied from `examples/LIBERO`. Images are rotated 180 degrees to match `[::-1, ::-1]`, video is H.264 at 20 fps, and no-op frames are dropped.
  - `finetune.sh` wraps Isaac-GR00T `examples/finetune.sh` (4.5).

### 4.3 Gateway HTTP and WebSocket endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` , `/presenter` , static assets | PWA and presenter page |
| WS | `/ws/phone` , `/ws/safety` , `/ws/presenter` | Section 3.5 |
| GET | `/healthz` | Liveness of sim, policy, Cosmos, Token Factory and HA, each with its last error |
| GET | `/api/registry` | Skills with live stats |
| GET | `/api/world` | Latest `WorldState` |
| GET | `/api/runs?limit=20` , `/api/runs/{run_id}` | Full trace: transcript, planner request and response, validator checks, Cosmos, timings |
| GET | `/api/metrics` | Latency histograms (e-stop, view lag, policy, ASR, LLM) |
| POST | `/api/dev/inject` (mock, or with `NOD_DEV=1`) | Inject gestures, utterances or e-stops for tests and judges |
| POST | `/api/scene/reset` (caregiver lease only) | Scripted table reset |

### 4.4 Mock mode (CPU only, no keys)

The judge or developer quickstart:

```bash
git clone https://github.com/Aadil12m/nod && cd nod
make fetch-mock        # downloads recordings/ + scene .mjb from the GitHub release (about 30-60 MB, [unverified size])
docker compose --profile mock up --build
# open http://localhost:8000          (laptop webcam works: localhost is a secure context)
# open http://localhost:8000/presenter (3D view + HUD)
```

Without Docker:

```bash
make mock              # runs: uv run --project gateway nod-gateway --mock & uv run --project viewer nod-viewer --mjb ...
```

- **NOD_MODE=mock wiring:**

  | Component | Mock behaviour |
  |---|---|
  | `FakePlanner` | Recorded real Nemotron responses keyed by transcript; deterministic regex rules otherwise |
  | `FakeASR` | Canned transcripts plus a text box |
  | `FakeCosmos` | Returns the verdict recorded with the run |
  | `FakeHome` | Logs and shows a toast |
  | `MockRunner` | Replays `poses.npz` at 20 Hz, freezes immediately on `ctl.estop`, honours `ctl.pause` |

- A **REPLAY** badge stays on screen whenever mock mode is on.
- **Keyboard fallback for judges without a camera:**

  | Key | Gesture |
  |---|---|
  | Left / Right | Head turn |
  | Hold Enter | Long blink |
  | B | Brow raise |
  | Space | STOP |

- **Runs before the first recordings exist:** `MockRunner` emits status and world messages from a scripted timeline, and the viewer shows the static scene.
- **The library grows nightly.** Every nightly-job episode (successes and real failures) becomes a mock run, keyed by `(skill_id, outcome, variant)`.

### 4.5 Compose sketch (live profile on the VM)

```yaml
x-gpu: &gpu {deploy: {resources: {reservations: {devices: [{driver: nvidia, count: 1, capabilities: [gpu, utility, compute, graphics]}]}}}}
services:
  policy:   {<<: *gpu, profiles: [live], build: {dockerfile: docker/policy.Dockerfile},
             command: "uv run python gr00t/eval/run_gr00t_server.py --model-path /ckpt/libero_goal --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper --host 0.0.0.0 --port 5555",
             environment: [HF_TOKEN], volumes: ["ckpt:/ckpt", "hf:/root/.cache/huggingface"]}   # port NOT published
  sim:      {<<: *gpu, profiles: [live], build: {dockerfile: docker/sim.Dockerfile},
             environment: [MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl, MUJOCO_EGL_DEVICE_ID=0, NVIDIA_DRIVER_CAPABILITIES=all],
             command: "python -m nod_sim.runner --policy tcp://policy:5555"}
  cosmos:   {<<: *gpu, profiles: [live], image: "vllm/vllm-openai:v0.29.0", environment: [HF_TOKEN],
             command: "nvidia/Cosmos-Reason2-8B --max-model-len 16384 --gpu-memory-utilization 0.40 --port 8001 --reasoning-parser qwen3",
             volumes: ["hf:/root/.cache/huggingface"]}
  viewer:   {profiles: [live, mock], build: {dockerfile: docker/viewer.Dockerfile}}
  gateway:  {profiles: [live, mock], build: {dockerfile: docker/gateway.Dockerfile}, env_file: .env,
             environment: ["NOD_MODE=${NOD_MODE:-mock}"], ports: ["127.0.0.1:8000:8000"]}
  homeassistant: {profiles: [live, hass], image: "ghcr.io/home-assistant/home-assistant:stable", volumes: ["./infra/hass:/config"]}
  caddy:    {profiles: [live], image: "caddy:2", ports: ["443:443", "80:80"], volumes: ["./infra/caddy/Caddyfile:/etc/caddy/Caddyfile"]}
volumes: {ckpt: {}, hf: {}}
```

- The vLLM image tag and flags come from the Nebius template and the cosmos-reason2 README. Running it next to GR00T on one GPU is **[unverified]**.
- Start `cosmos` before `policy`, because vLLM reserves a fraction of total memory at startup.
- Stop `cosmos` before fine-tuning.

### 4.6 Tests

| Layer | Test |
|---|---|
| `validator.py` | Table-driven pytest covering every rule in 5.2, plus property tests (hypothesis): no proposal ever passes with an unknown skill or a false precondition |
| `gestures.js` | `node --test` with recorded blendshape traces (JSONL), checking that long blink, natural blinking, talking, yawning, head turns and fps drops produce the expected events. A 10-minute "talking" trace must produce **zero** confirms |
| Mock end-to-end (in CI) | Run the gateway with `MockRunner`, script menu, utterance, focus, confirm, run and done through `/api/dev/inject`, then inject an e-stop mid-run and assert `estop.ack` and that the run is `aborted`, not resumed |
| Sim (GPU only, nightly) | `nod_eval` |

---

## 5. Safety layer

### 5.1 Principles

1. **The LLM proposes, the validator decides, the user confirms, and the sim enforces limits.** Each of these layers works without the others.
2. **The stop path is short and deterministic:** phone latch, then the gateway fast path, then the dedicated sim socket, then a check before every `env.step`. No LLM, no Cosmos and no Token Factory are involved.
3. **A false stop is cheap; a false confirm is dangerous.** Stop is easy to trigger; confirm is hard.
4. **We say what this is.** The MediaPipe model cards state the models are "not intended for human life-critical decisions" ([Face Mesh V2 card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)). The gesture stop is a supplementary stop. A real arm needs a hardware, safety-rated stop (ISO 13850).

### 5.2 Validator rules (pure functions, unit tested)

| Rule | Check | On failure |
|---|---|---|
| R1 schema | Planner output parses into `PlannerOutput` (pydantic) | Deterministic menu (5.7) |
| R2 known skill | `skill_id` exists in the registry, is enabled, and matches the current `scene_id` | Drop the option, log it |
| R3 objects | Every object a skill references exists in `WorldState.objects` | Drop |
| R4 preconditions | Every `pre` atom holds in the oracle `WorldState` | Drop; the reason is shown if the user asked for that skill |
| R5 not a no-op | Not all `effects` are already true | Drop |
| R6 reliability | Offered normally if `n >= 20` and `wilson_lo >= 0.60`. Offered with a "less reliable" badge and `confirm: double` if `wilson_lo` is 0.30-0.60. Hidden below 0.30 or when `n < 10` (custom skills use the n of their last eval) | Demote, badge or hide |
| R7 risk | `risk: high` (`stove_on`, `bowl_on_stove`) needs `caregiver_enabled=true` in the session plus a double confirm | Shown as "Needs caregiver"; cannot be confirmed |
| R8 mode | New motion only from `IDLE` or `AWAITING_CONFIRM`; never in `ESTOPPED`, `PAUSED`, `EXECUTING` or `COUNTDOWN` | Reject |
| R9 confirm binding | `proposal_id`, `option_id` and `nonce` match the live proposal. The confirm arrives after `shown_at + arm_delay_ms` and after the focus has rested on the option for at least 300 ms, and before `expires_at` (30 s). It comes from the session holding the `user` lease, the nonce is single-use, and the last heartbeat showed `face_present` | Reject; the phone shows "Try again" |
| R10 plan check | For `kind: plan`, a symbolic rollout must hold: each step's `pre` is satisfied by the world plus earlier `effects`. The plan's estimate is the product of in-chain success rates | Drop the plan; single steps are still offered |
| R11 spacing | At least 2 s between the end of one motion and the start of the next, unless the user confirmed that step ([VoicePilot](https://arxiv.org/abs/2404.04066) used 4 s) | Queue |
| R12 count | 1-3 options with no duplicates. The UI always adds a fixed "None of these" card | Truncate |
| R13 labels | The label shown is always the registry's canonical label. LLM text appears only as the `why` subtitle, at most 80 characters, plain text | Not applicable |
| R14 near user | Skills with `near_user: true` (`bowl_to_user`) run with a lower speed scale (the action is multiplied by 0.6) and need a double confirm on first use per session | Not applicable |

### 5.3 Execution-time limits (in the sim loop, every step)

- **Range check.** Every action component must be finite and inside [-1, 1]. On failure: `fault: nan_action` and a controlled stop.
- **Workspace box.** Predict the commanded end-effector position as current position + 0.05 × action xyz. If it leaves `limits.eef_box`, zero the translation for that step, and after 3 consecutive violations raise `fault: eef_out_of_box` and stop. Robosuite's `position_limits` is null in `osc_pose.json` **[checked]**, so this check is ours.
- **Step budget.** `max_steps` is 400 (20 s) by default and is tuned from the Day-1 step distribution. On reaching it: `fault: budget`, the run is `failed`, and a Cosmos post-check still runs.
- **Policy timeout.** `PolicyClient(timeout_ms=2000)`. On timeout: `fault: policy_timeout` and a controlled stop.

### 5.4 Cosmos and oracle decisions

**Pre-check.** It runs speculatively, in parallel for each displayed option, as soon as the proposal is shown, so the result is usually ready before the user confirms.

| Oracle preconditions | Cosmos pre-check | Decision |
|---|---|---|
| hold | yes | `proceed` |
| hold | no or unsure | `proceed`, with an "Unsure from camera" badge and a double confirm |
| fail | anything | `block` (R4 already removed the option) |

**Post-check:**

| Oracle success | Cosmos | Decision | What the user sees |
|---|---|---|---|
| true | yes | `done` | "Done" plus the next suggestions |
| false | no | `failed` | "That didn't work" with the Cosmos reason, then Retry / alternative / call caregiver (Nemotron Super recovery prompt) |
| true | no or unsure | `uncertain` | "I'm not sure it worked", a camera snapshot, Accept / Retry |
| false | yes | `uncertain` | Same |

The oracle exists only because this is a simulation. We say so on screen and in the README. On a real robot Cosmos would be the only check, and the nightly job reports how often it agrees with the oracle (precision and recall per skill). That number is part of the pitch.

### 5.5 E-stop semantics and latency

**Sources:**
- jaw-open held for 300 ms or more;
- the on-device keyword "stop";
- the on-screen STOP button (caregiver touch);
- the presenter's space bar.

**Semantics** (following ISO 13850 as summarised [here](https://machinerysafety101.com/2026/05/18/iso-13850-emergency-stop-requirements/)):

1. **Phone:**
   - At detection, latch locally at once: full-screen red STOPPED, vibration, and spoken "Stopped".
   - Send `estop` on `/ws/safety` every 100 ms until it receives `estop.ack` with the same `estop_seq`.
2. **Gateway `SafetyChannel.on_estop`:**
   - Runs synchronously in the WebSocket receive handler: push `ctl.estop` to the sim socket first, then do everything else.
   - Everything else means: set the session to `ESTOPPED`, cancel planner, Cosmos and ASR tasks, invalidate the open proposal, and bump `epoch`.
   - Duplicate `estop_seq` values are ignored.
3. **Sim:**
   - The control loop polls the e-stop socket (non-blocking) before every `env.step`.
   - The pacing sleep and the wait for a policy chunk both wait on the same `threading.Event`, polled every 5 ms, so neither delays the stop.
   - **Controlled stop (Category 1 analogue):** 4 steps of zero translation and rotation with the current gripper command held, so nothing is dropped; then physics stepping halts.
   - The `epoch` increments and any chunk from the old epoch is discarded. The sim publishes `sim.estop_ack`.
4. **Latched:**
   - Only `estop.reset` clears it: a long blink on the "Clear stop?" screen, or a caregiver touch.
   - After reset the state is `IDLE`. **The interrupted run is aborted, never resumed.**
   - When idle, STOP means cancel/back.

**Latency budget.** All figures are targets, measured and shown on the HUD from `estop.ack.lat_ms`.

| Segment | Target |
|---|---|
| Jaw hold (by design) | 300 ms |
| Phone latch after detection | 1 frame (at most 67 ms at 15 fps) |
| Phone to gateway | One-way network: measure (expected 15-60 ms within Europe **[unverified]**) |
| Gateway forward | 5 ms or less |
| Sim halt after receipt | One control step, 50 ms or less (p95) |
| **Acceptance** | Detection-to-halt p95 of 250 ms or less from the demo location (hard limit 500 ms). The phone latch is always immediate |

We run the sim synchronously, as the official eval does: the sim does not step while waiting for a chunk. That keeps our conditions identical to the 97.5% measurement. The wait is still interruptible, as described above. Async chunk refill (as in the streaming report's loop, or LeRobot async inference) is an optimisation we use only if the 20 Hz view stutters visibly. Inference takes about 80-100 ms per chunk against 400 ms of chunk execution.

### 5.6 Watchdogs (protective pause, not e-stop)

| Trigger | Action |
|---|---|
| No phone heartbeat for 500 ms | Gateway stops relaying `ctl.hb` |
| No `ctl.hb` at the sim for 500 ms | Sim performs a controlled stop: `PAUSED`, `fault: hb_lost` |
| `face_present=false` for 1000 ms while `EXECUTING` | Gateway sends `ctl.pause` |

A protective pause can be resumed: after a long blink on "Resume?", or cancelled. These thresholds are design parameters chosen between the interface report (300 ms and 500 ms) and the streaming report (600 ms and 1.5 s). Neither report sourced its numbers. Tune them on the phone in week 2.

### 5.7 Confirmation flow

1. **Trigger.** A brow raise or speech moves the state to `LISTENING` or `PLANNING`. The phone shows a spinner within 100 ms. Brow raise immediately shows the last prefetched suggestion set; otherwise it shows the deterministic menu (the top 3 feasible skills by `wilson_lo`, plus "Call caregiver" and "Lights"). If the user speaks, Nemotron proposes.
2. **Proposal.** 1-3 cards. Each card shows:
   - the canonical label;
   - the success rate ("19/20 in tests");
   - the estimated time;
   - the risk badge;
   - the pre-check status.

   The target object and destination are highlighted in the 3D view.
3. **Navigation.** A head turn moves focus. An option is armed 300 ms after it appears and once focus has rested on it for 300 ms.
4. **Confirm.** A long blink fills a ring around the focused card and fires when the eyes reopen. The gateway runs `validate_confirm`.
5. **Countdown, 1.0 s.** A spoken and on-screen "Putting the bowl on the plate". Stop or jaw cancels it.
6. **Executing.** Phone: big STOP hint, a progress bar, the step count. Presenter: live 3D view.
7. **Post-check.** Then done, failed or uncertain, as in 5.4. In per-step plan mode, the next step's card is pre-focused, and the user confirms again.

---

## 6. Gesture spec

All gesture logic is time-based (ms), never frame-based, and must work at 15 fps. Defaults come from the interface report and are **starting points to calibrate [unverified]**.

### 6.1 Gestures

| Gesture | Signal (MediaPipe blendshape, by name) | On / off threshold | Timing | Active when |
|---|---|---|---|---|
| **STOP** | `jawOpen` | 0.45 / 0.25 | Held 300 ms or more. While `LISTENING` with speech detected, 600 ms | Always. Moving: stop. Idle: cancel or back |
| STOP (voice) | TF.js speech-commands keyword "stop", on device | Probability 0.75 or more | Model window | Always. Not the Web Speech API, which is not on-device |
| STOP (touch) | Full-width button | | Tap | Always (caregiver) |
| **CONFIRM** | min(`eyeBlinkLeft`, `eyeBlinkRight`) | 0.55 / 0.35 | Closed 700 ms or more (calibrated range 600-1200 ms). Open gaps under 80 ms are ignored. A ring fills; fires on reopen | Only when armed, \|yaw\| < 20°, and no jaw activity during the hold |
| Double blink | Same signal | Same | Two closures of 100-500 ms each, gap 100-500 ms, then 500 ms quiet | Optional, for low-risk navigation only (e.g. "back"). **Never starts motion** |
| **MENU** | max(`browInnerUp`, mean(`browOuterUpLeft`, `browOuterUpRight`)) | 0.50 / 0.30 | Held 400 ms or more, then a 1 s lockout | `IDLE` only. Suppressed while speaking and when \|yaw\| > 20° |
| **NEXT / PREV** | Head yaw from the transformation matrix, relative to calibrated neutral | On: max(12°, 35% of the user's comfortable maximum). Off: half of that | Held 250 ms or more; the head must come back inside "off" before the next step; no wrap-around | `AWAITING_CONFIRM` (and menus) |

**Why the brief's double-blink confirm is changed.**
- People blink 12-20 times a minute, each blink lasting 150-400 ms ([Nystrom et al.](https://link.springer.com/article/10.3758/s13428-023-02333-9)).
- If blinks are treated as random, two natural blinks fall within 600 ms of each other about 1.4-3.6 times a minute. That is far too often for a signal that starts motion.
- Published blink systems use long blinks instead: 800 ms in [Blink-To-Live](https://www.nature.com/articles/s41598-023-34310-9), and a long/short split in [BlinkLink](https://www.cs.bu.edu/fac/betke/papers/GraumanBetke-UAIS-2003.pdf).
- Google's [Gameface](https://github.com/google/project-gameface) and Android Camera Switches offer no blink gesture at all.

### 6.2 Implementation notes

- **Model setup.** `@mediapipe/tasks-vision@1.0.1` (pinned and vendored), `runningMode: "VIDEO"`, `numFaces: 1`, `outputFaceBlendshapes: true`, `outputFacialTransformationMatrixes: true`, GPU delegate with CPU fallback. Model file: `face_landmarker.task` float16 v1 (3,758,596 bytes per the interface report).
- **Head yaw.** The matrix is a column-major 4x4: `yaw = atan2(-R[2][0], hypot(R[0][0], R[1][0]))`. The sign is **[unverified]**; calibration records which sign "turn left" gives. Do not mirror the frame before inference, because mirroring swaps the face's Left and Right blendshapes ([issue #6368](https://github.com/google-ai-edge/mediapipe/issues/6368)). Mirror only the preview.
- **Smoothing.** A One-Euro filter (or EMA with α about 0.5) on yaw. No smoothing on blink: use minimum-duration debouncing instead.
- **Ignore unstable readings.** Ignore blink and brow while |yaw| > 20°; readings are unstable during head rotation ([issue #5329](https://github.com/google-ai-edge/mediapipe/issues/5329)). Do not use `mouthPucker`, `cheekPuff`, `jawForward` or `noseSneer`.
- **Lockout.** After any recognised non-stop gesture, block other non-stop gestures for 500 ms. STOP is never blocked.
- **Yawns.** A yawn (jaw open plus eyes closed) yields STOP, which is the safe failure.
- **Face missing.** No face for 500 ms means the phone shows "I can't see you"; while executing, the gateway pauses after 1000 ms (5.6).
- **Frame rate.** If fps drops below 12 for 2 s, show a warning and lengthen the hold windows by 1 frame.

### 6.3 Calibration (about 2 minutes; stored on the phone; profile id sent in `hello`)

1. **Setup check.** Phone at eye level, 35-60 cm away. Proceed only if fps is 15 or more and face presence is 0.9 or more.
2. **Neutral, 10 s.** Record each blendshape's baseline mean and spread, and zero the head pose.
3. **Natural behaviour, 60 s.** The user watches a clip and chats. Record:
   - natural blink durations, and set the long blink to the 99th-percentile natural closure + 250 ms, clamped to 600-1200 ms;
   - natural brow and jaw peaks, and set each "on" threshold above the 99th percentile of its natural peak.
4. **Repetitions.** 5 long blinks, 5 brow raises, 3 jaw holds, and 3 comfortable head turns to each side. Set on = baseline + 50% of the gap to the median peak, and off = baseline + 30%.
5. **Separation check.** If a gesture's median peak is less than 0.10 above the natural 99th percentile, disable that gesture and offer an alternative set. Examples: confirm by holding a head turn for 1.5 s, or a dwell-and-voice "yes". This matters in bulbar-onset ALS.
6. **Validation game, 2 minutes.** Pass if 90% or more of intended gestures register, with zero false confirms and at most one false menu.
7. **Each session.** Re-zero for 5 s. The baseline drifts only while idle (time constant about 60 s). If misses rise, offer longer windows, since fatigue raises misses.

### 6.4 Accessibility notes for the pitch

- Head-turn-to-move plus confirm-to-select is standard two-switch step scanning, as in [Android Switch Access](https://support.google.com/accessibility/android/answer/6122836?hl=en).
- The defaults resemble Gameface (`hold_trigger_ms: 500`) and EyebrowClicker (brow held 500 ms, 89% success).
- Voice is optional and never the only way in. An unadapted recogniser reached 33-43% WER on ALS speech in [Project Euphonia](https://www.isca-archive.org/interspeech_2019/shor19_interspeech.pdf).

---

## 7. Nemotron usage

### 7.1 Model routing

| Path | Model (Token Factory id) | Base URL | Settings | Budget |
|---|---|---|---|---|
| Fast: utterance to 1-3 options; background prefetch after each state change | `nvidia/Nemotron-3_5-Lightning` (eu-north1; $0.06 / $0.24 per 1M tokens in/out) | `https://api.tokenfactory.nebius.com/v1/` **[unverified that the global URL serves it]** | `enable_thinking: false`, temperature 0.2, max_tokens 400, `response_format` json_schema | Timeout 2.5 s, then the deterministic menu |
| Plan: multi-step composition ("set up a snack"), recovery after failure, explaining a rejection | `nvidia/nemotron-3-super-120b-a12b` (us-central1; $0.30 / $0.90) | `https://api.tokenfactory.us-central1.nebius.com/v1/` (Nebius Nemotron page and cookbook) | `enable_thinking: true, low_effort: true`, temperature 0.6, max_tokens 1500 | Timeout 8 s, with a spoken "Thinking..." |
| Not used | `nvidia/Nemotron-3-Ultra-550b-a55b`, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | | Nano is the fallback if Lightning fails the smoke test | |

- **Always set `enable_thinking`.** Nebius's own test found Lightning with thinking left on "spent 506 of 512 output tokens on reasoning and returned truncated JSON" (nebius-ops report).
- **Cost.** Cost is negligible: about 3K tokens in and 300 out per call, roughly $0.0003 on Lightning and $0.0011 on Super. The $25 promo credits cover it.
- **Structured output is not confirmed for Nemotron.** The Token Factory JSON docs say to look for the "JSON mode" tag and test each model; Nemotron is not mentioned on that page ([docs](https://docs.tokenfactory.nebius.com/ai-models-inference/json.md), checked 2026-10-05). The Day-1 smoke test settles which mechanism works: `json_schema`, tools, or `json_object` plus Pydantic.

### 7.2 Request shape

```python
resp = await client.chat.completions.create(
    model=cfg.fast_model,
    messages=[{"role": "system", "content": SYSTEM_FAST},
              {"role": "user", "content": json.dumps(context)}],
    response_format={"type": "json_schema", "json_schema": build_schema(feasible_ids, plan_ids)},
    temperature=0.2, max_tokens=400, timeout=2.5,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
out = PlannerOutput.model_validate_json(resp.choices[0].message.content)   # R1
```

### 7.3 Prompt outline (`planner/prompts/fast.md`)

1. **Role.** "You plan for Nod, an assistive robot arm used by a person who cannot use their hands. You never move the arm. You pick up to 3 options from the FEASIBLE list. The person chooses with small facial gestures, so fewer, clearer options are better."
2. **Hard rules:**
   - Use only ids from the enum.
   - Never invent skills or objects.
   - Prefer higher `wilson_lo`.
   - Never offer `risk: high` skills unless the person explicitly asked for them.
   - For anything the arm cannot do (drink, food to mouth, things on the floor, people), offer `call_caregiver` and say why in `why`.
   - If the request is ambiguous, return `intent: clarify` with up to 3 options covering the interpretations.
   - `why` is at most 80 characters, plain words, addressed to the person.
3. **Context JSON:**
   - `profile` (preferred name, caregiver name, `caregiver_enabled`);
   - `time`;
   - `world` (true atoms only, in plain words);
   - `feasible[]`: id, canonical label, `success.alone`, `success.in_chain`, `est_s`, risk;
   - `infeasible[]`: id with the failing rule, so the model can explain;
   - `history` (last 5 runs with outcomes);
   - `input`: transcript, or `"MENU_OPENED_NO_SPEECH"`.
4. **Few-shot examples, 3 short ones.** "I'm hungry" gives a snack plan plus a single step plus caregiver. "Can you get my phone, it fell" gives caregiver plus an explanation. "Make it brighter" gives `lights_on`.

**Recovery prompt** (`plan.md`, Super): same context plus `failure: {skill_id, steps, cosmos_reason, oracle}`. It returns retry, an alternative skill reaching the same goal, or the caregiver.

### 7.4 Output schema (used as `response_format.json_schema`, or as the `parameters` of a single tool `propose_options` with `tool_choice: "required"`)

```json
{
  "name": "nod_plan",
  "strict": true,
  "schema": {
    "type": "object", "additionalProperties": false,
    "required": ["intent", "clarify_question", "options"],
    "properties": {
      "intent": {"type": "string", "enum": ["manipulate", "home", "caregiver", "clarify", "none"]},
      "clarify_question": {"type": ["string", "null"], "maxLength": 60},
      "options": {
        "type": "array", "minItems": 0, "maxItems": 3,
        "items": {
          "type": "object", "additionalProperties": false,
          "required": ["kind", "skill_id", "plan", "home_action", "why"],
          "properties": {
            "kind": {"type": "string", "enum": ["skill", "plan", "home"]},
            "skill_id": {"type": ["string", "null"], "enum": ["<FEASIBLE_IDS...>", null]},
            "plan": {"type": ["array", "null"], "minItems": 2, "maxItems": 4,
                     "items": {"type": "string", "enum": ["<POLICY_AND_CUSTOM_IDS...>"]}},
            "home_action": {"type": ["string", "null"], "enum": ["lights_on", "lights_off", "call_caregiver", null]},
            "why": {"type": "string", "maxLength": 80}
          }
        }
      }
    }
  }
}
```

The enum is rebuilt on every request from `feasible_skills()`. Constrained decoding (where it works) then makes infeasible skills impossible to name. The validator still re-checks everything, R1 to R14.

### 7.5 Failure handling

| Failure | Response |
|---|---|
| HTTP 429 or timeout | Deterministic menu; the HUD shows "Planner offline". Retry with backoff in the background |
| Schema failure | One retry using `json_object` with the schema in the prompt; then the deterministic menu |
| Every planner output | Recorded to `runs/<id>/planner_trace.json`, which also feeds `FakePlanner` |

### 7.6 Other NVIDIA models in Nod

| Model | Role | Hosting |
|---|---|---|
| GR00T N1.7 (`nvidia/GR00T-N1.7-LIBERO/libero_goal`, plus our fine-tune) | Actor | Dev VM |
| Cosmos-Reason2-8B | Scene checks before and after each skill | vLLM on the dev VM. Token Factory no longer hosts Cosmos: `nvidia/Cosmos3-Super-Reasoner` left serverless on Aug 31 2026 ([notice](https://docs.tokenfactory.nebius.com/august-2026-deprecation-notice)) |
| Cosmos3-Nano (16B, OpenMDW-1.1) | Measured alternative | A Nebius Serverless **Endpoint** from the `endpoint-cosmos3-reasoner` template, validated by Nebius on `gpu-rtx6000-a` in uk-south2 (serverless-ai-cookbook). If it beats Reason2-8B on agreement with the oracle in week 2, switch to it; it also earns the Serverless Endpoints bonus |
| Parakeet TDT 0.6B v3 (CC-BY-4.0) | ASR | `onnx-asr` int8 on CPU (652 MB encoder) |
| Cosmos-Reason2-2B | GR00T N1.7's vision-language backbone | Inside GR00T. A pitch line: "Cosmos reasons in the policy and checks its work" |

---

## 8. Day-1 GPU spike runbook (prove the GR00T path by Sat Oct 10)

Owner: the sim lead. Expected wall time: about 4-6 hours, of which about 1-2 GPU-hours are compute. Run on Tue Oct 6 or Wed Oct 7. If your Nebius console shows a maintenance notice for **Oct 7 06:00 UTC**, stop and start the VM before then and do not leave runs going across it. The nebius-ops report could not find the notice publicly; it may be per-tenant **[unverified]**.

### 8.0 Before you start (today, on a laptop)

1. **Hugging Face.** Request access to `nvidia/Cosmos-Reason2-2B` and `nvidia/Cosmos-Reason2-8B` on their model pages. Create a read token. Without 2B access no N1.7 checkpoint loads.
2. **Nebius.**
   - Sign up and add a card. The AI Cloud free trial is suspended, and the $25 Devpost and Builders grants are Token Factory credits only (nebius-ops report).
   - Find your **uk-south2** project id.
   - In Console, go to Administration > Limits > Quotas and confirm RTX PRO 6000 quota of 1 or more.
3. **Token Factory.** Create an API key at https://tokenfactory.nebius.com/project/api-keys and redeem the promo credits.

### 8.1 Create the VM (laptop)

The flags follow Nebius's compute quickstart as summarised by the nebius-ops report. If one is rejected, check `nebius compute instance create --help`.

```bash
curl -sSL https://artifacts.nebius.cloud/cli/install.sh | bash && exec -l $SHELL
nebius version
nebius profile create            # endpoint api.nebius.cloud; federation auth.nebius.com; parent-id = uk-south2 project id
export PROJECT_ID=<uk-south2-project-id>
nebius compute platform list --parent-id $PROJECT_ID | grep -A3 rtx6000    # expect gpu-rtx6000-a / 1gpu-24vcpu-218gb

export SUBNET_ID=$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')
cat > ci.yaml <<EOF
users:
  - name: nod
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys: ["$(cat ~/.ssh/id_ed25519.pub)"]
EOF
export VM_ID=$(nebius compute instance create --name nod-dev \
  --resources-platform gpu-rtx6000-a --resources-preset 1gpu-24vcpu-218gb \
  --boot-disk-managed-disk-name nod-dev-boot --boot-disk-managed-disk-type network_ssd \
  --boot-disk-managed-disk-size-gibibytes 300 --boot-disk-managed-disk-block-size-bytes 4096 \
  --boot-disk-managed-disk-source-image-family-image-family ubuntu24.04-cuda13.0 \
  --boot-disk-attach-mode READ_WRITE --cloud-init-user-data "$(cat ci.yaml)" \
  --network-interfaces "[{\"name\":\"eth0\",\"subnet_id\":\"$SUBNET_ID\",\"ip_address\":{},\"public_ip_address\":{}}]" \
  --format jsonpath='{.metadata.id}')
export VM_IP=$(nebius compute instance get --id $VM_ID --format json | jq -r '.status.network_interfaces[0].public_ip_address.address|split("/")[0]')
echo $VM_IP
ssh nod@$VM_IP
```

Use on-demand pricing for the spike. The default security group allows all ingress (nebius-ops report), so lock it down to ports 22 and 443 before running any service (`infra/nebius/security_group.sh`).

### 8.2 Install on the VM

```bash
# --- on the VM ---
nvidia-smi                                   # expect RTX PRO 6000 Blackwell, 96 GB
sudo apt-get update && sudo apt-get install -y git git-lfs ffmpeg libegl1-mesa-dev libglu1-mesa tmux jq
ffmpeg -version | head -1                    # Ubuntu 24.04 ships FFmpeg 6.x; torchcodec 0.8 needs 4-7
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T ~/Isaac-GR00T
cd ~/Isaac-GR00T && git log -1 --format='%h %ci'      # record the commit (design was checked at 51d4c89)
uv sync --python 3.12
export HF_TOKEN=hf_xxx                                 # also add to ~/.bashrc
uv run python -c "import gr00t; print('gr00t ok')"
uv run python -c "from huggingface_hub import hf_hub_download as d; print(d('nvidia/Cosmos-Reason2-2B','config.json'))"   # must NOT raise GatedRepoError
bash gr00t/eval/sim/LIBERO/setup_libero.sh             # builds the py3.12 LIBERO venv, pins mujoco==3.3.1; ends with "Env OK"
uv run hf download nvidia/GR00T-N1.7-LIBERO --include "libero_goal/config.json" "libero_goal/embodiment_id.json" \
  "libero_goal/model-*.safetensors" "libero_goal/model.safetensors.index.json" \
  "libero_goal/processor_config.json" "libero_goal/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO
du -sh checkpoints/GR00T-N1.7-LIBERO/libero_goal       # about 7 GB
mkdir -p ~/nod_spike
```

If `uv run` fails later with a Triton `ptx_get_version` error on this CUDA 13 host image, run `uv run bash scripts/patch_triton_cuda13.sh` (Isaac-GR00T README).

### 8.3 Gate G0: reproduce published single-skill success (official harness)

```bash
cd ~/Isaac-GR00T
tmux new -d -s policy "uv run python gr00t/eval/run_gr00t_server.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_goal --embodiment-tag LIBERO_PANDA \
  --use-sim-policy-wrapper --host 127.0.0.1 --port 5555 2>&1 | tee ~/nod_spike/server.log"
until gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python -c "import sys; from gr00t.policy.server_client import PolicyClient as P; sys.exit(0 if P(host='127.0.0.1', port=5555, timeout_ms=2000).ping() else 1)"; do
  sleep 10; done; echo "policy server up"         # first start also downloads Cosmos-Reason2-2B (several GB)

for T in put_the_wine_bottle_on_the_rack open_the_middle_drawer_of_the_cabinet put_the_cream_cheese_in_the_bowl; do
  gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 20 --n-envs 5 --n-action-steps 8 --max-episode-steps 720 --seed 0 \
    --policy-client-host 127.0.0.1 --policy-client-port 5555 \
    --env-name libero_sim/$T --video-dir ~/nod_spike/g0_videos/$T 2>&1 | tee ~/nod_spike/g0_$T.log | grep -i "success rate"
done
```

`--max-episode-steps 720` is the harness default **[checked]**. Episodes end early on success.

### 8.4 Gate G1: chain skills in one env with no reset

Save this file as `~/nod_spike/chain_test.py`.

```python
#!/usr/bin/env python
"""Nod Day-1 spike: chain LIBERO-Goal skills in ONE env with no reset (return-home between skills).
Run from ~/Isaac-GR00T with the LIBERO venv python. The policy server must be running on 127.0.0.1:5555."""
import argparse, json, os, time
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.utils import get_libero_path
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
from gr00t.policy.server_client import PolicyClient

SUITE = benchmark.get_benchmark_dict()["libero_goal"]()
TASKS = {SUITE.get_task(i).name: SUITE.get_task(i) for i in range(SUITE.get_num_tasks())}
def bddl(t): return os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
GOALS = {n: robosuite_parse_problem(bddl(t))["goal_state"] for n, t in TASKS.items()}
CHAINS = {
    "A": ["put_the_wine_bottle_on_the_rack", "open_the_middle_drawer_of_the_cabinet",
          "put_the_cream_cheese_in_the_bowl", "push_the_plate_to_the_front_of_the_stove"],
    "B": ["put_the_bowl_on_the_plate", "put_the_cream_cheese_in_the_bowl"],
}
KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LANG = "annotation.human.action.task_description"
FRAMES, STEP_MS = [], []

def act(xyz=(0.0, 0.0, 0.0), grip=1.0):
    a = {f"action.{k}": np.zeros(1, dtype=np.float32) for k in KEYS}
    for k, v in zip("xyz", xyz):
        a[f"action.{k}"][0] = v
    a["action.gripper"][0] = grip          # policy units: 1.0 = open (LiberoEnv rescales, binarizes, inverts)
    return a

def step(env, a):
    t0 = time.perf_counter()
    obs, _, _, _, info = env.step(a)
    STEP_MS.append((time.perf_counter() - t0) * 1000)
    if FRAMES is not None:
        FRAMES.append(obs["video.image"])
    return obs, info

def batch(obs):
    b = {}
    for k, v in obs.items():
        if k.startswith("video."):
            b[k] = np.ascontiguousarray(v, dtype=np.uint8)[None, None]
        elif k.startswith("state."):
            b[k] = np.asarray(v, dtype=np.float32).reshape(1, 1, -1)
        else:
            b[k] = (v,)
    return b

def eef(obs):
    return np.array([obs["state.x"][0], obs["state.y"][0], obs["state.z"][0]], dtype=np.float64)

def set_skill(env, name, obs):
    env._task_description = TASKS[name].language
    env._env.env.parsed_problem["goal_state"] = GOALS[name]
    obs[LANG] = TASKS[name].language

def run_skill(env, client, name, obs, budget, n_steps, lat):
    set_skill(env, name, obs)
    if env._env.check_success():
        return "already_true", 0, obs
    steps = 0
    while steps < budget:
        t0 = time.perf_counter()
        chunk, _ = client.get_action(batch(obs))
        lat.append((time.perf_counter() - t0) * 1000)
        for t in range(n_steps):
            a = {k: np.asarray(v[0, t], dtype=np.float32).reshape(-1) for k, v in chunk.items()}
            obs, info = step(env, a)
            steps += 1
            if info["success"]:
                return "success", steps, obs
            if steps >= budget:
                break
    return "timeout", steps, obs

def return_home(env, obs, home, max_steps=80):
    """Scripted (not GR00T): open gripper, lift to home height, move to home xyz. Rotation untouched."""
    for _ in range(5):
        obs, _ = step(env, act(grip=1.0))
    for phase in ("lift", "move"):
        for _ in range(max_steps):
            p = eef(obs)
            tgt = np.array([p[0], p[1], max(home[2], p[2])]) if phase == "lift" else home
            d = tgt - p
            if np.linalg.norm(d) < 0.01:
                break
            obs, _ = step(env, act(np.clip(d / 0.05, -1.0, 1.0), grip=1.0))
    return obs

def save_video(path, frames):
    import cv2
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for f in frames:
        vw.write(np.ascontiguousarray(f[..., ::-1]))
    vw.release()

def main():
    global FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="A", help="A, B, or single:<task_name>")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--n-action-steps", type=int, default=8)
    ap.add_argument("--no-home", action="store_true")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--video-dir", default="")
    ap.add_argument("--out", default=os.path.expanduser("~/nod_spike/chain_results.jsonl"))
    args = ap.parse_args()
    chain = [args.chain.split(":", 1)[1]] if args.chain.startswith("single:") else CHAINS[args.chain]
    first = TASKS[chain[0]]
    init_states = torch.load(os.path.join(get_libero_path("init_states"), first.problem_folder,
                                          first.init_states_file), weights_only=False)
    env = LiberoEnv(task_bddl_file=bddl(first), task_description=first.language)
    client = PolicyClient(host="127.0.0.1", port=args.port, timeout_ms=5000)
    assert client.ping(), "policy server not reachable on 127.0.0.1:%d" % args.port
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)
    lat, rows = [], []
    for trial in range(args.trials):
        FRAMES = [] if args.video_dir else None
        env._env.reset()
        obs = env._process_observation(env._env.set_init_state(init_states[trial % len(init_states)]))
        for _ in range(10):                                   # settle, as LeRobot's LIBERO wrapper does
            obs, _ = step(env, act(grip=1.0))
        home, res = eef(obs), []
        for i, name in enumerate(chain):
            if i > 0 and not args.no_home:
                obs = return_home(env, obs, home)
            t0 = time.time()
            status, steps, obs = run_skill(env, client, name, obs, args.budget, args.n_action_steps, lat)
            res.append({"skill": name, "status": status, "steps": steps, "wall_s": round(time.time() - t0, 2)})
            if status != "success":
                break
        ok = len(res) == len(chain) and all(r["status"] == "success" for r in res)
        row = {"trial": trial, "chain": args.chain, "home": not args.no_home, "results": res, "chain_ok": ok}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.video_dir and FRAMES:
            save_video(os.path.join(args.video_dir, f"{args.chain.replace(':', '_')}_{trial:02d}.mp4"), FRAMES)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("\n=== SUMMARY chain=%s trials=%d home=%s ===" % (args.chain, len(rows), not args.no_home))
    for i, name in enumerate(chain):
        att = sum(1 for r in rows if len(r["results"]) > i)
        okc = sum(1 for r in rows if len(r["results"]) > i and r["results"][i]["status"] == "success")
        stp = [r["results"][i]["steps"] for r in rows if len(r["results"]) > i and r["results"][i]["status"] == "success"]
        print(f"step {i+1} {name}: {okc}/{att} = {okc/max(att,1):.0%}  median steps {np.median(stp) if stp else float('nan'):.0f}")
    full = sum(r["chain_ok"] for r in rows)
    print(f"full chain: {full}/{len(rows)} = {full/max(len(rows),1):.0%}")
    l, s = np.array(lat), np.array(STEP_MS)
    print(f"policy call ms: p50 {np.percentile(l,50):.1f} p95 {np.percentile(l,95):.1f} max {l.max():.1f} (n={len(l)})")
    print(f"env.step ms:    p50 {np.percentile(s,50):.1f} p95 {np.percentile(s,95):.1f} (n={len(s)})")

if __name__ == "__main__":
    main()
```

Run it:

```bash
cd ~/Isaac-GR00T
PY=gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python
$PY ~/nod_spike/chain_test.py --chain single:put_the_wine_bottle_on_the_rack --trials 5     # smoke test of the harness
$PY ~/nod_spike/chain_test.py --chain A --trials 20 --video-dir ~/nod_spike/chainA 2>&1 | tee ~/nod_spike/g1_A.log
$PY ~/nod_spike/chain_test.py --chain A --trials 10 --no-home 2>&1 | tee ~/nod_spike/g1_A_nohome.log   # is return-home needed?
$PY ~/nod_spike/chain_test.py --chain B --trials 20 --video-dir ~/nod_spike/chainB 2>&1 | tee ~/nod_spike/g1_B.log
for T in put_the_bowl_on_the_plate put_the_cream_cheese_in_the_bowl push_the_plate_to_the_front_of_the_stove; do
  $PY ~/nod_spike/chain_test.py --chain single:$T --trials 20 2>&1 | tail -6; done                    # alone, same harness
```

If the harness itself breaks, the likely causes, in order:
1. The `torch.load` call on the init states.
2. Action dict shapes (each value must be a 1-D float32 array).
3. LIBERO's first-run config prompt, if `setup_libero.sh` was skipped.

Fix those before blaming the policy.

### 8.5 Gate G2: timing

Read these from the G1 summaries: `policy call ms` p95 and `env.step ms` p95. Also run `nvidia-smi --query-gpu=memory.used --format=csv` while the server is running.

### 8.6 Gate G3: Nemotron structured output (laptop or VM)

```bash
cat > ~/nod_spike/nemotron_smoke.py <<'EOF'
import os, time, json
from openai import OpenAI
KEY = os.environ["NEBIUS_API_KEY"]
URLS = {"global": "https://api.tokenfactory.nebius.com/v1/", "us": "https://api.tokenfactory.us-central1.nebius.com/v1/"}
SK = ["bowl_on_plate", "cheese_in_bowl", "bottle_on_rack", "open_middle_drawer", "plate_to_stove_front"]
schema = {"type": "object", "additionalProperties": False, "required": ["intent", "options"],
  "properties": {"intent": {"type": "string", "enum": ["manipulate", "home", "caregiver", "clarify", "none"]},
    "options": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "object", "additionalProperties": False,
      "required": ["kind", "skill_id", "home_action", "why"],
      "properties": {"kind": {"type": "string", "enum": ["skill", "home"]},
        "skill_id": {"type": ["string", "null"], "enum": SK + [None]},
        "home_action": {"type": ["string", "null"], "enum": ["lights_on", "call_caregiver", None]},
        "why": {"type": "string", "maxLength": 80}}}}}}
off = {"chat_template_kwargs": {"enable_thinking": False}}
SYS = "Pick up to 3 options for an assistive robot arm user. Use only the given enums. Reply as JSON."
PROMPTS = ["I'm a bit hungry", "I dropped my phone", "it's too dark in here", "tidy up the bottle", "open the drawer please"]
for name, url in URLS.items():
    c = OpenAI(base_url=url, api_key=KEY)
    try:
        ids = [m.id for m in c.models.list().data if "emotron" in m.id.lower()]
    except Exception as e:
        ids = repr(e)
    print(name, "models:", ids)
for M, url in [("nvidia/Nemotron-3_5-Lightning", URLS["global"]), ("nvidia/nemotron-3-super-120b-a12b", URLS["us"])]:
    c = OpenAI(base_url=url, api_key=KEY)
    for mode in ["json_schema", "tools"]:
        ok, lat = 0, []
        for p in PROMPTS * 2:
            t = time.time()
            try:
                kw = dict(model=M, messages=[{"role": "system", "content": SYS}, {"role": "user", "content": p}],
                          max_tokens=400, temperature=0.2, extra_body=off, timeout=15)
                if mode == "json_schema":
                    r = c.chat.completions.create(response_format={"type": "json_schema",
                        "json_schema": {"name": "opts", "schema": schema, "strict": True}}, **kw)
                    out = json.loads(r.choices[0].message.content)
                else:
                    r = c.chat.completions.create(tools=[{"type": "function", "function": {"name": "propose_options",
                        "parameters": schema}}], tool_choice="required", **kw)
                    out = json.loads(r.choices[0].message.tool_calls[0].function.arguments)
                assert all((o["skill_id"] in SK + [None]) for o in out["options"])
                ok += 1
            except Exception as e:
                print("  fail", M, mode, p, repr(e)[:160])
            lat.append(time.time() - t)
        lat.sort()
        print(f"{M} {mode}: valid {ok}/10, p50 {lat[4]:.2f}s, p95 {lat[9]:.2f}s")
EOF
uv run --no-project --with openai python ~/nod_spike/nemotron_smoke.py | tee ~/nod_spike/g3.log
```

### 8.7 Record and shut down

- Fill in `docs/runbooks/day1-results.md` with:
  - the commit hashes;
  - the G0 lines;
  - both G1 summaries;
  - G2 numbers;
  - the G3 table;
  - the GPU memory reading;
  - links to 3 sample chain videos.
- Copy `~/nod_spike` to the team bucket or a laptop: `scp -r nod@$VM_IP:~/nod_spike .`.
- Stop the VM: `nebius compute instance stop --id $VM_ID`. Disk is still billed.

### 8.8 Pass/fail criteria and decisions

| Gate | Pass | Partial | Fail |
|---|---|---|---|
| **G0** single skills, official harness | Each of 3 skills ≥ 17/20 | 14-16/20 on one skill: drop that skill | Any skill < 14/20 after one day of debugging, or a GatedRepoError that persists to Oct 10 |
| **G1** chain A, 20 trials, return-home | Every step ≥ 85% (conditional on reaching it) **and** full chain ≥ 14/20 | Full chain 10-13/20 | Full chain < 10/20 |
| G1 chain B | Measured only. If ≥ 60%, the "snack" plan becomes the video's centrepiece; otherwise the video uses chain A plus the custom skill from a fresh layout | | |
| **G2** timing | Policy p95 ≤ 150 ms and env.step p95 ≤ 30 ms | Policy p95 ≤ 300 ms: use TensorRT (`--trt-engine-path`) or `torch.compile` | Not a kill criterion |
| **G3** Nemotron | ≥ 9/10 valid, fast-path p95 ≤ 2.5 s | Use the other model or mode | Neither works: `json_object` plus Pydantic plus the deterministic menu (not a kill) |

**Decision on Oct 10:**

| Outcome | Tier |
|---|---|
| G0 and G1 pass | T1 |
| G0 passes, G1 partial | T2 |
| G0 passes, G1 fails | T3 |
| G0 fails | T4: scripted fallback; GR00T becomes a stretch goal (section 1.6) |

---

## 9. Plan, video and costs

### 9.1 Week by week

Workstreams:
- **S**: sim and policy.
- **G**: gateway, safety and LLM.
- **P**: PWA and gestures.
- **O**: ops, eval and video.

With fewer than four people, combine S+O and G+P.

**Week 1, Mon Oct 5 to Sun Oct 11: prove it**

| Days | Work |
|---|---|
| Mon-Tue | All: HF gating requests, Nebius card and quota, Token Factory keys. **O:** VM up. **S:** runbook 8.2-8.3 |
| Wed-Thu | **S:** G1 and G2, then start `nod_sim.runner` (interruptible loop, goal swap, return-home, oracle). **G:** `nod_protocol`, gateway skeleton, `SafetyChannel`, validator and its tests, G3. **P:** face tracker plus `gestures.js`, with recorded traces from 3 people on the laptop webcam and one Android phone |
| Fri | **O:** build `docker/policy` and `docker/sim` images; push to the Nebius registry; smoke-run them as a Serverless Job (the Nebius GR00T serverless smoke test was SMOKE_FAILED, so we own the image) |
| **Sat Oct 10** | **Kill gate.** Pick a tier; write `day1-results.md` |
| Sun | Buffer |

**Week 2, Oct 12-18: vertical slice, live and mock**

| Workstream | Work |
|---|---|
| **S** | `SimRunner` serving ZMQ; e-stop and controlled stop; pose publisher; capture; recorder; viewer with mjviser on the LIBERO scene (MJB export plus `model_patch`) |
| **G** | Session state machine; Parakeet ASR; NemotronPlanner and FakePlanner; deterministic menu; Cosmos client; proposal and confirm flow; presenter trace |
| **P** | Calibration flow; UI cards; `/ws/safety` with latch-and-resend; KWS "stop"; TTS |
| **O** | Caddy plus sslip.io TLS; compose live profile; nightly job running on GitHub cron (single skills plus chain A); mock library v1 from the first recordings |
| **S** (parallel) | `training/gen_demos.py` for `bowl_to_user`: BDDL with `main_table_handover_region`, scripted pick-place, 80 attempts, keep successes |

Milestone Sun Oct 18: end-to-end live and in mock. Brow, speak, Nemotron, card, long blink, GR00T run, Cosmos check, done. Plus e-stop with measured latency.

**Week 3, Oct 19-25: differentiate and harden**

| Workstream | Work |
|---|---|
| **S** | Fine-tune `bowl_to_user` (old plus new data, 5-10k steps, starting from the `libero_goal` checkpoint; fallback to N1.7-3B with the 20k recipe); eval all 11 skills for regression; register with stats. Stretch: `close_middle_drawer` |
| **G** | Recovery flow (Super); plan mode with symbolic check; Home Assistant (lights, caregiver notify); R6 badges from live stats |
| **P** | Fatigue handling; alternative gesture sets; accessibility review (contrast, text size, TTS on every state change). Informal tests with 3-5 people outside the team; record honest notes |
| **O** | Cosmos3-Nano versus Reason2-8B agreement test; offline 1080p re-render pipeline; README; mock `make fetch-mock` release assets |

**Feature freeze Sun Oct 25.**

**Week 4, Oct 26-29: ship**

| Days | Work |
|---|---|
| Mon-Tue | Record the video (on-demand VM; 2 sessions); edit |
| Wed | Devpost write-up, architecture image, honest-scope section, Nebius/NVIDIA feedback form (a required submission item), public repo check (license, README, mock quickstart tested on a clean machine) |
| **Thu Oct 29** | **Submit.** Fri Oct 30 before 10:00 PT is buffer only |

### 9.2 Demo video script (under 3:00; at least 1:00 continuous of modules in action)

The Devpost rules allow software-only Physical AI entries: "at least a 1-minute clip showing the physical hardware/robot actually operating, or, if your Project has no physical hardware component, the key application modules in action" ([rules](https://nebiusglobalaihackathon.devpost.com/rules), checked 2026-10-05). The same rules suggest "Use Nebius Serverless Jobs to... evaluate robot policies", which is exactly our nightly job.

| Time | Shot | Content |
|---|---|---|
| 0:00-0:15 | Persona (a team member acting, labelled as such) at a table, phone on a mount | "Of 85 people with ALS eligible for a robot arm, 32 got one, after about 197 days ([Spittel 2024](https://link.springer.com/article/10.1186/s42466-024-00342-3)). Then they spend 17% of their time switching joystick modes." |
| 0:15-0:25 | Title | "Nod: control a robot arm with your face and voice. GR00T acts, Nemotron plans, Cosmos checks, you decide." |
| 0:25-1:45 | **One continuous take**: split screen with phone (face plus cards) and presenter (3D view plus HUD) | Calibrated profile loads. Brow raise opens the menu. "I'm a bit hungry." Nemotron offers 3 cards with success rates. Head turn, long-blink ring, countdown; GR00T puts the bowl on the plate; Cosmos marks it done. Next step confirmed: cream cheese into the bowl. Mid-motion, **jaw-open STOP**: red latch, "stopped in 112 ms". Clear the stop; the run is aborted, not resumed. "Bring it to me" runs the custom skill to the handover zone. "Turn on the stove" is shown but locked: "Needs caregiver". "It's dark", and Home Assistant lights come on |
| 1:45-2:05 | Failure clip (real, from the nightly library) | A skill misses; Cosmos says "the cheese is beside the bowl"; Nemotron offers Retry; it succeeds |
| 2:05-2:35 | Architecture plus dashboards | Diagram; nightly Serverless Job table (alone versus in-chain success, Cosmos agreement); fine-tune run on Nebius; "safety never touches the LLM" |
| 2:35-2:55 | Honest scope and next steps | Sim only; LIBERO objects stand in for daily tasks; next is a real arm (e.g. SO-101 or a Kinova), taught near-face moves, and user studies with OT input |
| 2:55-3:00 | Close | Repo URL, "runs on your laptop in mock mode" |

The footage uses 1080p offline re-renders of the same logged runs, labelled "re-rendered from logged sim state", plus live screen capture for the UI.

### 9.3 Costs (uk-south2, Oct 5-29; prices from the nebius-ops report)

| Item | Assumption | On-demand | Cheaper option |
|---|---|---|---|
| Dev VM, RTX PRO 6000 at $1.80/h | About 60 GPU-hours (spike 6, development 40, recording 14) | $108 | $57 at $0.95/h preemptible for non-demo work only |
| Boot disk, 300 GiB network SSD at $0.071/GiB-month | About 0.85 month, billed while stopped | $18 | $18 |
| Nightly Serverless Job | 18 nights × about 1 h (including about 10 min provisioning) | $32 | $17 preemptible |
| Custom-skill fine-tune plus eval | 10 GPU-hours or less **[unverified estimate]** | $18 | $10 |
| Cosmos | Co-hosted on the dev VM | $0 | $0 |
| Cosmos3-Nano endpoint test (optional) | About 6 h including about 15 min cold starts | $11 | |
| Token Factory | About 5K calls | $2-8, covered by $25-50 of promo credit | |
| Object Storage | A few GB | under $2 **[unverified]** | |
| **Total on the card** | | **about $190** (budget cap $250) | **about $120** |

Keep all live demos and recordings on-demand: a preempted VM gets 60 s notice and stops.

---

## 10. Open questions and unverified claims

### 10.1 Conflicts between the research reports, and how we resolved them

| Topic | Conflict | Decision and reason |
|---|---|---|
| Confirm gesture | Brief and streaming report: double blink. Interface report: long blink | **Long blink.** The interface report cites blink statistics, published blink systems, and Google tools that leave blink out; the streaming report gives no source |
| Watchdog thresholds | Interface: heartbeat 300 ms / face 500 ms. Streaming: 600 ms / 1.5 s. Nebius-ops: about 300 ms | **Heartbeat every 100 ms, pause at 500 ms; face lost at 1000 ms.** None of the three is sourced; all are configurable and tuned in week 2 |
| Robot view transport | groot-sim: NVENC WebRTC. Interface: MuJoCo WASM. Streaming: mjviser/viser | **viser.** It is the only option measured end to end (5.19 MB scene once, about 1.9 KB per frame, 2.4 ms server time). WASM is a stretch for a static judge demo |
| H100 NVENC | groot-sim: "from memory" | Streaming cites NVIDIA's matrix: **H100 has no NVENC**. Moot, since we stream poses |
| GPU region | Streaming diagram lists us-central1 for RTX PRO 6000 | Nebius-ops cites quota docs: **0 default quota in us-central1, and no Serverless AI in eu-south1. Use uk-south2** |
| Core benchmark | Interface: build on RoboCasa tasks. groot-sim: LIBERO-Goal | **LIBERO-Goal.** It is the only checked single-scene suite with a public N1.7 checkpoint. The interface report's RoboCasa N1.7 numbers come from a non-public checkpoint, and it marked RoboCasa chaining as unverified itself |
| Inference timing | Streaming: async chunk refill. Official eval: synchronous | **Synchronous but interruptible**, to keep the published 97.5% protocol. Async only if the view stutters |
| Workspace clipping | Streaming: robosuite already clips to `position_limits` | **Not by default.** `osc_pose.json` has `position_limits: null` **[checked]**, so we clip in our loop |
| Policy server exposure | groot-sim: `--host 0.0.0.0` | `run_gr00t_server.py` has no `api_token` option **[checked]**. Bind to 127.0.0.1, or keep the port unpublished inside compose |
| Max episode steps | groot-sim: 300 | The harness default is 720 **[checked]**. Use 720 to reproduce G0; set the live budget from measured step counts |
| Cosmos hosting | Brief: Cosmos-Reason2-8B. Nebius-ops: Cosmos3-Nano has a validated Nebius template | **Reason2-8B co-hosted by default** (lower latency, no extra VM). Cosmos3-Nano tested in week 2 as an Endpoint; switch if it agrees with the oracle more often |
| Token Factory URL | Global versus regional | Use the regional URL for Super (Nebius cookbook). G3 tests both |

### 10.2 Unverified claims (with how and when we verify them)

| Claim | Verify by |
|---|---|
| GR00T success holds when chained in LIBERO-Goal | G1, Oct 6-10 |
| `torch.load(..., weights_only=False)` is needed for LIBERO init states under torch 2.9 | G1 harness smoke test |
| Return-home with rotation left uncorrected is enough | G1 home versus no-home; add orientation correction if needed |
| Fine-tuning can start from the `libero_goal` checkpoint rather than N1.7-3B | Week 3 dry run (10 steps) |
| Compute for the custom skill is 10 GPU-hours or less | Week 3 |
| Handover region at x about +0.10 to +0.16 is reachable | `gen_demos.py` success rate |
| Workspace box values in 3.5.4 | From demo trajectory extremes, Day 1 |
| Nemotron `json_schema` and tool calling work on Token Factory; the global URL serves Lightning | G3 |
| Cosmos-Reason2-8B judges LIBERO renders well, with latency around 1 s; vLLM guided JSON works with the reasoning parser | Week 2 agreement test on 100 labelled frames |
| vLLM and GR00T co-host on one 96 GB GPU alongside EGL rendering | Week 2 compose bring-up |
| mjviser renders the LIBERO scene correctly; MJB size; viser works in phone browsers | Week 2 |
| MediaPipe web reaches 15-30 fps on the target phone; the yaw sign | Week 1 traces |
| TF.js speech-commands (0.5.4, from 2023) still runs on current Chrome | Week 2 |
| Parakeet int8 CPU latency of 100-300 ms for a 3 s command | Week 2 |
| One-way phone-to-uk-south2 latency fits the e-stop budget | Day 1 (`ping`, then the HUD) |
| sslip.io plus Let's Encrypt works with Caddy for our hostnames; Nebius Tunnel latency | Week 2 |
| An Oct 7 06:00 UTC maintenance applies to our tenant | Console banner, `nebius compute maintenance list-active` |
| Serverless Job billing during provisioning | First nightly invoice line |
| Mock assets fit in about 30-60 MB | Week 2 |
| A fallback exists if Cosmos-Reason2-2B gating is slow (whether N1.6 checkpoints have an ungated backbone) | Check on Oct 5-6, while waiting for access |

### 10.3 Open questions for the team

1. **Location.** Where will the demo be recorded? That sets the region choice (RTT) and whether a US region needs a quota request.
2. **User input.** Can we get 30 minutes with an occupational therapist or a person with SCI or ALS for feedback on gestures and wording? Without it, the video must not imply user validation.
3. **Hardware.** Which phone is the reference device (Android model)? iOS Safari support is untested for MediaPipe GPU and TF.js KWS.
4. **Primary demo chain.** Chain B ("snack") or chain A plus the custom skill? Decide after G1.
5. **Undo skills.** Should the stretch undo be learned (fine-tuned) or scripted and labelled? Scripted is faster and honest.
6. **Brief correction.** Kinova Link 6 is an industrial cobot, not an assistive product ([The Robot Report](https://www.therobotreport.com/kinova-robotics-launches-link-6-the-first-canadian-industrial-collaborative-robot/)). Use JACO, iARM, Obi and Stretch as prior art in the pitch.

---

## Sources

**Code read for this document**

| Repo | Commit | Files |
|---|---|---|
| [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | `51d4c89`, 2026-08-20 | `README.md`, `examples/LIBERO/README.md`, `gr00t/eval/sim/LIBERO/{libero_env.py, setup_libero.sh}`, `gr00t/eval/run_gr00t_server.py`, `gr00t/eval/rollout_policy.py`, `gr00t/policy/{server_client.py, gr00t_policy.py}`, `gr00t/configs/data/embodiment_configs.py`, `scripts/deployment/README.md`, `getting_started/hardware_recommendation.md` |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `8f1084e` (the Isaac-GR00T submodule pin) | `bddl_files/libero_goal/*.bddl`, `benchmark/__init__.py`, `envs/env_wrapper.py`, `envs/bddl_base_domain.py`, `envs/bddl_utils.py`, `envs/problems/libero_tabletop_manipulation.py` |
| robosuite 1.4.0 | | `controllers/config/osc_pose.json` |
| [cosmos-reason2](https://github.com/nvidia-cosmos/cosmos-reason2) | `a3b4a1d` | `README.md` |

**Pages checked on 2026-10-05**
- https://nebiusglobalaihackathon.devpost.com/ and https://nebiusglobalaihackathon.devpost.com/rules
- https://tokenfactory.nebius.com/model-catalog.md
- https://docs.tokenfactory.nebius.com/ai-models-inference/json.md
- https://huggingface.co/nvidia/GR00T-N1.7-LIBERO

**From the four research reports.** The full source lists are in those reports. Key sources:
- Nebius docs (quotas, regions, pricing, Serverless, Tunnels); the nebius-physical-ai and serverless-ai-cookbook repos
- MediaPipe model cards and issues; Gameface
- Spittel 2024; Hutmacher 2025; Herlant 2016; Styler 2024; Nystrom et al.; Blink-To-Live; BlinkLink; Project Euphonia
- ISO 13850 summary; VoicePilot
- LiLo-VLA; LIBERO-PRO; LIBERO-Plus
- mjviser / viser; MuJoCo WASM; aiortc; NVIDIA NVENC matrix