# Nod: sim video streaming, safety layer and mock mode (research findings, Oct 5 2026)

## Bottom line

- **Don't stream video of the arm.** Stream body poses from the cloud sim and draw the scene in the browser at the screen's own resolution. That deals with critique (4), the lag, and critique (5), the low-res renders.
- **Fastest way to build it in 3 weeks: `mjviser`/`viser`, run as a separate "viewer process".** I tested it on a robosuite scene. Each browser downloads the scene once (5.19 MB), then each frame update is about 1.9 KB. That is about 37 KB/s at 20 Hz. The server spends about 2.4 ms per frame. No custom JavaScript is needed for the 3D view.
- **MuJoCo WASM in the browser, replaying qpos, also works.** I ran it in Node and the replayed body positions matched Python exactly (error 0). But the model file is 20–27 MB compressed and you have to write your own three.js renderer. Keep it as a stretch goal: a fully static judge demo.
- **Safety:** Nemotron proposes, a pure-function validator approves, the user confirms with a double blink. The e-stop path never touches the LLM. It sets a flag that the sim loop checks before every env step, and the pacing sleep wakes up on it.
- **Mock mode:** record states (not actions) from real GR00T runs. A mock runner replays them through the same interface as the real sim, so it needs no GPU.

---

## 1. Ways to show the cloud sim on the phone and laptop

| Option | Extra delay beyond one-way network time | Bandwidth at 20 fps | Effort | Main risks |
|---|---|---|---|---|
| MJPEG over HTTP (`multipart/x-mixed-replace`) | ~20–50 ms render/encode/decode (estimate). Grows without limit on weak links unless the server drops frames | ~1–3 MB/s at 720p (estimate) | ~0.5 day | TCP "can cause significant latency with poor network connections" because older frames must be sent before newer ones ([ArPiRobot](https://github.com/ArPiRobot/ArPiRobot-CameraStreaming/blob/main/README.md)). Safari, Chrome, Edge and Firefox display it natively ([Wikipedia](https://en.wikipedia.org/wiki/Motion_JPEG)). |
| WebSocket JPEG frames, newest frame wins | Same, but bounded because the server drops stale frames | Same as MJPEG | ~1 day | Bandwidth is still heavy on mobile data. |
| WebRTC via aiortc | Transitive Robotics measured 170 ms remote end to end, of which ~100 ms was camera/USB. Encode ~10 ms, decode ~10 ms, jitter buffer ~10 ms ([Transitive](https://transitiverobotics.com/blog/webrtc-latency-breakdown/), [ROS Discourse](https://discourse.openrobotics.org/t/where-does-latency-in-webrtc-video-streaming-come-from-an-analysis/54566)). | ~1–3 Mbps | 3–5 days including TURN | aiortc 1.15.0 encodes in software (libx264/libvpx through PyAV; checked in its source). Queue build-up reached 4–30 s in [aiortc#1192](https://github.com/aiortc/aiortc/issues/1192). H100 has 0 hardware video encoders (NVENC); L40S has 3 and RTX PRO 6000 Server has 4 ([NVIDIA matrix](https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new)). Phones on mobile data often need a TURN relay. |
| **Pose stream + browser rendering with mjviser/viser (recommended)** | Up to 16.7 ms (viser sends messages in 1/60 s windows) + ~2.4 ms server update (measured) + one browser frame | ~37 KB/s measured | 1–2 days | viser replaces older pose updates with newer ones when the link is slow, so frames don't queue up (from its source). Not tested on phone browsers. |
| Pose stream + MuJoCo WASM + your own three.js renderer | One browser frame only | ~3–7 KB/s | 4–7 days | Model file is large (see section 2). Official bindings are "still a WIP" ([README](https://github.com/google-deepmind/mujoco/blob/main/wasm/README.md)). |

**Recommendation**
- Use **mjviser/viser** for the 3D view on the laptop/TV.
- On the phone, use the 2D choice screen only. Add a small 256×256 "what the robot sees" inset sent as WebSocket JPEG at about 5 fps. Rendering 3D on the phone at the same time as MediaPipe has not been tested.
- The honest comparison for critique (4): well-built WebRTC is not much slower. What pose streaming really gets you is native resolution, ~40× less bandwidth, no TURN or codec setup, free multi-viewpoint, and mock mode reusing the same pipeline.
- Put a lag readout on the HUD: client time minus the server timestamp on each frame, with the clock offset estimated by ping.

## 2. Can GR00T run in the cloud while the browser only gets joint states (qpos)?

**Yes. I checked it end to end.** Scripts are in `/tmp/claude-0/-home-claude/c423faef-2384-54f9-89bc-a43666959303/scratchpad/`: `export_test.py`, `compile314.py`, `patch_test.py`, `slim.py`, `viser_test.py`, `viser_timing.py`, and `wasmtest/test.mjs`.

**What I ran**
- robosuite 1.4.0 PickPlace with a Panda, on MuJoCo 3.3.1 (the same version GR00T's LIBERO setup pins).
- Model: nq 37, 32 bodies, 72 meshes, 10 textures.
- Exported `env.model.get_xml()`, compiled it in MuJoCo 3.14.0, and saved it as a binary model file (MJB).
- Loaded that in `@mujoco/mujoco@3.14.0` under Node, wrote qpos, called `mj_kinematics`.

**Results**
- Body positions matched Python exactly (max error 0).
- Joint names, qpos addresses and body names were identical between 3.3.1 and 3.14.
- WASM start-up 273 ms; loading the MJB took 1.06 s; `mj_kinematics` about 11 µs per call.

**Pitfalls**

1. **MuJoCo version gap.**
   - GR00T's `setup_libero.sh` pins `mujoco==3.3.1` because robosuite 1.4.0's `mj_fullM` call broke in 3.10 ([Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)). The RoboCasa GR1 setup pins 3.2.6.
   - The earliest `@mujoco/mujoco` on npm is 3.6.0 (2026-03-13); the latest is 3.14.0 (2026-09-22) (npm registry).
   - MJB files are "version-specific" ([MuJoCo docs](https://github.com/google-deepmind/mujoco/blob/main/doc/programming/modeledit.rst)), so you can't ship the sim's own MJB.
   - Fix: recompile the robosuite XML with MuJoCo 3.6 or newer on the server, and check that joint and body names match.
2. **Asset paths.**
   - robosuite rewrites every mesh and texture path to an absolute path on the server (`resolve_asset_dependency`).
   - So compile on the server and ship the MJB, which is "standalone, do not refer to external files".
   - Avoid `mj_saveLastXML`. The docs say it "only saves models that have been loaded with mj_loadXML", and it saves whichever model was loaded last in the process.
3. **Changes made at reset are not in qpos.**
   - robosuite PickPlace moves its visual markers by writing `model.body_pos` at reset. LIBERO does the same for fixtures (`bddl_base_domain.py` around line 776) and flips `site_rgba`.
   - Measured: replaying qpos alone put 4 bodies 0.9 m off. After sending a per-episode patch of `body_pos`/`body_quat`, the error dropped to 0.2 mm (that is just the one-substep lag in the recorded positions), and to 0 in WASM.
   - Fix: add `model_patch` to every episode header, or stream body poses instead of qpos.
4. **File size.**
   - The MJB is 61 MB raw, 26.7 MB gzipped. Most of it is collision data (bounding-volume tree, 24 MB) and raw textures (16 MB).
   - Dropping collision meshes and shrinking textures to 256 px only got it to 44.6 MB raw / 20.1 MB gzipped.
   - viser sends compressed mesh files instead: 5.19 MB.
   - LIBERO kitchen scenes are probably bigger (not measured).
5. **Renderer.**
   - The official WASM demo only draws basic shapes; it has no mesh support.
   - [zalo/mujoco_wasm](https://github.com/zalo/mujoco_wasm) (official bindings 3.12, updated Aug 2026) handles meshes and textures from `mesh_vert` and `tex_data`. The Google AI [Panda demo](https://dev.to/googleai/building-a-gemini-powered-robotics-simulator-in-the-browser-with-mujoco-wasm-hjj) uses the same approach.
   - Objects from the bindings must be freed with `.delete()`.
   - The multi-threaded build needs COOP/COEP headers; the single-threaded build doesn't.
6. **mjviser needs MuJoCo 3.6 or newer** (PyPI 0.0.14; Apache-2.0), so it can't run inside the 3.3.1 sim environment. Run it as a separate process that receives poses from the sim.
   - Use `ViserMujocoScene.update_from_arrays(body_xpos, body_xmat)`.
   - Apply `model_patch` before building the scene, because static bodies are placed once at build time.
   - Switching scenes costs ~1.7 s to build plus the 5 MB transfer. Show it as a "moving to another room" transition (relevant to critique 1).

## 3. Safety layer: Nemotron proposes, a fixed validator approves

**Planner and validator**
- Nemotron runs on Token Factory with `response_format: json_schema`. Token Factory supports `json_schema`/`json_object` on models tagged "JSON mode" ([docs](https://docs.tokenfactory.nebius.com/ai-models-inference/json)). Whether Nemotron models carry that tag is **not verified**. Fallback: validate with pydantic, and on failure show a fixed menu.
- The validator is a pure function with unit tests. It checks:
  - `skill_id` is in the registry;
  - parameters name objects that are actually in the current scene;
  - preconditions hold against the sim's true state;
  - reliability: Wilson lower bound and `n` from the nightly job; low-reliability skills are demoted or labelled;
  - mode: no motion while e-stopped or paused;
  - every motion needs a double-blink confirm tied to `proposal_id`, `option_id` and an expiry time.
- During execution, a per-step limiter clips each action. robosuite's controller already clips the goal to `position_limits` (`control_utils.set_goal_position`).
- Related work to cite:
  - RoboGuard: cut unsafe plan execution "from over 92% to below 3%" ([arXiv 2503.07885](https://arxiv.org/abs/2503.07885)).
  - SayCan: weighs LLM choices by what the robot can actually do. Nod's measured success rates play that role ([arXiv 2204.01691](https://arxiv.org/abs/2204.01691), not re-fetched).

**E-stop**
- **On the phone:** latch a red STOPPED screen locally, then resend `estop` every 100 ms until it is acknowledged.
- **Gateway:** a fast path that never calls the LLM or waits on other work.
- **Sim:** a control thread sets `threading.Event` and increments `epoch`. The loop checks before every `env.step`. Any GR00T action chunk from an older epoch is thrown away.
- **Stop behaviour:** hold for 4 steps with zero position change (robosuite's OSC controller sets goal = current pose + delta). Keep the current gripper command so it never drops what it is holding. Then stop stepping the physics.
- **Standards framing:** ISO 13850 limits e-stops to Stop Category 0 or 1 ([gt-engineering](https://www.gt-engineering.it/en/insights/machinery-safety/stop-functions/)), and "resetting the emergency stop function must not restart the machine on its own" ([machinerysafety101](https://machinerysafety101.com/2026/05/18/iso-13850-emergency-stop-requirements/)).
  - After a reset, Nod goes to IDLE. The interrupted run is aborted, not resumed.
  - In the pitch, say the gesture e-stop is a supplementary stop: a real arm also needs a hardware, safety-rated stop.
- **Dead-man check:** if the phone heartbeat is missing for more than 600 ms, or the face is lost for more than 1.5 s, the sim does a protective pause. Both thresholds are my suggested parameters.
- **Voice "stop" on the device:** TF.js speech-commands includes "stop" in its default vocabulary and runs in the browser ([README](https://raw.githubusercontent.com/tensorflow/tfjs-models/master/speech-commands/README.md)). Not tested on the target phones.
- **Don't use GR00T's `MultiStepWrapper.step`** for live runs. It runs all `n_action_steps` inside one call, so it can't be interrupted mid-chunk (checked in the source). Write your own loop, like this:

```python
def run_skill(self, run):                       # Sim Runner process (LIBERO env, mujoco 3.3.1)
    epoch, step, q = self.epoch, 0, deque()
    fut = self.infer.submit(self._infer, self.obs, run.instruction, epoch)  # 1 worker thread owns PolicyClient (ZMQ REQ is not thread-safe)
    next_t = time.monotonic()
    while step < run.max_steps:
        if self.estop.is_set() or epoch != self.epoch: return self._controlled_stop(run, step)
        if time.monotonic() - self.last_hb > HB_TIMEOUT: return self._protective_pause(run, step)
        if not q:
            chunk = self._await(fut, epoch)          # concurrent.futures.wait(timeout=0.005) loop; None on estop/epoch change
            if chunk is None: continue
            q.extend(split_chunk(chunk)[:run.execution_horizon]); fut = None   # LIBERO eval uses 8
        a = self.validator.clip_action(q.popleft(), self.state())
        self.obs, _, _, _, info = self.env.step(a)   # build hold/zero actions in the SAME dict space (LiberoEnv binarizes+inverts gripper)
        self.pub_pose(step); step += 1
        if fut is None and len(q) <= run.refill_at:  # async refill, as in LeRobot async inference
            fut = self.infer.submit(self._infer, self.obs, run.instruction, epoch)
        if info["success"]: return self._done(run, step, True)
        next_t += 1 / 20                              # LIBERO control_freq=20
        self.estop.wait(timeout=max(0.0, next_t - time.monotonic()))  # pacing sleep that wakes instantly on e-stop
    return self._done(run, step, False)
```

**Supporting numbers**
- `PolicyClient` defaults to a 15 s timeout. Set it to 1–2 s. The client already rebuilds its socket after a timeout.
- GR00T N1.7 speed, in new action chunks per second: 12.8 Hz untuned PyTorch / 35.9 Hz TensorRT on RTX PRO 6000; 7.8 / 26.0 Hz on L40. Per the GR00T docs, a ~10 Hz chunk rate can sustain ~30 FPS execution with chunking plus async inference (`hardware_recommendation.md`).
- An 8-step chunk at 20 Hz lasts 400 ms, which is far more than inference time, so async refill keeps up.
- Background: LeRobot async inference ([docs](https://huggingface.co/docs/lerobot/async)) and real-time chunking ([arXiv 2506.07339](https://arxiv.org/abs/2506.07339)).
- GR00T N1.7's vision-language backbone is Cosmos-Reason2-2B (from the GR00T README). Useful line for the pitch.

## 4. Mock mode (runs without a GPU)

**Record states, not actions.**
- GR00T's `ReplayPolicy` replays dataset *actions* with no feedback. That only matches the original run if the start state and versions are identical. Replaying recorded states is exact.

**Each recorded run contains**
- `manifest.json`
- `poses.npz`:
  - `xpos`/`xquat` as float32 for the moving bodies (13 here);
  - `qpos`, `action`, `chunk_starts`.
  - About 0.5 KB per frame, so about 0.3 MB for a 30 s run.
- `model_patch.json`
- `policy_cam.mp4` (256×256 agent view and wrist view)
- `events.jsonl`
- `planner_trace.json`

**How it runs**
- `MockRunner` has the same ZMQ interface as `SimRunner`. It streams the recorded frames at 20 Hz and freezes on `estop`.
- The planner uses canned responses, or live Nemotron if a key is present.
- Start it with `NOD_MODE=mock` (CPU only: gateway + viewer + PWA).
- Show a "REPLAY" badge whenever mock mode is on.
- The nightly Nebius Serverless Job records every evaluation episode and updates both the registry stats and the mock library. Include real failures that Cosmos catches; that turns critique (5) into a feature.

**Optional static hosting**
- viser can record a scene to a file and play it back in a static page via `?playbackPath=` ([docs](https://viser.studio/main/embedded_visualizations/), described there as "experimental and not yet polished").
- Or use MuJoCo WASM with the MJB from section 2.

## 5. Component diagram

```
PHONE PWA (HTTPS required: getUserMedia is secure-context only, MDN)      LAPTOP/TV browser
 MediaPipe FaceLandmarker -> gesture FSM -> UI FSM (1-3 big cards)        Nod Robot View page
 E-STOP detector (jaw-hold | TF.js KWS "stop") -> local red latch          ├ iframe: viser client (three.js, native res)
 mic PCM16 chunks                                                          ├ policy-cam inset (WS JPEG ~5 fps)
        | wss /ws/phone (JSON + binary audio)                              └ HUD: view lag, e-stop latency, success rates
        v                                                                        ^ wss (viser protocol)
GPU VM (RTX PRO 6000: us-central1/eu-south1/uk-south2; or L40S eu-north1 per Nebius regions page)
 Caddy TLS -> GATEWAY (FastAPI/asyncio)
   session FSM | ESTOP fast path (no LLM) | heartbeat dead-man
   ASR -> Parakeet (same VM) | Planner -> Nemotron (Token Factory) | Validator (pure fn) <- Skill Registry JSON
   Cosmos-Reason2-8B (vLLM) pre/post checks | Home Assistant REST | Recorder/event log
        | ZMQ DEALER ctl (estop on dedicated socket)      ^ status/acks
        v                                                 |
 SIM RUNNER (LIBERO/robosuite, mujoco 3.3.1, EGL)  --ZMQ REQ--> GR00T N1.7 PolicyServer :5555 (GPU)
   interruptible 20 Hz loop, controlled stop, recorder
        | ZMQ PUB poses (msgpack float32)
        v
 VIEWER PROCESS (mujoco 3.14 + mjviser/viser, CPU) --> :8080 wss to browsers
NIGHTLY: Nebius Serverless Job -> eval N eps/skill -> registry stats + recordings (object storage) -> mock library
MOCK: Gateway + MockRunner(poses.npz) + Viewer on any laptop, no GPU
```

## 6. Message schemas

Every WebSocket message uses this envelope: `{"v":1,"type":"...","id":"<ulid>","ts":<ms>,"sess":"s_..","seq":<int>}`

**Phone to gateway**
- `hello {device, app_ver, resume_token}`
- `hb {face_present, track_conf, fps}` every 200 ms
- `ui {event:"focus"|"confirm"|"menu"|"cancel", proposal_id?, option_id?}`
- audio: binary frames `[utt_id u32][pcm16 16 kHz]`, then `utter_end {utt_id}`
- `estop {estop_seq, source:"jaw_hold"|"voice_stop"|"face_lost"|"button", t_detect, held_ms}`
- `estop_reset {estop_seq, confirm:"double_blink"}`

**Gateway to phone**
- `state {mode:"IDLE"|"LISTENING"|"PLANNING"|"AWAITING_CONFIRM"|"EXECUTING"|"PAUSED"|"ESTOPPED", run_id?, scene_id}`
- `proposal {proposal_id, expires_at, transcript, options:[{option_id, skill_id, params, label, success:{k,n,p,wilson_lo}, est_s, checks:[{rule, ok}]}], focus}`
- `run {run_id, phase:"precheck"|"executing"|"postcheck"|"done"|"failed"|"aborted", step, chunk, cosmos:{verdict, reason}?}`
- `estop_ack {estop_seq, stopped_at_step, lat_ms:{phone_gw, gw_sim, sim_stop}}`
- `notice {level, text, speak}`
- binary `thumb`

**Nemotron output (json_schema)**
- `{intent:"manipulate"|"home"|"caregiver"|"unclear", options:[{skill_id:<enum from registry>, params:{object:<enum from scene>}, label:"<=24 chars", why}] (1-3), clarify?}`

**Validator output**
- `{ok, options:[...+checks], rejected:[{skill_id, rule}]}`

**Registry entry**
- `{skill_id, ver, kind:"policy"|"home_assistant", scene_id, policy:{checkpoint, embodiment:"LIBERO_PANDA", instruction, execution_horizon:8, max_steps}, params_schema, pre:[...], post:[...], safety:{confirm:true, position_limits}, stats:{k, n, wilson_lo, median_s, eval_job, at}}`

**Gateway to sim (ZMQ, msgpack)**
- `ctl.load_scene {scene_id, model_hash, seed}`
- `ctl.run_skill {run_id, skill_id, instruction, execution_horizon, max_steps, epoch}`
- `ctl.estop {estop_seq}`
- `ctl.reset {epoch}`
- `ctl.hb {t}`

**Sim to gateway**
- `sim.status {run_id, phase, step, success}`
- `sim.estop_ack {...}`
- `sim.episode_header {scene_id, model_hash, mj_ver, body_names, dyn_body_ids, model_patch:{body_pos, body_quat, site_rgba?}}`

**Sim to viewer (PUB)**
- `pose {epoch, step, t_sim, t_wall, xpos:f32[nb,3], xquat:f32[nb,4], qpos:f32[nq]}`

**Sim to GR00T**
- Unchanged Isaac-GR00T `PolicyClient.get_action(obs)`: ZMQ REQ/REP with msgpack-numpy, optional `api_token`.

## Not verified

- mjviser/viser on phone browsers.
- Nemotron "JSON mode" support on Token Factory.
- MJB sizes for LIBERO kitchen scenes.
- Network round-trip times to Nebius regions: measure them.
- Bandwidth figures marked "estimate" in section 1.
- Nebius GPUs per region: I read a summarized version of the docs page; confirm in the console.

Sources: those linked above, plus [MuJoCo overview: MJB](https://github.com/google-deepmind/mujoco/blob/main/doc/overview.rst), [mjviser](https://github.com/mujocolab/mjviser), [primcast on H100 NVENC](https://primcast.com/h100), [MDN getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia), [Nebius regions](https://docs.nebius.com/overview/regions), and LIBERO and robosuite 1.4.0 sources (read locally).