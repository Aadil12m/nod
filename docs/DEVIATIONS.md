# Deviations from docs/design.md

Each area appends its own section. List what differs from the design, and why.

## protocol/ and registry/

All additions below are optional fields with defaults or new enumerated values, so every JSON
example in design section 3.5 still parses unchanged (tested in `protocol/tests/test_messages.py`).

### Message fields added

| Model (`type`) | Addition | Why |
|---|---|---|
| `Option` (inside `proposal`) | `badges: list[...]` with values `less_reliable`, `needs_caregiver`, `unsure_from_camera`, `near_user`, `scripted`, `first_use`; `locked: bool` | R6, R7, R14 and 5.4 describe badges and a "shown but locked" card, but the 3.5.3 JSON has no field to carry them |
| `Success` | `k: int \| None` | Cards show "19/20 in tests" (5.7), which needs k |
| `run` (`RunUpdate`) | `fault` | So the phone can say why a run failed |
| `ctl.run_skill` (`RunSkill`) | `action_scale` (default 1.0) | R14 needs the sim to scale actions by 0.6 for near-user skills |
| `ctl.run_scripted` (`RunScripted`) | `limits` (default box) | The workspace box should bound scripted moves too |
| `cosmos.check` | `proposal_id`, `option_id`; `run_id` nullable | Pre-checks run per displayed option before any run exists (5.4) |
| `sim.frames` `Frame` | optional `step` | So a frame can be matched to a `FrameRef` |

### Vocabularies the design left open

| Field | Values |
|---|---|
| `proposal.source` | `nemotron`, `menu` (deterministic menu, 5.7), `fake` (FakePlanner) |
| `option.precheck.status` | `pending`, `ok`, `unsure`, `block`, `error` (plus optional `reason`) |
| `welcome.lease` | `user`, `caregiver`, `none` |
| `hello.role` | `user`, `caregiver` |
| `sim.status.phase` | `idle`, `executing`, `paused`, `done`, `failed`, `aborted` (scripted runs use the same values) |
| `notice.level` | `info`, `warn`, `error` |

Open-ended strings stay `str`: `ui.menu.source`, `ui.cancel.source`, `ui.confirm.gesture` (6.3 allows
alternative confirm gestures) and `estop.reset.gesture`.

### Envelope and encoding

- ZMQ messages (3.5.4 to 3.5.6) carry `v` and `type` but not `id`, `ts`, `sess`, `seq`. The design
  states the envelope only for WebSocket messages, and ZMQ messages already correlate by `req_id`,
  `run_id` and `epoch`.
- Envelope fields have defaults in Python (`id` = new ULID, `ts` = now, `seq` = 0, `sess` = None), so
  the design's examples parse. Senders must still set `seq` and `sess`. The generated JSON Schema lists
  every field as required (serialization mode), so the PWA's dev-time validation does catch a missing
  envelope field.
- Binary fields use standard base64 in JSON (not pydantic's default URL-safe alphabet), so the browser
  can use `atob`.
- Pose arrays use the `{dtype: "<f4", shape, data}` map described in `protocol/README.md`.
- `PlannerOutput` does not reject an option whose fields do not match its `kind` (for example `kind:
  skill` with `skill_id: null`). Rejecting it would fail R1 for the whole output and fall back to the
  menu; instead the validator drops that option using `PlannerOption.is_consistent()`.

### Dependencies

`nod-protocol` depends on `pydantic>=2.9` (not just `>=2`), `msgpack` and `pyyaml` (the skills.yaml
loader lives with `SkillSpec`). numpy is an optional extra, needed only by the `pose` message, so the
gateway can import the package without numpy. ormsgpack is optional and used automatically.

### Registry (`registry/skills.yaml`)

- Extra `SkillSpec` fields beyond the 4.2.3 example: `scripted: {script, max_steps}` and `home:
  {action}` blocks (the counterpart of `policy:` for the other two kinds), `objects` (names checked by
  R3), `custom` (our trained skill), `enabled` (R2), `offer` (false for internal motions the planner and
  menu never show), `est_s` (time estimate until stats provide `median_s`). `SkillStats` gains
  `placeholder`, copied from the stats file at load.
- `bowl_on_plate` has four more preconditions than the 4.2.3 example (bowl not on the cabinet, not on
  the stove, not at the user, plate not pushed to the stove front). The rule throughout is that a
  learned skill that picks the bowl requires it at its training start; see the header of skills.yaml.
- `bowl_in_top_drawer` has a second effect, `open(wooden_cabinet_1_top_region)`, beyond its BDDL goal.
  The skill leaves the drawer open, and `open_middle_drawer` requires the top drawer closed. The
  oracle success check uses only the goal atom (`SkillSpec.goal_atom`, the first positive effect,
  which is always the BDDL goal). If Day-1 runs show the policy closes
  the drawer again, drop this effect.
- `bowl_to_user` has a delete effect, `not on(akita_black_bowl_1,plate_1)`, because it may start from
  the plate (chain B).
- `handover_retreat` has a scripted entry (it is in the 3.5.4 script list but not in the task list).
- Home entries use their action name as `skill_id` (`lights_on`, `lights_off`, `call_caregiver`).
- `reset_scene` and `return_home` have `offer: false`. Reset is caregiver-only through
  `POST /api/scene/reset` (4.3).
- `return_home` and `handover_retreat` have empty `effects`; the validator must exempt empty-effect
  skills from R5.
- The `bowl_to_user` checkpoint path `nod/GR00T-N1.7-LIBERO-nod/libero_goal_handover` is a name we
  chose; the fine-tune does not exist yet.

### Stats file (`registry/stats/latest.json`)

- Keeps the flat 4.2.5 layout (skill ids at the top level next to `chains`, `job_id`, `git_sha`, `at`)
  and adds `placeholder: bool` and `_note`. The committed file is a labelled placeholder with invented
  numbers so mock mode shows success rates and badges. `bowl_in_top_drawer` (alone) and several
  in-chain values sit in the R6 "less reliable" band on purpose.
- `chains.<name>` is `{skill_ids, k, n, wilson_lo, median_s, steps: [{k, n, wilson_lo}]}`, with
  `steps` conditional on reaching that step (as in the 8.4 summary). The design left this shape open.

## pwa/

### Protocol usage the gateway must match

| Topic | What the phone does | Why |
|---|---|---|
| Resume after a protective pause (5.6) | While PAUSED, the phone shows whatever proposal the gateway sends (the gateway in this repo offers a one-option "Resume?" proposal plus a fixed "Cancel"), and confirms or cancels it like any other proposal. Without such a proposal the paused screen only shows the STOP hint | 3.4 says "confirm resume" but 3.5 has no message for it; a proposal reuses R9. STOP while paused is still an e-stop |
| Duplicate `estop` | Resent every 100 ms until `estop.ack` with the same `estop_seq` | The gateway must ack every copy (it ignores the duplicate for the stop itself), or the phone keeps resending |
| `estop` before `welcome` | Sent with `sess: null` if the stop happens before the session exists | Safety first; the gateway should not drop it |
| Where acks arrive | Gateway messages are accepted from either socket | `estop.ack` on `/ws/safety` is natural, `state` on `/ws/phone` |
| Gateway forgot a stop | If the phone is latched and a `state` arrives with a lower `estop_seq` and a mode other than ESTOPPED, the phone sends its `estop` again | Covers a gateway restart; never resumes anything |
| Unlatch | The phone clears its red latch only after it sent `estop.reset` and a `state` arrives with mode other than ESTOPPED and `estop_seq` at least its own | ISO 13850: only an explicit reset clears the stop |
| "None of these" | `ui.cancel` with `source: "none_of_these"`; listening with no speech for 8 s sends `ui.cancel` with `source: "timeout"` | Open-ended `source` values |
| `ui.focus` | Not sent when focus lands on the fixed "None of these" card | It is not a proposal option |
| REPLAY badge | Read from `GET /healthz` field `mode` (`mock` shows the badge); a `mode` field in `welcome` is also honoured if added later | `Welcome` has no mode field |
| Presenter | Needs `state`, `run`, `proposal`, `estop.ack`, `notice` and `transcript` mirrored on `/ws/presenter`, in addition to `trace`, `metrics`, `cosmos.*`. It treats `state` ESTOPPED with `estop_seq` at least its own as an ack of its own stop. 3D view URL from `?viewer=`, else `/healthz` `viewer_url`, else port 8080 on the same host | `PRESENTER_OUT_TYPES` lists only the presenter-specific types |
| Keyboard mode heartbeat | With no camera (`?nocam=1` or camera failure), `hb.face_present` is true while the page is visible, `track_conf` and `fps` are 0, and `hello.calib.fps` is 0 | The keyboard fallback (4.4) must be able to confirm (R9 checks `face_present`). The gateway can spot keyboard sessions by `track_conf: 0` |
| Gesture telemetry | Keyboard gestures are sent as ordinary `gesture` messages; `scores` keys are `blink`, `brow`, `jaw` | The keyboard runs through the same gesture machine |

### Gesture and UI choices beyond the design text

- A long blink must **start** after the focused option is armed (300 ms after it appears and 300 ms
  after focus landed). The design only says the confirm must arrive after those times. This also
  stops a blink that began before a head turn from confirming the new option.
- Eyes closed longer than `long_blink_max_ms` (4 s) never confirm (dozing). The ring appears only
  after 250 ms of closure, so natural blinks do not flash it.
- Low frame rate lengthens the confirm, menu and turn holds by one frame, but not the jaw-hold STOP:
  a false stop is cheap.
- A face-missing frame cancels a long blink in progress at once; the jaw-hold timer is kept across
  missing frames until `face_lost` (500 ms) resets everything.
- Double blink is detected and sent as telemetry, but using it as "back" is off by default
  (`doubleBlinkBack`), because natural double blinks happen 1.4 to 3.6 times a minute (6.1).
- STOP from any channel is an e-stop when motion is possible: mode COUNTDOWN, EXECUTING, POSTCHECK
  or PAUSED, a run whose last phase is not terminal, or a `ui.confirm` sent less than 3 s ago with
  no answer. Otherwise it cancels (choose, listening, planning) or does nothing (idle).
- "Clear stop?" is offered 1.5 s after the latch and only once the stop is acked; the clearing
  long blink must start after the prompt appeared.
- An `aborted` run shows no result screen; the stopped screen already said so.
- Confirm `double` (R6, R7, R14) is done on the phone: the first long blink asks "once more", and
  only the second sends `ui.confirm`. Focus changes or 8 s reset it.
- Mic: opens when the gateway mode is LISTENING, or right after a brow raise (optimistic). The phone
  finds the start and end of speech with an energy VAD and sends `utter.start` at speech onset
  (with 300 ms of pre-roll), not when the mic opens. Utterances are capped at 10 s.

### Calibration (6.3)

- Implemented: setup check, neutral, natural behaviour, repetitions, separation check, a short
  validation game, per-session 5 s re-zero, and slow baseline drift while idle (bounded to 0.1).
  A "quick" plan (about 1 minute) exists for demos.
- The validation game is five prompts (one per gesture) with rests, not two minutes.
- The "on above the natural 99th percentile" rule is applied to brow and jaw. For blink, amplitude
  does not separate natural from deliberate closures; duration does (the long-blink length rule).
- Turns are disabled if the comfortable turn is less than 3 degrees above the "on" threshold.
- **Not built:** the alternative gesture sets of step 5 (confirm by holding a head turn 1.5 s,
  dwell-and-voice "yes"). A disabled gesture only shows a warning; a disabled jaw stop leaves the
  voice keyword and the STOP button.

### Libraries and files

- `@tensorflow/tfjs` is pinned to 3.21.0 because `@tensorflow-models/speech-commands` 0.5.4
  declares tfjs `^3` as its peer.
- `scripts/vendor.sh` downloads the npm packages from the npm registry tarballs (the same files
  jsdelivr serves; jsdelivr was blocked from the build machine), checks npm's sha512 integrity and
  pinned sha256 hashes for the models, and writes `vendor/manifest.json`. `face.js` and `kws.js`
  read that manifest (a GET, which works offline through the service worker) instead of probing
  files with HEAD. `vendor/` is git-ignored.
- The PWA does not validate messages against JSON Schema in the browser. Instead
  `tests/proto.test.mjs` validates every builder's output against `protocol/schema/*.json` in CI.

### Verified here, and not

- Verified: unit tests (gestures, calibration, UI machine, sockets, e-stop, protocol schema, audio
  helpers); a headless Chromium run against a stub gateway; and the real MediaPipe Face Landmarker
  (vendored, GPU delegate on SwiftShader) on a fake camera showing MediaPipe's test portrait: 52
  blendshapes, yaw and pitch, and yaw changing sign on a mirrored frame.
- Not verified: a real phone, real camera motion (blinks, turns, jaw), frame rates on Android, the
  keyword model on a live microphone, mic capture and resampling on a phone, `speechSynthesis`
  voices, and the real gateway.

## gateway/

### Safety layer (section 5)

- `ctl.estop.estop_seq` is a counter kept by the gateway, not the phone's number. E-stops come
  from several peers (phone safety socket, presenter, dev inject), each with its own `estop_seq`,
  so the sim sees one increasing sequence. The phone's `estop_seq` is still used for de-duplication
  and is echoed in its `estop.ack`.
- Duplicate e-stops are ignored per peer, keyed on (peer, client `estop_seq`), and only while the
  session is latched. After `estop.reset` a reused number from a reconnected phone stops again.
- `estop.ack` is sent when `sim.estop_ack` arrives, so it can carry `gw_sim` and `sim_halt`. If
  the sim does not answer within `estop_ack_wait_ms` (1000 ms) the ack is sent without those two
  fields, so the phone is never left without an answer. The session is latched at once either way.
- While an e-stop is unconfirmed by the sim, the watchdog relays no `ctl.hb`. The sim's own
  heartbeat watchdog then stops it even if the `ctl.estop` message was lost.
- A phone heartbeat with `latched: true` latches the gateway too (the phone may have stopped
  while its safety socket was down), except within 500 ms after a reset, to let old heartbeats
  drain.
- `lat_ms.phone_gw` uses the phone's send time directly. The ping-based clock offset is applied
  only when it shows more than 1000 ms of skew; values outside -50 to 10000 ms are dropped.
- `view_lag_ms` in `metrics` is not measured by the gateway (it needs the viewer) and stays null.

### Validator rules

- R6 does not apply to scripted skills or home actions; they have no policy success rate.
- R6 applies to plans as well: a plan is judged by its measured chain rate if there is one,
  otherwise by its weakest step.
- R9 extras beyond the table: the arm delay and the focus dwell are checked twice, on the
  gateway clock and on the phone's own timestamps (`ts`, `shown_at`, `focused_at`, compared only
  with each other, so no clock offset is needed); `double_blink` is rejected as a confirm gesture;
  and a `long_blink` confirm must be held at least `min_confirm_hold_ms` (600 ms).
- A double confirm (R6, R14 first use) works like this: the first valid confirm answers
  `confirm_result` with outcome `second`, and the proposal is sent again with a fresh nonce; the
  arm delay and the 300 ms focus dwell start again.
- Pre-checks (5.4) run per option after the card is shown. When one finishes, the proposal is sent
  again with the same `proposal_id` and nonce and the updated `precheck`. If a pre-check answers
  "unsure" after a single-confirm card was already confirmed, the run is aborted before any motion
  and the option is offered again with a double confirm.
- `post_decision` fills the cases the 5.4 table leaves open: oracle false with Cosmos unsure gives
  `failed`; without Cosmos the oracle decides alone; without an oracle Cosmos decides alone; with
  neither the outcome is `uncertain`.

### Session and messages

- Resuming from PAUSED goes through a normal confirm: the gateway sends a proposal with one option
  whose `option_id` is `resume` (purpose "resume"). It can only be confirmed while PAUSED. An
  aborted run is never resumed.
- Pauses, resumes and home actions are reported as `run` messages plus `notice`. Home actions
  (lights, caregiver) are `run` updates with `max_steps` 0, since they do not move the robot.
- Plan option labels are built from the registry ("X, then N more steps") rather than taken from
  the planner.
- The deterministic menu (5.7) shows the top 2 feasible, offerable, not high-risk policy skills by
  Wilson lower bound, plus "Call caregiver". R12 and the `Proposal` model cap options at 3, so
  "Lights" is reachable by voice or the planner, not from the menu.

### Endpoints and interfaces

- Added `GET /api/dev/state` (same gating as `/api/dev/inject`) so tests and demos can read the
  session's internals. With dev off, every `/api/dev/*` path answers 404.
- `POST /api/scene/reset` is authorised by an `X-Nod-Token` header equal to the `resume_token` of
  a phone holding the caregiver lease (or by dev mode).
- `Verifier.check` takes an extra `images` argument (the captured frames), so the session owns
  frame capture and the verifier stays a pure client.
- Sim link: replies to `ctl.world` may come on the DEALER socket or on PUB and are matched in
  order; `sim.ack` and `sim.frames` are matched by `req_id`. `IMMEDIATE` is set on the ctl and
  e-stop sockets so commands never queue while the sim is down; `estop()` raises at once if the sim
  is not connected.

### Mock mode

- The MockRunner replays recordings when `recordings/` has any: each run is a directory with
  `manifest.json` (`skill_id`, `success` or `outcome`, optional `steps`) and `poses.npz` (`xpos`,
  `xquat`, `qpos`). Otherwise it runs a scripted timeline that applies the skill's effects at the
  end. Frames are a fixed placeholder JPEG.
- The canned planner replies in `nod_gateway/planner/canned/*.json` are hand-written, not recorded
  from Nemotron; each has `recorded: false` and says so.

### Not verified here

- No GPU and no API keys were available, so the live paths are tested only against fakes and
  mocked HTTP: which structured-output mechanism Token Factory accepts for each Nemotron model, the
  exact onnx-asr call shape for Parakeet, vLLM guided JSON for Cosmos (a 400 falls back to plain
  text parsing), and the Home Assistant entity and notify service names
  (`light.ceiling_lights`, `notify/persistent_notification`, both configurable).

## Repo root, infra/, docker/, docs/runbooks/, .github/

### Layout

- The workflows live in `.github/workflows/` (where GitHub runs them), not `infra/github/workflows/`
  as in the design 4.1 tree.
- Extra files the design does not list: `ruff.toml` (root ruff settings for code without its own
  ruff table; `gateway/` keeps its stricter config), `infra/nebius/lib.sh` (shared helpers, every
  script has `--dry-run`), `infra/scripts/{fetch_mock.sh, mock.sh, smoke_mock.sh}`, `infra/tests/`
  (pytest), `infra/requirements-test.txt`, `docker/fetch_ckpt.sh`, `docker/eval_entrypoint.sh`,
  `recordings/README.md` and `docs/runbooks/day1-results.md` (the template design 8.7 refers to).
- `docs/runbooks/nemotron_smoke.py` extends the design 8.6 script: it tests the full 7.4 schema
  (identical to `nod_protocol.planner_response_format`, which a test checks) instead of the reduced
  one, adds the `json_object` fallback mode of 7.5, validates each reply field by field, and takes
  `--base-url`, `--modes`, `--models`, `--thinking` and `--out`. Its summary lines keep the
  design's form ("valid 9/10, p50 ..., p95 ...").

### compose.yaml (design 4.5)

- Mock mode uses separate services `gateway-mock` and `viewer-mock` instead of giving `gateway`
  and `viewer` both profiles. In mock mode the sim is the gateway's in-process MockRunner, so the
  viewer must subscribe to `gateway-mock:5563` rather than `sim:5563`, and the mock gateway needs
  `NOD_SIM_POSE_PUB=tcp://0.0.0.0:5563` so the viewer container can reach it.
- Added profile `eval` (one nightly-style eval on the VM) and service healthchecks; `policy` waits
  for `cosmos` to be healthy, which enforces "start cosmos before policy".
- `homeassistant` publishes 8123 on 127.0.0.1 only (for onboarding through an SSH tunnel); the
  design publishes nothing. Caddy also keeps 80 open for redirects, though the security group
  allows only 22 and 443 (Caddy then uses the TLS-ALPN challenge on 443).
- The sim service gets `NOD_RECORD_DIR=/data/runs/sim` and `NOD_ACTOR` (`scripted` selects the T4
  fallback actor) from the sim runner's CLI.
- `.env` is optional (`required: false`), so the mock profile runs without one.

### Images (docker/)

- The policy, sim and eval images pin Isaac-GR00T to commit `51d4c89f72fda44cbf77285c6a8114b52676b8a1`,
  the full hash of the `51d4c89` the design was checked against (confirmed with `git ls-remote` on
  2026-10-06, when it was still the repository HEAD). They use CUDA 12.8 devel/runtime on Ubuntu 24.04 like
  Isaac-GR00T's own Dockerfile.
- `setup_libero.sh` ends with a GPU smoke test that cannot run during `docker build`; the sim and
  eval images cut that block off and the runbook (demo-day.md step 2) runs an EGL check on the VM.
- Checkpoints are not baked into images. `docker/fetch_ckpt.sh` (the entrypoint of the policy and
  eval images) downloads them into a volume on first start.
- No image was built here (no daemon, no GPU). Only `docker compose config` was run.

### Not verified here

- Every Nebius CLI call (VM create, security groups, `nebius ai job create`, profile creation from
  a service-account file in CI, the job state names) follows the nebius-ops research report and is
  tested only in dry-run mode and against fake `nebius` and `aws` executables.
- The GitHub workflows were checked with actionlint, not run on GitHub.
- `eval/nod_eval` does not exist yet, so `docker/eval.Dockerfile` cannot build; the infra test
  marks this as an expected failure until it lands.
- The mock library release (`mock-v1`) has not been published, so `make fetch-mock` currently
  reports that the release is missing; mock mode works without it.

## sim/, viewer/, training/, eval/

### Session and scene (design 1.2, 1.3, 4.2.4)

- **Session BDDL.** The sim builds its one episode from `sim/bddl/nod/bring_the_bowl_to_me.bddl`,
  the LIBERO-Goal scene plus one invisible table region `main_table_handover_region`
  (x 0.08 to 0.16, y 0.06 to 0.14 in table coordinates), instead of from
  `put_the_bowl_on_the_plate.bddl`. Without the region the custom skill's goal atom could not be
  evaluated in the live session. `verify_single_scene()` checks that the ten LIBERO-Goal files
  are identical outside their language and goal blocks and that ours differs only by the region
  (BDDL `;` comments are ignored). Init states still come from `put_the_bowl_on_the_plate`.
- **Own observation replica.** `LiberoSession` reproduces Isaac-GR00T `LiberoEnv`'s observation
  and action conversion (51d4c89: keys, 180 degree image rotation, axis-angle state, gripper
  rescale, binarize and invert) instead of importing `gr00t.eval.sim.LIBERO.libero_env`, so the sim
  venv needs neither gr00t nor gymnasium. `skills.swap_skill` still accepts a real `LiberoEnv`.
- **Torch-free init states.** `libero.libero.benchmark` imports torch only to load init states;
  we read the task map module directly and unpickle the `.pruned_init` zip pickles with a
  restricted unpickler (numpy only), falling back to `torch.load` when torch exists.
- **Lazy cameras** (`LiberoSession` default; `--lazy-cameras` on the runner and nod_eval CLIs,
  which default to robosuite's per-step camera observables): the two policy images are rendered
  only when a chunk is requested. The images are the same; only CPU time changes.

### Runner (design 3.3, 3.5.4 to 3.5.6, 5.3, 5.5, 5.6)

- `SimRunner(env, policy, addrs, hz, *, registry, ...)` takes the env and the policy client as
  arguments (the design's signature has no keywords); `addrs=None` runs without sockets (tests,
  nod_eval) and `hz=None` runs unpaced.
- **E-stop timing.** `sim.estop_ack.t_halt` is taken after the 4 controlled-stop steps, so
  `t_halt - t_rx` includes them. Measured on the real scene on CPU at 20 Hz: 57 to 100 ms,
  p50 76 ms over 6 stops. `stopped_at_step` is the sim step at which the stop was received.
- **Controlled stop** holds the last gripper command (a carried object is not dropped; checked on
  the real scene) and sends zero translation and rotation; it runs unpaced.
- **Episode header on the pose socket.** `sim.episode_header` is published on the status PUB
  (as designed) and also on the pose PUB, repeated every 2 s, because the viewer subscribes
  only to the pose socket and must get the fixture placement even when it joins late.
- **Limits.** The eef box check predicts `eef + 0.05 * action_xyz` and zeroes the translation of
  that step; 3 consecutive violations fault `eef_out_of_box`. With the placeholder
  `DEFAULT_EEF_BOX` (z max 1.35) the scripted bottle lift had to be lowered; the box should be
  re-measured against GR00T's own trajectories on the GPU VM before it is tightened.
- **Scripted runs report success** when the skill's registry effects hold afterwards (oracle),
  like policy runs; `return_home` and other motions without effects succeed when they finish.
- `--hb-timeout-ms 0` disables the heartbeat watchdog (local tools, eval); the live default stays
  500 ms.
- **Scripted actor (T4) safety.** The runner refuses scripted stand-ins that were not verified
  in LIBERO unless `allow_unverified_scripts` is set; the CLI defaults to refusing. Verified:
  bowl_on_plate, cheese_in_bowl, bowl_to_user, bowl_on_stove, bowl_on_cabinet (2/3),
  bottle_on_cabinet, bottle_on_rack, return_home. Not working in LIBERO: open/close middle
  drawer, plate_to_stove_front, and return_home or handover_retreat from the handover or
  cabinet-top poses. A joint-space reset, or a handover region closer to the robot, would fix
  the last one (finding for the team).

### Recorder, mock library and scene (design 4.2.4, 4.2.5, 4.4)

- `policy_cam.mp4` is optional (`EpisodeRecorder(video=True)`, needs imageio-ffmpeg) and off by
  default, to keep the mock library small; `rerender.py` makes video from `sim_state` in
  `poses.npz` at any resolution.
- The seven recordings in `recordings/` come from the scripted actor, not GR00T (manifests say
  `"actor": "scripted"`), all from one session so that one `model_patch` fits all of them.
- **Scene file.** `export_scene.py` works in two steps (dump in the MuJoCo 3.3.1 sim venv,
  compile in the MuJoCo 3.14 viewer venv) because an MJB is version specific. The compiled scene
  is viewer-only: collision meshes dropped, textures shrunk to 256 px, untextured meshes over
  3000 faces decimated with fast-simplification. Body and joint names and qpos layout are checked
  against the sim, and forward kinematics on the saved file reproduce the sim's body positions to
  3.3e-5 m. It is still 43 MB raw, so it ships gzipped as `scene_libero_goal_v1.mjb.gz`
  (16.5 MB); the viewer loads `.gz` and falls back to `<name>.gz` when the `.mjb` path in
  compose does not exist. `infra/scripts/fetch_mock.sh` checks for the `.mjb` name only.

### Viewer (design 4.2.5)

- mjviser tracks the first moving body by default (shifts the whole scene); the viewer turns
  that off so overlays stay in world coordinates, and sets +z up and an initial camera.
- No `ZMQ_CONFLATE` (headers share the socket): the SUB keeps a small queue and only the newest
  pose of each wake-up is drawn.

### Eval (design 4.2.5)

- Added `--actor scripted` (T4 numbers, marked in the stats `_note`), `--policy-for
  CHECKPOINT=URL` for the custom checkpoint, and `--frames` for the Cosmos frames. Skills whose
  checkpoint has no server are skipped and chains cut before them; the nightly job therefore
  reports chain B as two steps until a second server serves `bowl_to_user`.
- `in_chain` counts chain steps after the first only (the first starts from a fresh table).
- `median_s` is simulated time (steps / 20), which equals wall time in the live 20 Hz loop.
- `cosmos_agree.py` uses its own stdlib HTTP client with the gateway's prompt suffix and reply
  parser (copied, so the eval image does not need the gateway package).

### Training (design 4.2.5)

- `gen_demos.py` writes LIBERO/robomimic-style HDF5 with both `actions` (robosuite units) and
  `actions_policy` (GR00T units, what the LeRobot data uses: gripper 1 = open), images in raw
  renderer order; `to_lerobot.py` rotates them. Start states are mixed (table, plate, plate plus
  cheese) because the registry says bowl_to_user is trained from both.
- `to_lerobot.py` writes H.264 (the NVIDIA demo data is AV1) and does not write `stats.json`:
  Isaac-GR00T's dataset factory generates it with fingerprints; our cross-check test does the
  same. Mixing with the original LIBERO-Goal data happens in `finetune.sh` through
  launch_finetune's `:`-separated dataset paths.

### docs/runbooks/chain_test.py

- Copied verbatim from design 8.4 (extracted programmatically, compiles), plus one added line,
  `# ruff: noqa: E401, E741`, because `make lint` runs ruff on docs/runbooks and the design's
  code has `import argparse, json, os, time` and a variable named `l`. Its LIBERO imports and
  task fields were checked against LIBERO 8f1084e; no provable bug was found, so nothing was
  changed. It was not run (needs gr00t, torch and a policy server). One edge case remains: the
  summary's `l.max()` raises if no policy call happened (every skill `already_true`).

## Integration (whole repo)

Changes made while wiring the areas together. Bug fixes are listed so other agents know the
behaviour changed.

- **`state.estop_seq` on the ESTOPPED state (gateway bug fix).** `SafetyChannel` latched the
  session before recording the client's `estop_seq`, so the `state` message that announces
  ESTOPPED carried the previous number (0 on the first stop). The phone and the presenter compare
  `state.estop_seq` with their own stop number (pwa/ section above), so the gateway now records
  the number first. Forwarding to the sim still happens before both.
- **Presenter "E-stop (last)" tile.** The gateway sends `estop.ack` only to the peer that sent the
  stop; it is not mirrored to `/ws/presenter`. The presenter page now also fills the tile from
  the `trace` of kind `estop` with `acked: true` (it carries the same `lat_ms` split), instead of
  the gateway mirroring another peer's ack, whose `estop_seq` could collide with the
  presenter's own numbering.
- **MockRunner and the viewer.** The MockRunner now behaves like `nod_sim.runner` towards the
  viewer: it publishes `sim.episode_header` on the status and the pose socket at start, on
  `ctl.load_scene` and every 2 s, filled from the mock library (`model_patch.json`,
  `body_names`, `model_hash` of the recordings; all recordings come from one session). Before,
  the header had an empty patch, so the viewer drew fixtures in their unpatched places. Scripted
  runs (`return_home`) now replay a recording too when the library has one, and between runs the
  last replayed pose is held instead of publishing empty arrays. The viewer ignores a pose with
  zero bodies ("no pose yet") instead of logging a warning 20 times a second.
- **`python -m nod_sim.runner --fake-env`.** Serves `nod_sim.fakes.FakeEnv` with the scripted
  actor and no policy. Used by `gateway/tests/test_real_sim_runner.py`, which runs the gateway's
  real `SimLink` against the real `SimRunner` in its own process (plan step to done with the
  runner's oracle, e-stop with the sim's own `sim.estop_ack`, heartbeat-loss pause). Mock mode
  is unchanged: it still uses the in-process MockRunner.
- **The scene file name.** `infra/scripts/mock.sh` and `fetch_mock.sh` accept
  `recordings/scene_libero_goal_v1.mjb.gz` as well as `.mjb` (the mock library ships the
  gzipped file; the viewer falls back to it by itself). Compose and the viewer image keep the
  `.mjb` path, which the viewer resolves to `.mjb.gz`.
- **Makefile and CI.** `make test` now also runs the stub-gateway browser test, the full mock
  end-to-end (`pwa/tests/e2e_full.mjs`, target `test-e2e`), eval, viewer and training, and every
  `test-*` target creates the venv or `node_modules` it needs on first use. New targets:
  `setup-viewer`, `setup-training`, `test-eval`, `test-viewer`, `test-training`, `test-e2e`,
  `screenshots`. `make lint` covers viewer, eval and training. CI gained an `e2e` job and a
  `viewer` job; the `sim-cpu` job also runs eval and training; the gateway job installs
  `sim/.venv` so the real-sim-runner test runs.
- **Plan steps keep the plan's ids.** Observed, not changed: the `run` messages of plan step 2
  and later carry the original plan proposal's `proposal_id` and `option_id` (not the id of the
  "Step 2 of 3" proposal that was confirmed); `plan_step` tells the steps apart. The full
  end-to-end test relies on `plan_step`.
