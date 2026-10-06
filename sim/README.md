# sim/: the LIBERO-Goal session and the interruptible control loop

`nod_sim` runs one unbroken LIBERO-Goal episode (design 1.2, 1.3, 4.2.4) and serves the gateway
over ZMQ (3.3, 3.5.4 to 3.5.6). Everything that touches the simulator lives here:

| Module | What |
|---|---|
| `runner.py` | `SimRunner`: 20 Hz loop, ZMQ sockets, e-stop and controlled stop (5.5), limits (5.3), pause and heartbeat watchdog (5.6), goal swapping, scripted actor (T4), CLI `python -m nod_sim.runner` |
| `scene.py` | `LiberoSession` (the session env), goal table, `verify_single_scene`, torch-free init-state loader |
| `skills.py` | `swap_skill`: new goal and instruction without a reset (works on our session, Isaac-GR00T `LiberoEnv` and a bare `OffScreenRenderEnv`) |
| `policy.py` | `Gr00tPolicy`: Isaac-GR00T policy server client (REQ/REP, msgpack plus msgpack-numpy), no gr00t install needed |
| `scripted.py` | waypoint controllers on privileged poses: `return_home`, `handover_retreat`, `pick_place`, stand-ins for policy skills (tier T4), also used by `training/gen_demos.py` |
| `oracle.py` | `world_state`: registry atoms through LIBERO's own `_eval_predicate` |
| `capture.py` | `ctl.capture` renders and JPEG encoding |
| `recorder.py` | per run: `manifest.json`, `poses.npz`, `model_patch.json`, `events.jsonl` (mock library format) |
| `rerender.py` | replay a recording with `set_state` and render frames or video (for the film) |
| `fakes.py` | `FakeEnv` (kinematic LIBERO-Goal stand-in) and `FakePolicy`, for CPU tests |
| `bddl/nod/bring_the_bowl_to_me.bddl` | the session BDDL: LIBERO-Goal plus the handover region |

## Two virtual environments

Python 3.11, created with uv inside this directory (both are git-ignored).

**`.venv`: fast CPU tests, no simulator.**

```bash
cd sim
uv venv -p 3.11 .venv
uv pip install --python .venv/bin/python -e "../protocol[numpy,fast]" -e ".[test,groot,media]"
.venv/bin/python -m pytest -q          # 57 passed, 1 skipped (the real-LIBERO file)
```

**`.venv-libero`: the real LIBERO-Goal scene on CPU** (what was installed and run here). No torch
and no GPU are needed: our loader reads the init states without torch, and we never import
`libero.libero.benchmark` (it imports torch).

```bash
cd sim
uv venv -p 3.11 .venv-libero
mkdir -p .venv-libero/src
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git .venv-libero/src/LIBERO
git -C .venv-libero/src/LIBERO checkout 8f1084e
uv pip install --python .venv-libero/bin/python "robosuite==1.4.0" "mujoco==3.3.1" "numpy==1.26.4" \
  "bddl==1.0.1" easydict termcolor pyyaml imageio "opencv-python-headless==4.10.0.84" "matplotlib==3.9.4" \
  "gym==0.25.2" cloudpickle future tqdm "PyOpenGL==3.1.10" pyzmq msgpack ormsgpack "pydantic>=2.9" \
  pytest h5py pillow msgpack-numpy imageio-ffmpeg
uv pip install --python .venv-libero/bin/python --no-deps -e .venv-libero/src/LIBERO --config-settings editable_mode=compat
uv pip install --python .venv-libero/bin/python -e ../protocol -e . -e ../eval
# LIBERO asks for its paths interactively on first import unless a config exists:
L=$PWD/.venv-libero/src/LIBERO/libero/libero
mkdir -p .venv-libero/libero_config
printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/../datasets\nassets: %s/assets\n' \
  "$L" "$L" "$L" "$L" "$L" > .venv-libero/libero_config/config.yaml
export LIBERO_CONFIG_PATH=$PWD/.venv-libero/libero_config
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa     # CPU (needs libOSMesa); egl on the GPU VM
.venv-libero/bin/python -m pytest -q tests/test_libero_real.py      # 6 passed
```

`--config-settings editable_mode=compat` matters: LIBERO's package layout is not found by the
default editable finder. A torch install from download.pytorch.org failed behind this
sandbox's proxy, which is what made the torch-free loader necessary. In the GPU images
(`docker/sim.Dockerfile`, `docker/eval.Dockerfile`) the LIBERO venv comes from Isaac-GR00T's
`setup_libero.sh` and does have torch; the code works either way.

## Running the sim

```bash
# live: GR00T policy server on 5555
python -m nod_sim.runner --policy tcp://policy:5555
# no GPU: scripted actor (T4), only stand-ins verified in LIBERO, no heartbeat watchdog
python -m nod_sim.runner --policy none --actor scripted --hb-timeout-ms 0 --lazy-cameras
```

`--lazy-cameras` renders the two policy cameras only when a chunk is requested (identical
images, about 10x faster on CPU). `--record-dir DIR` records every run.

## What was verified on the real scene (CPU, OSMesa, this repo's `.venv-libero`)

- **Single scene:** the ten LIBERO-Goal BDDL files are identical once the language and goal
  blocks are removed, and our session BDDL differs only by the handover region
  (`verify_single_scene`, `test_single_scene_claim`).
- **Goal table** matches the BDDL files; every registry atom is evaluable and false in the
  initial layout; observations match the Isaac-GR00T `libero_sim` keys and shapes.
- **Goal swap with no reset:** `bowl_on_plate`, then `return_home`, then `cheese_in_bowl` in one
  session; the first goal stays true after the swap.
- **E-stop** while carrying the bowl: run aborted, gripper command held closed during the
  controlled stop, bowl not dropped. Measured `t_halt - t_rx` over 6 stops at 20 Hz:
  57 to 100 ms, p50 76 ms (this includes the 4 controlled-stop steps, about 16 ms each).
- **Timing:** `env.step` about 16 ms without cameras; the two 256 px policy renders about 180 ms
  per observation on CPU (OSMesa). GR00T eval on the GPU uses EGL.
- **Scripted stand-ins** (`python -m nod_eval.run --actor scripted --episodes 3 --chains A,B
  --allow-unverified-scripts --lazy-cameras`, init states 0 to 2):

  | Skill | Alone | In chain |
  |---|---|---|
  | bowl_on_plate, cheese_in_bowl, bowl_to_user, bowl_on_stove, bottle_on_rack, bottle_on_cabinet | 3/3 each | cheese_in_bowl 3/3, bowl_to_user 3/3 (chain B) |
  | bowl_on_cabinet | 2/3 | |
  | open_middle_drawer, plate_to_stove_front | 0/3 | open_middle_drawer 0/3 (chain A step 2) |
  | stove_on, bowl_in_top_drawer | no scripted controller | |

  Chain B (bowl_on_plate, cheese_in_bowl, bowl_to_user with return_home between) 3/3; chain A
  0/3 (stops at the drawer). These are waypoint-controller numbers, not GR00T.
- **Known failures:** the drawer stand-ins (handle not reachable from above in LIBERO), the
  plate push (stalls before the plate), and `return_home` / `handover_retreat` from the
  extended handover or cabinet-top poses (the OSC servo drifts instead of retracting). The live
  sim therefore runs with `allow_unverified_scripts=False` (`VERIFIED_IN_LIBERO` in scripted.py).
- **Recording, re-rendering and the scene export** below.

GR00T itself was never run here (no GPU, no checkpoint). The policy client is tested against a
local REP server that speaks the Isaac-GR00T wire format (`tests/test_components.py`).

## Mock library and scene export

All recordings come from ONE session (LIBERO places fixtures randomly at each reset, so one
`model_patch` must be valid for every recording and the exported scene):

```bash
MUJOCO_GL=osmesa .venv-libero/bin/python scripts/record_mock_library.py --out ../recordings --scene-dir /tmp/nod_scene
# the viewer venv (MuJoCo 3.14) compiles the scene the sim dumped (MuJoCo 3.3.1):
../viewer/.venv/bin/python -m nod_viewer.export_scene compile --in-dir /tmp/nod_scene --out ../recordings/scene_libero_goal_v1.mjb
gzip -9 ../recordings/scene_libero_goal_v1.mjb      # the viewer loads .mjb.gz directly
```

`recordings/` now holds seven runs (bowl_on_plate, return_home, cheese_in_bowl, an e-stopped
return_home, return_home, bowl_to_user, a failed handover_retreat), `library.json` and the
scene (`scene_libero_goal_v1.mjb.gz`, 16.5 MB; 17 MB in total). Re-render a run:

```bash
.venv-libero/bin/python -m nod_sim.rerender ../recordings/00_bowl_on_plate_success --out /tmp/frames --w 1920 --h 1080
```

## Tests

| File | Venv | What |
|---|---|---|
| `tests/test_runner_safety.py` | `.venv` | e-stop within one step, controlled stop, epoch discard, eef box clipping and fault after 3 violations, NaN action, budget, policy timeout, heartbeat loss pause |
| `tests/test_runner_protocol.py` | `.venv` | the ZMQ protocol over ipc sockets: acks, e-stop socket, pause/resume, capture, pose stream with episode header, goal swap, scripted chain |
| `tests/test_components.py` | `.venv` | recorder, oracle, scripted controllers, goal swap helper, fingerprints, gripper convention, JPEG capture, GR00T client round trip |
| `tests/test_libero_real.py` | `.venv-libero` | the real-scene checks listed above (skipped without LIBERO) |
