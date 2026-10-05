# GR00T + simulation feasibility for Nod (checked 2026-10-05)

## Verdict

**Nod is feasible.** The core should be **GR00T N1.7, LIBERO-Goal checkpoint (`nvidia/GR00T-N1.7-LIBERO`, subfolder `libero_goal`), on a simulated Franka Panda in MuJoCo/robosuite.**

- **Published success:** 195/200, which is 97.5%, on LIBERO-Goal.
- **One shared scene.** LIBERO-Goal is the only LIBERO suite where all 10 tasks use the same scene. I compared the 10 BDDL task files directly. Their regions, fixtures, objects and starting layout are byte-identical. Only the instruction text, the objects of interest and the goal differ.
- **Chaining without reset is possible.** You can build one env and issue several skills in a row by changing the instruction. Resets come from the evaluation harness, not from LIBERO or robosuite (code references below).
- **Language matters here.** Because the layout never changes, the instruction is the only signal that tells the policy which task to do. This is the best answer to "benchmark replay".
- **Not verified yet: success when chained.** After skill A, the arm and objects start skill B outside the training distribution. This is the main risk. It costs about one GPU-hour to measure on Day 1.
- **Every other option is weaker for the core:**
  - RoboCasa: no official N1.7 checkpoint; N1.6 pick-and-place tasks are 19-63%.
  - RoboLab with the DROID checkpoint: 8.6% average zero-shot.
  - SimplerEnv, GR1 tabletop and BEHAVIOR: low success rates or one scene per task.
- **A custom assistive skill is a realistic stretch goal:** scripted demos in the same LIBERO-Goal scene, then continued training from the `libero_goal` checkpoint. Rough cost is 2-4 developer days and 10 GPU-hours or less.

Code I read (shallow clones under `/tmp/claude-0/-home-claude/c423faef-2384-54f9-89bc-a43666959303/scratchpad/repos/`):

| Repo | Commit | Date |
|---|---|---|
| Isaac-GR00T | `51d4c89` | 2026-08-20 |
| LIBERO | `8f1084e` | |
| lerobot | `8c920c4` | 2026-10-03 |
| robocasa (squarefk fork) | `d89d481` | |
| RoboLab | `ad45d4f` | |
| nebius-physical-ai | `bd61459` | 2026-10-05 |
| robosuite 1.4.0 wheel | from PyPI | |

---

## 1. Public GR00T checkpoints

| Repo id | Version / type | Contents and notes |
|---|---|---|
| `nvidia/GR00T-N1.7-3B` | N1.7 base, 3B | About 6 GB download. Usable without fine-tuning only for its built-in embodiments: DROID relative-EEF, XDOF, real G1, R1 Pro. |
| `nvidia/GR00T-N1.7-LIBERO` | N1.7 fine-tuned | Subfolders `libero_10`, `libero_goal`, `libero_object`, `libero_spatial`. Each holds about 6.9 GB of weights (4.99 GB + 1.92 GB safetensors). `libero_goal` is the step-18000 checkpoint. The whole repo is 142.8 GB because it includes optimizer states, so download with `--include`. |
| `nvidia/GR00T-N1.7-DROID` | N1.7 fine-tuned | DROID Franka; used zero-shot in RoboLab. |
| `nvidia/GR00T-N1.7-SimplerEnv-Bridge`, `nvidia/GR00T-N1.7-SimplerEnv-Fractal` | N1.7 fine-tuned | WidowX and Google Robot. |
| `nvidia/gr00t17-lerobot-libero_{spatial,object,goal,10}-640` | N1.7, LeRobot port | 95% / 100% / 98% / 93% over at least 50 episodes. HF API lists them as Apache-2.0. |
| `nvidia/GR00T-N1.6-3B` | N1.6 base | `ROBOCASA_PANDA_OMRON` and `GR1` are built-in embodiments, so it runs RoboCasa and GR1 tabletop without fine-tuning. License is "NVIDIA License", non-commercial (research or evaluation), which is fine for a hackathon. |
| `nvidia/GR00T-N1.6-{bridge,fractal,BEHAVIOR1k,G1-PnPAppleToPlate,DROID}`, `nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task` | N1.6 fine-tuned | All non-commercial. |
| `nvidia/GR00T-N1.5-3B`; community `youliangtan/gr00t-n1.5-libero-{spatial,goal,object,90,long}-posttrain` | N1.5 | 92% / 86% / 92% / 89.3% / 76%. The `n1.5-release` branch README lists these. LeRobot dropped N1.5 (`lerobot==0.5.1` is the last version that supports it). |
| `robocasa/robocasa365_checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000` | N1.5 on RoboCasa365 | Atomic-seen tasks 50.7%, composite-seen 14.8%. |

**Licenses.** The N1.7 weights use the NVIDIA Open Model License (HF card); the code is Apache-2.0. The README also claims "fully commercially licensable under Apache 2.0".

**Gated backbone.** Every N1.7 checkpoint loads the gated `nvidia/Cosmos-Reason2-2B` on first use. Without access, loading fails with `GatedRepoError`. **Request access today.**

**VRAM.**
- Inference: 16 GB or more.
- Fine-tuning: 40 GB or more. The default trains only the projector and the action head, and peaks under about 35 GB per GPU. Turning on `--tune-llm` or `--tune-visual` pushes this to 80 GB or more.

**Latency per action chunk.** Source: `scripts/deployment/README.md`; N1.7, 4 denoising steps, 1 camera.

| GPU | PyTorch eager | `torch.compile` | TensorRT |
|---|---|---|---|
| H100 | 85.8 ms | 48.6 ms | 27.9 ms |
| RTX PRO 6000 | 78.4 ms | 50.7 ms | 27.9 ms |
| L40 | 128.3 ms | 69.0 ms | 38.4 ms |
| AGX Thor (Jetson) | 112.8 ms | | 80.4 ms (12.4 Hz) |
| Orin | 354 ms | | 150.9 ms |

- LIBERO feeds 2 cameras, so expect somewhat higher latency (not measured).
- The LIBERO config predicts 16 actions per call (`gr00t/configs/data/embodiment_configs.py:193-209`); the eval executes 8 of them (`--n-action-steps 8`).
- LIBERO-Goal demos average 121.6 frames (52,042 frames / 428 episodes at 20 fps), so a skill takes about 6 s of sim time and about 15 policy calls. Real-time pacing is easy.

## 2. Simulation benchmarks with ready checkpoints

| Benchmark | Checkpoint | Success | Scene layout / can tasks chain? |
|---|---|---|---|
| **LIBERO-Goal** | N1.7-LIBERO `libero_goal` | 97.5% (no per-task breakdown published) | **One shared scene** (verified). Chainable. |
| LIBERO-Spatial | N1.7 | 97.65% | 10 different layouts; each task has its own starting layout (verified by hashing the init blocks). Not chainable. |
| LIBERO-Object | N1.7 | 98.45% | Different object set per task, and the target always sits at `floor_target_object_region` (verified), so the layout gives the task away. LIBERO-PRO and LIBERO-Plus show policies often ignore the language here. Avoid. |
| LIBERO-10 | N1.7 | 94.35% | 6 scenes. Only `LIVING_ROOM_SCENE2` holds 2 tasks (identical scene, verified). |
| LIBERO-90 | N1.5 community checkpoint only | 89.3% | 20 scenes with 2-7 tasks each, identical starting layout within a scene (verified). Fallback only. |
| RoboCasa (24 atomic tasks) | N1.6-3B base (no fine-tune needed); N1.7 needs your own 60k-step, 8-GPU, batch-512 fine-tune | 66.2% (N1.6); 70.8% (N1.7) | New kitchen layout each reset. The robot base is placed at each task's fixture (`init_robot_base_pos`), so tasks cluster by fixture. |
| RoboCasa GR1 tabletop | N1.6 base; N1.7 needs fine-tune | 47.6% / 44.5% | One scene per task. |
| SimplerEnv Bridge (N1.7) | | Average 62.3%: spoon on towel 78, carrot on plate 58, eggplant in basket 53, stack cube 48, eggplant in sink 2, close drawer 97, open drawer 100 | Fixed scene per task. |
| SimplerEnv Fractal (N1.7) | | Average 72.5%: coke can 100, pick object 94, move near 100, open drawer 65, close drawer 69, place in closed drawer 7 | Fixed scene per task. |
| RoboLab (Isaac Lab), zero-shot | N1.7-DROID | 412/4800 = **8.58%** over 120 tasks | Bright spots: BananaOnPlate 40/40, BananasInBinThreeTotal 38/40, UnstackRubiksCube 38/40, SauceBottlesCrate 33/40. Needs an RT-core GPU. |
| BEHAVIOR-1K | N1.6 | 26.3% average task progress | Not viable. |
| SO-100 / SO-101, DROID | | | Real-robot workflows only; no simulation checkpoints. |

**Exact LIBERO-Goal instruction strings.** Source: `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot` `meta/tasks.jsonl`, 428 episodes. These match the eval strings, which LIBERO derives from the file names (`benchmark/__init__.py:44-53`). Goal predicates are from the BDDL files.

| Instruction | Goal predicate |
|---|---|
| put the bowl on the plate | `On bowl plate` |
| put the wine bottle on the rack | `On bottle wine_rack_1_top_region` |
| open the top drawer and put the bowl inside | `In bowl wooden_cabinet_1_top_region` |
| put the cream cheese in the bowl | `On cream_cheese bowl` |
| put the wine bottle on top of the cabinet | |
| push the plate to the front of the stove | `On plate main_table_stove_front_region` |
| turn on the stove | `Turnon flat_stove_1` |
| put the bowl on the stove | |
| put the bowl on top of the cabinet | |
| open the middle drawer of the cabinet | `Open wooden_cabinet_1_middle_region` |

The policy only reliably understands these exact strings. Paraphrases are out of distribution, so Nemotron must map what the user says to a skill ID in a closed registry.

**Can you run task A, then task B, with no reset?** Yes. The evidence:
- **The GR00T env wrapper keeps stepping after success.** `gr00t/eval/sim/LIBERO/libero_env.py:104-109` builds `OffScreenRenderEnv(..., ignore_done=True)`. robosuite only raises "executing action in terminated episode" when `self.done` is set, and with `ignore_done=True` it never is (`robosuite/environments/base.py:375-376, 429`).
- **LIBERO's own "done" is only a flag.** It is set to `_check_success()` (`bddl_base_domain.py:806-807`) and nothing stops on it.
- **The instruction is fixed when the env is built** (`libero_env.py:110`) and re-inserted on every observation (`:157`). To switch skills, set `env._task_description`, or overwrite the language key in the observation dict you send.
- **The policy is stateless and takes the language on every call** (`gr00t/policy/gr00t_policy.py`, `Gr00tSimPolicyWrapper._get_action`). Changing instructions mid-session is therefore fine.
- **Success for any skill can be checked in the same env.** `_check_success` evaluates `parsed_problem["goal_state"]` (`envs/problems/libero_tabletop_manipulation.py:135-162`). All 10 BDDLs share the same object, fixture and region names. Parse each goal once with `libero.libero.envs.bddl_utils.robosuite_parse_problem(path)["goal_state"]` (`bddl_utils.py:87`), then swap it in or evaluate it with `_eval_predicate`.

**What the eval code does at episode end:**
- **Isaac-GR00T:**
  - `rollout_policy.py:603` forces `terminate_on_success=True`.
  - `MultiStepWrapper.step` sets `done` on success (`multistep_wrapper.py:359-361`) or on reaching max steps (`:321-326`).
  - The env runs inside `_RobustAsyncVectorEnv`, which uses gymnasium 0.29.1 inline autoreset, so finished envs are reset automatically.
  - Episodes are tallied in `_collect_rollout_episodes`, followed by a final `env.reset()`.
  - LIBERO `reset()` draws a fresh layout from the BDDL placement sampler. It does not use the fixed benchmark starting states.
- **LeRobot:**
  - `envs/libero.py:376-377` sets `terminated = done or is_success`.
  - Vector envs use `AutoresetMode.NEXT_STEP` (`:526`).
  - `reset()` restores the fixed benchmark starting states (`:344-346`) and then takes 10 no-op settle steps.
  - LeRobot does not pass `ignore_done`, so it defaults to False with a 1000-step horizon, and stepping past that raises. **For a continuous session, write your own loop with `ignore_done=True`.**

**Evidence that chaining degrades** (from papers; not checked in code):
- LiLo-VLA: pi0.5 scores 83% on the original task order but **0% when the same tasks are reordered**. Causes cited are "global layout and end-effector position change from skill to skill."
- LIBERO-PRO: replacing instructions with nonsense still produced nearly the same trajectories, i.e. memorisation.
- LIBERO-Plus (Table 10, as summarised): OpenVLA-OFT scores 53.2% on Goal-suite language tasks versus 79.5-99% on other suites. This supports "Goal is the suite where language matters".
- None of these papers evaluated GR00T.

## 3. RoboCasa and assistive fit

**The story fits better, the reliability is worse.** Tasks cluster around one fixture because the robot base is placed in front of it.

| Fixture | Task success with N1.6 |
|---|---|
| Sink | PnPCounterToSink 46.0, PnPSinkToCounter 50.0, TurnOnSinkFaucet 89.0, TurnOffSinkFaucet 93.5, TurnSinkSpout 87.0 |
| Coffee machine | CoffeeSetupMug 31.0, CoffeePressButton 98.5, CoffeeServeMug 63.5 |
| Microwave | TurnOn 91.5, TurnOff 96.0, PnPCounterToMicrowave 19.0, PnPMicrowaveToCounter 24.5 |

- A "make coffee" chain comes to about 31% × 98.5% × 63.5% ≈ 19%.
- A fixture-only sink chain (faucet on, then spout, then faucet off) comes to about 72%. That is usable for B-roll only.
- Chaining in RoboCasa also needs the same hacks as LIBERO: override `get_ep_meta()["lang"]` and write success checks per task.
- RoboCasa renders 512×512 cameras and downsizes them to 256 for the policy (`gymnasium_groot.py`), so 512px frames are available for video.

**No assistive checkpoints exist.** I found no GR00T or LeRobot checkpoint or dataset for feeding, drinking or fetching dropped items (search-based, so not proven exhaustive). Assistive Gym (PyBullet) and RCareWorld (Unity) exist but have no VLA checkpoints. LIBERO's assets include mugs (`porcelain_mug`, `red_coffee_mug`, `white_yellow_mug`) but no phone.

## 4. Custom assistive skill: fine-tune on about 50 simulated demos

**Recommended approach: keep the robot, cameras and scene, and add one skill.**
1. **Define the task.** Copy a LIBERO-Goal BDDL and add one invisible goal region near the front edge of the table, toward the user. Example: "bring the bowl to me", `On akita_black_bowl_1 main_table_user_region`. No new objects, so the existing 10 skills still see their training scene.
   - The robot base is at x = -0.66 (`mounted_panda.py:41-47`; table is 1.0 m long), so the reachable front edge is roughly x ≈ +0.1 to +0.18. That reach figure is not verified.
   - LIBERO ships `register_task_info` / `generate_bddl_from_task_info` (`libero/libero/utils/task_generation_utils.py`) and `scripts/create_libero_task_example.py`.
2. **Generate demos with a script.** Use waypoint control from privileged object poses with the OSC_POSE delta controller: above the bowl, descend, close, lift, move, release. Keep only runs where the goal predicate passes. 50 good demos is roughly 60-80 attempts at about 6-8 s of sim each, under 15 minutes of CPU. Engineering is about 1-2 days, mostly tuning bowl-rim grasps. LIBERO also has `scripts/collect_demonstration.py` for keyboard or SpaceMouse teleoperation.
3. **Convert to the GR00T LeRobot format** (v2.1 plus `meta/modality.json`; copy `examples/LIBERO/modality.json`):
   - 20 fps.
   - `observation.images.image` and `observation.images.wrist_image` at 256×256, **rotated 180° to match the `[::-1, ::-1]` in `libero_env.py:148-149`**.
   - `observation.state` with 8 values: eef xyz, axis-angle (3), gripper qpos (2).
   - `action` with 7 values.
   - **Match the gripper convention of the IPEC dataset exactly.** At run time the wrapper rescales the gripper action from [0,1] to [-1,1], binarises it and inverts it (`libero_env.py:68-95, 183-184`), so in policy units 1 means open.
   - Drop no-op frames, as the "no_noops" datasets do.
   - Encode videos as H.264. The IPEC datasets use AV1, and the README says torchcodec AV1 decoding is not guaranteed; `examples/SimplerEnv/convert_av1_to_h264.py` converts them.
4. **Train on old and new data together** so the 10 existing skills are not forgotten. N1.7 accepts multiple dataset paths with mixture weights (`ds_weights_alpha`). Use LIBERO-Goal (428 episodes) plus the new 50, starting from `checkpoints/GR00T-N1.7-LIBERO/libero_goal`. That `launch_finetune.py` accepts a fine-tuned checkpoint as its base is not verified; it should, since the format is the same. The fallback is the official 20k-step recipe starting from N1.7-3B.
   - Settings: `--embodiment-tag LIBERO_PANDA --state-dropout-prob 0.2`, 5-10k steps, global batch 32-64.
5. **Compute estimate (not verified):**
   - Reference points: GR00T N1 did 5,000 steps on 50 episodes in about 30 minutes on an H100 (dev.to guide); issue #581 reports N1.6 trains about 2× slower than N1.5.
   - So budget 2-5 hours on one H100 or RTX PRO 6000, plus about 1 hour of evaluation: 10 GPU-hours or less, under $45 even at H100 rates.
6. **Risk:** the existing skills may regress. Re-run all 11 skills afterwards; the nightly job does this anyway.

## 5. Commands for a single Ubuntu 22.04/24.04 GPU VM (headless)

Avoid Ubuntu 25.10/26.04: torchcodec 0.8 needs FFmpeg 4-7.

```bash
# (a) Install
sudo apt-get update && sudo apt-get install -y git git-lfs ffmpeg libegl1-mesa-dev libglu1-mesa
git lfs install && curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T && cd Isaac-GR00T
uv sync --python 3.12
uv run hf auth login                       # account must have Cosmos-Reason2-2B access
bash gr00t/eval/sim/LIBERO/setup_libero.sh # separate py3.12 venv; pins mujoco==3.3.1, robosuite 1.4.0, torch 2.9.0
uv run hf download nvidia/GR00T-N1.7-LIBERO --include "libero_goal/config.json" "libero_goal/embodiment_id.json" \
  "libero_goal/model-*.safetensors" "libero_goal/model.safetensors.index.json" \
  "libero_goal/processor_config.json" "libero_goal/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO

# (b) Terminal 1: policy server
uv run python gr00t/eval/run_gr00t_server.py --model-path checkpoints/GR00T-N1.7-LIBERO/libero_goal \
  --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper --host 0.0.0.0 --port 5555
# Terminal 2: official per-skill eval (this is also the nightly-job command)
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py --n-episodes 20 \
  --policy-client-host 127.0.0.1 --policy-client-port 5555 --max-episode-steps 300 \
  --env-name libero_sim/put_the_bowl_on_the_plate --n-action-steps 8 --n-envs 5
```

- **Headless rendering:** `libero_env.py:33-34` already defaults `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`. Pick the GPU with `MUJOCO_EGL_DEVICE_ID`. In Docker, set `NVIDIA_DRIVER_CAPABILITIES=all`.
- **Alternative single-venv route:** `pip install -e ".[groot,libero]"` in LeRobot, then `lerobot-eval --policy.type=groot --policy.base_model_path=nvidia/gr00t17-lerobot-libero_goal-640 --policy.embodiment_tag=libero_sim --env.type=libero --env.task=libero_goal`.

**Continuous-session runner.** This is a sketch I have not run. It goes in the LIBERO venv:

```python
env = gym.make("libero_sim/put_the_bowl_on_the_plate").unwrapped   # one scene for all 10 skills
robo = env._env.env
GOALS = {name: BDDLUtils.robosuite_parse_problem(path)["goal_state"] for ...}   # all 10 BDDLs
policy = PolicyClient(host="127.0.0.1", port=5555)
# batch(): video -> uint8[None,None]; state -> float32[None,None]; language -> (str,)
def run_skill(name, budget=300):
    robo.parsed_problem["goal_state"] = GOALS[name]
    obs["annotation.human.action.task_description"] = name.replace("_", " ")
    for each chunk: act,_ = policy.get_action(batch(obs)); for t in range(8):
        obs,_,_,_,info = env.step({k: v[0,t] for k,v in act.items()})
        if info["success"] or STOP.is_set(): ...
```

Two pieces to add:
- **Return-home move between skills:** OSC deltas with `action.xyz = clip((home - eef)/0.05, -1, 1)`, `action.gripper = [1.0]` (open), zero rotation. This keeps each skill's start pose close to training.
- **Swapping goals:** the success check reads `parsed_problem["goal_state"]`, so swapping it is enough.

**Higher-resolution frames** (no change to what the policy sees):
- `robo.sim.render(width=1280, height=720, camera_name="frontview")`. robosuite 1.4 enlarges the offscreen buffer automatically (`binding_utils.py:128-138`).
- Available cameras: `frontview`, `galleryview`, `agentview` (`libero_tabletop_manipulation.py:187-205`). You can also move a camera by editing `sim.model.cam_pos` / `cam_quat` at runtime, for example to a "user's eye" view. Flip the image as needed.
- **Better: re-render offline for the submission video.** Save `env._env.get_sim_state()` every step, then replay with `set_state` + `sim.forward()` and render at 1080p with better MSAA and shadows (`model.vis.quality.offsamples` / `shadowsize`; not verified to apply after the render context exists). The physics trajectory is identical, so this is honest.

## 6. Nebius Physical AI Workbench (github.com/nebius/nebius-physical-ai)

It is an "agent-facing control plane": the `npa` CLI, SkyPilot on Nebius Kubernetes, public GHCR images, and Claude/Codex skills.

**What exists for GR00T:**
- `npa workbench groot` with `deploy`, `download`, `finetune`, `eval`, `serve`, `infer` and `convert` commands (`docs/cli/groot.md`).
- An `npa-groot:0.1.0` image: Isaac-GR00T pinned to `3df8b38`, N1.7-3B.
- `workflows/testing/groot-1-7-finetune.yaml`. This only validates the training plumbing: 4 optimizer steps on 2× B200, then offline evaluation and RRD/MCAP files labelled "NOT A ROBOT ROLLOUT".

**What it does not do:**
- No closed-loop GR00T + LIBERO or RoboCasa recipe.
- Its LIBERO path is an unqualified BC-RNN candidate.
- The `robocasa` tool only does random rollouts and is a "validation candidate".

**Useful parts:**
- `golden-eval --serverless` shows how to submit a Nebius Serverless Job in a real image. That is the pattern for the nightly evaluation job.
- Caveat: `docs/workbench/cookbooks/serverless-tools-coverage.md` lists **GR00T serverless `infer` as SMOKE_FAILED**. Build your own image for the nightly job: Isaac-GR00T plus the LIBERO venv.
- The docs state that Isaac simulation needs an RT-core GPU (so not an H100).

## Recommended scene, chains and GPU

**Primary chain (A).** Every skill touches an object still in its starting state; the only drift is visual distractors and arm pose.

1. put the wine bottle on the rack
2. open the middle drawer of the cabinet
3. put the cream cheese in the bowl
4. push the plate to the front of the stove
5. turn on the stove — gate this behind caregiver confirmation, or leave it out

If each step independently matched the suite average, the full chain would succeed about 0.975^5 ≈ 88% of the time. That figure is not verified.

**Dependent chain (B), "set up a snack":** put the bowl on the plate → put the cream cheese in the bowl → custom "bring the bowl to me". Its intermediate states are out of distribution, so it must be measured.

**Registry fields:** exact instruction string, preconditions and effects (computed from LIBERO predicates), risk level, and success measured nightly both alone and in-chain.

**GPU choice: one RTX PRO 6000 (96 GB, $1.80/hr).** It fits GR00T (16 GB or more), Cosmos-Reason2-8B (32 GB or more) and MuJoCo EGL rendering on one GPU. It also has the RT cores needed for any Isaac/RoboLab B-roll, and NVENC for WebRTC streaming. The claim that the H100 lacks NVENC is from memory; please verify.

## Main risks, in order

1. **Chained-context drop (not verified).** Day-1 go/no-go: run chain A 20 times with return-home, which takes about 1 GPU-hour. Go if every step is at least 85% and the full chain at least 70%. Otherwise fall back to an honest "reset the table between skills" mode, or keep the best 3 skills.
2. **Cosmos-Reason2-2B gating.** Without access, no N1.7 checkpoint loads. Request it now. Fallbacks are the N1.6 or N1.5 paths.
3. **No undo skills in LIBERO-Goal** (no close-drawer, no turn-off-stove). The scene only moves forward. Add scripted undo moves, clearly labelled as scripted, a "reset table" action, or custom-trained undo skills.
4. **Brittle language.** Use exact training strings and a closed registry; Nemotron maps intent to IDs.
5. **Visuals.** The policy cameras are 256px and the LIBERO scene looks toy-like. Use a separate high-resolution camera plus the offline re-render. RoboCasa or RoboLab (BananaOnPlate is 40/40) can supply B-roll only.
6. **Dependency fragility.** There are two venvs, with pins on mujoco 3.3.1 (an API change in 3.10 breaks robosuite 1.4), torch 2.9, and FFmpeg below 8.
7. **The custom skill may make the existing skills worse.** Mitigate with joint training on old and new data and a full re-evaluation.
8. **Latency (critique 4).** Run the simulator and the policy on the same in-region VM, stream with NVENC WebRTC, and give UI feedback on the phone first. The e-stop is checked every control step (50 ms) plus a heartbeat timeout.
9. **Nebius workbench GR00T serverless path is unproven (SMOKE_FAILED).** Own the container.

## Sources

**Repos and code read**
- Isaac-GR00T: https://github.com/NVIDIA/Isaac-GR00T (README; `examples/{LIBERO,robocasa,robocasa-gr1-tabletop-tasks,SimplerEnv,RoboLab}/README.md`; `gr00t/eval/rollout_policy.py`, `gr00t/eval/sim/wrapper/multistep_wrapper.py`, `gr00t/eval/sim/LIBERO/{libero_env.py,setup_libero.sh}`; `getting_started/hardware_recommendation.md`; `scripts/deployment/README.md`; branches `n1d6` and `n1d5`)
- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
- LeRobot: https://github.com/huggingface/lerobot (`docs/source/groot.mdx`, `docs/source/robocasa.mdx`, `src/lerobot/envs/libero.py`)
- robocasa (fork used by GR00T): https://github.com/squarefk/robocasa
- RoboLab: https://github.com/NVlabs/RoboLab
- Nebius Physical AI: https://github.com/nebius/nebius-physical-ai
- robosuite 1.4.0 (PyPI)

**Hugging Face model and dataset pages**
- https://huggingface.co/nvidia/GR00T-N1.7-3B
- https://huggingface.co/api/models/nvidia/GR00T-N1.7-LIBERO/tree/main/libero_goal
- https://huggingface.co/collections/nvidia/gr00t-n17
- https://huggingface.co/nvidia/GR00T-N1.6-3B/raw/main/LICENSE
- https://huggingface.co/api/models/nvidia/gr00t17-lerobot-libero_goal-640
- https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot
- https://huggingface.co/RLobot-jun/gr00t-n1.7-robocasa-bc2-30k
- https://huggingface.co/nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task

**Leaderboards and documentation**
- https://github.com/robocasa-benchmark/leaderboard/blob/main/submissions_md/gr00t_n1.5_2026_05_19.md
- https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/sequential_static_manipulation/index.html

**Papers and articles**
- LiLo-VLA: https://yy-gx.github.io/LiLo-VLA/
- LIBERO-PRO: https://arxiv.org/abs/2510.03827
- LIBERO-Plus: https://arxiv.org/pdf/2510.13626
- GR00T N1.6 vs N1.5 training speed: https://github.com/NVIDIA/Isaac-GR00T/issues/581
- GR00T N1 fine-tuning guide: https://dev.to/yuk6ra/a-practical-guide-to-fine-tuning-and-inference-with-gr00t-n1-lerobot-on-a-custom-dataset-52ec
- NVIDIA SO-101 GR00T course: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/10-groot.html
- Assistive Gym: https://arxiv.org/abs/1910.04700