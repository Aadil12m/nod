# Day-1 GPU spike: prove the GR00T path (kill gate Sat Oct 10)

This is design section 8 as a checklist. Owner: the sim lead. Expect 4 to 6 hours of wall time,
of which 1 to 2 are GPU compute. Run it on Tue Oct 6 or Wed Oct 7. Record everything in
[day1-results.md](day1-results.md) as you go.

Files this runbook uses:

- `docs/runbooks/chain_test.py`: the G1 chaining harness (design 8.4), copied to the VM.
- [nemotron_smoke.py](nemotron_smoke.py): the G3 structured-output test (design 8.6).
- [infra/nebius/create_vm.sh](../../infra/nebius/create_vm.sh) and
  [infra/nebius/security_group.sh](../../infra/nebius/security_group.sh): the VM and its firewall.

> **Maintenance warning.** If your Nebius console shows a maintenance notice for
> **Oct 7 06:00 UTC**, stop and start the VM before then and do not leave runs going across it.
> Check with `nebius compute maintenance list-active`. The notice may be per-tenant and was not
> found publicly.

## 0. Before you start (laptop, today)

- [ ] **Hugging Face:** request access to
  [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) and
  [nvidia/Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B). Create a read token.
  **Without 2B access no N1.7 checkpoint loads** (`GatedRepoError`).
- [ ] **Nebius:** account with a card, the **uk-south2** project id, and RTX PRO 6000 quota of 1 or
  more (Console > Administration > Limits > Quotas). The $25 grants are Token Factory credits only.
- [ ] **Token Factory:** an API key from https://tokenfactory.nebius.com/project/api-keys.
- [ ] An SSH key at `~/.ssh/id_ed25519.pub` (or set `SSH_PUBKEY_FILE`).
- [ ] While waiting for HF access: check whether any N1.6 checkpoint has an ungated backbone, as a
  fallback if 2B access is slow (design 10.2).

The human-only items, with links, are in [docs/TEAM-TODO.md](../TEAM-TODO.md).

## 1. Create the VM (laptop)

```bash
curl -sSL https://artifacts.nebius.cloud/cli/install.sh | bash && exec -l $SHELL
nebius version
nebius profile create        # endpoint api.nebius.cloud; federation auth.nebius.com; parent-id = uk-south2 project id
export NEBIUS_PROJECT_ID=<uk-south2-project-id>
nebius compute platform list --parent-id $NEBIUS_PROJECT_ID | grep -A3 rtx6000   # expect gpu-rtx6000-a / 1gpu-24vcpu-218gb

infra/nebius/create_vm.sh --dry-run      # read the command and cloud-init first
infra/nebius/create_vm.sh                # on-demand; prints VM_ID, VM_IP and the sslip.io host names
export VM_ID=<printed> VM_IP=<printed>
```

- [ ] VM created **on-demand** (not preemptible) for the spike.
- [ ] **Lock the network down before running any service.** The default security group allows all
  ingress:

  ```bash
  SSH_CIDR=<your-ip>/32 infra/nebius/security_group.sh     # rules for 22 and 443 only
  nebius compute instance edit $VM_ID                       # add security_groups: [{id: <SG_ID>}] under spec.network_interfaces[0]
  nc -zv -w 3 $VM_IP 22; nc -zv -w 3 $VM_IP 8000             # 22 answers, 8000 must not
  ```

- [ ] If a flag is rejected, check `nebius compute instance create --help` and fix the script; the
  manual command is in design 8.1.
- [ ] Measure phone-to-VM round trip now (needed for the e-stop budget): `ping $VM_IP` from a
  laptop at the demo location, and note it in the results.

## 2. Install on the VM (design 8.2)

```bash
ssh nod@$VM_IP
nvidia-smi                                   # expect RTX PRO 6000 Blackwell, 96 GB
sudo apt-get update && sudo apt-get install -y git git-lfs ffmpeg libegl1-mesa-dev libglu1-mesa tmux jq
ffmpeg -version | head -1                    # Ubuntu 24.04 ships FFmpeg 6.x; torchcodec 0.8 needs 4-7
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T ~/Isaac-GR00T
cd ~/Isaac-GR00T && git log -1 --format='%h %ci'      # record it (the design was checked at 51d4c89)
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

- [ ] `gr00t ok`
- [ ] Cosmos-Reason2-2B `config.json` downloads (no `GatedRepoError`)
- [ ] `setup_libero.sh` ends with `Env OK`
- [ ] Checkpoint is about 7 GB
- [ ] If `uv run` later fails with a Triton `ptx_get_version` error on this CUDA 13 image, run
  `uv run bash scripts/patch_triton_cuda13.sh` (Isaac-GR00T README).

## 3. Gate G0: published single-skill success, official harness (design 8.3)

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

- [ ] Policy server answers `ping`
- [ ] Three "success rate" lines recorded. **Pass:** each skill 17/20 or better.
- [ ] While the server runs: `nvidia-smi --query-gpu=memory.used --format=csv` (for G2)

## 4. Gate G1: chain skills in one env with no reset (design 8.4)

Copy the harness to the VM (from the laptop, or straight from the repo if it is cloned on the VM):

```bash
scp docs/runbooks/chain_test.py nod@$VM_IP:~/nod_spike/chain_test.py
```

Then on the VM, with the policy server still running:

```bash
cd ~/Isaac-GR00T
PY=gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python
$PY ~/nod_spike/chain_test.py --chain single:put_the_wine_bottle_on_the_rack --trials 5     # harness smoke test
$PY ~/nod_spike/chain_test.py --chain A --trials 20 --video-dir ~/nod_spike/chainA 2>&1 | tee ~/nod_spike/g1_A.log
$PY ~/nod_spike/chain_test.py --chain A --trials 10 --no-home 2>&1 | tee ~/nod_spike/g1_A_nohome.log   # is return-home needed?
$PY ~/nod_spike/chain_test.py --chain B --trials 20 --video-dir ~/nod_spike/chainB 2>&1 | tee ~/nod_spike/g1_B.log
for T in put_the_bowl_on_the_plate put_the_cream_cheese_in_the_bowl push_the_plate_to_the_front_of_the_stove; do
  $PY ~/nod_spike/chain_test.py --chain single:$T --trials 20 2>&1 | tail -6; done                    # alone, same harness
```

- [ ] Harness smoke test passes. If it breaks, check in this order before blaming the policy:
  1. the `torch.load(..., weights_only=False)` call on the init states;
  2. action dict shapes (each value a 1-D float32 array);
  3. LIBERO's first-run config prompt, if `setup_libero.sh` was skipped.
- [ ] Chain A, 20 trials with return-home: per-step rates and full chain recorded.
  **Pass:** every step 85% or better (conditional on reaching it) and the full chain 14/20 or better.
- [ ] Chain A without return-home: recorded (decides whether return-home stays).
- [ ] Chain B: recorded (measured only; 60% or better makes the "snack" plan the video centrepiece).
- [ ] Single-skill rates for the three extra tasks in the same harness.
- [ ] Optional, for design 10.2: note the end-effector extremes from the successful runs, to fill
  `limits.eef_box` (design 3.5.4 holds placeholders).

## 5. Gate G2: timing (design 8.5)

- [ ] From the G1 summaries: `policy call ms` p95 and `env.step ms` p95.
- [ ] GPU memory from step 3.
- **Pass:** policy p95 150 ms or less and env.step p95 30 ms or less. Policy p95 up to 300 ms is
  partial: plan for TensorRT (`--trt-engine-path`) or `torch.compile`. Not a kill criterion.

## 6. Gate G3: Nemotron structured output (laptop or VM; design 8.6)

```bash
export NEBIUS_API_KEY=...
uv run --no-project --with openai python docs/runbooks/nemotron_smoke.py --out ~/nod_spike/g3.jsonl | tee ~/nod_spike/g3.log
```

- [ ] The "models served" lines show which URL serves Lightning and Super.
- [ ] Summary lines for each model and mode (`json_schema`, `tools`, `json_object`).
- **Pass:** 9/10 or more valid and fast-path p95 2.5 s or less. Otherwise use the other model or
  mode; if neither works, `json_object` plus Pydantic plus the deterministic menu (not a kill).
- [ ] Put the winning mode in `.env` as `NOD_PLANNER_STRUCTURED`.

## 7. Record and shut down (design 8.7)

- [ ] Fill in [day1-results.md](day1-results.md): commit hashes, G0 lines, both G1 summaries, G2
  numbers, the G3 table, the GPU memory reading, links to 3 sample chain videos.
- [ ] Copy the artifacts off the VM: `scp -r nod@$VM_IP:~/nod_spike .`
- [ ] **Stop the VM:** `nebius compute instance stop --id $VM_ID` (the disk is still billed).

## 8. Decide (Sat Oct 10; design 8.8)

| Gate | Pass | Partial | Fail |
|---|---|---|---|
| **G0** single skills, official harness | Each of 3 skills 17/20 or better | 14-16/20 on one skill: drop that skill | Any skill below 14/20 after one day of debugging, or a GatedRepoError that persists to Oct 10 |
| **G1** chain A, 20 trials, return-home | Every step 85% or better **and** full chain 14/20 or better | Full chain 10-13/20 | Full chain below 10/20 |
| G1 chain B | Measured only | | |
| **G2** timing | Policy p95 150 ms or less and env.step p95 30 ms or less | Policy p95 300 ms or less | Not a kill criterion |
| **G3** Nemotron | 9/10 or more valid, fast-path p95 2.5 s or less | Use the other model or mode | `json_object` plus Pydantic plus the deterministic menu |

| Outcome | Tier (design 1.6) |
|---|---|
| G0 and G1 pass | **T1:** GR00T for all policy skills, chained in one session |
| G0 passes, G1 partial | **T2:** offer in-chain only skills that measured 85% or better in chain; visible "reset table" |
| G0 passes, G1 fails | **T3:** one skill per scene reset, shown honestly as a scripted "tidy the table" step |
| G0 fails, or no Cosmos-Reason2-2B access by Oct 10 | **T4:** scripted waypoint skills (`NOD_ACTOR=scripted`); GR00T becomes a stretch goal |

- [ ] Tier written at the bottom of [day1-results.md](day1-results.md) and announced to the team.
