# Day-1 spike results (template)

Fill this in during the spike ([day1-spike.md](day1-spike.md)). Paste raw lines from the logs in
`~/nod_spike/`; do not round or retype numbers. Leave "TBD" where a step was not run, and say why.

| Field | Value |
|---|---|
| Date and operator | TBD |
| VM id, region, preset | TBD, uk-south2, gpu-rtx6000-a / 1gpu-24vcpu-218gb |
| Boot image | ubuntu24.04-cuda13.0 (driver: TBD) |
| Isaac-GR00T commit (`git log -1 --format='%h %ci'`) | TBD (design checked 51d4c89) |
| LIBERO submodule commit | TBD (design checked 8f1084e) |
| Checkpoint | nvidia/GR00T-N1.7-LIBERO / libero_goal, revision TBD |
| Nod repo commit | TBD |
| Triton CUDA 13 patch needed? | TBD |
| `torch.load(..., weights_only=False)` needed for init states? | TBD |
| Laptop to VM ping from the demo location (ms, min/avg/max) | TBD |

## G0: single skills, official harness

```
put_the_wine_bottle_on_the_rack:        TBD (paste the "success rate" line)
open_the_middle_drawer_of_the_cabinet:  TBD
put_the_cream_cheese_in_the_bowl:       TBD
```

Result: PASS / PARTIAL / FAIL. Notes: TBD

## G1: chaining in one env, no reset

Chain A, 20 trials, return-home (`g1_A.log`, the SUMMARY block):

```
TBD
```

Chain A, 10 trials, no return-home (`g1_A_nohome.log`):

```
TBD
```

Chain B, 20 trials (`g1_B.log`):

```
TBD
```

Same harness, alone, 20 trials each:

```
put_the_bowl_on_the_plate:                 TBD
put_the_cream_cheese_in_the_bowl:          TBD
push_the_plate_to_the_front_of_the_stove:  TBD
```

Result: PASS / PARTIAL / FAIL. Is return-home needed? TBD

End-effector extremes over successful runs (for `limits.eef_box`, design 3.5.4):
x [TBD, TBD], y [TBD, TBD], z [TBD, TBD]

Sample chain videos (links to the team bucket or drive):

1. TBD
2. TBD
3. TBD

## G2: timing

| Measure | p50 | p95 | max | n |
|---|---|---|---|---|
| Policy call (ms) | TBD | TBD | TBD | TBD |
| env.step (ms) | TBD | TBD | | TBD |
| GPU memory used with the server running (`nvidia-smi`) | TBD | | | |

Result: PASS / PARTIAL. Notes: TBD

## G3: Nemotron structured output

Models served:

```
global: TBD
us:     TBD
```

| Model | Mode | Valid | p50 (s) | p95 (s) |
|---|---|---|---|---|
| nvidia/Nemotron-3_5-Lightning | json_schema | TBD/10 | TBD | TBD |
| nvidia/Nemotron-3_5-Lightning | tools | TBD/10 | TBD | TBD |
| nvidia/Nemotron-3_5-Lightning | json_object | TBD/10 | TBD | TBD |
| nvidia/nemotron-3-super-120b-a12b | json_schema | TBD/10 | TBD | TBD |
| nvidia/nemotron-3-super-120b-a12b | tools | TBD/10 | TBD | TBD |
| nvidia/nemotron-3-super-120b-a12b | json_object | TBD/10 | TBD | TBD |

Result: PASS / PARTIAL / FAIL. Chosen `NOD_PLANNER_STRUCTURED`: TBD

## Decision

| Gate | Result |
|---|---|
| G0 | TBD |
| G1 | TBD |
| G2 | TBD |
| G3 | TBD |

**Tier: TBD** (T1 / T2 / T3 / T4, design 1.6). Decided on TBD by TBD.

Follow-ups (skills dropped, budgets changed, what to raise with Nebius or NVIDIA): TBD

Artifacts copied to: TBD. VM stopped at: TBD.
