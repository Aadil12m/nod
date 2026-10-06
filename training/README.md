# training/: the custom skill (`bowl_to_user`)

```bash
# 1. scripted demos in the LIBERO venv (80 attempts, keep successes; starts mixed: table, plate, plate+cheese)
MUJOCO_GL=egl python training/gen_demos.py --skill bowl_to_user --n 80 --keep-success --out training/data/bowl_to_user.hdf5
# 2. LeRobot v2.1 for Isaac-GR00T (training/.venv, see requirements.txt)
python training/to_lerobot.py --in training/data/bowl_to_user.hdf5 --out training/data/bowl_to_user_lerobot
# 3. fine-tune on the GPU VM from the libero_goal checkpoint, with the original LIBERO-Goal data mixed in
training/finetune.sh
# 4. evaluate the new checkpoint (custom skill alone and in chain B, optional regression on all skills)
CKPT=~/nod_ckpt/libero_goal_handover/checkpoint-5000 training/eval_custom.sh
```

Verified on CPU: `gen_demos.py` on the real LIBERO scene (5 attempts: 3/3 from the table,
1/2 from plate plus cheese with 0.05 translation noise), `to_lerobot.py` on those demos, and
Isaac-GR00T's own `generate_stats` and `LeRobotEpisodeLoader` reading the result (video decode
swapped for imageio, see `tests/test_training.py`). Not run: `finetune.sh` and
`eval_custom.sh` (GPU only); their flags follow Isaac-GR00T `examples/finetune.sh` and
`examples/LIBERO/README.md` at commit 51d4c89.

Tests: `training/.venv/bin/python -m pytest -q training/tests` (3 passed, 2 skipped; with
`ISAAC_GR00T_DIR` pointing at a checkout, 5 passed).
