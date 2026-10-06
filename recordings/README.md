# recordings/ (mock library)

This folder holds the mock-mode library. It is not in git (only this README is); fetch it with:

```bash
make fetch-mock        # infra/scripts/fetch_mock.sh: GitHub release asset, sha256-checked
```

Layout after the download:

| Path | What |
|---|---|
| `scene_libero_goal_v1.mjb` | The LIBERO-Goal scene compiled for MuJoCo 3.14, loaded by the viewer (`viewer/nod_viewer/export_scene.py` makes it in the sim image) |
| `<run_id>/manifest.json` | `skill_id`, and `success` or `outcome`, plus optional `steps` |
| `<run_id>/poses.npz` | `xpos`, `xquat`, `qpos` per step, replayed by the gateway's `MockRunner` at 20 Hz |
| `<run_id>/...` | Other recorder output (`model_patch.json`, `events.jsonl`, `policy_cam.mp4`), when present |

Every nightly-job episode, successes and real failures alike, can become a mock run (design 4.4).
To publish a new library from a local `recordings/` folder:

```bash
infra/scripts/fetch_mock.sh --pack dist/mock
gh release create mock-v2 dist/mock/nod-mock.tar.gz dist/mock/SHA256SUMS --title "Mock library v2"
# then bump NOD_MOCK_TAG (default mock-v1) in infra/scripts/fetch_mock.sh
```

Without this folder, mock mode still runs: the MockRunner plays a scripted timeline and the
presenter page shows no 3D scene.
