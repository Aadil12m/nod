# viewer/: the 3D view (mjviser/viser)

`nod_viewer.main` subscribes to the sim's pose stream (ZMQ PUB :5563) and draws the LIBERO-Goal
scene in the browser on port 8080 (design 3.2, 4.2.5). It runs MuJoCo 3.14 (mjviser needs 3.6 or
newer) while the sim is pinned to 3.3.1, so it is its own process and image.

```bash
uv venv -p 3.11 viewer/.venv
uv pip install --python viewer/.venv/bin/python -e "protocol[numpy,fast]" -e "viewer[export,test]"
viewer/.venv/bin/python -m nod_viewer.main --mjb recordings/scene_libero_goal_v1.mjb --sub tcp://127.0.0.1:5563 --port 8080
viewer/.venv/bin/python -m nod_viewer.main --mjb recordings/scene_libero_goal_v1.mjb --replay recordings/05_bowl_to_user_success
viewer/.venv/bin/python -m pytest -q viewer/tests        # 6 passed
```

- `--mjb` takes `.mjb`, `.mjb.gz` or `.xml`; a missing `.mjb` falls back to `<path>.gz`.
- The fixture placement (`model_patch`) comes from `--patch FILE`, from a replayed run's
  `model_patch.json`, or from the `sim.episode_header` the sim repeats on the pose socket every
  2 s; a changed patch rebuilds the scene.
- Overlays only the viewer draws: a translucent handover-zone box and a seated-user marker.
- The GUI shows the step, epoch and the measured view lag (receive time minus the sim's
  `t_wall`, meaningful on one host).
- `export_scene.py` makes the MJB in two steps (dump in the sim venv, compile here); see its
  docstring and sim/README.md.

Checked here: the viewer drew a live run from the real LIBERO sim (scripted `bowl_on_plate`,
episode header applied, view lag p50 about 4 ms on one host) and was screenshotted in headless
Chromium with software WebGL. On this CPU-only box it redrew about 4 poses per second while a
browser was connected (newest pose wins, so it lags by frames, not by queueing).
