# Run Nod on this Linux laptop (instructions for Claude Code)

You are Claude Code on Aadil's Linux laptop. Goal: get Nod running here in laptop mode, check
it works end to end, and help Aadil record the hackathon video. No hardware is involved: the
robot arm is simulated (MuJoCo, LIBERO-Goal scene) and runs on the CPU. No GPU, no robot.

Project: Nod, for the Nebius x NVIDIA Global AI Hackathon, Physical AI track. Deadline Fri
Oct 30 2026, 10:00 AM PT. A person who can't use their arms controls a robot arm with face
gestures (long blink = confirm, head turn = move focus, brow raise = menu, jaw hold or "stop" =
e-stop). Nemotron on Nebius Token Factory turns requests into a short menu of safe skills.

Repo: https://github.com/Aadil12m/nod (public). Full spec: `docs/design.md`.

## Rules

- **Never print, log, commit or echo the Token Factory API key.** Aadil sets it themselves with
  `export NEBIUS_API_KEY=...` (or puts it in `~/.nod_key` with `chmod 600`). Check only whether
  it is set: `[ -n "$NEBIUS_API_KEY" ] && echo set`.
- Zero spend: Aadil has only the free $25 Token Factory credit. Don't start paid cloud resources.
- Ask Aadil before running `sudo` (setup needs it once for the OSMesa graphics library).

## 1. Get the code

```bash
cd /home/nebius
[ -d nod ] || git clone https://github.com/Aadil12m/nod
cd nod && git pull
```

Needs `git` and `curl` (`sudo apt-get install -y git curl` if missing) and about 3 GB of disk.

## 2. One-time setup (about 1 to 5 minutes)

```bash
infra/scripts/setup_laptop.sh
```

It installs uv (which fetches Python 3.11 by itself), installs `libosmesa6 libgl1 libglib2.0-0`
with sudo if OSMesa is missing, builds three venvs (`sim/.venv-libero`, `gateway/.venv`,
`viewer/.venv`), clones LIBERO at commit 8f1084e, and exports the 3D scene to
`recordings/scene_libero_goal_v1.mjb`. It is safe to re-run. It was tested on a fresh clone on
Oct 22. If the scene export fails, laptop mode still works without the 3D view.

Known trouble spots:
- Not Ubuntu/Debian: install the OSMesa equivalents with the distro's package manager
  (Fedora: `mesa-libOSMesa`; Arch: `mesa`), then re-run.
- `ImportError` mentioning OSMesa or GL: check `ldconfig -p | grep OSMesa`.
- Disk full: delete `~/.cache/uv` after setup.

## 3. Check Nemotron with the key (optional, costs a few cents)

```bash
uv run --no-project --with openai python docs/runbooks/nemotron_smoke.py
```

It lists the Nemotron models the key can reach and checks structured output. The gateway uses
`nvidia/Nemotron-3_5-Lightning` at `https://api.tokenfactory.nebius.com/v1/` and
`nvidia/nemotron-3-super-120b-a12b` at `https://api.tokenfactory.us-central1.nebius.com/v1/`.
If a model ID isn't served, override with `NOD_FAST_MODEL`, `NOD_PLAN_MODEL`, `NOD_TF_FAST_URL`,
`NOD_TF_PLAN_URL` (the smoke test prints which ones exist). If the winning mode is not `auto`,
set `NOD_PLANNER_STRUCTURED` to it.

## 4. Run it

```bash
make laptop
```

Leave it running. It starts the sim (ports 5560 to 5563), the 3D viewer (8080) and the gateway
(8000). The scene takes about 20 seconds to load. Check health:

```bash
curl -s localhost:8000/healthz | python3 -m json.tool
```

Expect `"ok": true`, `sim.connected: true`, `sim.scene_loaded: true`, and `token_factory.ok`
true once a request has gone through (without a key it says the fixed menu is in use, which is
fine for testing but the video should show Nemotron).

Then open in Chrome:
- http://localhost:8000 : the phone screen (webcam reads the face; localhost counts as secure)
- http://localhost:8000/presenter?viewer=http://localhost:8080/ : big view with the 3D arm

Keyboard fallback (add `?nocam=1` to the phone URL for keyboard only):

| Key | Gesture |
|---|---|
| Left / Right | Head turn |
| Hold Enter | Long blink (confirm) |
| B | Brow raise (menu) |
| Space | STOP |

What works with the scripted arm (measured, 20 runs each, `registry/stats/scripted.json`):
bowl on plate, cream cheese in bowl, bring the bowl to me, bowl on stove, bottle on cabinet
(20/20 each), bowl on cabinet 18/20, bottle on rack 17/20, and the snack chain (bowl on plate,
cheese in bowl, bring it to me) 20/20. Drawer, plate push and stove knob are hidden on purpose.

## 5. Smoke test before recording

1. Phone page: open the menu with a brow raise (or B). Nemotron should offer the skill options.
2. Focus "bowl on the plate", confirm with a long blink. The arm moves in the presenter view.
3. Mid-move, press Space (or hold the jaw open). The arm must stop at once and the screen turns red.
4. Run the full snack chain once without stopping.

If any step fails, read the terminal output of `make laptop` and the newest folder in
`runs/`, fix the cause, and run `make -k test` before committing. Push fixes to `main`.

## 6. The video (under 3 minutes, YouTube)

Requirements: at least 1 minute of the app working, and it must show Nebius Token Factory and
an NVIDIA model (Nemotron). Suggested flow:

1. 0:00 to 0:20: the problem. People with tetraplegia can't use their arms; Nod lets them run a
   robot arm with blinks and their voice.
2. 0:20 to 2:00: live demo with screen recording (`ffmpeg` or OBS). Phone page and presenter
   side by side: ask for a snack, confirm with a blink, the arm runs the chain, then a STOP
   mid-move.
3. 2:00 to 2:40: how it works. Nemotron on Token Factory proposes a menu; a deterministic
   validator only allows measured, safe skills; the e-stop path bypasses the model. Say plainly
   that the arm is scripted in this demo and GR00T N1.7 is the drop-in learned policy.
4. 2:40 to 3:00: what's next (GR00T on a GPU, a real arm).

## 7. Devpost submission

Track: Physical AI. Include the YouTube link, the repo link, and a description saying: no
physical hardware (simulated arm on a laptop CPU); Nemotron via Nebius Token Factory; NVIDIA
open models used (Nemotron, with GR00T N1.7 as the learned actor once a GPU is available).
Submit by Oct 29 to leave a day of slack.
