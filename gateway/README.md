# nod-gateway

The Nod gateway: a FastAPI service that sits between the phone (PWA), the simulator and the
models. It owns the session state machine (design 3.4), the safety layer (section 5), the planner,
ASR, Cosmos and Home Assistant clients, and the presenter feed.

## Setup

```
cd gateway
/root/.local/bin/uv venv -p 3.11 .venv
/root/.local/bin/uv pip install -p .venv/bin/python -e "../protocol[numpy,fast]" -e ".[test]"
```

Add the `asr` extra (`-e ".[asr]"`) on a machine that should run Parakeet through onnx-asr.

## Running

Mock mode needs no GPU, no simulator and no API keys. It starts an in-process `MockRunner` that
speaks the sim's ZMQ protocol on 127.0.0.1, uses the fake planner, ASR, Cosmos and Home clients,
and enables `/api/dev/inject`.

```
.venv/bin/nod-gateway --mock            # http://127.0.0.1:8000/ (PWA) and /presenter
.venv/bin/nod-gateway --live --host 0.0.0.0 --port 8000
```

Every setting is an environment variable with the `NOD_` prefix (see `nod_gateway/config.py`):
for example `NOD_MODE=live`, `NOD_TF_API_KEY` (or `NEBIUS_API_KEY`), `NOD_SIM_CTL`,
`NOD_COSMOS_URL`, `NOD_HASS_URL`, `NOD_HASS_TOKEN`, `NOD_DEV=1`, `NOD_PLAN_MODE=per_step|whole`.
In live mode, without a Token Factory key the planner fails fast and the phone gets the
deterministic menu; without `NOD_COSMOS_URL` checks are skipped and the sim oracle decides;
without `NOD_HASS_URL` home actions use the fake. Each choice can be forced with `NOD_PLANNER`,
`NOD_ASR`, `NOD_COSMOS` and `NOD_HOME`, and `/healthz` shows what is in use and the last errors.

If `recordings/` (repo root) holds recorded runs, the MockRunner replays their poses: each run is
a directory with `manifest.json` (`skill_id`, and `success` or `outcome`, optional `steps`) and
`poses.npz` (`xpos`, `xquat`, `qpos`). Without recordings it runs a scripted timeline.

## Tests

```
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
```

The end-to-end tests (`tests/test_e2e.py`) start the real app under uvicorn with a MockRunner on
free ports and drive it over WebSockets, including an e-stop mid-run and a watchdog pause.

## Layout

| Module | Role |
|---|---|
| `config.py` | pydantic-settings `Config`, `NOD_MODE=live\|mock` |
| `app.py` | `create_app`, `Gateway` wiring, all HTTP and WebSocket endpoints |
| `safety.py` | `SafetyChannel`: e-stop fast path, heartbeat freshness, watchdog, face-lost pause |
| `validator.py` | pure rules R1 to R14, `feasible_skills`, `check_plan`, `validate_proposal`, `validate_confirm`, `post_decision` |
| `registry.py` | `load_registry`, `wilson_lower`, `deterministic_menu` |
| `session.py` | the 3.4 state machine, plans, pre and post checks, pause, resume, e-stop reset |
| `planner/` | `Planner` protocol, `NemotronPlanner` (Token Factory), `FakePlanner`, prompts, canned replies |
| `asr/`, `cosmos/`, `home/` | live clients (Parakeet, vLLM, Home Assistant) and fakes |
| `simlink.py` | ZMQ link to the sim (DEALER ctl, PUSH estop, SUB status) |
| `mock_runner.py` | `MockRunner`, a stand-in sim speaking the same protocol |
| `recorder.py`, `presenter.py`, `metrics.py`, `peers.py` | run traces, presenter fan-out, latency histograms, per-socket outboxes |

Differences from the design are listed under "gateway/" in `docs/DEVIATIONS.md`.
