# nod-protocol

The shared contract for Nod: pydantic v2 models for every message in design section 3.5, the skill
registry models (4.2.3), the planner contract (7.4), msgpack helpers for ZMQ, ULIDs, and the
symbolic atom helpers the validator uses. The JSON Schema in `schema/` is generated from these models.

Python 3.11 and 3.12. Core dependencies: `pydantic>=2.9`, `msgpack`, `pyyaml`.
Extras: `numpy` (needed only for the `pose` message), `fast` (ormsgpack, used automatically when
installed), `test`.

## Install

```bash
# from another package in this repo (gateway, sim, viewer, eval)
uv pip install -e "../protocol[numpy,fast]"
# or in pyproject.toml:  "nod-protocol[numpy] @ file:///${PROJECT_ROOT}/../protocol"

# for working on the protocol itself
cd protocol
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -e ".[numpy,fast,test]"
```

## Run the tests

```bash
cd protocol
.venv/bin/python -m pytest -q
```

## Regenerate the JSON Schema

No Makefile needed:

```bash
python protocol/gen_schema.py            # rewrite protocol/schema/*.json (run from the repo root)
python protocol/gen_schema.py --check    # CI: exit 1 if the committed files are stale
```

Use any interpreter that has the package's dependencies (for example `protocol/.venv/bin/python`).
The output is deterministic and committed. The tests fail if it is out of date.

`schema/` holds one file per message type, named after `type` (`ui.confirm.json`, `ctl.run_skill.json`,
...), the unions `ws_message.json`, `zmq_message.json`, `phone_in.json`, `safety_in.json`,
`presenter_in.json`, `gateway_out.json`, the documents `world_state.json`, `skill_spec.json`,
`skills_yaml.json`, `stats_file.json`, `planner_output.json`, `plan_request.json`, `cosmos_reply.json`,
`home_result.json`, an example of the 7.4 Nemotron schema, and `index.json` (type to file and model).

Message schemas are generated in pydantic "serialization" mode: every field is listed as required,
including the envelope (`v, type, id, ts, sess, seq`), because the sender always emits them. The PWA
should validate what it sends and what it receives against these files in development. Registry
documents use "validation" mode, where fields with defaults are optional.

## Usage

```python
from nod_protocol import (UiConfirm, Proposal, RunSkill, SimStatus, WorldState, SkillSpec,
                          parse_ws, to_ws, dumps, loads, SAFETY_IN, CTL, SIM_OUT, new_id, now_ms)

# WebSocket (JSON text frames)
msg = parse_ws(text)                    # any WebSocket message, by "type"
msg = parse_ws(text, SAFETY_IN)         # only hb / estop / estop.reset / ping
text = to_ws(msg)

# ZMQ (msgpack)
data = dumps(RunSkill(req_id=new_id("q"), run_id=new_id("r"), epoch=12, skill_id="bowl_on_plate",
                      instruction="put the bowl on the plate",
                      goal_ref="libero_goal/put_the_bowl_on_the_plate"))
cmd = loads(data, CTL)                  # restrict to ctl.* messages
```

Every function raises `pydantic.ValidationError` on invalid input.

### Message classes by link

| Link | Types (class) |
|---|---|
| `/ws/phone` in (`PHONE_IN`) | `hello` (Hello), `gesture` (Gesture), `ui.menu` (UiMenu), `ui.focus` (UiFocus), `ui.confirm` (UiConfirm), `ui.cancel` (UiCancel), `utter.start` (UtterStart), `utter.end` (UtterEnd), `utter.text` (UtterText), `ping` (Ping) |
| `/ws/safety` in (`SAFETY_IN`) | `hb` (Heartbeat), `estop` (EStop), `estop.reset` (EStopReset), `ping` |
| `/ws/presenter` in (`PRESENTER_IN`) | `estop`, `ping` |
| gateway out (`GATEWAY_OUT`) | `welcome` (Welcome), `pong` (Pong), `state` (State), `transcript` (Transcript), `proposal` (Proposal), `run` (RunUpdate), `estop.ack` (EStopAck), `notice` (Notice), `thumb` (Thumb) |
| presenter out | `trace` (Trace), `metrics` (Metrics), `cosmos.check` (CosmosCheck), `cosmos.result` (CosmosResult), plus the gateway-out messages |
| ZMQ ctl, gateway to sim (`CTL`) | `ctl.load_scene` (LoadScene), `ctl.run_skill` (RunSkill), `ctl.run_scripted` (RunScripted), `ctl.pause` (Pause), `ctl.resume` (Resume), `ctl.abort` (Abort), `ctl.capture` (Capture), `ctl.world` (WorldRequest), `ctl.hb` (CtlHeartbeat), `ctl.estop` (CtlEStop, on the dedicated e-stop socket) |
| ZMQ PUB, sim to gateway (`SIM_OUT`) | `sim.ack` (SimAck), `sim.status` (SimStatus), `sim.estop_ack` (SimEStopAck), `sim.world` (SimWorld), `sim.frames` (SimFrames), `sim.episode_header` (SimEpisodeHeader) |
| ZMQ PUB, sim to viewer | `pose` (Pose) |

WebSocket messages inherit `Envelope` (`v=1`, `type`, `id` = new ULID, `ts` = `now_ms()`, `sess=None`,
`seq=0` by default; the sender sets `sess` and `seq`). ZMQ messages carry only `v` and `type` plus
their own fields, as in the design. Unknown fields are ignored on input, for forward compatibility.

### Wire formats for binary data

- **Pose arrays** (`xpos`, `xquat`, `qpos`): in Python they are float32 `numpy.ndarray`. On the wire
  each one is a map `{"dtype": "<f4", "shape": [nb, 3], "data": <raw little-endian float32, C order>}`.
  In msgpack `data` is `bin`; in JSON it is standard base64. A reader without this package decodes
  with `np.frombuffer(m["data"], "<f4").reshape(m["shape"])`. Lists and ndarrays are accepted on
  input and cast to float32; NaN and inf are rejected; shapes are checked (`[nb,3]`, `[nb,4]`, `[nq]`).
- **JPEG frames** (`sim.frames[].jpeg`): `bytes`, msgpack `bin` on ZMQ, standard base64 (what `atob`
  reads) in JSON.

msgpack: `dumps` uses ormsgpack when installed, else msgpack-python; both produce standard msgpack
and read each other's output (tested). Force one with `backend="msgpack"`.

### Other models

- `WorldState`, `ObjPose`, `Eef`: oracle view. `SimWorld` is the same with `type: "sim.world"`;
  `.to_world()` and `SimWorld.from_world()` convert.
- `Option`, `Success`, `Check`, `Precheck`, `FixedOption`, `CosmosVerdict`, `LatMs`: parts of gateway messages.
- `PlannerOutput`, `PlannerOption`: the 7.4 schema. R1 is `PlannerOutput.model_validate_json(text)`.
  Options whose fields do not match their `kind` still parse; drop them with `PlannerOption.is_consistent()`.
  `planner_response_format(feasible_ids, plan_ids)` builds the 7.4 `json_schema` object with the
  enums filled in.
- `PlanRequest` (+ `Profile`, `FeasibleSkill`, `InfeasibleSkill`, `HistoryItem`, `FailureInfo`): the 7.3 context.
- `CosmosCheck`, `CosmosResult`, `CosmosReply` (the JSON Cosmos is asked to return), `HomeResult`.
- `SkillSpec` (+ `PolicySpec`, `ScriptedSpec`, `HomeSpec`, `CosmosQuestions`, `SkillStats`, `StatBlock`),
  `StatsFile`, `ChainStats`.

### Registry helpers

```python
from nod_protocol import load_skills, plan_rollout, registry_atoms, registry_objects, wilson_lower, unmet, holds

specs = load_skills("registry/skills.yaml", "registry/stats/latest.json")   # stats merged into spec.stats
r = plan_rollout(["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"], world.atoms, specs)   # R10
r.ok, r.failed_skill, r.unmet
```

- Atoms are `pred(arg,arg)` in lowercase with no spaces; literals are an atom or `not <atom>`.
- An atom missing from `WorldState.atoms` is unknown and satisfies neither `a` nor `not a`.
- `registry_atoms(specs)` lists every atom the sim oracle must report; `registry_objects(specs)`
  lists every object name the sim must include in `WorldState.objects` (rule R3).
- Skills with empty `effects` (`return_home`, `handover_retreat`) are arm motions, not world changes.
  R5 ("not all effects already true") is vacuously true for them, so the validator must exempt
  empty-effect skills from R5. They have `offer: false` and are never shown as cards anyway.
- `wilson_lower(k, n)` is the Wilson score lower bound (z = 1.96) used by R6 and the stats file.

### Ids and time

`new_ulid()` (monotonic within a process), `new_id("p")` gives `p_<ULID>`, `ulid_time_ms(ulid)`,
`is_ulid(text)`, and `now_ms()` (wall clock in integer ms).
