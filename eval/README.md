# eval/: the nightly evaluation

`nod_eval` measures every skill alone and in chains A and B with the same `SimRunner` the
live product uses (no gateway), then writes `stats.json` (`nod_protocol.StatsFile`) and scores
Cosmos verdicts against the oracle (design 3.1, 4.2.5; runbook docs/runbooks/nightly.md).

```bash
python -m nod_eval.run --registry registry/skills.yaml --episodes 20 --chains A,B --out OUT \
    [--actor policy|scripted] [--policy tcp://127.0.0.1:5555] [--policy-for CHECKPOINT=URL] [--lazy-cameras]
python -m nod_eval.publish --in OUT --out OUT/stats.json
python -m nod_eval.cosmos_agree --in OUT --registry registry/skills.yaml --url http://127.0.0.1:8001/v1
```

- `run.py`: alone episodes use init states `(offset + episode) % 50`; chain episodes reset once,
  then run `return_home` before every step after the first and stop at the first failure.
  Policy skills whose checkpoint has no server (the custom `bowl_to_user` in the nightly job)
  are skipped, and a chain is cut before the first skill that cannot run; both are recorded in
  `run_meta.json` and in the stats `_note`. Every run is recorded (`episodes/`), and frontview
  frames before and after each run are saved for Cosmos (`frames/`).
- `publish.py`: alone = every alone episode; in_chain = every chain step after the first;
  median_s = median simulated duration of successes (steps / 20 Hz = live wall time).
- `cosmos_agree.py`: asks the registry's `cosmos.pre`/`cosmos.post` questions on the saved frames
  (vLLM OpenAI API, or a seeded fake) and compares with the oracle (`pre_ok`, `effects_ok`):
  accuracy, coverage, false yes/no and Cohen's kappa per skill and phase.

Tests: `sim/.venv/bin/python -m pytest -q eval/tests` (8 passed; FakeEnv, fake verifier, a local
HTTP stand-in for vLLM). A 3-episode scripted run on the real LIBERO scene is summarised in
sim/README.md. Not run here: the GR00T actor (no GPU) and a real Cosmos server.
