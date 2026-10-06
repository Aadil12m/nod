# registry

- `skills.yaml`: the skill registry (design 4.2.3). A YAML list of `nod_protocol.SkillSpec` entries:
  the ten LIBERO-Goal policy skills, the custom `bowl_to_user`, the scripted motions
  (`return_home`, `reset_scene`, `close_middle_drawer`, `handover_retreat`) and the Home Assistant
  actions (`lights_on`, `lights_off`, `call_caregiver`). The header comment explains the atoms and
  the precondition policy.
- `stats/latest.json`: per-skill success alone and in-chain, written by the nightly job
  (`nod_eval/publish.py`). **The committed file is a placeholder** (`"placeholder": true`) with
  invented numbers so mock mode shows badges. Do not quote these numbers.

Load both with:

```python
from nod_protocol import load_skills
specs = load_skills("registry/skills.yaml", "registry/stats/latest.json")
```

Validation and the chain checks run in `protocol/tests/test_registry.py`
(`cd protocol && .venv/bin/python -m pytest -q tests/test_registry.py`).
