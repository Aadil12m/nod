"""Generated JSON Schema: deterministic, committed, and it accepts what the models emit."""

from __future__ import annotations

import json
import subprocess
import sys

import jsonschema
import pytest
import yaml

import nod_protocol as p
from conftest import PROTOCOL_DIR, SCHEMA_DIR, SKILLS_YAML, STATS_JSON, make_samples

sys.path.insert(0, str(PROTOCOL_DIR))
import gen_schema  # noqa: E402

SAMPLES = make_samples()


def _load(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text())


def test_generation_is_deterministic():
    a = {k: gen_schema.render(v) for k, v in gen_schema.build().items()}
    b = {k: gen_schema.render(v) for k, v in gen_schema.build().items()}
    assert a == b


def test_committed_schema_is_up_to_date(tmp_path):
    rc = gen_schema.main(["--check", "--out", str(SCHEMA_DIR)])
    assert rc == 0, "run: python protocol/gen_schema.py"
    # writing to a fresh dir produces byte-identical files
    gen_schema.main(["--out", str(tmp_path)])
    fresh = {f.name: f.read_text() for f in tmp_path.glob("*.json")}
    committed = {f.name: f.read_text() for f in SCHEMA_DIR.glob("*.json")}
    assert fresh == committed


def test_check_mode_detects_stale_files(tmp_path):
    gen_schema.main(["--out", str(tmp_path)])
    (tmp_path / "ping.json").write_text("{}\n")
    assert gen_schema.main(["--check", "--out", str(tmp_path)]) == 1


def test_cli_runs_as_a_script(tmp_path):
    out = subprocess.run([sys.executable, str(PROTOCOL_DIR / "gen_schema.py"), "--out", str(tmp_path)],
                         capture_output=True, text=True, check=True)
    assert "wrote" in out.stdout


def test_every_message_type_has_a_schema_file():
    index = _load("index.json")
    assert set(index["messages"]) == set(p.TYPE_TO_MODEL)
    for t, entry in index["messages"].items():
        assert (SCHEMA_DIR / entry["file"]).exists()
        assert _load(entry["file"])["properties"]["type"]["const"] == t


@pytest.mark.parametrize("msg", SAMPLES, ids=[m.type for m in SAMPLES])
def test_model_output_validates_against_schema(msg):
    data = json.loads(p.to_ws(msg))
    jsonschema.validate(data, _load(f"{msg.type}.json"))
    union = "zmq_message.json" if type(msg) in p.ZMQ_TYPES else "ws_message.json"
    jsonschema.validate(data, _load(union))


def test_schema_rejects_missing_envelope_fields():
    data = json.loads(p.to_ws(p.Ping(t0=1)))
    del data["seq"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, _load("ping.json"))


def test_registry_files_validate_against_document_schemas():
    jsonschema.validate(yaml.safe_load(SKILLS_YAML.read_text()), _load("skills_yaml.json"))
    jsonschema.validate(json.loads(STATS_JSON.read_text()), _load("stats_file.json"))


def test_all_schema_files_are_valid_json_schema():
    for f in SCHEMA_DIR.glob("*.json"):
        if f.name == "index.json":
            continue
        schema = json.loads(f.read_text())
        if f.name == "planner_response_format.example.json":
            schema = schema["schema"]
        jsonschema.Draft202012Validator.check_schema(schema)
