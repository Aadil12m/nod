#!/usr/bin/env python3
"""Generate protocol/schema/*.json from the nod_protocol models.

    python protocol/gen_schema.py            # (re)write protocol/schema/
    python protocol/gen_schema.py --check    # exit 1 if the committed files are stale
    python protocol/gen_schema.py --out DIR  # write somewhere else

Output is deterministic (sorted keys, 2-space indent, trailing newline), so the
generated files can be committed and diffed. Wire messages use pydantic's
"serialization" mode, in which every field with a default is listed as required,
because senders always emit it. Hand-authored documents (skills.yaml entries,
the stats file) use "validation" mode, in which defaulted fields are optional.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # use the source tree even if the package is not installed

import nod_protocol as p  # noqa: E402
from nod_protocol import codec  # noqa: E402

DRAFT = "https://json-schema.org/draft/2020-12/schema"


def _finish(schema: dict[str, Any], name: str, title: str | None = None) -> dict[str, Any]:
    schema = dict(schema)
    schema["$schema"] = DRAFT
    schema["$id"] = f"https://nod.invalid/schema/{name}"
    if title:
        schema["title"] = title
    return schema


def _model(cls: type[BaseModel], mode: str) -> dict[str, Any]:
    return cls.model_json_schema(mode=mode, by_alias=True)


def _adapter(tp: Any, mode: str) -> dict[str, Any]:
    return TypeAdapter(tp).json_schema(mode=mode, by_alias=True)


def build() -> dict[str, dict[str, Any]]:
    """Return {file name: schema}. One file per message type plus unions and documents."""
    out: dict[str, dict[str, Any]] = {}
    index: dict[str, Any] = {"messages": {}, "unions": {}, "documents": {}}

    for cls in codec.ALL_TYPES:
        type_name = cls.model_fields["type"].default
        fname = f"{type_name}.json"
        out[fname] = _finish(_model(cls, "serialization"), fname)
        link = "zmq" if cls in codec.ZMQ_TYPES else "ws"
        index["messages"][type_name] = {"file": fname, "model": cls.__name__, "link": link}

    unions = {
        "ws_message.json": (codec.WsMessage, "Any WebSocket message (discriminated on type)"),
        "zmq_message.json": (codec.ZmqAny, "Any ZMQ msgpack message (discriminated on type)"),
        "phone_in.json": (codec.PhoneIn, "Messages the gateway accepts on /ws/phone"),
        "safety_in.json": (codec.SafetyIn, "Messages the gateway accepts on /ws/safety"),
        "presenter_in.json": (codec.PresenterIn, "Messages the gateway accepts on /ws/presenter"),
        "gateway_out.json": (codec.GatewayOut, "Gateway to phone messages"),
    }
    for fname, (tp, title) in unions.items():
        out[fname] = _finish(_adapter(tp, "serialization"), fname, title)
        index["unions"][fname] = title

    documents = {
        "world_state.json": (p.WorldState, "serialization"),
        "planner_output.json": (p.PlannerOutput, "validation"),
        "plan_request.json": (p.PlanRequest, "serialization"),
        "cosmos_reply.json": (p.CosmosReply, "validation"),
        "home_result.json": (p.HomeResult, "serialization"),
        "skill_spec.json": (p.SkillSpec, "validation"),
        "stats_file.json": (p.StatsFile, "validation"),
    }
    for fname, (cls, mode) in documents.items():
        out[fname] = _finish(_model(cls, mode), fname)
        index["documents"][fname] = cls.__name__
    out["skills_yaml.json"] = _finish(_adapter(list[p.SkillSpec], "validation"), "skills_yaml.json",
                                      "registry/skills.yaml: a list of SkillSpec")
    index["documents"]["skills_yaml.json"] = "list[SkillSpec]"

    # The 7.4 schema as sent to Nemotron, with example enums (all policy ids).
    example = p.planner_response_format(["<FEASIBLE_IDS>"], ["<POLICY_AND_CUSTOM_IDS>"])
    out["planner_response_format.example.json"] = example
    index["documents"]["planner_response_format.example.json"] = "planner_response_format() output, enums are placeholders"

    out["index.json"] = {"protocol_version": p.PROTOCOL_VERSION, **index}
    return out


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "schema")
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing")
    args = ap.parse_args(argv)

    files = {name: render(s) for name, s in build().items()}
    existing = {f.name for f in args.out.glob("*.json")} if args.out.exists() else set()
    stale = sorted(existing - set(files))
    if args.check:
        bad = [n for n, text in files.items() if not (args.out / n).exists() or (args.out / n).read_text() != text]
        if bad or stale:
            print(f"schema out of date: changed {bad}, stale {stale}. Run: python protocol/gen_schema.py", file=sys.stderr)
            return 1
        print(f"schema up to date ({len(files)} files)")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name in stale:
        (args.out / name).unlink()
    for name, text in files.items():
        (args.out / name).write_text(text)
    print(f"wrote {len(files)} files to {args.out}" + (f", removed {stale}" if stale else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
