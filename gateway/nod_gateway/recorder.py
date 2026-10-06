"""Run and proposal traces (GET /api/runs, design 4.3, 7.5).

Layout on disk, one directory per id (proposal `p_...` or run `r_...`):

    runs/<id>/manifest.json        summary: kind, created, skill, outcome, timings
    runs/<id>/events.jsonl         every trace event, in order
    runs/<id>/planner_trace.json   planner request and response (feeds FakePlanner)

The newest records are also kept in memory for the API. With `root=None` nothing is written.
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from nod_protocol import now_ms

log = logging.getLogger(__name__)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class Recorder:
    def __init__(self, root: Path | None, keep: int = 200) -> None:
        self.root = Path(root) if root is not None else None
        self.keep = keep
        self.records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, rid: str) -> Path | None:
        if self.root is None or not _SAFE_ID.match(rid):
            return None
        d = self.root / rid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_json(self, rid: str, name: str, obj: Any) -> None:
        d = self._dir(rid)
        if d is None:
            return
        try:
            (d / name).write_text(json.dumps(obj, indent=2, default=str))
        except OSError as exc:  # never let tracing break the session
            log.warning("recorder: %s", exc)

    def start(self, rid: str, kind: str, **meta: Any) -> None:
        rec = {"id": rid, "kind": kind, "created_ms": now_ms(), "events": [], "files": {}, **meta}
        self.records[rid] = rec
        while len(self.records) > self.keep:
            self.records.popitem(last=False)
        self._write_json(rid, "manifest.json", self._manifest(rec))

    def _manifest(self, rec: dict) -> dict:
        return {k: v for k, v in rec.items() if k not in ("events", "files")}

    def event(self, rid: str, kind: str, data: dict[str, Any]) -> None:
        rec = self.records.get(rid)
        ev = {"t": now_ms(), "kind": kind, "data": data}
        if rec is not None:
            rec["events"].append(ev)
        d = self._dir(rid)
        if d is not None:
            try:
                with (d / "events.jsonl").open("a") as f:
                    f.write(json.dumps(ev, default=str) + "\n")
            except OSError as exc:
                log.warning("recorder: %s", exc)

    def file(self, rid: str, name: str, obj: Any) -> None:
        rec = self.records.get(rid)
        if rec is not None:
            rec["files"][name] = obj
        self._write_json(rid, name, obj)

    def update(self, rid: str, **fields: Any) -> None:
        rec = self.records.get(rid)
        if rec is None:
            return
        rec.update(fields)
        self._write_json(rid, "manifest.json", self._manifest(rec))

    def list(self, limit: int = 20) -> list[dict]:
        items = list(self.records.values())[-limit:][::-1]
        return [self._manifest(r) | {"n_events": len(r["events"])} for r in items]

    def get(self, rid: str) -> dict | None:
        rec = self.records.get(rid)
        if rec is not None:
            return rec
        d = self.root / rid if (self.root is not None and _SAFE_ID.match(rid)) else None
        if d is None or not (d / "manifest.json").exists():
            return None
        out = json.loads((d / "manifest.json").read_text())
        ev = d / "events.jsonl"
        out["events"] = [json.loads(line) for line in ev.read_text().splitlines()] if ev.exists() else []
        pt = d / "planner_trace.json"
        out["files"] = {"planner_trace.json": json.loads(pt.read_text())} if pt.exists() else {}
        return out
