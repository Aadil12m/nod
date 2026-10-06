"""Turn an eval run into `stats.json` (design 4.2.5), the `nod_protocol.StatsFile` format.

    python -m nod_eval.publish --in /mnt/results/<date> --out /mnt/results/<date>/stats.json

    {skill_id: {alone: {k, n, wilson_lo, median_s}, in_chain: {...}},
     chains: {name: {skill_ids, k, n, wilson_lo, median_s, steps: [{k, n, wilson_lo, median_s}]}},
     job_id, git_sha, at}

* alone: every `kind: alone` episode of the skill. in_chain: every chain step after the first
  (conditional on reaching it). k counts oracle successes; wilson_lo is the 95% Wilson lower
  bound; median_s is the median simulated duration (steps / 20 Hz) of the successful runs, which
  equals wall time in the live 20 Hz loop.
* chains: k full-chain successes of n episodes; median_s includes the return_home motions;
  steps[i] is conditional on reaching step i.
* A run with the scripted actor says so in `_note`: those are T4 numbers, not GR00T.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from nod_protocol import StatBlock, StatsFile


def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    with open(run_dir / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _median(xs: Iterable[float]) -> float | None:
    xs = list(xs)
    return round(statistics.median(xs), 2) if xs else None


def block(rows: list[dict]) -> StatBlock:
    k = sum(1 for r in rows if r["success"])
    return StatBlock.from_counts(k, len(rows), _median(r["sim_s"] for r in rows if r["success"]))


def build_stats(rows: list[dict], meta: dict | None = None, *, job_id: str | None = None,
                git_sha: str | None = None, at: str | None = None) -> StatsFile:
    meta = meta or {}
    alone: dict[str, list[dict]] = defaultdict(list)
    in_chain: dict[str, list[dict]] = defaultdict(list)
    summaries: dict[str, list[dict]] = defaultdict(list)
    chain_steps: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        kind = r.get("kind")
        if kind == "alone":
            alone[r["skill_id"]].append(r)
        elif kind == "chain":
            chain_steps[(r["chain"], r["step"])].append(r)
            if r["step"] > 0:
                in_chain[r["skill_id"]].append(r)
        elif kind == "chain_summary":
            summaries[r["chain"]].append(r)
    data: dict = {}
    for sid in sorted(set(alone) | set(in_chain)):
        entry = {}
        if alone.get(sid):
            entry["alone"] = block(alone[sid]).model_dump()
        if in_chain.get(sid):
            entry["in_chain"] = block(in_chain[sid]).model_dump()
        data[sid] = entry
    chains = {}
    for name, ss in sorted(summaries.items()):
        skill_ids = ss[0]["skill_ids"]
        k = sum(1 for s in ss if s["success"])
        full = StatBlock.from_counts(k, len(ss), _median(s["sim_s"] for s in ss if s["success"]))
        steps = [block(chain_steps.get((name, i), [])).model_dump() for i in range(len(skill_ids))]
        chains[name] = {"skill_ids": skill_ids, "k": full.k, "n": full.n, "wilson_lo": full.wilson_lo,
                        "median_s": full.median_s, "steps": steps}
    notes = []
    if meta.get("actor") == "scripted":
        notes.append("actor: scripted waypoint controllers (tier T4), not GR00T")
    if meta.get("max_steps"):
        notes.append(f"step budget overridden to {meta['max_steps']}")
    for name, why in (meta.get("chain_notes") or {}).items():
        notes.append(f"chain {name} {why}")
    if meta.get("skipped"):
        notes.append("skipped: " + "; ".join(f"{k} ({v})" for k, v in sorted(meta["skipped"].items())))
    data.update({
        "chains": chains,
        "job_id": job_id or meta.get("job_id") or os.environ.get("NOD_JOB_ID"),
        "git_sha": git_sha or meta.get("git_sha") or os.environ.get("NOD_GIT_SHA"),
        "at": at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "placeholder": False,
    })
    if notes:
        data["_note"] = ". ".join(notes)
    return StatsFile.model_validate(data)


def publish(run_dir: Path, out: Path, **kw) -> StatsFile:
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    stats = build_stats(load_rows(run_dir), meta, **kw)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(stats.dump_json())
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write stats.json from a nod_eval.run output folder")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--git-sha", default=None)
    a = ap.parse_args(argv)
    stats = publish(Path(a.inp), Path(a.out), job_id=a.job_id, git_sha=a.git_sha)
    for sid, s in sorted(stats.skills.items()):
        al = f"{s.alone.k}/{s.alone.n} (lo {s.alone.wilson_lo:.2f})" if s.alone else "-"
        ic = f"{s.in_chain.k}/{s.in_chain.n} (lo {s.in_chain.wilson_lo:.2f})" if s.in_chain else "-"
        print(f"{sid:24s} alone {al:20s} in_chain {ic}")
    for name, c in stats.chains.items():
        print(f"chain {name}: {c.k}/{c.n} (lo {c.wilson_lo:.2f}) " + " ".join(f"{b.k}/{b.n}" for b in c.steps))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
