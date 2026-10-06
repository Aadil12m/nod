"""Skill registry models (4.2.3) and the nightly stats file (4.2.5).

`registry/skills.yaml` is a YAML list of `SkillSpec` entries. `registry/stats/latest.json`
is a `StatsFile`: flat, keyed by skill id, next to `chains`, `job_id`, `git_sha`, `at`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from .atoms import Rollout, parse_literal, rollout
from .base import ConfirmPolicy, HomeAction, Risk, ScriptName, SkillKind, StrictModel


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for k successes in n trials (0.0 if n == 0)."""
    if n <= 0:
        return 0.0
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k} n={n}")
    p = k / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - margin) / (1 + z2 / n))


# ---------------------------------------------------------------- stats


class StatBlock(StrictModel):
    """Measured success: k of n episodes, Wilson lower bound, median duration of successes."""

    k: int = Field(ge=0)
    n: int = Field(ge=0)
    wilson_lo: float = Field(ge=0.0, le=1.0)
    median_s: float | None = None

    @model_validator(mode="after")
    def _k_le_n(self) -> "StatBlock":
        if self.k > self.n:
            raise ValueError(f"k ({self.k}) > n ({self.n})")
        return self

    @property
    def p(self) -> float:
        return self.k / self.n if self.n else 0.0

    @classmethod
    def from_counts(cls, k: int, n: int, median_s: float | None = None) -> "StatBlock":
        return cls(k=k, n=n, wilson_lo=round(wilson_lower(k, n), 4), median_s=median_s)


class SkillStats(StrictModel):
    """Per-skill stats: alone (fresh table) and in_chain (after other skills, no reset)."""

    alone: StatBlock | None = None
    in_chain: StatBlock | None = None
    placeholder: bool = False  # set at load when the stats file is a placeholder


class ChainStats(StrictModel):
    """Whole-chain result from the nightly job (e.g. chain "A")."""

    skill_ids: list[str]
    k: int = Field(ge=0)
    n: int = Field(ge=0)
    wilson_lo: float = Field(ge=0.0, le=1.0)
    median_s: float | None = None
    steps: list[StatBlock] = Field(default_factory=list)  # per step, conditional on reaching it


class StatsFile(StrictModel):
    """`registry/stats/latest.json`. Skill entries sit at the top level, keyed by skill id."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)
    __pydantic_extra__: dict[str, SkillStats] = Field(init=False)

    note: str | None = Field(default=None, alias="_note")
    placeholder: bool = False
    job_id: str | None = None
    git_sha: str | None = None
    at: str | None = None  # ISO 8601 UTC
    chains: dict[str, ChainStats] = Field(default_factory=dict)

    @property
    def skills(self) -> dict[str, SkillStats]:
        return dict(self.__pydantic_extra__ or {})

    def dump_json(self) -> str:
        """Stable, human-diffable JSON (aliases, sorted keys, 2-space indent, defaults omitted)."""
        return json.dumps(self.model_dump(mode="json", by_alias=True, exclude_defaults=True), indent=2, sort_keys=True) + "\n"


def load_stats(path: str | Path) -> StatsFile:
    return StatsFile.model_validate_json(Path(path).read_text())


# ---------------------------------------------------------------- skills


class PolicySpec(StrictModel):
    checkpoint: str
    embodiment: str = "LIBERO_PANDA"
    instruction: str  # exact LIBERO instruction; the policy is conditioned on this string
    n_action_steps: int = 8
    max_steps: int = 400


class ScriptedSpec(StrictModel):
    script: ScriptName
    max_steps: int = 160


class HomeSpec(StrictModel):
    action: HomeAction


class CosmosQuestions(StrictModel):
    pre: str | None = None
    post: str | None = None


class SkillSpec(StrictModel):
    """One registry entry (4.2.3). Fields marked "addition" are documented in docs/DEVIATIONS.md."""

    skill_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    ver: int = 1
    kind: SkillKind
    label: str  # canonical label shown on cards (R13)
    scene_id: str | None = None  # None for home actions
    policy: PolicySpec | None = None  # kind == policy
    scripted: ScriptedSpec | None = None  # addition: kind == scripted
    home: HomeSpec | None = None  # addition: kind == home
    goal_ref: str | None = None  # "libero_goal/<task>" or "nod/<task>"; policy skills only
    pre: list[str] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)  # addition: objects referenced (R3)
    risk: Risk = "low"
    confirm: ConfirmPolicy = "single"
    near_user: bool = False
    custom: bool = False  # addition: trained by us, not a LIBERO-Goal benchmark task
    enabled: bool = True  # addition: R2 "is enabled"
    offer: bool = True  # addition: false for internal motions the planner/menu never offers
    est_s: float | None = None  # addition: static time estimate until stats give median_s
    cosmos: CosmosQuestions | None = None
    undo: str | None = None
    stats: SkillStats = Field(default_factory=SkillStats)

    @field_validator("pre", "effects")
    @classmethod
    def _literals_parse(cls, v: list[str]) -> list[str]:
        for text in v:
            parse_literal(text)  # raises ValueError on malformed literals
        return v

    @model_validator(mode="after")
    def _kind_blocks(self) -> "SkillSpec":
        blocks = {"policy": self.policy, "scripted": self.scripted, "home": self.home}
        if blocks[self.kind] is None:
            raise ValueError(f"kind {self.kind!r} needs a {self.kind!r} block")
        extra = [k for k, b in blocks.items() if b is not None and k != self.kind]
        if extra:
            raise ValueError(f"kind {self.kind!r} must not have {extra} blocks")
        if self.kind == "policy" and not self.goal_ref:
            raise ValueError("policy skills need goal_ref")
        if self.kind != "policy" and self.goal_ref:
            raise ValueError("only policy skills have goal_ref")
        if self.kind != "home" and not self.scene_id:
            raise ValueError("policy and scripted skills need scene_id")
        return self

    @property
    def instruction(self) -> str | None:
        return self.policy.instruction if self.policy else None

    @property
    def goal_atom(self) -> str | None:
        """The oracle success atom: the first positive effect (the BDDL goal for policy skills)."""
        atoms = self.success_atoms
        return atoms[0] if atoms else None

    @property
    def success_atoms(self) -> list[str]:
        """Positive effects: the atoms the oracle checks after the run (post-check)."""
        return [e for e in self.effects if not e.startswith("not ")]


def load_skills(path: str | Path, stats: str | Path | StatsFile | None = None) -> list[SkillSpec]:
    """Load and cross-check `skills.yaml`, optionally merging a stats file into `stats`."""
    import yaml  # pyyaml is a core dependency; imported here to keep import time low

    raw: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a YAML list of skills")
    specs = [SkillSpec.model_validate(item) for item in raw]
    problems = check_registry(specs)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    if stats is not None:
        sf = stats if isinstance(stats, StatsFile) else load_stats(stats)
        specs = apply_stats(specs, sf)
    return specs


def apply_stats(specs: list[SkillSpec], sf: StatsFile) -> list[SkillSpec]:
    """Return copies of `specs` with `stats` filled from the stats file (unknown ids ignored)."""
    table = sf.skills
    out = []
    for s in specs:
        st = table.get(s.skill_id)
        if st is None:
            out.append(s)
            continue
        merged = st.model_copy(update={"placeholder": sf.placeholder or st.placeholder})
        out.append(s.model_copy(update={"stats": merged}))
    return out


def check_registry(specs: list[SkillSpec]) -> list[str]:
    """Cross-entry checks: unique ids, undo targets exist, unique policy instructions."""
    problems: list[str] = []
    ids = [s.skill_id for s in specs]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        problems.append(f"duplicate skill ids {dupes}")
    known = set(ids)
    for s in specs:
        if s.undo is not None and s.undo not in known:
            problems.append(f"{s.skill_id}: undo {s.undo!r} is not a skill")
    instr = [s.instruction for s in specs if s.instruction]
    if len(instr) != len(set(instr)):
        problems.append("two policy skills share an instruction")
    return problems


def plan_rollout(skill_ids: list[str], atoms: dict[str, bool], specs: list[SkillSpec]) -> Rollout:
    """R10 symbolic rollout of a list of skill ids against a world's atoms.

    An unknown skill id fails at its step with `unmet == ["unknown skill"]`.
    """
    by_id = {s.skill_id: s for s in specs}
    for i, sid in enumerate(skill_ids):
        if sid not in by_id:
            return Rollout(ok=False, failed_step=i, failed_skill=sid, unmet=["unknown skill"], atoms=dict(atoms))
    return rollout(((sid, by_id[sid].pre, by_id[sid].effects) for sid in skill_ids), atoms)


def registry_atoms(specs: list[SkillSpec]) -> list[str]:
    """Every atom mentioned in any pre or effects, sorted. The sim oracle reports all of these."""
    out: set[str] = set()
    for s in specs:
        for t in (*s.pre, *s.effects):
            out.add(parse_literal(t).atom)
    return sorted(out)


def registry_objects(specs: list[SkillSpec]) -> list[str]:
    """Every object named in any skill's `objects`, sorted. The sim reports all of these in
    `WorldState.objects` so rule R3 can check them."""
    return sorted({o for s in specs for o in s.objects})
