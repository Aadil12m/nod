"""Skill registry with live stats (design 4.2.2, 4.2.3) and the deterministic menu (5.7 step 1).

`load_registry` reads `registry/skills.yaml` through `nod_protocol.load_skills` and merges
`registry/stats/latest.json`. The `Registry` object is immutable after load and is handed to
the pure validator.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from nod_protocol import (
    ChainStats,
    PlannerOption,
    PlannerOutput,
    SkillSpec,
    StatBlock,
    StatsFile,
    load_skills,
    load_stats,
    registry_atoms,
    registry_objects,
)
from nod_protocol import wilson_lower as _wilson_lower

if TYPE_CHECKING:
    from .validator import Feasibility


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound (re-exported from nod_protocol so there is one implementation)."""
    return _wilson_lower(k, n, z)


@dataclass(frozen=True)
class Registry:
    skills: dict[str, SkillSpec]
    stats: StatsFile | None = None
    scene_id: str = "libero_goal_v1"
    atoms: tuple[str, ...] = field(default=())
    objects: tuple[str, ...] = field(default=())

    @classmethod
    def from_specs(
        cls, specs: list[SkillSpec], stats: StatsFile | None = None, scene_id: str = "libero_goal_v1"
    ) -> Registry:
        return cls(
            skills={s.skill_id: s for s in specs},
            stats=stats,
            scene_id=scene_id,
            atoms=tuple(registry_atoms(specs)),
            objects=tuple(registry_objects(specs)),
        )

    def get(self, skill_id: str | None) -> SkillSpec | None:
        return self.skills.get(skill_id) if skill_id else None

    def __getitem__(self, skill_id: str) -> SkillSpec:
        return self.skills[skill_id]

    def __contains__(self, skill_id: object) -> bool:
        return skill_id in self.skills

    def __iter__(self) -> Iterator[SkillSpec]:
        return iter(self.skills.values())

    def __len__(self) -> int:
        return len(self.skills)

    def home_skill(self, action: str) -> SkillSpec | None:
        for s in self.skills.values():
            if s.kind == "home" and s.home is not None and s.home.action == action:
                return s
        return None

    def stat(self, skill_id: str, basis: str) -> StatBlock | None:
        s = self.skills.get(skill_id)
        if s is None:
            return None
        return s.stats.in_chain if basis == "in_chain" else s.stats.alone

    def plan_ids(self) -> list[str]:
        """Ids allowed inside a plan (7.4): enabled, offerable policy skills of this scene."""
        return [
            s.skill_id
            for s in self.skills.values()
            if s.kind == "policy" and s.enabled and s.offer and s.scene_id == self.scene_id
        ]

    def chain_for(self, skill_ids: list[str]) -> ChainStats | None:
        """Measured whole-chain stats when the plan equals a chain the nightly job ran."""
        if not self.stats:
            return None
        for ch in self.stats.chains.values():
            if ch.skill_ids == list(skill_ids):
                return ch
        return None

    @property
    def placeholder_stats(self) -> bool:
        return bool(self.stats and self.stats.placeholder)

    def public_view(self) -> list[dict]:
        """JSON for GET /api/registry: every skill with its live stats."""
        return [s.model_dump(mode="json") for s in self.skills.values()]


def load_registry(path: Path, stats: Path | None, scene_id: str = "libero_goal_v1") -> Registry:
    """Load skills.yaml and merge the stats file if it exists."""
    sf = load_stats(stats) if stats is not None and Path(stats).exists() else None
    specs = load_skills(path, sf)
    return Registry.from_specs(specs, sf, scene_id)


# ---------------------------------------------------------------- deterministic menu (5.7 step 1)

MENU_SKILL_SLOTS = 2  # plus "Call my caregiver" = 3 cards, the R12 maximum (see docs/DEVIATIONS.md)


def deterministic_menu(feasible: list[Feasibility], reg: Registry) -> PlannerOutput:
    """The menu shown when there is no speech and no prefetched suggestion, or the planner fails.

    Top feasible skills by `wilson_lo` (ties broken by registry order), excluding high-risk
    skills, plus "Call my caregiver". The result goes through `validate_proposal`
    like any planner output, so it obeys every rule.
    """
    order = {sid: i for i, sid in enumerate(reg.skills)}
    candidates = [
        f
        for f in feasible
        if f.ok and f.risk != "high" and f.kind != "home" and (spec := reg.get(f.skill_id)) is not None and spec.offer
    ]
    candidates.sort(key=lambda f: (-(f.success.wilson_lo if f.success else 0.0), order.get(f.skill_id, 0)))
    options = [
        PlannerOption(kind="skill", skill_id=f.skill_id, plan=None, home_action=None, why="")
        for f in candidates[:MENU_SKILL_SLOTS]
    ]
    if reg.home_skill("call_caregiver") is not None:
        options.append(PlannerOption(kind="home", skill_id=None, plan=None, home_action="call_caregiver", why=""))
    return PlannerOutput(intent="none", clarify_question=None, options=options)
