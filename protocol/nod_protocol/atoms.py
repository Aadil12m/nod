"""Ground atoms and literals in registry form, plus a STRIPS-style symbolic rollout.

An atom is `pred(arg,arg,...)`: lowercase predicate, lowercase args, no spaces,
e.g. "on(akita_black_bowl_1,plate_1)" or "turnon(flat_stove_1)". Predicates are
the lowercased LIBERO BDDL predicates (On, In, Open, Close, TurnOn, TurnOff).

A literal is an atom or "not <atom>". `pre` lists literals that must hold;
`effects` lists literals that become true after a successful run (a negated
effect is a delete effect, e.g. the bowl leaves the plate).

Unknown atoms (absent from the world's `atoms` map) never satisfy a literal,
positive or negated. That is the safe reading for preconditions.

These are pure functions with no I/O, usable from the gateway validator (R4, R5,
R10), the sim oracle and the tests.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

_ATOM_RE = re.compile(r"^([a-z][a-z0-9_]*)\(([a-z0-9_]+(?:,[a-z0-9_]+)*)\)$")
_NOT_PREFIX = "not "


@dataclass(frozen=True)
class Lit:
    """A parsed literal: `atom` is canonical, `neg` is True for "not atom"."""

    atom: str
    neg: bool
    pred: str
    args: tuple[str, ...]

    def __str__(self) -> str:
        return f"not {self.atom}" if self.neg else self.atom


def parse_atom(text: str) -> tuple[str, tuple[str, ...]]:
    """Parse a canonical atom into (pred, args). Raises ValueError if malformed."""
    m = _ATOM_RE.match(text)
    if not m:
        raise ValueError(f"malformed atom {text!r}; expected pred(arg,arg) in lowercase with no spaces")
    return m.group(1), tuple(m.group(2).split(","))


def parse_literal(text: str) -> Lit:
    """Parse "pred(a,b)" or "not pred(a,b)" (exactly one space after "not")."""
    neg = text.startswith(_NOT_PREFIX)
    atom = text[len(_NOT_PREFIX):] if neg else text
    pred, args = parse_atom(atom)
    return Lit(atom=atom, neg=neg, pred=pred, args=args)


def holds(literal: str | Lit, atoms: Mapping[str, bool]) -> bool:
    """True if the literal is satisfied. Unknown atoms satisfy nothing."""
    lit = literal if isinstance(literal, Lit) else parse_literal(literal)
    value = atoms.get(lit.atom)
    if value is None:
        return False
    return (not value) if lit.neg else bool(value)


def unmet(literals: Iterable[str], atoms: Mapping[str, bool]) -> list[str]:
    """The literals that do not hold (empty list means all hold)."""
    return [lit for lit in literals if not holds(lit, atoms)]


def apply_effects(atoms: Mapping[str, bool], effects: Iterable[str]) -> dict[str, bool]:
    """Return a new atom map with the effects applied (positive -> True, negated -> False)."""
    out = dict(atoms)
    for text in effects:
        lit = parse_literal(text)
        out[lit.atom] = not lit.neg
    return out


def atoms_of(literals: Iterable[str]) -> set[str]:
    """The set of atoms mentioned by some literals (sign dropped)."""
    return {parse_literal(t).atom for t in literals}


@dataclass
class Rollout:
    """Result of `rollout`. When `ok` is False, `failed_step` and `unmet` say why."""

    ok: bool
    failed_step: int | None = None
    failed_skill: str | None = None
    unmet: list[str] = field(default_factory=list)
    atoms: dict[str, bool] = field(default_factory=dict)  # world after the last applied step


def rollout(
    steps: Iterable[tuple[str, Iterable[str], Iterable[str]]],
    atoms: Mapping[str, bool],
) -> Rollout:
    """Symbolic plan check (R10).

    `steps` yields (skill_id, pre, effects). Each step's `pre` must hold in the
    world produced by the initial atoms plus all earlier effects.
    """
    state = dict(atoms)
    for i, (skill_id, pre, effects) in enumerate(steps):
        missing = unmet(pre, state)
        if missing:
            return Rollout(ok=False, failed_step=i, failed_skill=skill_id, unmet=missing, atoms=state)
        state = apply_effects(state, effects)
    return Rollout(ok=True, atoms=state)
