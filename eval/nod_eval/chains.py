"""The evaluated chains (design 1.4). Skill ids are registry ids (registry/skills.yaml).

| Chain | Steps | Why |
|---|---|---|
| A, independent (primary) | bottle_on_rack, open_middle_drawer, cheese_in_bowl, plate_to_stove_front | every skill acts on objects untouched by earlier steps |
| B, dependent ("snack") | bowl_on_plate, cheese_in_bowl, bowl_to_user | the bowl is no longer where training put it |

Between steps the scripted `return_home` runs (design 1.4 mitigation 1), as it does live.
"""

from __future__ import annotations

CHAINS: dict[str, list[str]] = {
    "A": ["bottle_on_rack", "open_middle_drawer", "cheese_in_bowl", "plate_to_stove_front"],
    "B": ["bowl_on_plate", "cheese_in_bowl", "bowl_to_user"],
}


def parse_chains(text: str | None) -> dict[str, list[str]]:
    """'A,B' -> {'A': [...], 'B': [...]}. Also accepts an ad-hoc chain 'name=skill+skill+...'."""
    out: dict[str, list[str]] = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, steps = part.split("=", 1)
            out[name.strip()] = [s.strip() for s in steps.split("+") if s.strip()]
        elif part in CHAINS:
            out[part] = list(CHAINS[part])
        else:
            raise ValueError(f"unknown chain {part!r}; known: {sorted(CHAINS)} or name=skill+skill")
    return out


def available_prefix(steps: list[str], available: set[str]) -> list[str]:
    """The leading steps that can run. A chain cannot skip a step, so it stops at the first gap."""
    out = []
    for s in steps:
        if s not in available:
            break
        out.append(s)
    return out
