"""Goal swapping (design 1.3, 4.2.4 skills.py).

All ten LIBERO-Goal tasks share one scene, so switching skills inside a running episode needs
two assignments and no reset:
  * the instruction the policy sees (`env._task_description` in Isaac-GR00T's LiberoEnv, which
    `_process_observation` copies into every observation), and
  * the goal `_check_success()` evaluates (`parsed_problem["goal_state"]`).
"""

from __future__ import annotations

from typing import Any


def _problem_env(env: Any):
    """Find the LIBERO problem env (the object holding parsed_problem) behind any wrapper."""
    for path in (("problem",), ("_env", "env", "env"), ("_env", "env"), ("env",)):
        obj = env
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "parsed_problem"):
            return obj
    if hasattr(env, "parsed_problem"):
        return env
    raise TypeError(f"cannot find parsed_problem behind {type(env).__name__}")


def swap_skill(env: Any, instruction: str, goal: list) -> None:
    """Make `env` pursue `goal` under `instruction` from its current state.

    Accepts a `nod_sim.scene.LiberoSession`, an Isaac-GR00T `LiberoEnv` (`_env` is the
    OffScreenRenderEnv, `_env.env` the problem env) or a bare OffScreenRenderEnv.
    """
    pe = _problem_env(env)
    pe.parsed_problem["goal_state"] = [list(s) for s in goal]
    if hasattr(env, "_task_description"):  # Isaac-GR00T LiberoEnv
        env._task_description = instruction
    if hasattr(env, "instruction"):  # LiberoSession
        env.instruction = instruction
