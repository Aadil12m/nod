"""Shared helpers for the infra tests: repo paths, running scripts, fake CLIs on PATH."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
VENV_BIN = Path(sys.executable).parent


def which(tool: str) -> str | None:
    """Find a tool in this venv first (shellcheck-py, actionlint-py), then on PATH."""
    p = VENV_BIN / tool
    if p.exists():
        return str(p)
    return shutil.which(tool)


def run(cmd: list[str], env: dict[str, str] | None = None, cwd: Path = REPO, timeout: float = 60,
        check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a command with a clean, predictable environment plus `env`."""
    base = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
    }
    base.update(env or {})
    cp = subprocess.run(cmd, cwd=cwd, env=base, capture_output=True, text=True, timeout=timeout)
    if check and cp.returncode != 0:
        raise AssertionError(f"{cmd} failed ({cp.returncode}):\n{cp.stdout}\n{cp.stderr}")
    return cp


def write_exe(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def repo() -> Path:
    return REPO
