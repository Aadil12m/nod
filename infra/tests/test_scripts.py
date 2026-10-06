"""Shell scripts (syntax, shellcheck), fetch_mock.sh, the mock smoke script, the Makefile and workflows."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import pytest
import yaml
from conftest import REPO, run, which

SCRIPTS = sorted(
    [*REPO.glob("infra/nebius/*.sh"), *REPO.glob("infra/scripts/*.sh"), *REPO.glob("docker/*.sh")]
)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(REPO)))
def test_bash_syntax(script):
    cp = run(["bash", "-n", str(script)])
    assert cp.returncode == 0, cp.stderr


def test_scripts_are_executable_and_strict():
    for s in SCRIPTS:
        text = s.read_text()
        assert text.startswith("#!/usr/bin/env bash"), s
        if s.name != "lib.sh":
            assert "set -euo pipefail" in text, s
            assert s.stat().st_mode & 0o111, f"{s} is not executable"


@pytest.mark.skipif(which("shellcheck") is None, reason="shellcheck not installed (pip install shellcheck-py)")
def test_shellcheck_clean():
    cp = run([which("shellcheck"), "-x", *map(str, SCRIPTS)])
    assert cp.returncode == 0, cp.stdout


# ------------------------------------------------------------------------ fetch_mock.sh

FETCH = str(REPO / "infra" / "scripts" / "fetch_mock.sh")


def _make_library(root: Path) -> Path:
    rec = root / "recordings"
    run_dir = rec / "r_01TEST"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('{"skill_id": "bowl_on_plate", "success": true}')
    (run_dir / "poses.npz").write_bytes(b"not really npz")
    (rec / "scene_libero_goal_v1.mjb").write_bytes(b"mjb")
    (rec / "README.md").write_text("kept in git")
    return rec


def test_fetch_mock_pack_then_fetch_round_trip(tmp_path):
    src = _make_library(tmp_path / "src")
    release = tmp_path / "release"
    cp = run(["bash", FETCH, "--pack", str(release)], env={"NOD_RECORDINGS_DIR": str(src)})
    assert cp.returncode == 0, cp.stderr
    with tarfile.open(release / "nod-mock.tar.gz") as tf:
        names = tf.getnames()
    assert "./README.md" not in names and "./r_01TEST/manifest.json" in names

    dest = tmp_path / "dest"
    cp = run(["bash", FETCH], env={"NOD_RECORDINGS_DIR": str(dest), "NOD_MOCK_BASE_URL": release.as_uri()})
    assert cp.returncode == 0, cp.stderr
    assert (dest / "r_01TEST" / "manifest.json").is_file()
    assert "1 recorded runs" in cp.stdout and "scene_libero_goal_v1.mjb present" in cp.stdout


def test_fetch_mock_rejects_bad_checksum(tmp_path):
    src = _make_library(tmp_path / "src")
    release = tmp_path / "release"
    run(["bash", FETCH, "--pack", str(release)], env={"NOD_RECORDINGS_DIR": str(src)}, check=True)
    (release / "SHA256SUMS").write_text(hashlib.sha256(b"other").hexdigest() + "  nod-mock.tar.gz\n")
    dest = tmp_path / "dest"
    cp = run(["bash", FETCH], env={"NOD_RECORDINGS_DIR": str(dest), "NOD_MOCK_BASE_URL": release.as_uri()})
    assert cp.returncode == 1 and "checksum mismatch" in cp.stderr
    assert not dest.exists()


def test_fetch_mock_rejects_path_traversal(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    evil = tmp_path / "evil.txt"
    evil.write_text("x")
    with tarfile.open(release / "nod-mock.tar.gz", "w:gz") as tf:
        tf.add(evil, arcname="../escape.txt")
    digest = hashlib.sha256((release / "nod-mock.tar.gz").read_bytes()).hexdigest()
    (release / "SHA256SUMS").write_text(f"{digest}  nod-mock.tar.gz\n")
    cp = run(["bash", FETCH], env={"NOD_RECORDINGS_DIR": str(tmp_path / "dest"), "NOD_MOCK_BASE_URL": release.as_uri()})
    assert cp.returncode == 1 and "refusing archive" in cp.stderr
    assert not (tmp_path / "escape.txt").exists()


def test_fetch_mock_missing_release_explains_mock_still_works(tmp_path):
    cp = run(["bash", FETCH], env={"NOD_RECORDINGS_DIR": str(tmp_path / "d"),
                                   "NOD_MOCK_BASE_URL": (tmp_path / "nothing").as_uri()})
    assert cp.returncode == 1
    assert "Mock mode still runs" in cp.stderr


# ------------------------------------------------------------------------ the mock gateway as a process

GATEWAY_BIN = REPO / "gateway" / ".venv" / "bin" / "nod-gateway"


@pytest.mark.skipif(not GATEWAY_BIN.exists(), reason="gateway/.venv not set up (make setup-gateway)")
def test_smoke_mock_gateway():
    cp = run(["bash", str(REPO / "infra" / "scripts" / "smoke_mock.sh"), str(GATEWAY_BIN)], timeout=120)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "smoke: OK" in cp.stdout


# ------------------------------------------------------------------------ Makefile

REQUIRED_TARGETS = ["setup", "test", "mock", "schema", "vendor", "fetch-mock", "lint", "spike-help"]


@pytest.mark.parametrize("target", REQUIRED_TARGETS)
def test_make_targets_exist(target):
    cp = run(["make", "-n", target])
    assert cp.returncode == 0, cp.stderr


def test_make_help_and_spike_help():
    help_out = run(["make", "help"], check=True).stdout
    for t in REQUIRED_TARGETS:
        assert f"  {t} " in help_out, t
    spike = run(["make", "spike-help"], check=True).stdout
    assert "docs/runbooks/day1-spike.md" in spike and "nemotron_smoke.py" in spike


# ------------------------------------------------------------------------ GitHub workflows

def _workflow(name: str) -> dict:
    return yaml.safe_load((REPO / ".github" / "workflows" / name).read_text())


def test_ci_covers_protocol_gateway_pwa_and_lint():
    wf = _workflow("ci.yml")
    assert {"protocol", "gateway", "pwa", "infra"} <= set(wf["jobs"])
    text = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "ruff" in text and "make test-pwa" in text and "make test-protocol" in text


def test_nightly_is_disabled_by_default_and_calls_job_script():
    wf = _workflow("nightly.yml")
    on = wf[True] if True in wf else wf["on"]  # YAML 1.1 reads the bare key `on` as True
    assert on["schedule"][0]["cron"]
    job = wf["jobs"]["eval"]
    assert "vars.NOD_NIGHTLY_ENABLED == 'true'" in job["if"]
    steps = " ".join(str(s.get("run", "")) for s in job["steps"])
    assert "infra/nebius/job_nightly.sh" in steps
    for secret in ("NEBIUS_SA_JSON", "NEBIUS_PROJECT_ID", "NOD_EVAL_IMAGE", "NOD_BUCKET_ID", "NOD_HF_SECRET"):
        assert f"secrets.{secret}" in (REPO / ".github" / "workflows" / "nightly.yml").read_text()


@pytest.mark.skipif(which("actionlint") is None, reason="actionlint not installed (pip install actionlint-py)")
def test_actionlint_clean():
    cp = run([which("actionlint"), *map(str, sorted((REPO / ".github" / "workflows").glob("*.yml")))])
    assert cp.returncode == 0, cp.stdout + cp.stderr
