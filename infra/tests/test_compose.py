"""compose.yaml (design 4.5) and the Dockerfiles: parse, profiles, and the exposure rules of design 3.3."""

from __future__ import annotations

import json
import re
import shutil

import pytest
import yaml
from conftest import REPO, run

DOCKER = shutil.which("docker")
needs_docker = pytest.mark.skipif(DOCKER is None, reason="docker CLI not installed")

GPU_SERVICES = {"policy", "sim", "cosmos", "eval"}


def compose_config(profile: str) -> dict:
    cp = run([DOCKER, "compose", "-f", "compose.yaml", "--profile", profile, "config", "--format", "json"],
             env={"DOCKER_CONFIG": str(REPO / ".no-docker-config")})
    assert cp.returncode == 0, cp.stderr
    return json.loads(cp.stdout)


def test_compose_is_valid_yaml_with_expected_services():
    doc = yaml.safe_load((REPO / "compose.yaml").read_text())
    names = set(doc["services"])
    assert {"policy", "sim", "cosmos", "viewer", "gateway", "homeassistant", "caddy"} <= names
    assert {"gateway-mock", "viewer-mock", "eval"} <= names
    for name, svc in doc["services"].items():
        assert svc.get("profiles"), f"{name} has no profile, so a bare `docker compose up` would start it"


@needs_docker
@pytest.mark.parametrize("profile,expected", [
    ("mock", {"gateway-mock", "viewer-mock"}),
    ("live", {"policy", "sim", "cosmos", "viewer", "gateway", "homeassistant", "caddy"}),
    ("hass", {"homeassistant"}),
    ("eval", {"eval"}),
])
def test_profiles_select_services(profile, expected):
    cfg = compose_config(profile)
    assert set(cfg["services"]) == expected


@needs_docker
def test_mock_profile_is_cpu_only_and_loopback_only():
    cfg = compose_config("mock")
    for name, svc in cfg["services"].items():
        assert "deploy" not in svc or not svc["deploy"].get("resources"), f"{name} reserves a GPU in mock mode"
        for p in svc.get("ports", []):
            assert p.get("host_ip") == "127.0.0.1", f"{name} publishes {p} beyond loopback"
    gw = cfg["services"]["gateway-mock"]
    assert gw["environment"]["NOD_MODE"] == "mock"
    assert gw["build"]["args"]["GATEWAY_EXTRAS"] == ""
    assert any(p["published"] == "8000" and p["target"] == 8000 for p in gw["ports"])


@needs_docker
def test_live_profile_exposes_only_caddy():
    cfg = compose_config("live")
    for name, svc in cfg["services"].items():
        for p in svc.get("ports", []):
            if name == "caddy":
                assert p["target"] in (80, 443)
            else:
                assert p.get("host_ip") == "127.0.0.1", f"{name} publishes {p} on a public interface"
    # The GR00T server has no auth (design 3.3): never published at all.
    assert not cfg["services"]["policy"].get("ports")
    assert not cfg["services"]["sim"].get("ports")
    assert not cfg["services"]["cosmos"].get("ports")
    gw = cfg["services"]["gateway"]
    assert gw["environment"]["NOD_MODE"] == "live"
    assert gw["environment"]["NOD_SIM_CTL"] == "tcp://sim:5560"
    assert gw["environment"]["NOD_COSMOS_URL"].startswith("http://cosmos:8001")


@needs_docker
def test_gpu_services_reserve_one_gpu_with_graphics():
    cfg = compose_config("live")
    for name in GPU_SERVICES - {"eval"}:
        devs = cfg["services"][name]["deploy"]["resources"]["reservations"]["devices"]
        assert devs[0]["count"] == 1 and "graphics" in devs[0]["capabilities"], name
    sim_env = cfg["services"]["sim"]["environment"]
    assert sim_env["MUJOCO_GL"] == "egl" and sim_env["PYOPENGL_PLATFORM"] == "egl"


@needs_docker
def test_cosmos_command_matches_design():
    cmd = compose_config("live")["services"]["cosmos"]["command"]
    joined = " ".join(cmd)
    assert cmd[0] == "nvidia/Cosmos-Reason2-8B"
    for flag in ("--max-model-len=16384", "--gpu-memory-utilization=0.40", "--port=8001", "--reasoning-parser=qwen3"):
        assert flag in joined


def _copy_sources(dockerfile: str) -> list[str]:
    srcs = []
    text = (REPO / "docker" / dockerfile).read_text().replace("\\\n", " ")
    for line in text.splitlines():
        m = re.match(r"\s*COPY\s+(.*)", line)
        if not m or "--from=" in m.group(1):
            continue
        parts = [p for p in m.group(1).split() if not p.startswith("--")]
        srcs.extend(parts[:-1])
    return srcs


# Directories other areas of the repo have not written yet. The image cannot be built until they
# exist; listed here so the gap is visible instead of silently passing.
PENDING = {"eval": "eval/nod_eval (nightly harness) is not in the repo yet"}


@pytest.mark.parametrize("dockerfile", sorted(p.name for p in (REPO / "docker").glob("*.Dockerfile")))
def test_dockerfile_copy_sources_exist(dockerfile):
    missing = [s for s in _copy_sources(dockerfile) if not (REPO / s).exists()]
    pending = [s for s in missing if s in PENDING]
    hard = [s for s in missing if s not in PENDING]
    assert not hard, f"{dockerfile} copies missing paths {hard}"
    if pending:
        pytest.xfail(f"{dockerfile}: {[PENDING[s] for s in pending]}")


def test_dockerignore_keeps_secrets_and_venvs_out():
    ign = (REPO / ".dockerignore").read_text().split()
    for pat in (".env", "**/.venv", "sa.json", "*.pem", ".git"):
        assert pat in ign


def test_compose_build_files_exist():
    doc = yaml.safe_load((REPO / "compose.yaml").read_text())
    for name, svc in doc["services"].items():
        if "build" in svc:
            assert (REPO / svc["build"]["dockerfile"]).is_file(), name
