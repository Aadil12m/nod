"""Configuration surface and docs: .env.example, Caddyfile, Home Assistant config, README and runbooks."""

from __future__ import annotations

import ast
import re

import pytest
import yaml
from conftest import REPO

ENV_EXAMPLE = (REPO / ".env.example").read_text()
# Names mentioned in .env.example, commented out or not.
ENV_NAMES = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", ENV_EXAMPLE, flags=re.M))


def _config_fields() -> list[str]:
    tree = ast.parse((REPO / "gateway" / "nod_gateway" / "config.py").read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Config")
    return [s.target.id for s in cls.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]


def test_env_example_lists_every_gateway_setting():
    fields = _config_fields()
    assert "mode" in fields and "hb_timeout_ms" in fields
    missing = []
    for f in fields:
        names = {"NEBIUS_API_KEY", "NOD_TF_API_KEY"} if f == "tf_api_key" else {f"NOD_{f.upper()}"}
        if not names & ENV_NAMES:
            missing.append(f)
    assert not missing, f"add these Config fields to .env.example: {missing}"


def test_env_example_lists_every_compose_and_caddy_variable():
    compose = (REPO / "compose.yaml").read_text()
    used = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)", compose))
    # `- NAME` entries without a value are passed through from the environment / .env
    used |= set(re.findall(r"^\s*-\s+([A-Z][A-Z0-9_]+)\s*$", compose, flags=re.M))
    used |= set(re.findall(r"\{\$([A-Z_][A-Z0-9_]*)", (REPO / "infra" / "caddy" / "Caddyfile").read_text()))
    assert used, "no variables found; the regexes are wrong"
    assert not used - ENV_NAMES, f"missing from .env.example: {sorted(used - ENV_NAMES)}"


def test_env_example_has_no_real_secrets():
    for line in ENV_EXAMPLE.splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if any(s in key for s in ("KEY", "TOKEN", "SECRET")):
            assert value.strip() == "", f"{key} must be empty in .env.example"


def test_caddyfile_routes():
    text = (REPO / "infra" / "caddy" / "Caddyfile").read_text()
    assert "reverse_proxy gateway:8000" in text and "reverse_proxy viewer:8080" in text
    assert "{$NOD_DOMAIN" in text and "{$VIEW_DOMAIN" in text
    assert text.count("{") == text.count("}")


class _HassLoader(yaml.SafeLoader):
    """Home Assistant YAML allows !secret / !include; accept them as plain strings."""


_HassLoader.add_multi_constructor("!", lambda loader, suffix, node: f"!{suffix} {node.value}")


def test_home_assistant_config():
    doc = yaml.load((REPO / "infra" / "hass" / "configuration.yaml").read_text(), Loader=_HassLoader)
    assert "demo" in doc and "api" in doc  # demo lights; REST API for the gateway
    assert doc["http"]["server_port"] == 8123
    assert "default_config" not in doc  # no cloud, discovery or analytics


DOCS = [REPO / "README.md", REPO / "docs" / "TEAM-TODO.md", *sorted((REPO / "docs" / "runbooks").glob("*.md"))]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_docs_relative_links_resolve(doc):
    text = doc.read_text()
    broken = []
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        path = target.split("#", 1)[0]
        if not (doc.parent / path).exists():
            broken.append(target)
    assert not broken, f"broken relative links in {doc.name}: {broken}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_docs_have_no_em_dashes(doc):
    assert "—" not in doc.read_text()


def test_required_docs_exist():
    for rel in ("README.md", "LICENSE", "docs/TEAM-TODO.md", "docs/runbooks/day1-spike.md",
                "docs/runbooks/demo-day.md", "docs/runbooks/nightly.md", "docs/runbooks/day1-results.md",
                "docs/runbooks/nemotron_smoke.py"):
        assert (REPO / rel).is_file(), rel


def test_license_is_full_apache_2():
    text = (REPO / "LICENSE").read_text()
    assert text.lstrip().startswith("Apache License")
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in text
    assert "END OF TERMS AND CONDITIONS" in text


def test_readme_covers_the_required_sections():
    text = (REPO / "README.md").read_text()
    for needle in ("docker compose --profile mock up", "Built on NVIDIA Cosmos", "CC-BY-4.0",
                   "NVIDIA Open Model License", "not a medical device", "Nemotron-3_5-Lightning",
                   "nemotron-3-super-120b-a12b", "GR00T-N1.7-LIBERO", "Cosmos-Reason2-8B", "Parakeet",
                   "Serverless Job", "oracle", "Spittel"):
        assert needle in text, needle
