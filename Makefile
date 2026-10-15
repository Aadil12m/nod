# Nod developer tasks. `make` or `make help` lists them.
# Python venvs live inside each package (protocol/.venv, gateway/.venv, sim/.venv, viewer/.venv,
# training/.venv, infra/.venv) and are created with uv (https://docs.astral.sh/uv/). Node 20+ is
# needed for the PWA tests. Every test-* target creates the venvs it needs on first use (that
# needs network access to PyPI and npm), so `make test` works from a clean checkout.
# The browser tests need a Chromium: PLAYWRIGHT_BROWSERS_PATH or CHROME_PATH (see pwa/README.md).

SHELL := /bin/bash
.DEFAULT_GOAL := help

UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
PY311 ?= 3.11
NPM ?= npm

PROTO_PY := protocol/.venv/bin/python
GW_PY := gateway/.venv/bin/python
INFRA_PY := infra/.venv/bin/python
SIM_PY := sim/.venv/bin/python
VIEWER_PY := viewer/.venv/bin/python
TRAIN_PY := training/.venv/bin/python
PWA_MODULES := pwa/node_modules/.package-lock.json

.PHONY: help setup setup-protocol setup-gateway setup-pwa setup-infra setup-sim setup-viewer setup-training \
        test test-protocol test-gateway test-pwa test-pwa-e2e test-e2e test-infra test-sim test-eval \
        test-viewer test-training \
        mock mock-docker smoke screenshots schema schema-check vendor fetch-mock \
        lint compose-check spike-help clean

help: ## list targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------- setup
setup: setup-protocol setup-gateway setup-pwa setup-infra setup-sim setup-viewer setup-training ## create every venv and install the PWA dev tools

setup-protocol: ## protocol/.venv with test extras
	$(UV) venv -q --allow-existing -p $(PY311) protocol/.venv
	$(UV) pip install -q -p $(PROTO_PY) -e "protocol[numpy,fast,test]"

setup-gateway: ## gateway/.venv (mock mode and tests; add the asr extra by hand for Parakeet)
	$(UV) venv -q --allow-existing -p $(PY311) gateway/.venv
	$(UV) pip install -q -p $(GW_PY) -e "protocol[numpy,fast]" -e "gateway[test]"

setup-pwa: ## npm dev tools for the PWA tests (no build step)
	cd pwa && $(NPM) ci --no-audit --no-fund

setup-infra: ## infra/.venv: pytest, pyyaml, ruff, shellcheck and actionlint wheels, openai
	$(UV) venv -q --allow-existing -p $(PY311) infra/.venv
	$(UV) pip install -q -p $(INFRA_PY) -r infra/requirements-test.txt -e "protocol"

setup-sim: ## sim/.venv for the CPU-only sim and eval tests (the real LIBERO venv is in sim/README.md)
	$(UV) venv -q --allow-existing -p $(PY311) sim/.venv
	$(UV) pip install -q -p $(SIM_PY) -e "protocol[numpy,fast]" -e "sim[test,groot,media]" -e "eval[test]"

setup-viewer: ## viewer/.venv: MuJoCo 3.14, mjviser, viser (the 3D view for mock mode and the presenter)
	$(UV) venv -q --allow-existing -p $(PY311) viewer/.venv
	$(UV) pip install -q -p $(VIEWER_PY) -e "protocol[numpy,fast]" -e "viewer[test]"

setup-training: ## training/.venv: LeRobot conversion and its tests (gen_demos itself needs the LIBERO venv)
	$(UV) venv -q --allow-existing -p $(PY311) training/.venv
	$(UV) pip install -q -p $(TRAIN_PY) -r training/requirements.txt -e "protocol[numpy]" -e sim

# First-use bootstrap: a test target whose venv is missing creates it.
$(PROTO_PY):
	@$(MAKE) --no-print-directory setup-protocol
$(GW_PY):
	@$(MAKE) --no-print-directory setup-gateway
$(INFRA_PY):
	@$(MAKE) --no-print-directory setup-infra
$(SIM_PY):
	@$(MAKE) --no-print-directory setup-sim
$(VIEWER_PY):
	@$(MAKE) --no-print-directory setup-viewer
$(TRAIN_PY):
	@$(MAKE) --no-print-directory setup-training
$(PWA_MODULES):
	@$(MAKE) --no-print-directory setup-pwa

# ---------------------------------------------------------------------------- tests
test: test-protocol test-gateway test-pwa test-pwa-e2e test-e2e test-sim test-eval test-viewer test-training test-infra ## run every CPU test suite (creates missing venvs)

test-protocol: | $(PROTO_PY) ## protocol: pytest and the schema check
	$(PROTO_PY) -m pytest -q protocol/tests
	cd protocol && .venv/bin/python gen_schema.py --check

test-gateway: | $(GW_PY) $(SIM_PY) ## gateway: pytest (mock end-to-end, and the real sim runner on FakeEnv)
	cd gateway && .venv/bin/python -m pytest -q

test-pwa: | $(PWA_MODULES) ## PWA: node --test unit tests
	cd pwa && $(NPM) test --silent

test-pwa-e2e: | $(PWA_MODULES) ## PWA: Chromium end-to-end against the stub gateway (needs a Chromium; CHROME_PATH)
	cd pwa && node tests/e2e.mjs

test-e2e: | $(PWA_MODULES) $(GW_PY) ## full mock end-to-end: Chromium + real PWA + real gateway (--mock)
	cd pwa && node tests/e2e_full.mjs

screenshots: | $(PWA_MODULES) $(GW_PY) ## re-take docs/screenshots/*.png with the full mock end-to-end test
	cd pwa && node tests/e2e_full.mjs --screenshots ../docs/screenshots

test-infra: | $(INFRA_PY) ## infra: compose, YAML, shell scripts, dry runs, workflows, .env.example, smoke script
	$(INFRA_PY) -m pytest -q infra/tests

test-sim: | $(SIM_PY) ## sim: CPU tests on FakeEnv (the real-LIBERO file skips without LIBERO)
	cd sim && .venv/bin/python -m pytest -q

test-eval: | $(SIM_PY) ## eval: nod_eval on FakeEnv (runs in sim/.venv)
	$(SIM_PY) -m pytest -q eval/tests

test-viewer: | $(VIEWER_PY) ## viewer: scene loading, pose feed, patching (uses recordings/ if present)
	$(VIEWER_PY) -m pytest -q viewer/tests

test-training: | $(TRAIN_PY) ## training: demo layout and LeRobot conversion (Isaac-GR00T checks skip without ISAAC_GR00T_DIR)
	$(TRAIN_PY) -m pytest -q training/tests

smoke: | $(GW_PY) ## start the mock gateway as a process and probe its HTTP endpoints
	infra/scripts/smoke_mock.sh

# ---------------------------------------------------------------------------- run
mock: ## mock mode without Docker: gateway (and viewer if available) on http://localhost:8000
	UV=$(UV) infra/scripts/mock.sh

laptop: ## zero-cost live mode: real LIBERO sim (scripted arm) on CPU + Nemotron on Token Factory
	UV=$(UV) infra/scripts/laptop.sh

eval-scripted: ## measure the scripted arm on the real scene (CPU) and write registry/stats/scripted.json
	LIBERO_CONFIG_PATH=$(CURDIR)/sim/.venv-libero/libero_config MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
	  sim/.venv-libero/bin/python -m nod_eval.run --registry registry/skills.yaml --episodes 20 --chains A,B \
	  --out results/scripted --actor scripted --lazy-cameras
	sim/.venv-libero/bin/python -m nod_eval.publish --in results/scripted --out registry/stats/scripted.json

mock-docker: ## mock mode with Docker Compose
	docker compose --profile mock up --build

# ---------------------------------------------------------------------------- assets
schema: ## regenerate protocol/schema/*.json from nod_protocol
	cd protocol && .venv/bin/python gen_schema.py

schema-check: ## fail if protocol/schema is stale
	cd protocol && .venv/bin/python gen_schema.py --check

vendor: ## download pinned MediaPipe, face model and TF.js into pwa/vendor (hash-checked)
	sh pwa/scripts/vendor.sh

fetch-mock: ## download the mock library and viewer scene into recordings/ (GitHub release)
	infra/scripts/fetch_mock.sh

# ---------------------------------------------------------------------------- checks
lint: | $(INFRA_PY) ## ruff (Python), shellcheck (scripts), actionlint (workflows), compose config
	infra/.venv/bin/ruff check protocol gateway sim viewer eval training infra docs/runbooks
	cd gateway && ../infra/.venv/bin/ruff format --check .
	@if [ -x infra/.venv/bin/shellcheck ]; then \
	  infra/.venv/bin/shellcheck -x infra/nebius/*.sh infra/scripts/*.sh docker/*.sh; \
	else echo "lint: shellcheck skipped (make setup-infra)"; fi
	@if [ -x infra/.venv/bin/actionlint ]; then \
	  infra/.venv/bin/actionlint .github/workflows/*.yml; \
	else echo "lint: actionlint skipped (make setup-infra)"; fi
	@$(MAKE) --no-print-directory compose-check

compose-check: ## docker compose config for every profile (no daemon needed)
	@for p in mock live hass eval; do \
	  docker compose -f compose.yaml --profile $$p config --quiet || exit 1; \
	  echo "compose profile $$p: ok"; \
	done

spike-help: ## print the Day-1 GPU spike checklist (docs/runbooks/day1-spike.md)
	@echo "Day-1 GPU spike (design section 8). Full checklist: docs/runbooks/day1-spike.md"
	@echo
	@echo "  0. Laptop: HF access to nvidia/Cosmos-Reason2-2B and -8B, Nebius card + uk-south2 quota,"
	@echo "     Token Factory key (docs/TEAM-TODO.md)"
	@echo "  1. VM:     export NEBIUS_PROJECT_ID=...; infra/nebius/create_vm.sh; infra/nebius/security_group.sh"
	@echo "  2. VM:     install Isaac-GR00T + LIBERO venv + checkpoint (runbook 8.2)"
	@echo "  3. G0:     official harness, 3 skills x 20 episodes, each >= 17/20"
	@echo "  4. G1:     docs/runbooks/chain_test.py --chain A --trials 20 (full chain >= 14/20)"
	@echo "  5. G2:     policy call p95 <= 150 ms, env.step p95 <= 30 ms"
	@echo "  6. G3:     NEBIUS_API_KEY=... uv run --no-project --with openai python docs/runbooks/nemotron_smoke.py"
	@echo "  7. Record: docs/runbooks/day1-results.md, copy ~/nod_spike, STOP THE VM"
	@echo
	@echo "  Dry runs (no CLI, no network): infra/nebius/create_vm.sh --dry-run"

clean: ## remove caches (keeps venvs, node_modules, recordings)
	find . -name __pycache__ -type d -prune -not -path "*/.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache */.pytest_cache .ruff_cache */.ruff_cache
