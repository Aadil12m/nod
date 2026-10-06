"""infra/nebius scripts (design 8.1, 3.1): required settings, dry runs, and real runs against fake CLIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import REPO, run, write_exe

NEB = REPO / "infra" / "nebius"
PUBKEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGZha2VrZXlmb3J0ZXN0cw nod@test"


@pytest.fixture
def keyfile(tmp_path: Path) -> Path:
    p = tmp_path / "id_ed25519.pub"
    p.write_text(PUBKEY + "\n")
    return p


# ------------------------------------------------------------------------ create_vm.sh

def test_create_vm_requires_project_id():
    cp = run(["bash", str(NEB / "create_vm.sh"), "--dry-run"])
    assert cp.returncode == 1
    assert "NEBIUS_PROJECT_ID is not set" in cp.stderr and "TODO" in cp.stderr


def test_create_vm_dry_run_renders_command_and_cloud_init(keyfile):
    cp = run(["bash", str(NEB / "create_vm.sh"), "--dry-run"],
             env={"NEBIUS_PROJECT_ID": "project-e00test", "SSH_PUBKEY_FILE": str(keyfile),
                  "NOD_REPO_URL": "https://github.com/example/nod.git"})
    assert cp.returncode == 0, cp.stderr
    cmd, _, cloud_init = cp.stdout.partition("--- cloud-init ---\n")
    for part in ("nebius compute instance create", "--parent-id project-e00test", "--resources-platform gpu-rtx6000-a",
                 "--resources-preset 1gpu-24vcpu-218gb", "ubuntu24.04-cuda13.0",
                 "--boot-disk-managed-disk-size-gibibytes 300", "TODO-subnet-id"):
        assert part in cmd, part
    assert "--preemptible" not in cmd  # on-demand by default (demos and recordings)
    doc = yaml.safe_load(cloud_init)
    assert cloud_init.startswith("#cloud-config")
    assert doc["users"][0]["name"] == "nod"
    assert doc["users"][0]["ssh_authorized_keys"] == [PUBKEY]
    assert any("git clone https://github.com/example/nod.git" in " ".join(c) for c in doc["runcmd"])


def test_create_vm_preemptible_flags(keyfile):
    cp = run(["bash", str(NEB / "create_vm.sh"), "--dry-run"],
             env={"NEBIUS_PROJECT_ID": "p", "SSH_PUBKEY_FILE": str(keyfile), "PREEMPTIBLE": "1"})
    assert cp.returncode == 0
    assert "--preemptible-on-preemption stop" in cp.stdout and "--follows-spot-price" in cp.stdout


def test_create_vm_rejects_quotes_in_inputs(tmp_path):
    bad = tmp_path / "bad.pub"
    bad.write_text('ssh-ed25519 AAAA" evil\n')
    cp = run(["bash", str(NEB / "create_vm.sh"), "--dry-run"],
             env={"NEBIUS_PROJECT_ID": "p", "SSH_PUBKEY_FILE": str(bad)})
    assert cp.returncode == 1 and "unexpected character" in cp.stderr


def test_create_vm_live_path_with_fake_cli(tmp_path, keyfile):
    """Not a dry run: a fake `nebius` returns ids; the script must print VM_ID, VM_IP and sslip.io names."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    write_exe(bindir / "nebius", f"""#!/usr/bin/env bash
echo "$*" >> {log}
case "$1 $2 $3" in
  "vpc subnet list") echo subnet-e00abc ;;
  "compute instance create") echo computeinstance-e00vm ;;
  "compute instance get") echo '{{"status":{{"network_interfaces":[{{"public_ip_address":{{"address":"203.0.113.7/32"}}}}]}}}}' ;;
  *) echo "unexpected: $*" >&2; exit 9 ;;
esac
""")
    cp = run(["bash", str(NEB / "create_vm.sh")],
             env={"NEBIUS_PROJECT_ID": "project-e00test", "SSH_PUBKEY_FILE": str(keyfile),
                  "PATH": f"{bindir}:/usr/bin:/bin"})
    assert cp.returncode == 0, cp.stderr
    assert "VM_ID=computeinstance-e00vm" in cp.stdout
    assert "VM_IP=203.0.113.7" in cp.stdout
    assert "NOD_DOMAIN=nod.203-0-113-7.sslip.io" in cp.stdout
    calls = log.read_text()
    assert "--parent-id project-e00test" in calls and "subnet-e00abc" in calls


# ------------------------------------------------------------------------ security_group.sh

def test_security_group_dry_run_opens_only_22_and_443():
    cp = run(["bash", str(NEB / "security_group.sh"), "--dry-run"],
             env={"NEBIUS_PROJECT_ID": "p", "SSH_CIDR": "198.51.100.4/32"})
    assert cp.returncode == 0, cp.stderr
    ingress = [ln for ln in cp.stdout.splitlines() if "--ingress-destination-ports" in ln]
    ports = sorted(ln.split("--ingress-destination-ports ")[1].split()[0] for ln in ingress)
    assert ports == ["22", "443"]
    ssh = next(ln for ln in ingress if ln.rstrip().endswith(" 22"))
    assert "198.51.100.4/32" in ssh
    assert "8000" not in cp.stdout.split("Attach the group")[0]


# ------------------------------------------------------------------------ job_nightly.sh

JOB_ENV = {
    "NEBIUS_PROJECT_ID": "project-e00test",
    "NOD_EVAL_IMAGE": "cr.uk-south2.nebius.cloud/e00reg/nod-eval:abc123",
    "NOD_BUCKET_ID": "bucket-e00res",
    "NOD_HF_SECRET": "mbsec-e00hf",
    "NOD_EVAL_DATE": "2026-10-12",
    "NOD_ALLOW_MAINTENANCE_WINDOW": "1",  # keep the tests independent of the wall clock
}


@pytest.mark.parametrize("missing", ["NEBIUS_PROJECT_ID", "NOD_EVAL_IMAGE", "NOD_BUCKET_ID", "NOD_HF_SECRET"])
def test_job_requires_each_setting(missing):
    env = {k: v for k, v in JOB_ENV.items() if k != missing}
    cp = run(["bash", str(NEB / "job_nightly.sh"), "--dry-run"], env=env)
    assert cp.returncode == 1
    assert f"{missing} is not set" in cp.stderr and "TODO" in cp.stderr


def test_job_dry_run_command():
    cp = run(["bash", str(NEB / "job_nightly.sh"), "--dry-run", "--wait", "--fetch-stats", "out.json"],
             env={**JOB_ENV, "NOD_CKPT_BUCKET_ID": "bucket-e00ckpt", "NOD_BUCKET_NAME": "nod-results"})
    assert cp.returncode == 0, cp.stderr
    out = cp.stdout
    for part in ("nebius ai job create", "--parent-id project-e00test", "--platform gpu-rtx6000-a",
                 "--preset 1gpu-24vcpu-218gb", "--timeout 2h", "--restart-policy on-failure", "--preemptible",
                 "--volume bucket-e00res:/mnt/results:rw", "--volume bucket-e00ckpt:/ckpt:rw",
                 "--env-secret HF_TOKEN=mbsec-e00hf", "--env NOD_EVAL_DATE=2026-10-12", "--env NOD_EVAL_SMOKE=0",
                 "nebius ai job logs", "s3://nod-results/2026-10-12/stats.json",
                 "--endpoint-url https://storage.uk-south2.nebius.cloud"):
        assert part in out, part
    assert "--name nod-eval-20261012-" in out


def test_job_smoke_and_on_demand():
    cp = run(["bash", str(NEB / "job_nightly.sh"), "--dry-run", "--smoke"], env={**JOB_ENV, "PREEMPTIBLE": "0"})
    assert cp.returncode == 0
    assert "--env NOD_EVAL_SMOKE=1" in cp.stdout and "nod-eval-smoke-" in cp.stdout
    assert "--preemptible" not in cp.stdout


def _fake_nebius(tmp_path: Path, final_state: str) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    counter = tmp_path / "polls"
    counter.write_text("0")
    write_exe(bindir / "nebius", f"""#!/usr/bin/env bash
case "$1 $2 $3" in
  "ai job create") echo aijob-e00job ;;
  "ai job get")
    n=$(cat {counter}); echo $((n+1)) > {counter}
    if [ "$n" -lt 2 ]; then echo '{{"status":{{"state":"RUNNING"}}}}'; else echo '{{"status":{{"state":"{final_state}"}}}}'; fi ;;
  "ai job logs") echo "[eval] done: /mnt/results/2026-10-12/stats.json" ;;
  *) echo "unexpected: $*" >&2; exit 9 ;;
esac
""")
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"bowl_on_plate": {"alone": {"k": 19, "n": 20}}, "job_id": "aijob-e00job"}))
    write_exe(bindir / "aws", f"""#!/usr/bin/env bash
# aws s3 cp s3://bucket/date/stats.json DEST --endpoint-url URL
[ "$1 $2" = "s3 cp" ] || exit 9
cp {stats} "$4"
""")
    return bindir


def test_job_wait_and_fetch_with_fake_cli(tmp_path):
    bindir = _fake_nebius(tmp_path, "COMPLETED")
    dest = tmp_path / "registry" / "stats" / "latest.json"
    gh_out = tmp_path / "gh_output"
    cp = run(["bash", str(NEB / "job_nightly.sh"), "--wait", "--fetch-stats", str(dest)],
             env={**JOB_ENV, "PATH": f"{bindir}:/usr/bin:/bin", "NOD_POLL_S": "0", "NOD_BUCKET_NAME": "nod-results",
                  "GITHUB_OUTPUT": str(gh_out)})
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "JOB_ID=aijob-e00job" in cp.stdout
    assert cp.stdout.count("RUNNING") == 2 and "COMPLETED" in cp.stdout
    assert json.loads(dest.read_text())["job_id"] == "aijob-e00job"
    outputs = dict(line.split("=", 1) for line in gh_out.read_text().splitlines())
    assert outputs == {"job_id": "aijob-e00job", "job_name": outputs["job_name"], "date": "2026-10-12"}


def test_job_wait_reports_failure(tmp_path):
    bindir = _fake_nebius(tmp_path, "FAILED")
    cp = run(["bash", str(NEB / "job_nightly.sh"), "--wait"],
             env={**JOB_ENV, "PATH": f"{bindir}:/usr/bin:/bin", "NOD_POLL_S": "0"})
    assert cp.returncode == 1
    assert "ended in state FAILED" in cp.stderr
    assert "job logs (tail)" in cp.stdout  # logs are printed before failing


def test_cloud_init_template_parses():
    text = (NEB / "cloud-init.yaml").read_text()
    assert text.startswith("#cloud-config")
    doc = yaml.safe_load(text)
    assert doc["users"][0]["sudo"] == "ALL=(ALL) NOPASSWD:ALL"
    assert "__SSH_PUBKEY__" in text and "__NOD_REPO_URL__" in text
