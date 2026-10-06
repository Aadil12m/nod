#!/usr/bin/env bash
# Create the Nod dev VM: one RTX PRO 6000 (96 GB) in uk-south2 (design 3.1, 8.1).
#
#   export NEBIUS_PROJECT_ID=<uk-south2 project id>     # TODO: Console > project selector, or `nebius profile current`
#   infra/nebius/create_vm.sh                          # on-demand (use this for the spike and demos)
#   PREEMPTIBLE=1 infra/nebius/create_vm.sh            # cheaper, for non-demo work only
#   infra/nebius/create_vm.sh --dry-run                # print the command and cloud-init; change nothing
#
# Environment:
#   NEBIUS_PROJECT_ID  required. The uk-south2 project id.
#   SUBNET_ID          default: the first subnet in the project (`nebius vpc subnet list`).
#   SSH_PUBKEY_FILE    default: ~/.ssh/id_ed25519.pub
#   NOD_REPO_URL       optional git URL cloned to ~/nod on first boot.
#   VM_NAME            default nod-dev.     DISK_GIB  default 300 (network SSD).
#   IMAGE_FAMILY       default ubuntu24.04-cuda13.0.
#   PREEMPTIBLE=1      preemptible instance (60 s notice, then stop). Never for demos or recordings.
#
# Flags follow the Nebius compute quickstart as summarised in docs/research/nebius-ops.md 2.4.
# If one is rejected, check `nebius compute instance create --help`.
# After it prints the IP: lock the network down with infra/nebius/security_group.sh BEFORE
# starting any service (the default security group allows all ingress).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=infra/nebius/lib.sh
source "$HERE/lib.sh"

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

VM_NAME="${VM_NAME:-nod-dev}"
DISK_GIB="${DISK_GIB:-300}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu24.04-cuda13.0}"
SSH_PUBKEY_FILE="${SSH_PUBKEY_FILE:-$HOME/.ssh/id_ed25519.pub}"
NOD_REPO_URL="${NOD_REPO_URL:-}"

need NEBIUS_PROJECT_ID "your uk-south2 project id (Nebius Console, project selector; region must be $NEBIUS_REGION)"

if [[ -f "$SSH_PUBKEY_FILE" ]]; then
  SSH_PUBKEY="$(tr -d '\n' < "$SSH_PUBKEY_FILE")"
elif [[ "$DRY_RUN" == "1" ]]; then
  SSH_PUBKEY="ssh-ed25519 AAAA...TODO-your-public-key"
else
  die "no public key at $SSH_PUBKEY_FILE (set SSH_PUBKEY_FILE, or run ssh-keygen -t ed25519)"
fi
# Both values are pasted into YAML and a sed replacement: refuse quotes, | & and backslashes.
for v in "$SSH_PUBKEY" "$NOD_REPO_URL"; do
  [[ "$v" != *[\"\'\|\&\\]* ]] || die "unexpected character (quote, |, & or backslash) in: $v"
done

# Render cloud-init from the template.
CLOUD_INIT="$(sed -e "s|__SSH_PUBKEY__|${SSH_PUBKEY}|g" -e "s|__NOD_REPO_URL__|${NOD_REPO_URL}|g" "$HERE/cloud-init.yaml")"
[[ "$(head -n 1 <<<"$CLOUD_INIT")" == "#cloud-config" ]] || die "cloud-init must start with #cloud-config"

if [[ -z "${SUBNET_ID:-}" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    SUBNET_ID="TODO-subnet-id"
  else
    have_cli
    SUBNET_ID="$(nebius vpc subnet list --parent-id "$NEBIUS_PROJECT_ID" --format jsonpath='{.items[0].metadata.id}')"
    [[ -n "$SUBNET_ID" ]] || die "no subnet found in project $NEBIUS_PROJECT_ID"
  fi
fi

args=(
  nebius compute instance create
  --parent-id "$NEBIUS_PROJECT_ID"
  --name "$VM_NAME"
  --resources-platform "$NOD_PLATFORM"
  --resources-preset "$NOD_PRESET"
  --boot-disk-managed-disk-name "${VM_NAME}-boot"
  --boot-disk-managed-disk-type network_ssd
  --boot-disk-managed-disk-size-gibibytes "$DISK_GIB"
  --boot-disk-managed-disk-block-size-bytes 4096
  --boot-disk-managed-disk-source-image-family-image-family "$IMAGE_FAMILY"
  --boot-disk-attach-mode READ_WRITE
  --cloud-init-user-data "$CLOUD_INIT"
  --network-interfaces "[{\"name\":\"eth0\",\"subnet_id\":\"$SUBNET_ID\",\"ip_address\":{},\"public_ip_address\":{}}]"
  --format "jsonpath={.metadata.id}"
)
if [[ "${PREEMPTIBLE:-0}" == "1" ]]; then
  note "preemptible: the VM gets 60 s notice and stops; do not use it for the demo or recordings"
  args+=(--preemptible-on-preemption stop --recovery-policy fail --follows-spot-price)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  run "${args[@]}"
  echo "--- cloud-init ---"
  echo "$CLOUD_INIT"
  exit 0
fi

have_cli
VM_ID="$("${args[@]}")"
[[ -n "$VM_ID" ]] || die "instance create returned no id"
echo "VM_ID=$VM_ID"

# The public IP appears once the instance is running.
VM_IP=""
for _ in $(seq 1 60); do
  VM_IP="$(nebius compute instance get --id "$VM_ID" --format json \
    | jq -r '.status.network_interfaces[0].public_ip_address.address // empty | split("/")[0]')" || true
  [[ -n "$VM_IP" ]] && break
  sleep 10
done
[[ -n "$VM_IP" ]] || die "no public IP after 10 minutes; check: nebius compute instance get --id $VM_ID"

D="$(ip_dashed "$VM_IP")"
cat <<EOF

VM_ID=$VM_ID
VM_IP=$VM_IP

Next steps:
  1. infra/nebius/security_group.sh            # ports 22 and 443 only, before any service starts
  2. ssh nod@$VM_IP                            # cloud-init needs a few minutes on first boot
  3. Put these in .env on the VM (TLS through Caddy and sslip.io, design 3.1):
       NOD_DOMAIN=nod.$D.sslip.io
       VIEW_DOMAIN=view.$D.sslip.io
  4. Follow docs/runbooks/day1-spike.md from section 8.2.
  Stop when idle (disk is still billed): nebius compute instance stop --id $VM_ID
EOF
