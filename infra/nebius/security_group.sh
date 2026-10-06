#!/usr/bin/env bash
# Lock the Nod VM down to TCP 22 (SSH) and 443 (Caddy) inbound (design 3.1, 8.1).
# The default Nebius security group allows ALL ingress, so run this before starting any service.
#
#   export NEBIUS_PROJECT_ID=<uk-south2 project id>   # TODO
#   export VM_ID=<instance id from create_vm.sh>      # TODO
#   infra/nebius/security_group.sh             # create the group and rules, then print how to attach
#   infra/nebius/security_group.sh --dry-run   # print the commands only
#
# Environment:
#   NETWORK_ID   default: the first network in the project (`nebius vpc network list`)
#   SSH_CIDR     source range for SSH (default 0.0.0.0/0; set your team's IP/32 if you can)
#   SG_NAME      default nod-sg
#
# Port 80 stays closed: Caddy then gets certificates with the TLS-ALPN-01 challenge on 443.
# Flags follow docs/research/nebius-ops.md 2.5 (nebius vpc security-group / security-rule).
# The egress rule syntax and attaching the group are marked [unverified] there.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=infra/nebius/lib.sh
source "$HERE/lib.sh"

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

SG_NAME="${SG_NAME:-nod-sg}"
SSH_CIDR="${SSH_CIDR:-0.0.0.0/0}"
INGRESS_PORTS=(22 443)

need NEBIUS_PROJECT_ID "your uk-south2 project id"

if [[ -z "${NETWORK_ID:-}" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    NETWORK_ID="TODO-network-id"
  else
    have_cli
    NETWORK_ID="$(nebius vpc network list --parent-id "$NEBIUS_PROJECT_ID" --format jsonpath='{.items[0].metadata.id}')"
    [[ -n "$NETWORK_ID" ]] || die "no network found in project $NEBIUS_PROJECT_ID"
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  SG="TODO-security-group-id"
  run nebius vpc security-group create --parent-id "$NEBIUS_PROJECT_ID" --name "$SG_NAME" \
    --network-id "$NETWORK_ID" --format "jsonpath={.metadata.id}"
else
  have_cli
  SG="$(nebius vpc security-group create --parent-id "$NEBIUS_PROJECT_ID" --name "$SG_NAME" \
    --network-id "$NETWORK_ID" --format 'jsonpath={.metadata.id}')"
  [[ -n "$SG" ]] || die "security group create returned no id"
  echo "SG_ID=$SG"
fi

for P in "${INGRESS_PORTS[@]}"; do
  src="0.0.0.0/0"
  [[ "$P" == "22" ]] && src="$SSH_CIDR"
  run nebius vpc security-rule create --parent-id "$SG" --name "in-$P" --access allow \
    --protocol tcp --ingress-source-cidrs "$src" --ingress-destination-ports "$P"
done
# Outbound: Token Factory, Hugging Face, registries, Let's Encrypt.
run nebius vpc security-rule create --parent-id "$SG" --name out-all --access allow \
  --protocol any --egress-destination-cidrs 0.0.0.0/0

cat <<EOF

Attach the group to the VM's network interface (TODO, [unverified] non-interactive form):
  nebius compute instance edit ${VM_ID:-<VM_ID>}
  # add under spec.network_interfaces[0]:
  #   security_groups: [{id: $SG}]
Then check from your laptop that only 22 and 443 answer, for example:
  nc -zv -w 3 <VM_IP> 22; nc -zv -w 3 <VM_IP> 443; nc -zv -w 3 <VM_IP> 8000   # the last must fail
EOF
