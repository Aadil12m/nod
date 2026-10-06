#!/usr/bin/env bash
# Shared helpers for the infra/nebius scripts. Source it; do not run it.
#
# Every script accepts --dry-run (or DRY_RUN=1): commands that would change anything in Nebius
# are printed, shell-quoted, instead of being run. Read-only lookups are skipped in dry-run mode
# too, and the value named in the TODO is used instead, so dry runs need no CLI and no network.

# Settings that are the same for every script (design 3.1). Override through the environment.
: "${NEBIUS_REGION:=uk-south2}"
: "${NOD_PLATFORM:=gpu-rtx6000-a}"
: "${NOD_PRESET:=1gpu-24vcpu-218gb}"

DRY_RUN="${DRY_RUN:-0}"

die() { echo "error: $*" >&2; exit 1; }
note() { echo "-- $*" >&2; }

# need VAR "what it is and where to find it": fail with a clear TODO if VAR is empty.
need() {
  local name="$1" hint="$2"
  if [[ -z "${!name:-}" ]]; then
    die "$name is not set. TODO: $hint"
  fi
}

# run CMD...: run it, or print it in dry-run mode.
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

# have_cli: the nebius CLI is installed (not needed for dry runs).
have_cli() {
  command -v nebius >/dev/null 2>&1 || die "nebius CLI not found. Install: curl -sSL https://artifacts.nebius.cloud/cli/install.sh | bash"
}

# dashed IPv4 for sslip.io host names: 1.2.3.4 -> 1-2-3-4
ip_dashed() { echo "${1//./-}"; }
