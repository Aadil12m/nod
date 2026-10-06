#!/usr/bin/env bash
# Download the mock library (recorded runs plus the viewer scene) into recordings/ (design 4.4).
#
#   infra/scripts/fetch_mock.sh                 # what `make fetch-mock` runs
#   infra/scripts/fetch_mock.sh --pack DIR      # build the release assets from ./recordings into DIR
#
# The library is published as two assets on a GitHub release (tag $NOD_MOCK_TAG, default mock-v1):
#   nod-mock.tar.gz   recordings/<run_id>/{manifest.json, poses.npz, ...} and scene_libero_goal_v1.mjb
#   SHA256SUMS        sha256 of the tarball, checked before anything is extracted
# Upload them with: gh release create mock-v1 DIR/nod-mock.tar.gz DIR/SHA256SUMS
#
# Environment:
#   NOD_GITHUB_REPO     owner/name (default: parsed from the origin remote)
#   NOD_MOCK_TAG        release tag (default mock-v1)
#   NOD_MOCK_BASE_URL   full base URL instead of the GitHub release (file:// works; used by tests)
#   NOD_RECORDINGS_DIR  target directory (default <repo>/recordings)
#
# Mock mode works without the library: the MockRunner then plays a scripted timeline and the
# presenter has no 3D scene. So a failed download is reported, never fatal to the rest of setup.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${NOD_RECORDINGS_DIR:-$REPO_ROOT/recordings}"
TAG="${NOD_MOCK_TAG:-mock-v1}"
ASSET="nod-mock.tar.gz"

die() { echo "fetch-mock: $*" >&2; exit 1; }

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

if [[ "${1:-}" == "--pack" ]]; then
  OUT="${2:?--pack needs an output directory}"
  [[ -d "$DEST" ]] || die "nothing to pack: $DEST does not exist"
  mkdir -p "$OUT"
  # Pack everything under recordings/ except the README that lives in git.
  tar -C "$DEST" --exclude=./README.md -czf "$OUT/$ASSET" .
  (cd "$OUT" && echo "$(sha256_of "$ASSET")  $ASSET" > SHA256SUMS)
  echo "fetch-mock: wrote $OUT/$ASSET and $OUT/SHA256SUMS"
  exit 0
fi
[[ $# -eq 0 ]] || die "unknown argument: $1 (use --pack DIR, or no arguments)"

if [[ -n "${NOD_MOCK_BASE_URL:-}" ]]; then
  BASE="${NOD_MOCK_BASE_URL%/}"
else
  GH_REPO="${NOD_GITHUB_REPO:-}"
  if [[ -z "$GH_REPO" ]]; then
    url="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
    # git@github.com:owner/nod.git or https://github.com/owner/nod(.git)
    url="${url%.git}"
    case "$url" in
      git@github.com:*) GH_REPO="${url#git@github.com:}" ;;
      https://github.com/*) GH_REPO="${url#https://github.com/}" ;;
    esac
  fi
  [[ -n "$GH_REPO" ]] || die "cannot tell which GitHub repo hosts the release. TODO: set NOD_GITHUB_REPO=owner/nod"
  BASE="https://github.com/$GH_REPO/releases/download/$TAG"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "fetch-mock: $BASE/$ASSET"
if ! curl -fsSL --retry 3 -o "$TMP/SHA256SUMS" "$BASE/SHA256SUMS" \
   || ! curl -fSL --retry 3 --progress-bar -o "$TMP/$ASSET" "$BASE/$ASSET"; then
  die "download failed. Has release '$TAG' been published? Mock mode still runs without it (scripted timeline, no 3D scene)."
fi

want="$(awk -v a="$ASSET" '$2 == a || $2 == "*"a {print $1}' "$TMP/SHA256SUMS")"
[[ -n "$want" ]] || die "SHA256SUMS has no line for $ASSET"
got="$(sha256_of "$TMP/$ASSET")"
[[ "$want" == "$got" ]] || die "checksum mismatch for $ASSET (want $want, got $got)"

# Refuse absolute paths and .. in the archive before extracting.
if tar -tzf "$TMP/$ASSET" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
  die "refusing archive with absolute or parent paths"
fi

mkdir -p "$DEST"
tar -C "$DEST" -xzf "$TMP/$ASSET"
runs="$(find "$DEST" -name manifest.json | wc -l | tr -d ' ')"
scene="missing"
# The scene ships gzipped (scene_libero_goal_v1.mjb.gz); the viewer reads either form.
[[ -f "$DEST/scene_libero_goal_v1.mjb" || -f "$DEST/scene_libero_goal_v1.mjb.gz" ]] && scene="present"
echo "fetch-mock: $runs recorded runs in $DEST; scene_libero_goal_v1.mjb $scene"
