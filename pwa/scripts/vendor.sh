#!/bin/sh
# Download the pinned browser libraries and models into pwa/vendor/ so the PWA
# works offline and does not depend on a CDN during a demo ("make vendor").
#
#   @mediapipe/tasks-vision 1.0.1         -> vendor/mediapipe/ (bundle + wasm)
#   face_landmarker.task (float16, v1)    -> vendor/models/
#   @tensorflow/tfjs 3.21.0               -> vendor/tfjs/tf.min.js
#   @tensorflow-models/speech-commands 0.5.4 and its 18w model -> vendor/speech-commands/
#
# npm packages come from the npm registry tarballs (the files jsdelivr serves);
# every download is checked against a pinned hash. face.js and kws.js read
# vendor/manifest.json and fall back to the CDN when it is missing.
#
# Usage: sh pwa/scripts/vendor.sh   (re-running is safe; files are replaced)
set -eu

HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT="$HERE/vendor"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
sha512b64() { openssl dgst -sha512 -binary "$1" | openssl base64 -A; }

fetch() { # url dest
  echo "  get $1"
  curl -fsSL --retry 3 -o "$2" "$1"
}

check256() { # file expected
  got=$(sha256 "$1")
  [ "$got" = "$2" ] || { echo "hash mismatch for $1: $got" >&2; exit 1; }
}

npm_pkg() { # tarball-url sha512-integrity dir
  fetch "$1" "$TMP/pkg.tgz"
  got="sha512-$(sha512b64 "$TMP/pkg.tgz")"
  [ "$got" = "$2" ] || { echo "integrity mismatch for $1" >&2; exit 1; }
  rm -rf "$TMP/package" && tar -xzf "$TMP/pkg.tgz" -C "$TMP"
}

echo "Vendoring into $OUT"
mkdir -p "$OUT/mediapipe/wasm" "$OUT/models" "$OUT/tfjs" "$OUT/speech-commands/18w"

npm_pkg https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-1.0.1.tgz \
  'sha512-rvRE2FmAZ6ZxKSw7wq+e+jQDpN3t1B/tD2mJz9SmAzb1msoDkd4dMoE4wAh8Z30Um0PQwLiHr9QtomhmXk3aUQ=='
cp "$TMP/package/vision_bundle.mjs" "$OUT/mediapipe/"
cp "$TMP/package/wasm/"* "$OUT/mediapipe/wasm/"

fetch https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task "$OUT/models/face_landmarker.task"
check256 "$OUT/models/face_landmarker.task" 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

npm_pkg https://registry.npmjs.org/@tensorflow/tfjs/-/tfjs-3.21.0.tgz \
  'sha512-khcARd3/872llL/oF4ouR40qlT71mylU66PGT8kHP/GJ5YKj44sv8lDRjU7lOVlJK7jsJFWEsNVHI3eMc/GWNQ=='
cp "$TMP/package/dist/tf.min.js" "$OUT/tfjs/"

npm_pkg https://registry.npmjs.org/@tensorflow-models/speech-commands/-/speech-commands-0.5.4.tgz \
  'sha512-r0c/MvC15/09xWujx1pKe6mA0nta+4jQWDXGkqfSVkXLo8ARrwcZ4mTGLlfvT43ySfidiveUo0m+P51+UK821Q=='
cp "$TMP/package/dist/speech-commands.min.js" "$OUT/speech-commands/"

SC=https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.5/browser_fft/18w
for f in model.json:be053f904fe26d1fab2be824d943e936d9f04c9ea3d238d3c58fa2a6225b1ef7 \
         metadata.json:06f92997a0f6b3e5f11d70251222350f1e0f6717280f7edd151d4001482665ac \
         group1-shard1of2:79b7ef9d7577b333ee39b0d6e8dfebdb03be559a645bc389307ef84da414a30b \
         group1-shard2of2:129b9a7d501f81e62710d494efcba67ed98b3f3032dd28d7fe20cbc8cd5bee1f; do
  name=${f%%:*}; hash=${f#*:}
  fetch "$SC/$name" "$OUT/speech-commands/18w/$name"
  check256 "$OUT/speech-commands/18w/$name" "$hash"
done

cat > "$OUT/manifest.json" <<EOF
{
  "mediapipe": "@mediapipe/tasks-vision@1.0.1",
  "face_model": "face_landmarker/float16/1",
  "tfjs": "@tensorflow/tfjs@3.21.0",
  "speech_commands": "@tensorflow-models/speech-commands@0.5.4",
  "speech_commands_model": "v0.5/browser_fft/18w"
}
EOF
echo "Done. $(du -sh "$OUT" | cut -f1) in $OUT"
