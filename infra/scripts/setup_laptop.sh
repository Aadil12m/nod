#!/usr/bin/env bash
# One-time setup for laptop mode on Linux (Ubuntu/Debian tested): the CPU LIBERO venv, the
# gateway and the 3D viewer. Safe to re-run. Needs git, curl and about 3 GB of disk.
#
#   infra/scripts/setup_laptop.sh        # then: export NEBIUS_API_KEY=...; make laptop
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  echo "setup: installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

# MuJoCo renders off-screen with OSMesa on a CPU.
if ! ldconfig -p 2>/dev/null | grep -q libOSMesa; then
  echo "setup: installing libosmesa6 (needs sudo)"
  sudo apt-get update -y && sudo apt-get install -y libosmesa6 libgl1 libglib2.0-0
fi

V=sim/.venv-libero
"$UV" venv -q --allow-existing -p 3.11 "$V"
mkdir -p "$V/src"
if [[ ! -d "$V/src/LIBERO/.git" ]]; then
  git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git "$V/src/LIBERO"
fi
git -C "$V/src/LIBERO" checkout -q 8f1084e
"$UV" pip install -q --python "$V/bin/python" "robosuite==1.4.0" "mujoco==3.3.1" "numpy==1.26.4" \
  "bddl==1.0.1" easydict termcolor pyyaml imageio "opencv-python-headless==4.10.0.84" "matplotlib==3.9.4" \
  "gym==0.25.2" cloudpickle future tqdm "PyOpenGL==3.1.10" pyzmq msgpack ormsgpack "pydantic>=2.9" \
  pytest h5py pillow msgpack-numpy imageio-ffmpeg
"$UV" pip install -q --python "$V/bin/python" --no-deps -e "$V/src/LIBERO" --config-settings editable_mode=compat
"$UV" pip install -q --python "$V/bin/python" -e protocol -e sim -e eval

L="$PWD/$V/src/LIBERO/libero/libero"
mkdir -p "$V/libero_config"
printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/../datasets\nassets: %s/assets\n' \
  "$L" "$L" "$L" "$L" "$L" > "$V/libero_config/config.yaml"

"$UV" venv -q --allow-existing -p 3.11 gateway/.venv
"$UV" pip install -q -p gateway/.venv/bin/python -e "protocol[numpy,fast]" -e gateway
"$UV" venv -q --allow-existing -p 3.11 viewer/.venv
"$UV" pip install -q -p viewer/.venv/bin/python -e "protocol[numpy,fast]" -e viewer

# The 3D view needs the compiled scene; export it from the real LIBERO scene once.
if [[ ! -f recordings/scene_libero_goal_v1.mjb && ! -f recordings/scene_libero_goal_v1.mjb.gz ]]; then
  mkdir -p recordings
  # Two steps: dump with LIBERO's MuJoCo 3.3.1, compile with the viewer's newer MuJoCo.
  { LIBERO_CONFIG_PATH="$PWD/$V/libero_config" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=viewer \
      "$V/bin/python" -m nod_viewer.export_scene dump --out-dir recordings/.scene_dump \
    && viewer/.venv/bin/python -m nod_viewer.export_scene compile --in-dir recordings/.scene_dump \
      --out recordings/scene_libero_goal_v1.mjb; } \
    || echo "setup: scene export failed; laptop mode still works without the 3D view"
fi

echo
echo "setup: done. Next:"
echo "  export NEBIUS_API_KEY=<your Token Factory key>"
echo "  make laptop      # then open http://localhost:8000 and http://localhost:8000/presenter"
