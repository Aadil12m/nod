"""Export the LIBERO-Goal session scene for the browser viewer (design 4.2.5, streaming report 2).

MJB files are MuJoCo-version specific and the sim is pinned to MuJoCo 3.3.1 while mjviser needs
3.6 or newer, so the export has two steps in two environments:

1. `dump` (sim / LIBERO venv, MuJoCo 3.3.1): build the session env, write the procedural MJCF
   (`env.model.get_xml()`, with absolute asset paths) and a layout file: body and joint names,
   qpos addresses and the live `body_pos`/`body_quat` of every body (LIBERO places fixtures at
   reset, which the XML does not know).

       python -m nod_viewer.export_scene dump --out-dir /tmp/scene

2. `compile` (viewer venv, MuJoCo >= 3.6, same machine so the asset paths resolve): compile the
   XML, check that body and joint names and qpos addresses match the layout, write the fixture
   placement into the model, make it viewer-only (drop collision-only mesh geoms, shrink
   textures to <= 256 px; physics is never stepped) and save the MJB.

       python -m nod_viewer.export_scene compile --in-dir /tmp/scene --out recordings/scene_libero_goal_v1.mjb

`all` runs both when one interpreter can import LIBERO and a new enough MuJoCo (not the case in
our pinned images; kept for completeness).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

SCENE_ID = "libero_goal_v1"


# ---------------------------------------------------------------------------- step 1: dump


def dump(out_dir: Path, init_index: int = 0, env=None) -> dict:
    """Write scene.xml and layout.json from a live LIBERO session (MuJoCo 3.3.1 side)."""
    import mujoco

    if env is None:
        from nod_sim.scene import make_session_env

        env = make_session_env(init_index=init_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml = env.problem.model.get_xml()
    (out_dir / "scene.xml").write_text(xml)
    raw = env.sim.model._model if hasattr(env.sim.model, "_model") else env.sim.model
    names = [mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(raw.nbody)]
    joints = [mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(raw.njnt)]
    layout = {
        "scene_id": SCENE_ID,
        "mj_ver": mujoco.__version__,
        "body_names": names,
        "joint_names": joints,
        "jnt_qposadr": raw.jnt_qposadr.tolist(),
        "nq": int(raw.nq),
        "body_pos": raw.body_pos.tolist(),
        "body_quat": raw.body_quat.tolist(),
        "qpos": env.sim.data.qpos.tolist(),
        "xpos": env.sim.data.xpos.tolist(),
        "model_patch": env.episode_info().model_patch,
    }
    (out_dir / "layout.json").write_text(json.dumps(layout))
    return layout


# ---------------------------------------------------------------------------- step 2: compile


def _shrink_textures(xml: str, tex_dir: Path, max_px: int) -> str:
    from PIL import Image

    tex_dir.mkdir(parents=True, exist_ok=True)

    def sub(m: re.Match) -> str:
        path = m.group(1)
        if not path.lower().endswith(".png") or not os.path.exists(path):
            return m.group(0)
        out = tex_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", path.strip("/")))
        if not out.exists():
            im = Image.open(path)
            # cube and skybox textures are tall strips: keep their aspect ratio
            limit = (max_px, max_px * 6) if im.height > 2 * im.width else (max_px, max_px)
            im.thumbnail(limit)
            im.save(out)
        return f'file="{out}"'

    return re.sub(r'file="([^"]+)"', sub, xml)


def _decimate_meshes(spec, mesh_dir: Path, max_faces: int) -> int:
    """Point big meshes that carry no texture at decimated copies (viewer only).

    The original files are read (not the compiled vertices, which MuJoCo has already moved to
    the mesh inertial frame), simplified with fast-simplification and written as STL. Meshes
    drawn with a textured material keep their UVs and are left alone. Returns the number of
    meshes replaced; a no-op if trimesh or fast-simplification is missing.
    """
    try:
        import fast_simplification
        import trimesh
    except ImportError:
        return 0
    textured_mats = {m.name for m in spec.materials if any(t for t in m.textures)}
    textured_meshes = {g.meshname for g in spec.geoms if g.material in textured_mats}
    mesh_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for me in spec.meshes:
        path = me.file
        if me.name in textured_meshes or not path.lower().endswith((".stl", ".obj")) or not os.path.exists(path):
            continue
        out = mesh_dir / (re.sub(r"[^A-Za-z0-9_.-]", "_", path.strip("/")) + f".{max_faces}.stl")
        if not out.exists():
            mesh = trimesh.load(path, force="mesh", process=False)
            if len(mesh.faces) <= max_faces:
                continue
            pts, faces = fast_simplification.simplify(np.asarray(mesh.vertices, np.float32),
                                                      np.asarray(mesh.faces), 1.0 - max_faces / len(mesh.faces))
            trimesh.Trimesh(pts, faces, process=False).export(out)
        me.file = str(out)
        count += 1
    return count


def compile_scene(in_dir: Path, out: Path, *, slim: bool = True, max_tex_px: int = 256,
                  max_faces: int = 3000) -> dict:
    """Compile scene.xml with this MuJoCo, check names against layout.json, save an MJB."""
    import mujoco

    layout = json.loads((in_dir / "layout.json").read_text())
    xml = (in_dir / "scene.xml").read_text()
    decimated = 0
    if slim:
        xml = _shrink_textures(xml, in_dir / "tex_small", max_tex_px)
    spec = mujoco.MjSpec.from_string(xml)
    removed = 0
    if slim:
        # robosuite convention: group 0 = collision, group 1 = visual. Collision meshes are
        # never drawn by the viewer and dominate the file size.
        for g in list(spec.geoms):
            if g.group == 0 and g.type == mujoco.mjtGeom.mjGEOM_MESH:
                spec.delete(g)
                removed += 1
        used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
        for me in list(spec.meshes):
            if me.name not in used:
                spec.delete(me)
        decimated = _decimate_meshes(spec, in_dir / "mesh_small", max_faces)
        spec.compiler.boundmass = 1e-4  # viewer-only model: never stepped
        spec.compiler.boundinertia = 1e-8
    m = spec.compile()
    report = check_layout(m, layout)
    if not report["ok"]:
        raise SystemExit(f"compiled model does not match the sim model: {report}")
    apply_layout(m, layout)
    out.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(m, str(out), None)
    # verify: forward kinematics on the saved model reproduces the sim's body positions
    m2 = mujoco.MjModel.from_binary_path(str(out))
    d2 = mujoco.MjData(m2)
    d2.qpos[:] = np.asarray(layout["qpos"])
    mujoco.mj_kinematics(m2, d2)
    err = float(np.abs(d2.xpos - np.asarray(layout["xpos"])).max())
    report.update({"removed_collision_meshes": removed, "decimated_meshes": decimated, "bytes": out.stat().st_size, "max_xpos_err_m": err,
                   "mj_ver": mujoco.__version__})
    (out.with_suffix(".json")).write_text(json.dumps({"scene_id": layout["scene_id"], "sim_mj_ver": layout["mj_ver"],
                                                       **report}, indent=1))
    return report


def check_layout(m, layout: dict) -> dict:
    import mujoco

    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    ok = (bodies == layout["body_names"] and joints == layout["joint_names"]
          and m.jnt_qposadr.tolist() == layout["jnt_qposadr"] and m.nq == layout["nq"])
    return {"ok": bool(ok), "nbody": int(m.nbody), "nq": int(m.nq), "same_bodies": bodies == layout["body_names"],
            "same_joints": joints == layout["joint_names"]}


def apply_layout(m, layout: dict) -> None:
    """Write the sim's live body_pos/body_quat (fixture placement) into the model."""
    m.body_pos[:] = np.asarray(layout["body_pos"])
    m.body_quat[:] = np.asarray(layout["body_quat"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Export the LIBERO-Goal scene for the viewer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="sim venv: write scene.xml + layout.json")
    d.add_argument("--out-dir", type=Path, required=True)
    d.add_argument("--init-index", type=int, default=0)
    c = sub.add_parser("compile", help="viewer venv: compile scene.xml to an MJB")
    c.add_argument("--in-dir", type=Path, required=True)
    c.add_argument("--out", type=Path, default=Path("recordings/scene_libero_goal_v1.mjb"))
    c.add_argument("--no-slim", action="store_true")
    a = sub.add_parser("all", help="both steps in one interpreter")
    a.add_argument("--out", type=Path, default=Path("recordings/scene_libero_goal_v1.mjb"))
    a.add_argument("--work-dir", type=Path, default=Path("/tmp/nod_scene_export"))
    args = ap.parse_args(argv)
    if args.cmd == "dump":
        layout = dump(args.out_dir, args.init_index)
        print(f"wrote {args.out_dir}/scene.xml and layout.json ({len(layout['body_names'])} bodies, MuJoCo {layout['mj_ver']})")
    elif args.cmd == "compile":
        print(json.dumps(compile_scene(args.in_dir, args.out, slim=not args.no_slim), indent=1))
    else:
        dump(args.work_dir)
        print(json.dumps(compile_scene(args.work_dir, args.out), indent=1))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
