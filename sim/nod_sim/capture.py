"""Frame capture for Cosmos checks and the phone's "what the robot sees" inset (3.5.4 ctl.capture).

`render` returns an upright uint8 image. LIBERO/robosuite render with the OpenGL convention
(row 0 at the bottom), so the raw buffer is flipped vertically, as in design 4.2.4.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from nod_protocol import Frame


def render(env: Any, camera: str, w: int, h: int) -> np.ndarray:
    """Render one camera at w x h. Accepts a `SimEnv` or a raw LIBERO/GR00T env."""
    if hasattr(env, "render") and hasattr(env, "eval_atom"):  # a SimEnv adapter
        return env.render(camera, w, h)
    sim = env._env.sim if hasattr(env, "_env") else env.sim  # GR00T LiberoEnv or OffScreenRenderEnv
    return np.ascontiguousarray(sim.render(width=w, height=h, camera_name=camera)[::-1])


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    """JPEG-encode an RGB uint8 image with Pillow, or OpenCV if Pillow is missing."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except ImportError:
        pass
    try:
        import cv2

        ok, data = cv2.imencode(".jpg", img[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            return data.tobytes()
    except ImportError:
        pass
    raise RuntimeError("JPEG encoding needs Pillow or OpenCV (pip install pillow)")


def capture_frames(env: Any, cameras: list[str], w: int, h: int, step: int | None = None) -> list[Frame]:
    """Render and JPEG-encode each requested camera (ctl.capture -> sim.frames)."""
    frames = []
    for cam in cameras:
        img = render(env, cam, w, h)
        frames.append(Frame(camera=cam, w=int(img.shape[1]), h=int(img.shape[0]), jpeg=encode_jpeg(img), step=step))
    return frames
