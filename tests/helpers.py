from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from normal2disp.n2d.image_utils import _ensure_system_site_packages

_DUMMY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9YfD+VsAAAAASUVORK5CYII="
)


def write_two_island_asset(directory: Path) -> Path:
    obj_data = """\
o TwoIslands
mtllib two_islands.mtl
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
v 3 0 0
v 3 1 0
v 2 1 0
vt 0.05 0.05
vt 0.95 0.05
vt 0.95 0.95
vt 0.05 0.95
vt 1.95 0.05
vt 1.05 0.05
vt 1.95 0.95
vt 1.05 0.95
usemtl matA
f 1/1 2/2 3/3
f 1/1 3/3 4/4
usemtl matB
f 5/5 6/6 7/7
f 5/5 7/7 8/8
"""
    mtl_data = """\
newmtl matA
Kd 1.0 0.0 0.0
newmtl matB
Kd 0.0 1.0 0.0
"""

    obj_path = directory / "two_islands.obj"
    (directory / "two_islands.mtl").write_text(mtl_data, encoding="utf-8")
    obj_path.write_text(obj_data, encoding="utf-8")
    return obj_path


def write_dummy_png(path: Path, size: int = 1) -> None:
    if size <= 1:
        path.write_bytes(_DUMMY_PNG)
        return

    image = np.full((size, size, 3), 128, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to write dummy PNG to {path}")


def read_exr_pixels(path: Path) -> np.ndarray:
    _ensure_system_site_packages()
    import OpenImageIO as oiio  # type: ignore

    buf = oiio.ImageBuf(str(path))
    spec = buf.spec()
    array = np.array(buf.get_pixels(oiio.FLOAT))
    return array.reshape(spec.height, spec.width, spec.nchannels)
