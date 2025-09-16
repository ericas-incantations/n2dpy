"""Mesh inspection helpers for the ``n2d inspect`` CLI command."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .core import MeshLoadError

__all__ = ["run_inspect"]

_LOGGER = logging.getLogger(__name__)


def run_inspect(mesh_path: Path) -> Dict[str, Any]:
    """Inspect ``mesh_path`` and return a serialisable report."""

    resolved_path = mesh_path.expanduser().resolve()
    try:
        from pyassimp import errors as pyassimp_errors
        from pyassimp import load as assimp_load
    except ImportError as exc:  # pragma: no cover - handled in runtime
        raise MeshLoadError("pyassimp is required for mesh inspection") from exc

    try:
        with assimp_load(str(resolved_path)) as scene:
            if not scene.meshes:
                raise MeshLoadError(f"Mesh '{resolved_path}' does not contain any geometry")
            mesh = scene.meshes[0]
            uv_sets = _extract_uv_sets(mesh)
    except pyassimp_errors.AssimpError as exc:
        raise MeshLoadError(f"Failed to load mesh '{resolved_path}': {exc}") from exc

    report = {
        "mesh_path": str(resolved_path),
        "uv_sets": uv_sets,
    }
    _LOGGER.debug("Inspection report generated: %s", report)
    return report


def _extract_uv_sets(mesh: Any) -> List[Dict[str, Any]]:
    uv_sets: List[Dict[str, Any]] = []
    coords_sequences = getattr(mesh, "texturecoords", [])
    component_counts = getattr(mesh, "numuvcomponents", [])

    for index, coords in enumerate(coords_sequences):
        if coords is None:
            continue
        components = int(component_counts[index]) if index < len(component_counts) else 0
        if components < 2:
            continue

        uv_array = np.asarray(coords, dtype=np.float64)
        if uv_array.ndim != 2 or uv_array.shape[0] == 0:
            udims: List[int] = []
        else:
            uv_2d = uv_array[:, :2]
            finite_mask = np.isfinite(uv_2d).all(axis=1)
            if not finite_mask.any():
                udims = []
            else:
                valid_uv = uv_2d[finite_mask]
                tiles_u = np.floor(valid_uv[:, 0])
                tiles_v = np.floor(valid_uv[:, 1])
                tiles = 1001 + tiles_u.astype(int) + 10 * tiles_v.astype(int)
                udims = sorted({int(tile) for tile in tiles})

        uv_sets.append(
            {
                "name": f"UV{index}",
                "chart_count": 0,
                "udims": udims,
            }
        )

    return uv_sets
