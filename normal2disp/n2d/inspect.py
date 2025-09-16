"""Mesh inspection helpers for the ``n2d inspect`` CLI command."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


import numpy as np

from .core import MeshLoadError
from .mesh_utils import assimp_mesh_to_trimesh, compute_triangle_tangent_frames


__all__ = ["run_inspect"]

_LOGGER = logging.getLogger(__name__)

_UV_EPSILON = 1e-5
_VECTOR_EPSILON = 1e-6


def run_inspect(mesh_path: Path, loader: str = "auto") -> Dict[str, Any]:
    """Inspect ``mesh_path`` and return a serialisable report."""

    resolved_path = mesh_path.expanduser().resolve()
    loader_choice = loader.lower()
    if loader_choice not in {"auto", "pyassimp"}:
        raise MeshLoadError(
            f"Unsupported loader '{loader}'. Only 'auto' and 'pyassimp' are available in this phase."
        )

    _ensure_pyassimp_dependencies()

    try:
        from pyassimp import errors as pyassimp_errors
        from pyassimp import load as assimp_load
    except ImportError as exc:  # pragma: no cover - handled in runtime
        raise MeshLoadError("pyassimp is required for mesh inspection") from exc

    try:
        with assimp_load(str(resolved_path)) as scene:
            if scene is None or not scene.meshes:
                raise MeshLoadError(f"Mesh '{resolved_path}' does not contain any geometry")

            mesh = scene.meshes[0]
            try:
                tri_mesh = assimp_mesh_to_trimesh(mesh)
            except ValueError as exc:
                raise MeshLoadError(str(exc)) from exc

            uv_sets = _extract_uv_sets(mesh)
            uv_reports: Dict[str, Any] = {}
            for name, uv_coords in uv_sets.items():
                uv_reports[name] = _analyse_uv_set(tri_mesh, uv_coords, uv_set_name=name)

            report = {"path": str(resolved_path), "uv_sets": uv_reports}
            _LOGGER.debug("Inspection report generated: %s", report)
            return report
    except pyassimp_errors.AssimpError as exc:  # pragma: no cover - depends on asset
        raise MeshLoadError(f"Failed to load mesh '{resolved_path}': {exc}") from exc


def _extract_uv_sets(mesh: Any) -> Dict[str, np.ndarray]:
    coords_sequences = getattr(mesh, "texturecoords", [])
    component_counts = getattr(mesh, "numuvcomponents", [])

    uv_sets: Dict[str, np.ndarray] = {}

    for index, coords in enumerate(coords_sequences):
        if coords is None:
            continue
        components = int(component_counts[index]) if index < len(component_counts) else 0
        if components < 2:
            continue

        uv_array = np.asarray(coords, dtype=np.float64)
        if uv_array.ndim != 2 or uv_array.shape[0] == 0:
            continue
        uv_sets[f"UV{index}"] = uv_array[:, :2]

    return uv_sets


def _analyse_uv_set(tri_mesh: Any, uv_coords: np.ndarray, *, uv_set_name: str) -> Dict[str, Any]:
    vertices = np.asarray(tri_mesh.vertices, dtype=np.float64)
    faces = np.asarray(tri_mesh.faces, dtype=np.int64)
    if uv_coords.shape[0] != vertices.shape[0]:
        raise MeshLoadError(
            f"UV set '{uv_set_name}' has {uv_coords.shape[0]} coordinates for {vertices.shape[0]} vertices"
        )

    uv = np.asarray(uv_coords[:, :2], dtype=np.float64)
    valid_vertex_mask = np.isfinite(uv).all(axis=1)
    valid_faces_mask = np.all(valid_vertex_mask[faces], axis=1)

    if not valid_faces_mask.any():
        return {"chart_count": 0, "udims": [], "charts": []}

    faces_subset = faces[valid_faces_mask]
    adjacency = _build_uv_adjacency(faces_subset, uv, eps=_UV_EPSILON)
    islands = _connected_components(adjacency)

    tangents, bitangents, _normals, orientation = compute_triangle_tangent_frames(
        vertices, faces_subset, uv
    )

    used_vertices = np.unique(faces_subset.reshape(-1))
    udims = _compute_udims(uv, used_vertices)

    charts = _build_charts(
        faces_subset,
        islands,
        uv,
        tangents,
        bitangents,
        orientation,
        adjacency,
    )

    return {"chart_count": len(charts), "udims": udims, "charts": charts}


def _ensure_pyassimp_dependencies() -> None:
    try:
        import distutils  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        try:
            distutils_module = importlib.import_module("setuptools._distutils")
            sys.modules.setdefault("distutils", distutils_module)
            sys.modules.setdefault(
                "distutils.sysconfig",
                importlib.import_module("setuptools._distutils.sysconfig"),
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
            raise MeshLoadError(
                "pyassimp requires setuptools to provide distutils on this Python version"
            ) from exc

    system_site = Path("/usr/lib/python3/dist-packages")
    system_site_str = str(system_site)
    if system_site.exists() and system_site_str not in sys.path:
        sys.path.append(system_site_str)


def _build_uv_adjacency(
    faces: np.ndarray,
    uv: np.ndarray,
    *,
    eps: float,
) -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(len(faces))]
    edge_to_faces: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}

    for face_index, face in enumerate(faces):
        for offset in range(3):
            vi = int(face[offset])
            vj = int(face[(offset + 1) % 3])

            if not _uv_vertex_finite(uv, vi) or not _uv_vertex_finite(uv, vj):
                continue

            key = (vi, vj) if vi < vj else (vj, vi)
            edge_to_faces.setdefault(key, []).append((face_index, vi, vj))

    for uses in edge_to_faces.values():
        if len(uses) < 2:
            continue

        for idx_a in range(len(uses)):
            face_a, a_v0, a_v1 = uses[idx_a]
            uv_a0 = uv[a_v0]
            uv_a1 = uv[a_v1]

            for idx_b in range(idx_a + 1, len(uses)):
                face_b, b_v0, b_v1 = uses[idx_b]
                uv_b0 = uv[b_v0]
                uv_b1 = uv[b_v1]

                if _uv_edge_matches(uv_a0, uv_a1, uv_b0, uv_b1, eps):
                    adjacency[face_a].append(face_b)
                    adjacency[face_b].append(face_a)

    return [sorted(set(neighbours)) for neighbours in adjacency]


def _connected_components(adjacency: Sequence[Sequence[int]]) -> List[List[int]]:
    if not adjacency:
        return []

    visited = np.zeros(len(adjacency), dtype=bool)
    components: List[List[int]] = []

    for start in range(len(adjacency)):
        if visited[start]:
            continue
        stack = [start]
        component: List[int] = []
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            visited[current] = True
            component.append(current)
            for neighbour in adjacency[current]:
                if not visited[neighbour]:
                    stack.append(neighbour)
        components.append(component)

    return components


def _build_charts(
    faces: np.ndarray,
    islands: Sequence[Sequence[int]],
    uv: np.ndarray,
    tangents: np.ndarray,
    bitangents: np.ndarray,
    orientation: np.ndarray,
    adjacency: Sequence[Sequence[int]],
) -> List[Dict[str, Any]]:
    charts: List[Dict[str, Any]] = []

    for chart_id, island in enumerate(islands, start=1):
        face_indices = np.array(island, dtype=int)
        island_faces = faces[face_indices]
        vertex_indices = np.unique(island_faces.reshape(-1))
        uv_coords = uv[vertex_indices]
        finite_mask = np.isfinite(uv_coords).all(axis=1)

        if finite_mask.any():
            uv_valid = uv_coords[finite_mask]
            bbox_min = uv_valid.min(axis=0)
            bbox_max = uv_valid.max(axis=0)
            bbox_uv = [
                float(bbox_min[0]),
                float(bbox_min[1]),
                float(bbox_max[0]),
                float(bbox_max[1]),
            ]
        else:
            bbox_uv = [0.0, 0.0, 0.0, 0.0]

        orientation_values = orientation[face_indices]
        neg_count = int(np.sum(orientation_values < 0))
        pos_count = int(np.sum(orientation_values > 0))
        mirrored = neg_count > pos_count

        flip_u = False
        flip_v = False
        if mirrored:
            flip_u, flip_v = _determine_mirror_axis(adjacency, island, tangents, bitangents)

        charts.append(
            {
                "id": chart_id,
                "face_count": int(len(island)),
                "bbox_uv": bbox_uv,
                "mirrored": mirrored,
                "flip_u": flip_u,
                "flip_v": flip_v,
            }
        )

    return charts


def _determine_mirror_axis(
    adjacency: Sequence[Sequence[int]],
    island: Sequence[int],
    tangents: np.ndarray,
    bitangents: np.ndarray,
) -> Tuple[bool, bool]:
    island_set = set(int(face_index) for face_index in island)
    processed_pairs = set()
    t_inversions = 0
    t_pairs = 0
    b_inversions = 0
    b_pairs = 0

    for face_index in island_set:
        for neighbour in adjacency[face_index]:
            if neighbour not in island_set:
                continue
            pair = (min(face_index, neighbour), max(face_index, neighbour))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            tangent_a = tangents[face_index]
            tangent_b = tangents[neighbour]
            bitangent_a = bitangents[face_index]
            bitangent_b = bitangents[neighbour]

            if _vector_valid(tangent_a) and _vector_valid(tangent_b):
                t_pairs += 1
                if float(np.dot(tangent_a, tangent_b)) < 0.0:
                    t_inversions += 1

            if _vector_valid(bitangent_a) and _vector_valid(bitangent_b):
                b_pairs += 1
                if float(np.dot(bitangent_a, bitangent_b)) < 0.0:
                    b_inversions += 1

    if t_pairs == 0 and b_pairs == 0:
        return True, False

    t_ratio = t_inversions / t_pairs if t_pairs else 0.0
    b_ratio = b_inversions / b_pairs if b_pairs else 0.0

    if t_ratio > b_ratio:
        return True, False
    if b_ratio > t_ratio:
        return False, True
    return True, False


def _compute_udims(uv: np.ndarray, vertex_indices: np.ndarray) -> List[int]:
    if vertex_indices.size == 0:
        return []

    coords = uv[vertex_indices]
    finite_mask = np.isfinite(coords).all(axis=1)
    coords = coords[finite_mask]
    if coords.size == 0:
        return []

    tiles_u = np.floor(coords[:, 0]).astype(int)
    tiles_v = np.floor(coords[:, 1]).astype(int)
    tiles = 1001 + tiles_u + 10 * tiles_v
    return sorted(int(tile) for tile in set(tiles))


def _uv_vertex_finite(uv: np.ndarray, index: int) -> bool:
    coord = uv[index]
    return bool(np.isfinite(coord).all())


def _uv_edge_matches(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    eps: float,
) -> bool:
    return (_uv_points_close(a0, b0, eps) and _uv_points_close(a1, b1, eps)) or (
        _uv_points_close(a0, b1, eps) and _uv_points_close(a1, b0, eps)
    )


def _uv_points_close(p: np.ndarray, q: np.ndarray, eps: float) -> bool:
    return float(np.max(np.abs(p - q))) <= eps


def _vector_valid(vec: np.ndarray, eps: float = _VECTOR_EPSILON) -> bool:
    return bool(np.linalg.norm(vec) > eps)

