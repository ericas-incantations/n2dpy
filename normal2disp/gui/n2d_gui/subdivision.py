"""Loop subdivision and displacement sampling utilities for the viewport."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import availability depends on runtime environment
    import OpenImageIO as oiio  # type: ignore
except Exception as exc:  # pragma: no cover - handled at runtime
    oiio = None  # type: ignore
    _OIIO_IMPORT_ERROR = exc
else:
    _OIIO_IMPORT_ERROR = None

__all__ = [
    "HeightField",
    "LoopSubdivisionCache",
    "DisplacementResult",
    "MeshBuffers",
    "load_height_field",
    "generate_displacement",
]


_EPSILON = 1e-7


class HeightFieldError(RuntimeError):
    """Raised when a height field cannot be loaded or sampled."""


@dataclass(frozen=True)
class HeightTile:
    """A single EXR tile storing height data."""

    tile: int
    data: np.ndarray
    amplitude: Optional[float]
    units: Optional[str]

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.data.shape[0]), int(self.data.shape[1])


class HeightField:
    """Collection of UDIM height tiles with bilinear sampling."""

    def __init__(self, tiles: Mapping[int, HeightTile]) -> None:
        if not tiles:
            raise HeightFieldError("No EXR tiles were provided for displacement preview")

        self._tiles: Dict[int, HeightTile] = dict(tiles)
        self._default_tile = next(iter(self._tiles))

    @property
    def amplitude(self) -> Optional[float]:
        for tile in self._tiles.values():
            if tile.amplitude is not None:
                return float(tile.amplitude)
        return None

    @property
    def units(self) -> Optional[str]:
        for tile in self._tiles.values():
            if tile.units:
                return str(tile.units)
        return None

    @staticmethod
    def _udim_from_uv(uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = uv[:, 0]
        v = uv[:, 1]
        tile_u = np.floor(u)
        tile_v = np.floor(v)
        tiles = 1001 + tile_u.astype(np.int64) + tile_v.astype(np.int64) * 10
        local_u = u - tile_u
        local_v = v - tile_v
        return tiles, np.stack([local_u, local_v], axis=1)

    @staticmethod
    def _clamp01(values: np.ndarray) -> np.ndarray:
        return np.clip(values, 0.0, 1.0 - _EPSILON)

    def sample(self, uv: np.ndarray) -> np.ndarray:
        """Return bilinearly sampled height values for ``uv`` coordinates."""

        if uv.ndim != 2 or uv.shape[1] < 2:
            raise HeightFieldError("UV coordinates must be an array of shape (N, 2)")

        if not np.isfinite(uv).all():
            raise HeightFieldError("UV coordinates contain NaN or infinite values")

        tiles, local_uv = self._udim_from_uv(uv)
        local_uv = self._clamp01(local_uv)
        heights = np.zeros(len(uv), dtype=np.float64)

        for tile_id in np.unique(tiles):
            mask = tiles == tile_id
            if tile_id not in self._tiles:
                raise HeightFieldError(f"Missing UDIM tile {tile_id} in baked EXRs")

            tile = self._tiles[tile_id]
            tile_data = tile.data
            height, width = tile_data.shape

            local = local_uv[mask]
            if height <= 1 or width <= 1:
                heights[mask] = float(tile_data[0, 0])
                continue

            u = local[:, 0] * (width - 1)
            v = local[:, 1] * (height - 1)

            x0 = np.floor(u).astype(np.int64)
            y0 = np.floor(v).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, width - 1)
            y1 = np.clip(y0 + 1, 0, height - 1)
            sx = u - x0
            sy = v - y0

            h00 = tile_data[y0, x0]
            h10 = tile_data[y0, x1]
            h01 = tile_data[y1, x0]
            h11 = tile_data[y1, x1]

            interp_top = h00 * (1.0 - sx) + h10 * sx
            interp_bottom = h01 * (1.0 - sx) + h11 * sx
            heights[mask] = interp_top * (1.0 - sy) + interp_bottom * sy

        return heights


def _read_exr_channel(image_input: "oiio.ImageInput") -> np.ndarray:
    spec = image_input.spec()
    if spec.nchannels < 1:
        raise HeightFieldError("EXR does not contain any channels")

    channel_names = list(spec.channelnames)
    try:
        channel_index = channel_names.index("height")
    except ValueError:
        channel_index = 0

    pixels = image_input.read_image(
        channel_index,
        channel_index + 1,
        oiio.FLOAT if oiio is not None else None,
    )
    if pixels is None:
        raise HeightFieldError("Failed to read height data from EXR")

    array = np.asarray(pixels, dtype=np.float32)
    array = array.reshape(spec.height, spec.width)
    return array


def _extract_udim(path: Path) -> Optional[int]:
    match = re.search(r"(\d{4})", path.name)
    if match:
        return int(match.group(1))
    return None


def load_height_field(paths: Sequence[Path]) -> HeightField:
    """Load baked EXR files into a :class:`HeightField` instance."""

    if oiio is None:  # pragma: no cover - depends on optional dependency
        raise HeightFieldError(
            "OpenImageIO is required for displacement preview"
            + (f": {_OIIO_IMPORT_ERROR}" if _OIIO_IMPORT_ERROR else "")
        )

    tiles: Dict[int, HeightTile] = {}

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise HeightFieldError(f"Displacement EXR not found: {path}")

        image_input = oiio.ImageInput.open(str(path))
        if image_input is None:
            raise HeightFieldError(f"Failed to open EXR: {path}")

        try:
            data = _read_exr_channel(image_input)
            spec = image_input.spec()
            amplitude = spec.getattribute("n2d:amplitude")
            units = spec.getattribute("n2d:units")
        finally:
            image_input.close()

        tile_id = _extract_udim(path)
        if tile_id is None:
            tile_id = 1001

        tiles[tile_id] = HeightTile(
            tile=tile_id,
            data=np.asarray(data, dtype=np.float32),
            amplitude=float(amplitude) if amplitude is not None else None,
            units=str(units) if units else None,
        )

    return HeightField(tiles)


@dataclass
class MeshBuffers:
    """Container for vertex, UV, and face buffers."""

    positions: np.ndarray
    uv: np.ndarray
    faces: np.ndarray

    def copy(self) -> "MeshBuffers":
        return MeshBuffers(
            positions=self.positions.copy(),
            uv=self.uv.copy(),
            faces=self.faces.copy(),
        )


def _ensure_contiguous(mesh: MeshBuffers) -> MeshBuffers:
    return MeshBuffers(
        positions=np.ascontiguousarray(mesh.positions, dtype=np.float64),
        uv=np.ascontiguousarray(mesh.uv, dtype=np.float64),
        faces=np.ascontiguousarray(mesh.faces, dtype=np.int64),
    )


def _build_vertex_adjacency(faces: np.ndarray, vertex_count: int) -> Tuple[
    Tuple[Tuple[int, ...], ...],
    Tuple[Tuple[int, ...], ...],
    Dict[Tuple[int, int], Tuple[int, Optional[int]]],
]:
    neighbors: MutableMapping[int, set[int]] = {i: set() for i in range(vertex_count)}
    boundary_neighbors: MutableMapping[int, set[int]] = {i: set() for i in range(vertex_count)}
    edge_faces: Dict[Tuple[int, int], Tuple[int, Optional[int]]] = {}

    for face_index, face in enumerate(faces):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        for start, end, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            neighbors[start].add(end)
            neighbors[start].add(opposite)

            key = (min(start, end), max(start, end))
            existing = edge_faces.get(key)
            if existing is None:
                edge_faces[key] = (face_index, None)
            else:
                first_face, second_face = existing
                if second_face is None and first_face != face_index:
                    edge_faces[key] = (first_face, face_index)

    for (v0, v1), face_pair in edge_faces.items():
        if face_pair[1] is None:
            boundary_neighbors[v0].add(v1)
            boundary_neighbors[v1].add(v0)

    neighbor_tuple = tuple(tuple(sorted(vals)) for vals in neighbors.values())
    boundary_tuple = tuple(tuple(sorted(vals)) for vals in boundary_neighbors.values())
    return neighbor_tuple, boundary_tuple, edge_faces


def _loop_beta(valence: int) -> float:
    if valence == 0:
        return 0.0
    if valence == 3:
        return 3.0 / 16.0
    return 3.0 / (8.0 * float(valence))


def _reposition_vertices(
    positions: np.ndarray,
    uv: np.ndarray,
    neighbors: Tuple[Tuple[int, ...], ...],
    boundary_neighbors: Tuple[Tuple[int, ...], ...],
) -> Tuple[np.ndarray, np.ndarray]:
    vertex_count = positions.shape[0]
    new_positions = np.zeros_like(positions)
    new_uv = np.zeros_like(uv)

    for vertex_index in range(vertex_count):
        boundary = boundary_neighbors[vertex_index]
        if len(boundary) >= 2:
            nb0, nb1 = boundary[:2]
            new_positions[vertex_index] = (
                positions[vertex_index] * 0.75 + (positions[nb0] + positions[nb1]) * 0.125
            )
            new_uv[vertex_index] = uv[vertex_index] * 0.75 + (uv[nb0] + uv[nb1]) * 0.125
            continue

        neighbor_list = neighbors[vertex_index]
        valence = len(neighbor_list)
        if valence == 0:
            new_positions[vertex_index] = positions[vertex_index]
            new_uv[vertex_index] = uv[vertex_index]
            continue

        beta = _loop_beta(valence)
        weight = 1.0 - float(valence) * beta
        neighbor_positions = positions[np.array(neighbor_list, dtype=np.int64)]
        neighbor_uv = uv[np.array(neighbor_list, dtype=np.int64)]

        new_positions[vertex_index] = (
            positions[vertex_index] * weight + neighbor_positions.sum(axis=0) * beta
        )
        new_uv[vertex_index] = uv[vertex_index] * weight + neighbor_uv.sum(axis=0) * beta

    return new_positions, new_uv


def _create_edge_vertices(
    positions: np.ndarray,
    uv: np.ndarray,
    faces: np.ndarray,
    edge_faces: Mapping[Tuple[int, int], Tuple[int, Optional[int]]],
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], int]]:
    edge_vertices: list[np.ndarray] = []
    edge_uvs: list[np.ndarray] = []
    edge_indices: Dict[Tuple[int, int], int] = {}

    base_count = positions.shape[0]
    for (v0, v1), (face_a, face_b) in edge_faces.items():
        key = (v0, v1)
        if face_b is None:
            position = (positions[v0] + positions[v1]) * 0.5
            uv_value = (uv[v0] + uv[v1]) * 0.5
        else:
            face0 = faces[face_a]
            face1 = faces[face_b]

            def _opposite(face: np.ndarray) -> int:
                for value in (int(face[0]), int(face[1]), int(face[2])):
                    if value != v0 and value != v1:
                        return value
                return int(v0)

            opp0 = _opposite(face0)
            opp1 = _opposite(face1)
            position = (positions[v0] + positions[v1]) * 3.0 / 8.0 + (
                positions[opp0] + positions[opp1]
            ) * 1.0 / 8.0
            uv_value = (uv[v0] + uv[v1]) * 3.0 / 8.0 + (uv[opp0] + uv[opp1]) * 1.0 / 8.0

        edge_indices[(min(v0, v1), max(v0, v1))] = base_count + len(edge_vertices)
        edge_vertices.append(position)
        edge_uvs.append(uv_value)

    if edge_vertices:
        edge_pos_array = np.asarray(edge_vertices, dtype=np.float64)
        edge_uv_array = np.asarray(edge_uvs, dtype=np.float64)
    else:
        edge_pos_array = np.empty((0, 3), dtype=np.float64)
        edge_uv_array = np.empty((0, 2), dtype=np.float64)

    return edge_pos_array, edge_uv_array, edge_indices


def _loop_subdivide_once(mesh: MeshBuffers) -> MeshBuffers:
    vertex_count = mesh.positions.shape[0]
    neighbors, boundary_neighbors, edge_faces = _build_vertex_adjacency(mesh.faces, vertex_count)

    new_positions, new_uv = _reposition_vertices(
        mesh.positions, mesh.uv, neighbors, boundary_neighbors
    )

    edge_positions, edge_uv, edge_indices = _create_edge_vertices(
        mesh.positions, mesh.uv, mesh.faces, edge_faces
    )

    combined_positions = np.vstack([new_positions, edge_positions])
    combined_uv = np.vstack([new_uv, edge_uv])

    new_faces = []
    for face in mesh.faces:
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        ab = edge_indices[(min(a, b), max(a, b))]
        bc = edge_indices[(min(b, c), max(b, c))]
        ca = edge_indices[(min(c, a), max(c, a))]

        new_faces.extend(
            (
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca),
            )
        )

    faces_array = np.asarray(new_faces, dtype=np.int64)
    return MeshBuffers(
        positions=np.asarray(combined_positions, dtype=np.float64),
        uv=np.asarray(combined_uv, dtype=np.float64),
        faces=faces_array,
    )


class LoopSubdivisionCache:
    """Cache Loop subdivision levels for a mesh."""

    def __init__(self, mesh: MeshBuffers) -> None:
        self._levels: Dict[int, MeshBuffers] = {0: _ensure_contiguous(mesh)}

    def mesh_for_level(self, level: int) -> MeshBuffers:
        if level < 0:
            raise ValueError("Subdivision level must be non-negative")
        if level > 5:
            raise ValueError("Subdivision level above 5 is not supported in preview")

        for current in range(1, level + 1):
            if current not in self._levels:
                self._levels[current] = _loop_subdivide_once(self._levels[current - 1])
        return self._levels[level]


def _compute_vertex_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions)
    tri_a = positions[faces[:, 0]]
    tri_b = positions[faces[:, 1]]
    tri_c = positions[faces[:, 2]]

    face_normals = np.cross(tri_b - tri_a, tri_c - tri_a)
    normals[faces[:, 0]] += face_normals
    normals[faces[:, 1]] += face_normals
    normals[faces[:, 2]] += face_normals

    lengths = np.linalg.norm(normals, axis=1)
    mask = lengths > 1e-12
    normals[mask] /= lengths[mask][:, None]
    normals[~mask] = np.array([0.0, 1.0, 0.0])
    return normals


@dataclass
class DisplacementResult:
    """Final buffers and statistics for a displacement preview."""

    level: int
    positions: np.ndarray
    normals: np.ndarray
    faces: np.ndarray
    heights: np.ndarray
    build_ms: float
    sample_ms: float
    displace_ms: float

    @property
    def triangle_count(self) -> int:
        return int(self.faces.shape[0])


def generate_displacement(
    cache: LoopSubdivisionCache,
    height_field: HeightField,
    level: int,
    scale: float,
    *,
    precomputed_heights: Optional[Mapping[int, np.ndarray]] = None,
    height_cache: Optional[MutableMapping[int, np.ndarray]] = None,
) -> DisplacementResult:
    start_time = time.perf_counter()
    mesh = cache.mesh_for_level(level)
    mesh_copy = mesh.copy()
    build_ms = (time.perf_counter() - start_time) * 1000.0

    heights: Optional[np.ndarray] = None
    if precomputed_heights and level in precomputed_heights:
        heights = np.asarray(precomputed_heights[level], dtype=np.float64)

    if heights is None:
        sample_start = time.perf_counter()
        heights = height_field.sample(mesh_copy.uv)
        sample_ms = (time.perf_counter() - sample_start) * 1000.0
        if height_cache is not None:
            height_cache[level] = heights.copy()
    else:
        sample_ms = 0.0

    normals = _compute_vertex_normals(mesh_copy.positions, mesh_copy.faces)
    displace_start = time.perf_counter()
    displaced_positions = mesh_copy.positions + normals * (heights[:, None] * float(scale))
    displaced_normals = _compute_vertex_normals(displaced_positions, mesh_copy.faces)
    displace_ms = (time.perf_counter() - displace_start) * 1000.0

    return DisplacementResult(
        level=level,
        positions=np.asarray(displaced_positions, dtype=np.float32),
        normals=np.asarray(displaced_normals, dtype=np.float32),
        faces=mesh_copy.faces.copy(),
        heights=np.asarray(heights, dtype=np.float32),
        build_ms=build_ms,
        sample_ms=sample_ms,
        displace_ms=displace_ms,
    )
