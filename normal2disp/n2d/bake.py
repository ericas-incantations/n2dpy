"""Displacement bake orchestration."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

from .core import ImageIOError, SolverError, TextureAssignmentError, UDIMError
from .image_utils import (
    expand_udim_pattern,
    load_texture_info,
    read_texture_pixels,
    write_exr_channels,
)
from .inspect import MeshInfo, UVSetInfo, inspect_mesh
from .uv_raster import TileRasterResult, rasterize_uv_charts


NormalizationMode = Literal["auto", "xyz", "xy", "none"]
LoaderMode = Literal["auto", "pyassimp", "blender"]


_LOGGER = logging.getLogger(__name__)


@dataclass
class BakeOptions:
    """Options that configure the displacement baking process."""

    uv_set: Optional[str] = None
    y_is_down: bool = False
    normalization: NormalizationMode = "auto"
    max_slope: float = 10.0
    amplitude: float = 1.0
    cg_tol: float = 1e-6
    cg_maxiter: int = 10000
    deterministic: bool = False
    processes: Optional[int] = None
    loader: LoaderMode = "auto"
    export_sidecars: bool = False
    on_progress: Optional[Callable[[str, float], None]] = None
    material_overrides: Mapping[str, str] = field(default_factory=dict)
    validate_only: bool = False
    mesh_info: Optional[MeshInfo] = None
    resolved_assignments: Optional[Dict[int, Dict[str, object]]] = None


def bake(
    mesh_path: Path,
    normal_pattern: Optional[str],
    output_pattern: str,
    options: Optional[BakeOptions] = None,
) -> Tuple[List[Path], List[str], List[Path]]:
    """Bake displacement maps from tangent-space normal maps."""

    bake_options = options or BakeOptions()
    _configure_deterministic_env(bake_options.deterministic)

    mesh_info = bake_options.mesh_info or inspect_mesh(mesh_path, loader=bake_options.loader)
    uv_set_name = _resolve_uv_set(mesh_info, bake_options.uv_set)
    uv_info = mesh_info.uv_sets[uv_set_name]

    assignments = bake_options.resolved_assignments or resolve_material_textures(
        mesh_info, uv_set_name, normal_pattern, bake_options.material_overrides
    )

    missing_summary = _format_missing_summary(assignments)
    if missing_summary:
        raise TextureAssignmentError(f"Missing UDIM tiles detected: {missing_summary}")

    tile_resolutions = _collect_tile_resolutions(assignments)
    log_lines: List[str] = []
    if not tile_resolutions:
        if uv_info.charts:
            raise TextureAssignmentError("Cannot bake because no texture tiles were resolved.")
        log_lines.append("UV set contains no charts; nothing to bake.")
        return [], log_lines, []

    raster_results = rasterize_uv_charts(
        uv_info, tile_resolutions, deterministic=bake_options.deterministic
    )

    sidecar_paths: List[Path] = []
    if bake_options.export_sidecars:
        sidecar_paths = export_sidecars(
            mesh_info,
            uv_set_name,
            assignments,
            deterministic=bake_options.deterministic,
            y_is_down=bake_options.y_is_down,
            precomputed=raster_results,
        )

    if bake_options.validate_only:
        log_lines.append("Validation mode: displacement solve skipped.")
        return [], log_lines, sidecar_paths

    if not raster_results:
        log_lines.append("No charts intersect resolved tiles; nothing to bake.")
        return [], log_lines, sidecar_paths

    if not output_pattern:
        raise TextureAssignmentError("Output pattern is required to bake displacement maps")

    tile_paths = _collect_tile_paths(assignments)
    if not tile_paths:
        raise TextureAssignmentError("No normal maps were resolved for the bake")

    amplitude = float(bake_options.amplitude)
    guard = _max_slope_guard(float(bake_options.max_slope))

    output_paths: List[Path] = []
    multi_tile = len(raster_results) > 1
    total_tiles = len(raster_results) or 1

    for index, tile_result in enumerate(raster_results):
        tile = int(tile_result.tile)
        normal_path = tile_paths.get(tile)
        if normal_path is None:
            raise TextureAssignmentError(f"No normal map found for UDIM {tile}")

        tile_heights, tile_logs = _solve_tile(
            tile_result,
            normal_path,
            bake_options,
            guard,
        )
        log_lines.extend(tile_logs)

        output_path = _resolve_output_path(output_pattern, tile, multi_tile)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tile_heights *= amplitude
        height_data = np.ascontiguousarray(tile_heights.astype(np.float32))
        metadata = {
            "n2d:space": "tangent",
            "n2d:units": "texel",
            "n2d:amplitude": amplitude,
            "Software": "normal2disp",
            "SourceMesh": str(mesh_info.path),
        }
        write_exr_channels(output_path, {"height": height_data}, metadata=metadata)
        output_paths.append(output_path)
        log_lines.append(f"Wrote tile {tile} to {output_path}")

        if bake_options.on_progress is not None:
            progress = float(index + 1) / float(total_tiles)
            bake_options.on_progress(f"tile {tile}", progress)

    return output_paths, log_lines, sidecar_paths


def resolve_material_textures(
    mesh_info: MeshInfo,
    uv_set: str,
    default_pattern: Optional[str],
    overrides: Mapping[str, str],
) -> Dict[int, Dict[str, object]]:
    """Resolve texture assignments per material and expand UDIM tiles."""

    if uv_set not in mesh_info.uv_sets:
        available = ", ".join(sorted(mesh_info.uv_sets)) or "<none>"
        raise TextureAssignmentError(f"UV set '{uv_set}' not found. Available sets: {available}")

    uv_info = mesh_info.uv_sets[uv_set]
    materials = dict(mesh_info.materials)

    normalized_overrides: Dict[int, str] = {}
    for key, pattern in overrides.items():
        material_id = _resolve_material_key(materials, key)
        if material_id is None:
            raise TextureAssignmentError(f"Unknown material override target '{key}'")
        normalized_overrides[material_id] = pattern

    assignments: Dict[int, Dict[str, object]] = {}

    for material_id, name in materials.items():
        pattern = normalized_overrides.get(material_id, default_pattern)

        required_tiles = set(
            int(tile) for tile in uv_info.per_material_udims.get(material_id, set())
        )

        tile_map: Dict[int, Path] = {}
        if pattern:
            tile_map = expand_udim_pattern(pattern)
        elif required_tiles:
            raise TextureAssignmentError(
                f"Material {material_id} ('{name}') requires textures but no pattern was provided"
            )

        found_tiles = set(tile_map.keys())
        missing_tiles = required_tiles - found_tiles

        assignments[material_id] = {
            "material_name": name,
            "pattern": pattern,
            "tiles_found": found_tiles,
            "tiles_required": required_tiles,
            "missing_tiles": missing_tiles,
            "tile_paths": tile_map,
        }

    return assignments


def _resolve_material_key(materials: Mapping[int, str], key: str) -> Optional[int]:
    try:
        index = int(key)
    except ValueError:
        candidates = [
            material_id for material_id, name in materials.items() if name.lower() == key.lower()
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise TextureAssignmentError(f"Material name '{key}' is ambiguous")
        return None
    else:
        return index if index in materials else None


def export_sidecars(
    mesh_info: MeshInfo,
    uv_set: str,
    assignments: Mapping[int, Mapping[str, object]],
    *,
    deterministic: bool = False,
    y_is_down: bool = False,
    precomputed: Optional[List[TileRasterResult]] = None,
) -> List[Path]:
    """Rasterise charts and write sidecar assets for ``uv_set``."""

    if uv_set not in mesh_info.uv_sets:
        available = ", ".join(sorted(mesh_info.uv_sets)) or "<none>"
        raise TextureAssignmentError(f"UV set '{uv_set}' not found. Available sets: {available}")

    uv_info = mesh_info.uv_sets[uv_set]
    tile_resolutions = _collect_tile_resolutions(assignments)
    if not tile_resolutions:
        raise TextureAssignmentError(
            "Cannot export sidecars because no texture tiles were resolved."
        )

    if precomputed is not None:
        raster_results = precomputed
    else:
        raster_results = rasterize_uv_charts(uv_info, tile_resolutions, deterministic=deterministic)

    output_dir = _default_sidecar_directory(mesh_info.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_channel = "-Y" if y_is_down else "+Y"
    sidecar_paths: List[Path] = []

    for result in raster_results:
        base_name = f"{uv_set}_{result.tile}"

        chart_path = output_dir / f"{base_name}_chart_id.exr"
        write_exr_channels(chart_path, {"chart_id": result.chart_mask})
        sidecar_paths.append(chart_path)

        boundary_path = output_dir / f"{base_name}_boundary.exr"
        write_exr_channels(boundary_path, {"boundary": result.boundary_mask})
        sidecar_paths.append(boundary_path)

        chart_table_path = output_dir / f"{base_name}_chart_table.json"
        table = _build_chart_table_payload(result, uv_info, y_channel)
        with chart_table_path.open("w", encoding="utf-8") as handle:
            json.dump(table, handle, indent=2)
            handle.write("\n")
        sidecar_paths.append(chart_table_path)

    return sidecar_paths


def _collect_tile_resolutions(
    assignments: Mapping[int, Mapping[str, object]]
) -> Dict[int, Tuple[int, int]]:
    tile_sizes: Dict[int, Tuple[int, int]] = {}
    tile_sources: Dict[int, List[Tuple[int, Path, Tuple[int, int]]]] = {}

    for material_id, data in assignments.items():
        tile_paths: Mapping[int, Path] = data.get("tile_paths", {})  # type: ignore[assignment]
        for tile, path in tile_paths.items():
            info = load_texture_info(Path(path))
            resolution = (info.height, info.width)
            existing = tile_sizes.get(tile)
            tile_sources.setdefault(tile, []).append((material_id, Path(path), resolution))

            if existing is None:
                tile_sizes[tile] = resolution
            elif existing != resolution:
                message = _format_resolution_conflict(tile, tile_sources[tile])
                raise TextureAssignmentError(message)

    return tile_sizes


def _format_resolution_conflict(
    tile: int, sources: Iterable[Tuple[int, Path, Tuple[int, int]]]
) -> str:
    parts = []
    for material_id, path, (height, width) in sources:
        parts.append(f"material {material_id} → {path} ({width}×{height})")
    joined = "; ".join(parts)
    return f"UDIM {tile} has mixed resolutions: {joined}"


def _default_sidecar_directory(mesh_path: Path) -> Path:
    return mesh_path.parent / f"{mesh_path.stem}_sidecars"


def _build_chart_table_payload(
    result: TileRasterResult, uv_info: UVSetInfo, y_channel: str
) -> Dict[str, object]:
    charts_payload = []
    for entry in sorted(result.charts, key=lambda chart: chart.local_id):
        charts_payload.append(
            {
                "id": int(entry.local_id),
                "flip_u": bool(entry.flip_u),
                "flip_v": bool(entry.flip_v),
                "pixel_count": int(entry.pixel_count),
                "bbox_uv": [float(value) for value in entry.bbox_uv],
                "bbox_px": [int(value) for value in entry.bbox_px],
            }
        )

    return {
        "tile": int(result.tile),
        "uv_set": uv_info.name,
        "charts": charts_payload,
        "y_channel": y_channel,
    }


def _resolve_uv_set(mesh_info: MeshInfo, requested: Optional[str]) -> str:
    if requested is not None:
        if requested not in mesh_info.uv_sets:
            available = ", ".join(sorted(mesh_info.uv_sets)) or "<none>"
            raise TextureAssignmentError(
                f"UV set '{requested}' not found. Available sets: {available}"
            )
        return requested

    if not mesh_info.uv_sets:
        raise TextureAssignmentError("Mesh does not contain any UV sets")

    return sorted(mesh_info.uv_sets)[0]


def _configure_deterministic_env(enabled: bool) -> None:
    if not enabled:
        return

    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ.setdefault("PYTHONHASHSEED", "0")


def _collect_tile_paths(assignments: Mapping[int, Mapping[str, object]]) -> Dict[int, Path]:
    tile_paths: Dict[int, Path] = {}
    for data in assignments.values():
        mapping: Mapping[int, Path] = data.get("tile_paths", {})  # type: ignore[assignment]
        for tile, path in mapping.items():
            tile_int = int(tile)
            tile_paths.setdefault(tile_int, Path(path))
    return dict(sorted(tile_paths.items()))


def _resolve_output_path(pattern: str, tile: int, multi_tile: bool) -> Path:
    normalized = Path(pattern).expanduser()
    pattern_str = str(normalized)

    placeholder: Optional[str] = None
    if "<UDIM>" in pattern_str:
        placeholder = "<UDIM>"
    elif "%04d" in pattern_str:
        placeholder = "%04d"

    if placeholder is None:
        if multi_tile:
            raise UDIMError("Output pattern must include <UDIM> when baking multiple tiles")
        return normalized

    formatted = pattern_str.replace(placeholder, f"{tile:04d}")
    return Path(formatted)


def _determine_normal_policy(channels: int, normalization: NormalizationMode) -> Tuple[str, bool]:
    mode = normalization.lower()
    if mode == "auto":
        decode_mode = "xyz" if channels >= 3 else "xy"
        normalize_vectors = True
    elif mode == "xyz":
        if channels < 3:
            raise TextureAssignmentError(
                "XYZ normalization requires a normal map with at least three channels"
            )
        decode_mode = "xyz"
        normalize_vectors = True
    elif mode == "xy":
        if channels < 2:
            raise TextureAssignmentError(
                "XY normalization requires a normal map with at least two channels"
            )
        decode_mode = "xy"
        normalize_vectors = True
    elif mode == "none":
        decode_mode = "xyz" if channels >= 3 else "xy"
        normalize_vectors = False
    else:  # pragma: no cover - defensive programming
        raise TextureAssignmentError(f"Unknown normalization mode '{normalization}'")

    if decode_mode == "xyz" and channels < 3:
        raise TextureAssignmentError("Normal map does not provide enough channels for XYZ decoding")
    if decode_mode == "xy" and channels < 2:
        raise TextureAssignmentError("Normal map does not provide enough channels for XY decoding")

    return decode_mode, normalize_vectors


def _max_slope_guard(max_slope: float) -> float:
    if max_slope <= 0:
        return 1.0
    return 1.0 / math.sqrt(1.0 + float(max_slope) * float(max_slope))


def _solve_tile(
    tile_result: TileRasterResult,
    normal_path: Path,
    options: BakeOptions,
    guard: float,
) -> Tuple[np.ndarray, List[str]]:
    pixels = read_texture_pixels(normal_path)
    height = int(tile_result.height)
    width = int(tile_result.width)

    if pixels.shape[0] != height or pixels.shape[1] != width:
        raise ImageIOError(
            f"Texture '{normal_path}' resolution {pixels.shape[1]}×{pixels.shape[0]} "
            f"does not match expected {width}×{height}"
        )
    if pixels.shape[2] < 2:
        raise TextureAssignmentError("Normal maps must have at least two channels")

    decode_mode, normalize_vectors = _determine_normal_policy(
        pixels.shape[2], options.normalization
    )

    base_x = np.asarray(pixels[..., 0], dtype=np.float64) * 2.0 - 1.0
    base_y = np.asarray(pixels[..., 1], dtype=np.float64) * 2.0 - 1.0
    if decode_mode == "xyz":
        base_z = np.asarray(pixels[..., 2], dtype=np.float64) * 2.0 - 1.0
    else:
        squared = np.clip(1.0 - base_x * base_x - base_y * base_y, 0.0, None)
        base_z = np.sqrt(squared)

    if not (
        np.all(np.isfinite(base_x)) and np.all(np.isfinite(base_y)) and np.all(np.isfinite(base_z))
    ):
        raise ImageIOError(f"Texture '{normal_path}' contains invalid normal components")

    tile_heights = np.zeros((height, width), dtype=np.float64)
    chart_logs: List[str] = []

    for entry in sorted(tile_result.charts, key=lambda chart: chart.local_id):
        mask = tile_result.chart_mask == entry.local_id
        if not np.any(mask):
            continue

        solution, iterations, residual = _solve_chart(
            int(tile_result.tile),
            int(entry.chart_id),
            mask,
            base_x,
            base_y,
            base_z,
            normalize_vectors,
            guard,
            options.y_is_down,
            bool(entry.flip_u),
            bool(entry.flip_v),
            float(options.cg_tol),
            int(options.cg_maxiter),
        )
        tile_heights[mask] = solution
        chart_logs.append(
            f"Tile {tile_result.tile} chart {entry.chart_id}: iter={iterations}, residual={residual:.3e}"
        )

    if not chart_logs:
        chart_logs.append(f"Tile {tile_result.tile}: no chart pixels")

    return tile_heights, chart_logs


def _solve_chart(
    tile: int,
    chart_id: int,
    mask: np.ndarray,
    base_x: np.ndarray,
    base_y: np.ndarray,
    base_z: np.ndarray,
    normalize_vectors: bool,
    guard: float,
    y_is_down: bool,
    flip_u: bool,
    flip_v: bool,
    cg_tol: float,
    cg_maxiter: int,
) -> Tuple[np.ndarray, int, float]:
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return np.zeros(0, dtype=np.float64), 0, 0.0

    height, width = mask_bool.shape

    x = np.array(base_x[mask_bool], dtype=np.float64, copy=True)
    y = np.array(base_y[mask_bool], dtype=np.float64, copy=True)
    z = np.array(base_z[mask_bool], dtype=np.float64, copy=True)

    if y_is_down:
        y *= -1.0
    if flip_u:
        x *= -1.0
    if flip_v:
        y *= -1.0

    if normalize_vectors:
        length = np.sqrt(np.maximum(x * x + y * y + z * z, 1e-20))
        x /= length
        y /= length
        z /= length

    z = np.maximum(z, guard)

    du_values = -x / z
    dv_values = -y / z

    if not (
        np.all(np.isfinite(du_values)) and np.all(np.isfinite(dv_values)) and np.all(np.isfinite(z))
    ):
        raise SolverError(f"Invalid slope values for tile {tile} chart {chart_id}")

    du = np.zeros((height, width), dtype=np.float64)
    dv = np.zeros((height, width), dtype=np.float64)
    z_full = np.zeros((height, width), dtype=np.float64)

    du[mask_bool] = du_values
    dv[mask_bool] = dv_values
    z_full[mask_bool] = z

    divergence = _compute_divergence(du, dv, mask_bool)
    index_map, positions = _build_index_map(mask_bool)
    laplacian = _build_laplacian(mask_bool, index_map, positions)
    rhs = divergence[mask_bool]

    anchor = _select_anchor_index(z_full, positions, width, index_map)
    _apply_anchor(laplacian, rhs, anchor)

    matrix = laplacian.tocsr()
    diag = matrix.diagonal()
    inv_diag = np.where(diag != 0.0, 1.0 / diag, 1.0)
    preconditioner = LinearOperator(matrix.shape, matvec=lambda x: inv_diag * x)

    stats = _CGStats()
    solution, info = cg(
        matrix,
        rhs,
        tol=cg_tol,
        maxiter=cg_maxiter,
        M=preconditioner,
        callback=stats,
    )

    if info != 0:
        residual = float(np.linalg.norm(matrix @ solution - rhs))
        raise SolverError(
            f"CG solver failed for tile {tile} chart {chart_id}: info={info}, residual={residual:.3e}"
        )

    residual = float(np.linalg.norm(matrix @ solution - rhs))
    return solution.astype(np.float64, copy=False), stats.iterations, residual


def _compute_divergence(du: np.ndarray, dv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    du = np.asarray(du, dtype=np.float64)
    dv = np.asarray(dv, dtype=np.float64)
    mask_bool = mask.astype(bool)

    du_right = np.empty_like(du)
    du_right[:, :-1] = du[:, 1:]
    du_right[:, -1] = du[:, -1]
    right_inside = np.zeros_like(mask_bool, dtype=bool)
    right_inside[:, :-1] = mask_bool[:, 1:]
    flux_right = 0.5 * (du + np.where(right_inside, du_right, du))

    du_left = np.empty_like(du)
    du_left[:, 1:] = du[:, :-1]
    du_left[:, 0] = du[:, 0]
    left_inside = np.zeros_like(mask_bool, dtype=bool)
    left_inside[:, 1:] = mask_bool[:, :-1]
    flux_left = 0.5 * (du + np.where(left_inside, du_left, du))

    dv_down = np.empty_like(dv)
    dv_down[:-1, :] = dv[1:, :]
    dv_down[-1, :] = dv[-1, :]
    down_inside = np.zeros_like(mask_bool, dtype=bool)
    down_inside[:-1, :] = mask_bool[1:, :]
    flux_down = 0.5 * (dv + np.where(down_inside, dv_down, dv))

    dv_up = np.empty_like(dv)
    dv_up[1:, :] = dv[:-1, :]
    dv_up[0, :] = dv[0, :]
    up_inside = np.zeros_like(mask_bool, dtype=bool)
    up_inside[1:, :] = mask_bool[:-1, :]
    flux_up = 0.5 * (dv + np.where(up_inside, dv_up, dv))

    divergence = (flux_right - flux_left) + (flux_down - flux_up)
    divergence[~mask_bool] = 0.0
    return divergence


def _build_index_map(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    index_map = -np.ones(mask.shape, dtype=np.int32)
    positions = np.argwhere(mask)
    for idx, (y, x) in enumerate(positions):
        index_map[int(y), int(x)] = idx
    return index_map, positions


def _build_laplacian(
    mask: np.ndarray, index_map: np.ndarray, positions: np.ndarray
) -> sparse.lil_matrix:
    count = positions.shape[0]
    laplacian = sparse.lil_matrix((count, count), dtype=np.float64)
    height, width = mask.shape

    for y, x in positions:
        row = int(index_map[int(y), int(x)])
        degree = 0
        for offset_y, offset_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = int(y) + offset_y
            nx = int(x) + offset_x
            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx]:
                neighbour = int(index_map[ny, nx])
                laplacian[row, neighbour] = -1.0
                degree += 1
        laplacian[row, row] = float(degree)

    return laplacian


def _apply_anchor(matrix: sparse.lil_matrix, rhs: np.ndarray, anchor: int) -> None:
    matrix.rows[anchor] = [anchor]
    matrix.data[anchor] = [1.0]
    rhs[anchor] = 0.0

    for row_index in range(matrix.shape[0]):
        if row_index == anchor:
            continue
        row = matrix.rows[row_index]
        data = matrix.data[row_index]
        for idx, column in enumerate(row):
            if column == anchor:
                row.pop(idx)
                data.pop(idx)
                break


def _select_anchor_index(
    z_full: np.ndarray, positions: np.ndarray, width: int, index_map: np.ndarray
) -> int:
    if positions.size == 0:
        return 0

    linear_positions = positions.astype(int, copy=False)
    z_mask = z_full[linear_positions[:, 0], linear_positions[:, 1]]
    if z_mask.size == 0:
        return 0

    max_z = float(np.max(z_mask))
    candidates = np.flatnonzero(np.isclose(z_mask, max_z, rtol=1e-6, atol=1e-9))
    if candidates.size == 0:
        candidates = np.array([0], dtype=int)

    linear_indices = linear_positions[:, 0] * width + linear_positions[:, 1]
    best = int(candidates[np.argmin(linear_indices[candidates])])
    anchor_position = linear_positions[best]
    return int(index_map[int(anchor_position[0]), int(anchor_position[1])])


def _format_missing_summary(assignments: Mapping[int, Mapping[str, object]]) -> str:
    parts: List[str] = []
    for material_id, data in sorted(assignments.items()):
        missing = sorted(int(tile) for tile in data.get("missing_tiles", set()))
        if missing:
            name = data.get("material_name", "<unknown>")
            parts.append(f"{material_id} ({name}): {missing}")
    return "; ".join(parts)


class _CGStats:
    """Track iteration counts for the conjugate gradient solver."""

    def __init__(self) -> None:
        self.iterations = 0

    def __call__(self, _vector: np.ndarray) -> None:
        self.iterations += 1
