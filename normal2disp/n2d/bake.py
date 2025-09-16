"""Displacement bake orchestration (stub for future phases)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from typing import Literal

from .core import TextureAssignmentError
from .image_utils import expand_udim_pattern, load_texture_info, write_exr_channels
from .inspect import MeshInfo, UVSetInfo
from .uv_raster import TileRasterResult, rasterize_uv_charts

NormalizationMode = Literal["auto", "xyz", "xy", "none"]
LoaderMode = Literal["auto", "pyassimp", "blender"]


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


def bake(
    mesh_path: Path,
    normal_pattern: str,
    output_pattern: str,
    options: Optional[BakeOptions] = None,
) -> Tuple[List[Path], List[str], List[Path]]:
    """Placeholder for the future baking pipeline implementation."""

    raise NotImplementedError("Bake pipeline will be implemented in a later phase.")


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
