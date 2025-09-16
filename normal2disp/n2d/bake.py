"""Displacement bake orchestration (stub for future phases)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple
from typing import Literal

from .core import TextureAssignmentError
from .image_utils import expand_udim_pattern
from .inspect import MeshInfo

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
