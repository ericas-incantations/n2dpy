"""Displacement bake orchestration (stub for future phases)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from typing import Literal

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
