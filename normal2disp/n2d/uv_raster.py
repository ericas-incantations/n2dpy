"""Rasterisation utilities for UV charts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from .inspect import ChartInfo, UVSetInfo

__all__ = ["TileChartEntry", "TileRasterResult", "rasterize_uv_charts"]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TileChartEntry:
    """Summary information for a chart within a particular UDIM tile."""

    local_id: int
    chart_id: int
    flip_u: bool
    flip_v: bool
    pixel_count: int
    bbox_uv: Tuple[float, float, float, float]
    bbox_px: Tuple[int, int, int, int]


@dataclass(frozen=True)
class TileRasterResult:
    """Rasterisation output for a single UDIM tile."""

    tile: int
    width: int
    height: int
    chart_mask: np.ndarray
    boundary_mask: np.ndarray
    charts: List[TileChartEntry]


def rasterize_uv_charts(
    uv_info: UVSetInfo,
    tile_resolutions: Mapping[int, Tuple[int, int]],
    *,
    deterministic: bool = False,
) -> List[TileRasterResult]:
    """Rasterise the charts for ``uv_info`` using ``tile_resolutions``."""

    mesh_data = uv_info.mesh_data
    face_chart_ids = mesh_data.face_chart_ids
    chart_lookup: Dict[int, ChartInfo] = {chart.id: chart for chart in uv_info.charts}

    tile_faces: Dict[int, List[int]] = {}
    cross_tile_counts: Dict[int, int] = {}

    for face_index, tiles in enumerate(mesh_data.face_udims):
        if not tiles:
            continue
        if len(tiles) == 1:
            tile = next(iter(tiles))
            tile_faces.setdefault(tile, []).append(face_index)
        else:
            for tile in tiles:
                cross_tile_counts[tile] = cross_tile_counts.get(tile, 0) + 1

    if cross_tile_counts:
        warning = ", ".join(f"{tile}: {count}" for tile, count in sorted(cross_tile_counts.items()))
        _LOGGER.warning(
            "UV set %s has faces spanning multiple UDIM tiles (triangles per tile: %s)",
            uv_info.name,
            warning,
        )

    for tile in tile_faces:
        if tile not in tile_resolutions:
            raise ValueError(f"No resolution available for UDIM {tile}")

    results: List[TileRasterResult] = []

    for tile in sorted(tile_faces):
        height, width = tile_resolutions[tile]
        chart_faces = _collect_chart_faces_for_tile(tile_faces[tile], face_chart_ids)
        if deterministic:
            sorted_chart_ids = sorted(chart_faces)
        else:
            sorted_chart_ids = list(sorted(chart_faces))

        chart_mask = np.zeros((height, width), dtype=np.int32)
        chart_entries: List[TileChartEntry] = []

        tile_u, tile_v = _tile_to_indices(tile)
        for local_offset, chart_id in enumerate(sorted_chart_ids, start=1):
            face_indices = chart_faces[chart_id]
            uv_bounds: List[np.ndarray] = []
            for face_index in face_indices:
                face = mesh_data.faces[face_index]
                uv_coords = mesh_data.uv[face]
                local_uv = _localise_uv(uv_coords, tile_u, tile_v)
                uv_bounds.append(local_uv)
                _rasterize_triangle(chart_mask, local_uv, width, height, local_offset)

            uv_concat = np.concatenate(uv_bounds, axis=0) if uv_bounds else np.zeros((0, 2))
            bbox_uv = _compute_uv_bbox(uv_concat)
            pixel_count, bbox_px = _compute_pixel_stats(chart_mask, local_offset)

            chart = chart_lookup.get(chart_id)
            flip_u = chart.flip_u if chart else False
            flip_v = chart.flip_v if chart else False

            chart_entries.append(
                TileChartEntry(
                    local_id=local_offset,
                    chart_id=chart_id,
                    flip_u=flip_u,
                    flip_v=flip_v,
                    pixel_count=pixel_count,
                    bbox_uv=bbox_uv,
                    bbox_px=bbox_px,
                )
            )

        boundary_mask = _compute_boundary_mask(chart_mask)
        results.append(
            TileRasterResult(
                tile=tile,
                width=width,
                height=height,
                chart_mask=chart_mask,
                boundary_mask=boundary_mask,
                charts=chart_entries,
            )
        )

    return results


def _collect_chart_faces_for_tile(
    face_indices: Sequence[int], face_chart_ids: np.ndarray
) -> Dict[int, List[int]]:
    mapping: Dict[int, List[int]] = {}
    for face_index in face_indices:
        chart_id = int(face_chart_ids[face_index])
        if chart_id <= 0:
            continue
        mapping.setdefault(chart_id, []).append(face_index)
    return mapping


def _tile_to_indices(tile: int) -> Tuple[int, int]:
    offset = tile - 1001
    u = offset % 10
    v = offset // 10
    return u, v


def _localise_uv(uv_coords: np.ndarray, tile_u: int, tile_v: int) -> np.ndarray:
    local = np.empty_like(uv_coords, dtype=np.float64)
    local[:, 0] = uv_coords[:, 0] - float(tile_u)
    local[:, 1] = uv_coords[:, 1] - float(tile_v)
    return np.clip(local, 0.0, 1.0)


def _rasterize_triangle(
    mask: np.ndarray,
    uv_coords: np.ndarray,
    width: int,
    height: int,
    value: int,
) -> None:
    points = _uv_to_pixels(uv_coords, width, height)
    cv2.fillConvexPoly(mask, points, int(value), lineType=cv2.LINE_8)


def _uv_to_pixels(uv_coords: np.ndarray, width: int, height: int) -> np.ndarray:
    u = np.clip(uv_coords[:, 0], 0.0, 1.0)
    v = np.clip(uv_coords[:, 1], 0.0, 1.0)
    x = np.floor(np.clip(u * width, 0.0, width - 1e-6)).astype(np.int32)
    y = np.floor(np.clip((1.0 - v) * height, 0.0, height - 1e-6)).astype(np.int32)
    return np.stack([x, y], axis=1)


def _compute_uv_bbox(uv_coords: np.ndarray) -> Tuple[float, float, float, float]:
    if uv_coords.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    u_min = float(np.min(uv_coords[:, 0]))
    v_min = float(np.min(uv_coords[:, 1]))
    u_max = float(np.max(uv_coords[:, 0]))
    v_max = float(np.max(uv_coords[:, 1]))
    return (u_min, v_min, u_max, v_max)


def _compute_pixel_stats(
    mask: np.ndarray, chart_value: int
) -> Tuple[int, Tuple[int, int, int, int]]:
    positions = np.argwhere(mask == chart_value)
    if positions.size == 0:
        return 0, (0, 0, 0, 0)

    y_coords = positions[:, 0]
    x_coords = positions[:, 1]
    y0 = int(y_coords.min())
    y1 = int(y_coords.max()) + 1
    x0 = int(x_coords.min())
    x1 = int(x_coords.max()) + 1
    return int(len(positions)), (x0, y0, x1, y1)


def _compute_boundary_mask(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(mask, dtype=np.uint8)

    # Horizontal differences
    left = mask[:, :-1]
    right = mask[:, 1:]
    diff = (left != right) & ((left > 0) | (right > 0))
    boundary[:, :-1][diff] = 1
    boundary[:, 1:][diff] = 1

    # Vertical differences
    top = mask[:-1, :]
    bottom = mask[1:, :]
    diff = (top != bottom) & ((top > 0) | (bottom > 0))
    boundary[:-1, :][diff] = 1
    boundary[1:, :][diff] = 1

    return boundary

