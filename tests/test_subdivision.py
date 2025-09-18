"""Tests for GUI subdivision and displacement helpers."""

from __future__ import annotations

import numpy as np
import pytest

from normal2disp.gui.n2d_gui.subdivision import (
    HeightField,
    HeightFieldError,
    HeightTile,
    LoopSubdivisionCache,
    MeshBuffers,
    generate_displacement,
)


def _plane_mesh() -> MeshBuffers:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    uv = np.array(
        [
            [0.0, 0.0],
            [0.999, 0.0],
            [0.999, 0.999],
            [0.0, 0.999],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return MeshBuffers(positions=positions, uv=uv, faces=faces)


def test_height_field_sampling_across_tiles() -> None:
    tile_1001 = HeightTile(
        tile=1001,
        data=np.zeros((2, 2), dtype=np.float32),
        amplitude=1.5,
        units="texel",
    )
    tile_1011 = HeightTile(
        tile=1011,
        data=np.full((2, 2), 4.0, dtype=np.float32),
        amplitude=None,
        units=None,
    )

    field = HeightField({1001: tile_1001, 1011: tile_1011})

    samples = field.sample(np.array([[0.25, 0.25], [0.25, 1.25]], dtype=np.float64))
    assert np.allclose(samples[0], 0.0)
    assert np.allclose(samples[1], 4.0)

    with pytest.raises(HeightFieldError):
        field.sample(np.array([[1.2, 0.1]], dtype=np.float64))


def test_generate_displacement_and_cache() -> None:
    mesh = _plane_mesh()
    cache = LoopSubdivisionCache(mesh)

    height_tile = HeightTile(
        tile=1001,
        data=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        amplitude=1.0,
        units="texel",
    )
    field = HeightField({1001: height_tile})

    height_cache: dict[int, np.ndarray] = {}

    result = generate_displacement(cache, field, level=0, scale=1.0, height_cache=height_cache)

    assert result.triangle_count == 2
    assert np.allclose(result.heights[:4], [0.0, 1.0, 3.0, 2.0], atol=5e-3)
    assert np.allclose(result.positions[:4, 2], result.heights[:4], atol=5e-3)
    normal_lengths = np.linalg.norm(result.normals[:4], axis=1)
    assert np.allclose(normal_lengths, 1.0, atol=5e-3)
    assert np.all(result.normals[:4, 2] > 0)
    assert 0 in height_cache

    reused = generate_displacement(
        cache,
        field,
        level=0,
        scale=2.0,
        precomputed_heights=height_cache,
    )
    assert np.allclose(reused.positions[:4, 2], result.heights[:4] * 2.0, atol=1e-4)
    assert np.allclose(reused.heights[:4], result.heights[:4], atol=1e-4)


def test_subdivision_level_bounds() -> None:
    cache = LoopSubdivisionCache(_plane_mesh())
    with pytest.raises(ValueError):
        cache.mesh_for_level(-1)
    with pytest.raises(ValueError):
        cache.mesh_for_level(6)
