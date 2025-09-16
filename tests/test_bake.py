from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .helpers import read_exr_pixels, write_dummy_png, write_two_island_asset


def _sidecar_dir(mesh_path: Path) -> Path:
    return mesh_path.parent / f"{mesh_path.stem}_sidecars"


def _run_bake_command(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, capture_output=True, text=True)


def test_bake_export_sidecars_produces_assets(tmp_path: Path) -> None:
    obj_path = write_two_island_asset(tmp_path)
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    write_dummy_png(texture_dir / "normal_1001.png", size=256)
    write_dummy_png(texture_dir / "normal_1002.png", size=256)

    cmd = [
        sys.executable,
        "-m",
        "normal2disp.n2d.cli",
        "bake",
        str(obj_path),
        "--validate-only",
        "--uv-set",
        "UV0",
        "--normal",
        str(texture_dir / "normal_<UDIM>.png"),
        "--export-sidecars",
    ]

    result = _run_bake_command(cmd)
    report = json.loads(result.stdout)
    assert "sidecars" in report

    sidecar_dir = _sidecar_dir(obj_path)
    expected_tiles = [1001, 1002]
    for tile in expected_tiles:
        chart_id_path = sidecar_dir / f"UV0_{tile}_chart_id.exr"
        boundary_path = sidecar_dir / f"UV0_{tile}_boundary.exr"
        table_path = sidecar_dir / f"UV0_{tile}_chart_table.json"

        assert chart_id_path.exists()
        assert boundary_path.exists()
        assert table_path.exists()

        chart_pixels = read_exr_pixels(chart_id_path)
        boundary_pixels = read_exr_pixels(boundary_path)
        assert chart_pixels.shape[2] == 1
        assert boundary_pixels.shape[2] == 1

        chart_mask = np.squeeze(chart_pixels)
        boundary_mask = np.squeeze(boundary_pixels)
        assert chart_mask.shape == boundary_mask.shape == (256, 256)
        assert np.max(chart_mask) >= 1
        assert np.count_nonzero(boundary_mask) > 0

        table = json.loads(table_path.read_text(encoding="utf-8"))
        assert table["tile"] == tile
        assert table["uv_set"] == "UV0"
        assert table["y_channel"] in {"+Y", "-Y"}
        assert table["charts"]
        for entry in table["charts"]:
            assert entry["pixel_count"] > 0
            assert len(entry["bbox_uv"]) == 4
            assert len(entry["bbox_px"]) == 4


def test_bake_export_sidecars_deterministic(tmp_path: Path) -> None:
    obj_path = write_two_island_asset(tmp_path)
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    write_dummy_png(texture_dir / "normal_1001.png", size=128)
    write_dummy_png(texture_dir / "normal_1002.png", size=128)

    base_cmd = [
        sys.executable,
        "-m",
        "normal2disp.n2d.cli",
        "bake",
        str(obj_path),
        "--validate-only",
        "--uv-set",
        "UV0",
        "--normal",
        str(texture_dir / "normal_<UDIM>.png"),
        "--export-sidecars",
        "--deterministic",
    ]

    _run_bake_command(base_cmd)
    sidecar_dir = _sidecar_dir(obj_path)
    first_masks = {
        path.name: read_exr_pixels(path) for path in sorted(sidecar_dir.glob("UV0_*_chart_id.exr"))
    }

    _run_bake_command(base_cmd)
    second_masks = {
        path.name: read_exr_pixels(path) for path in sorted(sidecar_dir.glob("UV0_*_chart_id.exr"))
    }

    assert first_masks.keys() == second_masks.keys()
    for name in first_masks:
        assert np.array_equal(first_masks[name], second_masks[name])


def test_bake_export_sidecars_mixed_resolution_fails(tmp_path: Path) -> None:
    obj_path = write_two_island_asset(tmp_path)
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    write_dummy_png(texture_dir / "matA_1001.png", size=128)
    write_dummy_png(texture_dir / "matA_1002.png", size=128)
    write_dummy_png(texture_dir / "matB_1001.png", size=64)
    write_dummy_png(texture_dir / "matB_1002.png", size=64)

    cmd = [
        sys.executable,
        "-m",
        "normal2disp.n2d.cli",
        "bake",
        str(obj_path),
        "--validate-only",
        "--uv-set",
        "UV0",
        "--mat-normal",
        f"matA={texture_dir / 'matA_<UDIM>.png'}",
        "--mat-normal",
        f"matB={texture_dir / 'matB_<UDIM>.png'}",
        "--export-sidecars",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "mixed resolutions" in (result.stderr or result.stdout)
