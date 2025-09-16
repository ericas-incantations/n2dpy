import json
import subprocess
import sys
from pathlib import Path

from .helpers import write_dummy_png, write_two_island_asset


def test_inspect_cli_reports_mirroring(tmp_path: Path) -> None:
    obj_path = write_two_island_asset(tmp_path)


    json_path = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "-m",
        "normal2disp.n2d.cli",
        "inspect",
        str(obj_path),
        "--inspect-json",
        str(json_path),
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert result.returncode == 0

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert "uv_sets" in report

    uv_sets = report["uv_sets"]
    assert isinstance(uv_sets, dict)
    assert uv_sets

    first_uv_name = sorted(uv_sets)[0]
    uv_info = uv_sets[first_uv_name]
    assert uv_info["chart_count"] >= 2
    assert uv_info["udims"]
    assert report.get("materials")

    mirrored_charts = [chart for chart in uv_info["charts"] if chart["mirrored"]]
    assert mirrored_charts, "expected at least one mirrored chart"

    for chart in mirrored_charts:
        flip_flags = [bool(chart["flip_u"]), bool(chart["flip_v"])]
        assert sum(flip_flags) == 1
        assert "material_id" in chart
        assert chart.get("material_name") in {"matA", "matB"}

    non_mirrored = [chart for chart in uv_info["charts"] if not chart["mirrored"]]
    assert non_mirrored, "expected at least one non-mirrored chart"

    chart_materials = {chart["material_name"] for chart in uv_info["charts"]}
    assert {"matA", "matB"}.issubset(chart_materials)


def test_bake_validate_only_reports_missing_tiles(tmp_path: Path) -> None:
    obj_path = write_two_island_asset(tmp_path)
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    write_dummy_png(texture_dir / "normal_1001.png")


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
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0

    report = json.loads(result.stdout)
    materials = {material["name"]: material for material in report["materials"]}
    assert materials["matA"]["missing_tiles"] == []
    assert materials["matB"]["missing_tiles"] == [1002]


def test_bake_validate_only_with_overrides_succeeds(tmp_path: Path) -> None:
    obj_path = write_two_island_asset(tmp_path)
    texture_dir = tmp_path / "textures"
    texture_dir.mkdir()
    write_dummy_png(texture_dir / "normal_1001.png")
    write_dummy_png(texture_dir / "other_1002.png")


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
        f"matA={texture_dir / 'normal_<UDIM>.png'}",
        "--mat-normal",
        f"matB={texture_dir / 'other_<UDIM>.png'}",
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    report = json.loads(result.stdout)
    materials = {material["name"]: material for material in report["materials"]}
    assert materials["matA"]["tiles_found"] == [1001]
    assert materials["matB"]["tiles_found"] == [1002]
    assert materials["matA"]["missing_tiles"] == []
    assert materials["matB"]["missing_tiles"] == []

