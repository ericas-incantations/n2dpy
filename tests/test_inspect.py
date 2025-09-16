import json
import subprocess
import sys
from pathlib import Path


def _write_two_island_obj(path: Path) -> None:
    obj_data = """\
o TwoIslands
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
v 3 0 0
v 3 1 0
v 2 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vt 1 0
vt 0 0
vt 0 1
vt 1 1
f 1/1 2/2 3/3
f 1/1 3/3 4/4
f 5/5 6/6 7/7
f 5/5 7/7 8/8
"""
    path.write_text(obj_data, encoding="utf-8")


def test_inspect_cli_reports_mirroring(tmp_path: Path) -> None:
    obj_path = tmp_path / "two_islands.obj"
    _write_two_island_obj(obj_path)

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

    mirrored_charts = [chart for chart in uv_info["charts"] if chart["mirrored"]]
    assert mirrored_charts, "expected at least one mirrored chart"

    for chart in mirrored_charts:
        flip_flags = [bool(chart["flip_u"]), bool(chart["flip_v"])]
        assert sum(flip_flags) == 1

    non_mirrored = [chart for chart in uv_info["charts"] if not chart["mirrored"]]
    assert non_mirrored, "expected at least one non-mirrored chart"
