"""Command line interface for the ``n2d`` tool."""

from __future__ import annotations

import json
import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
from rich.console import Console
from rich.table import Table

from . import get_version
from .core import ImageIOError, MeshLoadError, TextureAssignmentError, UDIMError
from .bake import BakeOptions, export_sidecars, resolve_material_textures
from .inspect import _ensure_pyassimp_dependencies, inspect_mesh, run_inspect

__all__ = ["main"]

_LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
    _LOGGER.debug("Logging configured (level=%s)", logging.getLevelName(level))


def _probe_module(module_name: str) -> Tuple[bool, str | None]:
    if module_name in {"pyassimp", "OpenImageIO"}:
        try:
            _ensure_pyassimp_dependencies()
        except MeshLoadError:
            # Dependencies missing; fall back to standard import attempt.
            pass

    try:
        module = import_module(module_name)
    except Exception as exc:  # pragma: no cover - import failure depends on environment
        return False, str(exc)

    version = getattr(module, "__version__", None)
    if module_name == "OpenImageIO" and version is None:
        version = getattr(module, "VERSION", None)
        if isinstance(version, int):
            version = str(version)
    detail = f"v{version}" if version else None
    return True, detail


def _parse_material_override(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise click.BadParameter(
            "Material overrides must use the form NameOrIndex=pattern", param="--mat-normal"
        )

    key, pattern = value.split("=", 1)
    key = key.strip()
    pattern = pattern.strip()
    if not key or not pattern:
        raise click.BadParameter(
            "Material overrides must use the form NameOrIndex=pattern", param="--mat-normal"
        )

    return key, pattern


@click.group()
@click.option("--verbose", is_flag=True, help="Enable verbose logging output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Entrypoint for the ``n2d`` command."""

    _configure_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["console"] = Console()
    ctx.obj["verbose"] = bool(verbose)


@main.command()
def version() -> None:
    """Show package version and capability probes."""

    pkg_version = get_version()
    click.echo(f"normal2disp {pkg_version}")

    for module_name in ("OpenImageIO", "pyassimp"):
        available, detail = _probe_module(module_name)
        status = "yes" if available else "no"
        if detail:
            status = f"{status} ({detail})"
        click.echo(f"{module_name}: {status}")


@main.command(name="inspect")
@click.argument("mesh_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option(
    "--inspect-json",
    type=click.Path(path_type=Path),
    help="Write the inspection report to this JSON file.",
)
@click.option(
    "--loader",
    type=click.Choice(["auto", "pyassimp"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Mesh loader backend",
)
@click.pass_context
def inspect_command(
    ctx: click.Context,
    mesh_path: Path,
    inspect_json: Path | None,
    loader: str,
) -> None:
    """Inspect mesh UV sets and UDIM coverage."""

    try:
        report = run_inspect(mesh_path, loader=loader)
    except MeshLoadError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safeguard
        raise click.ClickException(f"Failed to inspect mesh: {exc}") from exc

    ctx_obj: Dict[str, Any] = ctx.obj if isinstance(ctx.obj, dict) else {}
    console = ctx_obj.get("console")
    if not isinstance(console, Console):
        console = Console()
    table = Table(title=str(mesh_path))
    table.add_column("UV Set")
    table.add_column("Charts", justify="right")
    table.add_column("UDIMs")

    uv_sets: Dict[str, Dict[str, Any]] = dict(report.get("uv_sets", {}))
    if not uv_sets:
        table.add_row("—", "0", "—")
    else:
        for uv_name in sorted(uv_sets):
            uv_set = uv_sets[uv_name]
            udims = uv_set.get("udims", [])
            udim_text = ", ".join(str(tile) for tile in udims) if udims else "—"
            table.add_row(uv_name, str(uv_set.get("chart_count", 0)), udim_text)

    console.print(table)

    verbose_enabled = bool(ctx_obj.get("verbose"))
    if verbose_enabled and uv_sets:
        chart_table = Table(title="Chart Flags")
        chart_table.add_column("UV Set")
        chart_table.add_column("Chart", justify="right")
        chart_table.add_column("Faces", justify="right")
        chart_table.add_column("Mirrored")
        chart_table.add_column("Flip U")
        chart_table.add_column("Flip V")

        material_table = Table(title="Chart Materials")
        material_table.add_column("UV Set")
        material_table.add_column("Chart", justify="right")
        material_table.add_column("Material ID", justify="right")
        material_table.add_column("Material Name")

        row_count = 0
        material_row_count = 0
        for uv_name in sorted(uv_sets):
            for chart in uv_sets[uv_name].get("charts", []):
                row_count += 1
                chart_table.add_row(
                    uv_name,
                    str(chart.get("id", "")),
                    str(chart.get("face_count", "")),
                    "yes" if chart.get("mirrored") else "no",
                    "yes" if chart.get("flip_u") else "no",
                    "yes" if chart.get("flip_v") else "no",
                )
                material_row_count += 1
                material_table.add_row(
                    uv_name,
                    str(chart.get("id", "")),
                    str(chart.get("material_id", "")),
                    chart.get("material_name", ""),
                )

        if row_count:
            console.print(chart_table)
        if material_row_count:
            console.print(material_table)

    if inspect_json is not None:
        inspect_json.parent.mkdir(parents=True, exist_ok=True)
        with inspect_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        console.print(f"[green]Wrote inspection report to {inspect_json}[/green]")


@main.command(name="bake")
@click.argument("mesh_path", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--normal", "default_normal", type=str, help="Default normal map pattern")
@click.option(
    "--mat-normal",
    "material_normals",
    multiple=True,
    help="Material override in the form NameOrIndex=pattern",
)
@click.option("--uv-set", "uv_set", type=str, help="UV set to use (defaults to first)")
@click.option("--y-is-down", is_flag=True, help="Treat +Y normals as pointing down")
@click.option(
    "--normalization",
    type=click.Choice(["auto", "xyz", "xy", "none"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Normal map normalization policy",
)
@click.option("--validate-only", is_flag=True, help="Only validate texture assignments")
@click.option(
    "--export-sidecars", "export_sidecars_flag", is_flag=True, help="Write chart masks and tables"
)
@click.option("--deterministic", is_flag=True, help="Force deterministic processing order")
@click.option(
    "--inspect-json",
    type=click.Path(path_type=Path),
    help="Write validation report to this JSON file.",
)
@click.option(
    "--loader",
    type=click.Choice(["auto", "pyassimp"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Mesh loader backend",
)
@click.pass_context
def bake_command(
    ctx: click.Context,
    mesh_path: Path,
    default_normal: str | None,
    material_normals: Tuple[str, ...],
    uv_set: str | None,
    y_is_down: bool,
    normalization: str,
    validate_only: bool,
    export_sidecars_flag: bool,
    deterministic: bool,
    inspect_json: Path | None,
    loader: str,
) -> None:
    """Validate and orchestrate displacement baking."""

    overrides: Dict[str, str] = {}
    for entry in material_normals:
        key, pattern = _parse_material_override(entry)
        overrides[key] = pattern

    try:
        mesh_info = inspect_mesh(mesh_path, loader=loader)
    except MeshLoadError as exc:
        raise click.ClickException(str(exc)) from exc

    uv_set_name = uv_set
    if uv_set_name is None:
        if mesh_info.uv_sets:
            uv_set_name = sorted(mesh_info.uv_sets)[0]
        else:
            raise click.ClickException("Mesh does not contain any UV sets")

    options = BakeOptions(
        uv_set=uv_set_name,
        y_is_down=y_is_down,
        normalization=normalization,
        loader=loader,
        export_sidecars=export_sidecars_flag,
        deterministic=deterministic,
    )
    ctx.obj = ctx.obj or {}
    if isinstance(ctx.obj, dict):
        ctx.obj["bake_options"] = options

    if not validate_only:
        raise click.ClickException("Only --validate-only mode is supported in this phase")

    try:
        assignments = resolve_material_textures(mesh_info, uv_set_name, default_normal, overrides)
    except (TextureAssignmentError, ImageIOError, UDIMError) as exc:
        raise click.ClickException(str(exc)) from exc

    materials_report = []
    missing_materials: Dict[int, Dict[str, object]] = {}
    for material_id, data in sorted(assignments.items()):
        tiles_found = sorted(int(tile) for tile in data["tiles_found"])
        tiles_required = sorted(int(tile) for tile in data["tiles_required"])
        missing_tiles = sorted(int(tile) for tile in data["missing_tiles"])
        if missing_tiles:
            missing_materials[material_id] = data

        materials_report.append(
            {
                "id": material_id,
                "name": data["material_name"],
                "pattern": data["pattern"],
                "tiles_found": tiles_found,
                "tiles_required": tiles_required,
                "missing_tiles": missing_tiles,
                "tile_paths": {str(tile): str(path) for tile, path in data["tile_paths"].items()},
            }
        )

    sidecar_paths: List[Path] = []
    if export_sidecars_flag and not missing_materials:
        try:
            sidecar_paths = export_sidecars(
                mesh_info,
                uv_set_name,
                assignments,
                deterministic=deterministic,
                y_is_down=y_is_down,
            )
        except (TextureAssignmentError, ImageIOError) as exc:
            raise click.ClickException(str(exc)) from exc

    report = {"mesh": str(mesh_path), "uv_set": uv_set_name, "materials": materials_report}
    if sidecar_paths:
        report["sidecars"] = [str(path) for path in sidecar_paths]

    click.echo(json.dumps(report, indent=2))

    if inspect_json is not None:
        inspect_json.parent.mkdir(parents=True, exist_ok=True)
        with inspect_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")

    if missing_materials:
        summary = ", ".join(
            f"{material_id} ({assignments[material_id]['material_name']}): {sorted(assignments[material_id]['missing_tiles'])}"
            for material_id in sorted(missing_materials)
        )
        raise click.ClickException(f"Missing UDIM tiles detected: {summary}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
