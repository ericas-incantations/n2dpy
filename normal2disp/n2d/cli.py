"""Command line interface for the ``n2d`` tool."""

from __future__ import annotations

import json
import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Tuple


import click
from rich.console import Console
from rich.table import Table

from . import get_version
from .core import MeshLoadError
from .inspect import _ensure_pyassimp_dependencies, run_inspect

__all__ = ["main"]

_LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
    _LOGGER.debug("Logging configured (level=%s)", logging.getLevelName(level))


def _probe_module(module_name: str) -> Tuple[bool, str | None]:
    if module_name == "pyassimp":
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

        row_count = 0
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

        if row_count:
            console.print(chart_table)
    if inspect_json is not None:
        inspect_json.parent.mkdir(parents=True, exist_ok=True)
        with inspect_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        console.print(f"[green]Wrote inspection report to {inspect_json}[/green]")

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

