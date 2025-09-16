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
from .core import MeshLoadError
from .inspect import run_inspect

__all__ = ["main"]

_LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
    _LOGGER.debug("Logging configured (level=%s)", logging.getLevelName(level))


def _probe_module(module_name: str) -> Tuple[bool, str | None]:
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
@click.pass_context
def inspect_command(ctx: click.Context, mesh_path: Path, inspect_json: Path | None) -> None:
    """Inspect mesh UV sets and UDIM coverage."""

    try:
        report = run_inspect(mesh_path)
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

    uv_sets: List[Dict[str, Any]] = list(report.get("uv_sets", []))
    if not uv_sets:
        table.add_row("—", "0", "—")
    else:
        for uv_set in uv_sets:
            udims = uv_set.get("udims", [])
            udim_text = ", ".join(str(tile) for tile in udims) if udims else "—"
            table.add_row(uv_set.get("name", "UV"), str(uv_set.get("chart_count", 0)), udim_text)

    console.print(table)

    if inspect_json is not None:
        inspect_json.parent.mkdir(parents=True, exist_ok=True)
        with inspect_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        console.print(f"[green]Wrote inspection report to {inspect_json}[/green]")
