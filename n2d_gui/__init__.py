"""Top-level package wrapper that exposes the GUI modules."""

from __future__ import annotations

from importlib import import_module
from typing import Any
import sys as _sys

_base_pkg = import_module("normal2disp.gui.n2d_gui")

__all__ = getattr(_base_pkg, "__all__", [])


def __getattr__(name: str) -> Any:
    return getattr(_base_pkg, name)


for _module_name in (
    "app",
    "backend",
    "jobs",
    "image_provider",
    "models",
    "subdivision",
    "viewport",
):
    _sys.modules[f"{__name__}.{_module_name}"] = import_module(
        f"normal2disp.gui.n2d_gui.{_module_name}"
    )
