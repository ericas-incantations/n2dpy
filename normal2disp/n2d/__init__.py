"""Core package metadata and helpers for normal2disp."""

from __future__ import annotations

from importlib import metadata

__all__ = ["__version__", "get_version"]

__version__ = "0.1.0"


def get_version() -> str:
    """Return the installed package version."""
    try:
        return metadata.version("normal2disp")
    except metadata.PackageNotFoundError:
        return __version__
