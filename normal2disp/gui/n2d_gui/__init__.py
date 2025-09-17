"""GUI package for the normal2disp application."""

from __future__ import annotations

from pathlib import Path

__all__ = ["qml_path"]


def qml_path() -> Path:
    """Return the root directory that stores the QML resources."""
    return Path(__file__).resolve().parent.parent / "qml"
