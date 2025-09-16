"""Core types and exceptions for normal2disp."""

from __future__ import annotations

__all__ = ["N2DError", "MeshLoadError"]


class N2DError(Exception):
    """Base exception for normal2disp errors."""


class MeshLoadError(N2DError):
    """Raised when a mesh cannot be loaded."""
