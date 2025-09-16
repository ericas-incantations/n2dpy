"""Core types and exceptions for normal2disp."""

from __future__ import annotations

__all__ = [
    "N2DError",
    "MeshLoadError",
    "ImageIOError",
    "UDIMError",
    "TextureAssignmentError",
]


class N2DError(Exception):
    """Base exception for normal2disp errors."""


class MeshLoadError(N2DError):
    """Raised when a mesh cannot be loaded."""


class ImageIOError(N2DError):
    """Raised when image input/output fails."""


class UDIMError(N2DError):
    """Raised when UDIM expansion or validation fails."""


class TextureAssignmentError(N2DError):
    """Raised when CLI texture assignment arguments are invalid."""
