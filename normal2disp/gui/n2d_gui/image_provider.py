"""Image provider for QML previews."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtGui import QColor, QImage
from PySide6.QtQuick import QQuickImageProvider

__all__ = ["N2DImageProvider"]


class N2DImageProvider(QQuickImageProvider):
    """Provide preview textures to the QML layer via ``image://`` URLs."""

    def __init__(self) -> None:
        super().__init__(QQuickImageProvider.Image)
        self._normal_image: Optional[QImage] = None
        self._revision = 0

    # Qt Quick calls ``requestImage`` from the GUI thread. We simply provide the
    # cached QImage (or a transparent fallback) without additional processing.
    def requestImage(self, identifier: str, size, requested_size) -> QImage:  # type: ignore[override]
        if identifier.startswith("normal") and self._normal_image is not None:
            if size is not None:
                size.setWidth(self._normal_image.width())
                size.setHeight(self._normal_image.height())
            return self._normal_image

        fallback = QImage(2, 2, QImage.Format_RGBA8888)
        fallback.fill(QColor(0, 0, 0, 0))
        if size is not None:
            size.setWidth(fallback.width())
            size.setHeight(fallback.height())
        return fallback

    def set_normal_image(self, image_path: Path) -> str:
        """Load ``image_path`` and return a cache-busting ``image://`` URL."""

        path = image_path.expanduser()
        image = QImage(str(path))
        if image.isNull():
            raise ValueError(f"Failed to load normal map preview: {path}")

        self._normal_image = image
        self._revision += 1
        return f"image://n2d/normal?r={self._revision}"

    def clear_normal_image(self) -> str:
        """Clear the cached normal image and invalidate the preview URL."""

        self._normal_image = None
        self._revision += 1
        return ""
