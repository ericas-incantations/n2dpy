"""Image I/O helpers (implemented in later phases)."""

from __future__ import annotations

import glob
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .core import ImageIOError, UDIMError

__all__ = ["TextureInfo", "load_texture_info", "expand_udim_pattern"]

_LOGGER = logging.getLogger(__name__)

_UDIM_TOKEN = "<UDIM>"
_PRINTF_TOKEN = "%04d"


@dataclass(frozen=True)
class TextureInfo:
    """Metadata about a texture file discovered via OpenImageIO."""

    path: Path
    width: int
    height: int
    channels: int
    pixel_type: str
    colorspace: Optional[str]


def load_texture_info(path: Path) -> TextureInfo:
    """Load image metadata using OpenImageIO."""

    resolved = path.expanduser()

    _ensure_system_site_packages()
    try:
        import OpenImageIO as oiio
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ImageIOError("OpenImageIO is required for texture loading in this phase") from exc

    input_handle = oiio.ImageInput.open(str(resolved))
    if input_handle is None:  # pragma: no cover - depends on asset
        raise ImageIOError(f"Failed to open image '{resolved}': {oiio.geterror()}")

    try:
        spec = input_handle.spec()
        width = int(spec.width)
        height = int(spec.height)
        channels = int(spec.nchannels)
        pixel_type = str(spec.format)

        colorspace = None
        for attribute_name in ("oiio:ColorSpace", "oiio:colorspace"):
            value = spec.getattribute(attribute_name)
            if value:
                colorspace = str(value)
                break

        if colorspace and colorspace.lower() == "srgb":
            _LOGGER.warning(
                "Texture '%s' reports sRGB colorspace; treating as linear normal map.", resolved
            )

        return TextureInfo(
            path=resolved,
            width=width,
            height=height,
            channels=channels,
            pixel_type=pixel_type,
            colorspace=colorspace,
        )
    finally:
        input_handle.close()


def expand_udim_pattern(pattern: str) -> Dict[int, Path]:
    """Expand a UDIM filename pattern into discovered tiles on disk."""

    normalized = str(Path(pattern).expanduser())
    placeholder = _detect_placeholder(normalized)

    if placeholder is None:
        path = Path(normalized)
        if not path.exists():
            return {}
        load_texture_info(path)
        tile = _extract_tile_from_path(path)
        if tile is None:
            tile = 1001
        return {int(tile): path}

    glob_pattern = _pattern_to_glob(normalized, placeholder)
    regex = _pattern_to_regex(normalized, placeholder)

    matches: Dict[int, Path] = {}
    for candidate in sorted(glob.glob(glob_pattern)):
        match = regex.fullmatch(candidate)
        if not match:
            continue
        tile_value = int(match.group("udim"))
        if tile_value in matches:
            raise UDIMError(
                f"Multiple textures found for UDIM {tile_value} matching pattern '{pattern}'"
            )
        matches[tile_value] = Path(candidate)
        load_texture_info(Path(candidate))

    return dict(sorted(matches.items()))


def _detect_placeholder(pattern: str) -> Optional[str]:
    if _UDIM_TOKEN in pattern:
        return _UDIM_TOKEN
    if _PRINTF_TOKEN in pattern:
        return _PRINTF_TOKEN
    return None


def _pattern_to_glob(pattern: str, placeholder: str) -> str:
    replacement = "[0-9][0-9][0-9][0-9]"
    return pattern.replace(placeholder, replacement)


def _pattern_to_regex(pattern: str, placeholder: str) -> re.Pattern[str]:
    escaped = re.escape(pattern)
    token = re.escape(placeholder)
    pattern_regex = escaped.replace(token, r"(?P<udim>[0-9]{4})")
    return re.compile(pattern_regex)


def _extract_tile_from_path(path: Path) -> Optional[int]:
    match = re.search(r"(1\d{3})", path.name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:  # pragma: no cover - defensive
            return None
    return None


def _ensure_system_site_packages() -> None:
    system_site = Path("/usr/lib/python3/dist-packages")
    system_site_str = str(system_site)
    if system_site.exists() and system_site_str not in sys.path:
        sys.path.append(system_site_str)
