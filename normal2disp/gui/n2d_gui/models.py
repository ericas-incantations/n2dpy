"""Model definitions placeholder for milestone P0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PlaceholderOption:
    """Represents a placeholder option entry."""

    name: str


def default_options() -> List[PlaceholderOption]:
    """Return an empty list for now."""

    return []
