"""Standalone entry point for PySide6 deployment tools."""

from __future__ import annotations

import sys

from normal2disp.gui.n2d_gui.app import main


def run() -> int:
    """Launch the GUI and return the exit code."""
    return main()


if __name__ == "__main__":
    sys.exit(run())
