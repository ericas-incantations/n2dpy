"""Application entry point for the normal2disp GUI."""

from __future__ import annotations

import sys

from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication

from . import qml_path
from .backend import Backend

__all__ = ["main"]


def main() -> int:
    """Launch the GUI application."""
    QQuickStyle.setStyle("Material")

    app = QApplication(sys.argv)
    app.setOrganizationName("normal2disp")
    app.setApplicationName("normal2disp GUI")

    engine = QQmlApplicationEngine()

    backend = Backend()
    engine.rootContext().setContextProperty("appBackend", backend)

    qml_dir = qml_path()
    engine.addImportPath(str(qml_dir))
    main_window = qml_dir / "MainWindow.qml"
    engine.load(str(main_window))

    if not engine.rootObjects():
        return -1

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
