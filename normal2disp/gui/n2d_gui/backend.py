"""Qt bridge exposing normal2disp functionality to QML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PySide6.QtCore import QObject, Property, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog

from .jobs import InspectJob


class Backend(QObject):
    """QObject bridge that exposes mesh inspection to the QML layer."""

    meshPathChanged = Signal()
    materialsChanged = Signal()
    uvSetsChanged = Signal()
    udimTilesChanged = Signal()
    inspectSummaryChanged = Signal()
    warningSummaryChanged = Signal()
    statusMessageChanged = Signal()
    logTextChanged = Signal()
    inspectRunningChanged = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._mesh_path: str = ""
        self._materials: List[Dict[str, Any]] = []
        self._uv_sets: List[Dict[str, Any]] = []
        self._udim_tiles: List[int] = []
        self._inspect_summary: str = ""
        self._warning_summary: str = ""
        self._status_message: str = "Ready"
        self._log_lines: List[str] = [self._status_message]
        self._inspect_running = False
        self._inspect_job: Optional[InspectJob] = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(150)
        self._poll_timer.timeout.connect(self._poll_job_queue)

    # ------------------------------------------------------------------
    # Properties exposed to QML
    # ------------------------------------------------------------------
    @Property(str, notify=meshPathChanged)
    def meshPath(self) -> str:
        return self._mesh_path

    @Property("QVariant", notify=materialsChanged)
    def materials(self) -> List[Dict[str, Any]]:
        return list(self._materials)

    @Property(int, notify=materialsChanged)
    def materialCount(self) -> int:
        return len(self._materials)

    @Property("QVariant", notify=uvSetsChanged)
    def uvSets(self) -> List[Dict[str, Any]]:
        return list(self._uv_sets)

    @Property(int, notify=uvSetsChanged)
    def uvSetCount(self) -> int:
        return len(self._uv_sets)

    @Property("QVariant", notify=udimTilesChanged)
    def udimTiles(self) -> List[int]:
        return list(self._udim_tiles)

    @Property(int, notify=udimTilesChanged)
    def udimTileCount(self) -> int:
        return len(self._udim_tiles)

    @Property(str, notify=inspectSummaryChanged)
    def inspectSummary(self) -> str:
        return self._inspect_summary

    @Property(str, notify=warningSummaryChanged)
    def warningSummary(self) -> str:
        return self._warning_summary

    @Property(str, notify=statusMessageChanged)
    def statusMessage(self) -> str:
        return self._status_message

    @Property(str, notify=logTextChanged)
    def logText(self) -> str:
        return "\n".join(self._log_lines)

    @Property(bool, notify=inspectRunningChanged)
    def inspectRunning(self) -> bool:
        return self._inspect_running

    # ------------------------------------------------------------------
    # Invokable methods
    # ------------------------------------------------------------------
    @Slot(str, result=str)
    def browseMesh(self, start_path: str = "") -> str:
        """Open a file dialog to choose a mesh and kick off inspection."""

        directory = start_path or str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            None,
            "Select mesh",
            directory,
            "Meshes (*.fbx *.obj *.gltf *.glb);;All files (*)",
        )
        if selected:
            self._set_mesh_path(selected)
            self.runInspect(selected)
        return selected

    @Slot(str, result=bool)
    def runInspect(self, mesh_path: str) -> bool:
        """Inspect ``mesh_path`` in a background process."""

        if self._inspect_running:
            # Ignore repeated requests while a job is active.
            self._append_log("Inspect already running; ignoring new request.")
            return False

        path = Path(mesh_path).expanduser()
        if not mesh_path:
            self._set_status_message("Select a mesh to inspect.")
            return False
        if not path.exists():
            self._set_status_message(f"Mesh not found: {path}")
            self._append_log(f"Mesh not found: {path}")
            return False

        self._set_mesh_path(str(path))
        self._set_status_message(f"Inspecting {path.name}…")
        self._append_log(f"Running inspect on {path}")

        self._inspect_job = InspectJob(path)
        self._inspect_job.start()
        self._inspect_running = True
        self.inspectRunningChanged.emit()
        self._poll_timer.start()
        return True

    @Slot(str)
    def setMeshPath(self, mesh_path: str) -> None:
        """Allow QML to set the current mesh path without inspecting."""

        self._set_mesh_path(mesh_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _set_mesh_path(self, mesh_path: str) -> None:
        normalised = str(Path(mesh_path)) if mesh_path else ""
        if self._mesh_path != normalised:
            self._mesh_path = normalised
            self.meshPathChanged.emit()

    def _set_status_message(self, message: str) -> None:
        if self._status_message != message:
            self._status_message = message
            self.statusMessageChanged.emit()

    def _append_log(self, line: str) -> None:
        self._log_lines.append(line)
        self.logTextChanged.emit()

    def _set_inspect_data(self, payload: Dict[str, Any]) -> None:
        materials = payload.get("materials", [])
        uv_sets_map = payload.get("uv_sets", {})

        self._materials = [
            {"id": entry.get("id"), "name": entry.get("name", "Unnamed material")}
            for entry in materials
        ]
        self.materialsChanged.emit()

        uv_sets: List[Dict[str, Any]] = []
        tiles: List[int] = []
        mirrored_sets: List[str] = []

        for name, uv_info in sorted(uv_sets_map.items()):
            charts: Iterable[Dict[str, Any]] = uv_info.get("charts", [])
            has_mirroring = any(
                chart.get("mirrored") or chart.get("flip_u") or chart.get("flip_v")
                for chart in charts
            )
            if has_mirroring:
                mirrored_sets.append(name)

            set_tiles = sorted(int(tile) for tile in uv_info.get("udims", []))
            tiles.extend(set_tiles)

            uv_sets.append(
                {
                    "name": name,
                    "chart_count": int(uv_info.get("chart_count", 0)),
                    "udims": set_tiles,
                }
            )

        self._uv_sets = uv_sets
        self.uvSetsChanged.emit()

        self._udim_tiles = sorted(set(tiles))
        self.udimTilesChanged.emit()

        material_count = len(self._materials)
        uv_set_count = len(self._uv_sets)
        tile_count = len(self._udim_tiles) if self._udim_tiles else 1

        tile_label = "tile" if tile_count == 1 else "tiles"
        self._inspect_summary = (
            f"{material_count} material{'s' if material_count != 1 else ''}, "
            f"{uv_set_count} UV set{'s' if uv_set_count != 1 else ''}, "
            f"{tile_count} UDIM {tile_label}"
        )
        self.inspectSummaryChanged.emit()

        if mirrored_sets:
            self._warning_summary = "Mirrored charts detected in: " + ", ".join(
                sorted(set(mirrored_sets))
            )
        else:
            self._warning_summary = ""
        self.warningSummaryChanged.emit()

        self._set_status_message(f"Inspect complete — {self._inspect_summary}")
        self._append_log(f"Inspect complete: {self._inspect_summary}")
        if self._warning_summary:
            self._append_log(f"Warning: {self._warning_summary}")

        if self._udim_tiles and len(self._udim_tiles) > 1:
            joined_tiles = ", ".join(str(tile) for tile in self._udim_tiles)
            self._append_log(f"Multiple UDIM tiles detected: {joined_tiles}")

    def _poll_job_queue(self) -> None:
        if not self._inspect_job:
            self._poll_timer.stop()
            return

        message = self._inspect_job.poll()
        if message is None:
            return

        self._poll_timer.stop()
        self._inspect_running = False
        self.inspectRunningChanged.emit()

        try:
            if message.get("ok"):
                payload = message.get("data", {})
                self._set_inspect_data(payload)
            else:
                error_message = message.get("error", "Inspection failed")
                self._set_status_message(error_message)
                self._append_log(f"Inspect failed: {error_message}")
                traceback_lines = message.get("traceback")
                if traceback_lines:
                    self._append_log(traceback_lines)
                self._materials = []
                self.materialsChanged.emit()
                self._uv_sets = []
                self.uvSetsChanged.emit()
                self._udim_tiles = []
                self.udimTilesChanged.emit()
                self._inspect_summary = ""
                self.inspectSummaryChanged.emit()
                self._warning_summary = ""
                self.warningSummaryChanged.emit()
        finally:
            self._inspect_job.cleanup()
            self._inspect_job = None
