"""Qt bridge exposing normal2disp functionality to QML."""

from __future__ import annotations

import re
import tempfile
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional

import numpy as np
from PySide6.QtCore import (
    QByteArray,
    QMetaObject,
    QObject,
    Property,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QDesktopServices, QVector3D
from PySide6.QtWidgets import QFileDialog

try:  # pragma: no cover - optional dependency during headless testing
    from PySide6.QtQuick3D import QQuick3DGeometry
except ImportError:  # pragma: no cover - Quick3D unavailable in some CI setups
    QQuick3DGeometry = None  # type: ignore

from .jobs import BakeJob, InspectJob
from .subdivision import (
    DisplacementResult,
    HeightField,
    LoopSubdivisionCache,
    MeshBuffers,
    generate_displacement,
    load_height_field,
)

if TYPE_CHECKING:
    from .image_provider import N2DImageProvider


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
    meshSourceChanged = Signal()
    normalPathChanged = Signal()
    normalTextureChanged = Signal()
    normalPreviewPathChanged = Signal()
    normalEnabledChanged = Signal()
    selectedTileChanged = Signal()
    bakeRunningChanged = Signal()
    progressValueChanged = Signal()
    progressDetailChanged = Signal()
    outputDirectoryChanged = Signal()
    canOpenOutputChanged = Signal()
    canRevealOutputChanged = Signal()
    latestOutputPathChanged = Signal()
    displacementEnabledChanged = Signal()
    displacementBusyChanged = Signal()
    displacementLevelChanged = Signal()
    displacementPreviewScaleChanged = Signal()
    displacementDirtyChanged = Signal()
    displacementStatusChanged = Signal()
    displacementGeometryChanged = Signal()
    displacementAmplitudeChanged = Signal()
    advancedWarningsChanged = Signal()
    lightAzimuthChanged = Signal()
    lightElevationChanged = Signal()

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
        self._mesh_source: str = ""
        self._normal_path: str = ""
        self._normal_texture_url: str = ""
        self._normal_preview_path: str = ""
        self._normal_enabled = True
        self._normal_enabled_snapshot = True
        self._selected_tile: Optional[int] = None
        self._pending_tile_selection: Optional[int] = None
        self._bake_job: Optional[BakeJob] = None
        self._bake_running = False
        self._progress_value = 0.0
        self._progress_total_charts = 0
        self._progress_finished_charts = 0
        self._progress_stage_frac = 0.0
        self._progress_current_tile: Optional[int] = None
        self._progress_current_chart: Optional[int] = None
        self._progress_stage_name: str = ""
        self._progress_last_iter: Optional[tuple[int, float]] = None
        self._progress_detail: str = ""
        self._output_directory: str = ""
        self._latest_outputs: List[str] = []
        self._latest_sidecars: List[str] = []
        self._latest_output_path: str = ""
        self._can_open_output = False
        self._can_reveal_output = False
        self._displacement_enabled = False
        self._displacement_busy = False
        self._displacement_level = 0
        self._displacement_preview_scale = 1.0
        self._displacement_dirty = False
        self._displacement_status = ""
        self._displacement_geometry: Optional[QQuick3DGeometry] = None
        self._displacement_thread: Optional[threading.Thread] = None
        self._displacement_request_id = 0
        self._height_field: Optional[HeightField] = None
        self._height_cache: Dict[int, np.ndarray] = {}
        self._subdivision_cache: Optional[LoopSubdivisionCache] = None
        self._mesh_buffers: Optional[MeshBuffers] = None
        self._preview_amplitude: Optional[float] = None
        self._preview_units: Optional[str] = None
        self._tile_warnings: List[str] = []
        self._light_azimuth: float = 35.0
        self._light_elevation: float = -35.0
        self._mesh_tile_preferences: Dict[str, int] = {}

        self._temp_dir = Path(tempfile.gettempdir()) / "n2d_gui"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._temp_mesh_path: Optional[Path] = None
        self._image_provider: Optional[N2DImageProvider] = None

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

    @Property(str, notify=meshSourceChanged)
    def meshSource(self) -> str:
        return self._mesh_source

    @Property(str, notify=normalPathChanged)
    def normalPath(self) -> str:
        return self._normal_path

    @Property(str, notify=normalTextureChanged)
    def normalTextureUrl(self) -> str:
        return self._normal_texture_url

    @Property(str, notify=normalPreviewPathChanged)
    def normalPreviewPath(self) -> str:
        return self._normal_preview_path

    @Property(bool, notify=normalEnabledChanged)
    def normalEnabled(self) -> bool:
        return self._normal_enabled

    @Property(int, notify=selectedTileChanged)
    def selectedTile(self) -> int:
        return int(self._selected_tile) if self._selected_tile is not None else 0

    @Property(bool, notify=bakeRunningChanged)
    def bakeRunning(self) -> bool:
        return self._bake_running

    @Property(float, notify=progressValueChanged)
    def progressValue(self) -> float:
        return float(self._progress_value)

    @Property(str, notify=progressDetailChanged)
    def progressDetail(self) -> str:
        return self._progress_detail

    @Property(str, notify=outputDirectoryChanged)
    def outputDirectory(self) -> str:
        return self._output_directory

    @Property(bool, notify=canOpenOutputChanged)
    def canOpenOutput(self) -> bool:
        return self._can_open_output

    @Property(bool, notify=canRevealOutputChanged)
    def canRevealLatestOutput(self) -> bool:
        return self._can_reveal_output

    @Property(str, notify=latestOutputPathChanged)
    def latestOutputPath(self) -> str:
        return self._latest_output_path

    @Property(bool, notify=displacementEnabledChanged)
    def displacementEnabled(self) -> bool:
        return self._displacement_enabled

    @Property(bool, notify=displacementBusyChanged)
    def displacementBusy(self) -> bool:
        return self._displacement_busy

    @Property(int, notify=displacementLevelChanged)
    def displacementLevel(self) -> int:
        return int(self._displacement_level)

    @Property(bool, notify=displacementDirtyChanged)
    def displacementDirty(self) -> bool:
        return self._displacement_dirty

    @Property(float, notify=displacementPreviewScaleChanged)
    def displacementPreviewScale(self) -> float:
        return float(self._displacement_preview_scale)

    @Property(str, notify=displacementStatusChanged)
    def displacementStatus(self) -> str:
        return self._displacement_status

    @Property("QVariant", notify=displacementGeometryChanged)
    def displacementGeometry(self) -> Optional[QQuick3DGeometry]:
        return self._displacement_geometry

    @Property(str, notify=displacementAmplitudeChanged)
    def displacementAmplitude(self) -> str:
        if self._preview_amplitude is None:
            return ""
        units = self._preview_units or ""
        if units:
            return f"Amplitude {self._preview_amplitude:g} ({units})"
        return f"Amplitude {self._preview_amplitude:g}"

    @Property("QVariant", notify=advancedWarningsChanged)
    def advancedWarnings(self) -> List[str]:
        return list(self._tile_warnings)

    @Property(float, notify=lightAzimuthChanged)
    def lightAzimuth(self) -> float:
        return float(self._light_azimuth)

    @Property(float, notify=lightElevationChanged)
    def lightElevation(self) -> float:
        return float(self._light_elevation)

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
        self._ensure_polling()
        return True

    @Slot(str)
    def setMeshPath(self, mesh_path: str) -> None:
        """Allow QML to set the current mesh path without inspecting."""

        self._set_mesh_path(mesh_path)

    @Slot(str)
    def setNormalPath(self, normal_path: str) -> None:
        """Set the current normal map or UDIM pattern for preview."""

        normalised = normal_path.strip()
        if normalised:
            normalised = str(Path(normalised).expanduser())

        pattern, tile_override = self._prepare_normal_pattern(normalised)

        if self._normal_path != pattern:
            self._normal_path = pattern
            self.normalPathChanged.emit()
            if pattern:
                self._append_log(f"Normal map selected: {pattern}")
            else:
                self._append_log("Normal map cleared")

        if tile_override is not None:
            self._pending_tile_selection = tile_override
            if tile_override in self._udim_tiles:
                self._set_selected_tile(tile_override)

        if not pattern:
            self._pending_tile_selection = None
            self._set_selected_tile(None)

        self._update_normal_texture()

    @Slot(bool)
    def setNormalEnabled(self, enabled: bool) -> None:
        """Enable or disable the normal-map contribution in the viewport."""

        if self._normal_enabled == enabled:
            return

        self._normal_enabled = enabled
        self.normalEnabledChanged.emit()
        state = "enabled" if enabled else "disabled"
        self._set_status_message(f"Normal map preview {state}")
        self._append_log(f"Normal map preview {state}")

    @Slot(bool)
    def setDisplacementEnabled(self, enabled: bool) -> None:
        """Toggle displacement preview generation."""

        if enabled == self._displacement_enabled:
            return

        if enabled:
            if not self._latest_outputs:
                message = "Bake a displacement map before enabling the preview."
                self._set_status_message(message)
                self._append_log(message)
                return
            if self._mesh_buffers is None:
                message = "Mesh data unavailable for displacement preview."
                self._set_status_message(message)
                self._append_log(message)
                return
            if not self._ensure_height_field():
                return
            if not self._ensure_subdivision_cache():
                return

            self._displacement_enabled = True
            self.displacementEnabledChanged.emit()
            self._normal_enabled_snapshot = self._normal_enabled
            if self._normal_enabled:
                self._normal_enabled = False
                self.normalEnabledChanged.emit()
            self._append_log("Displacement preview enabled")
            self._set_status_message("Generating displacement preview…")
            self._set_displacement_dirty(False)
            self._start_displacement_job()
        else:
            self._disable_displacement()

    @Slot(int)
    def setDisplacementLevel(self, level: int) -> None:
        """Set the Loop subdivision level for the displacement preview."""

        clamped = max(0, min(5, int(level)))
        if clamped == self._displacement_level:
            return

        self._displacement_level = clamped
        self.displacementLevelChanged.emit()
        self._append_log(f"Subdivision level set to {clamped}")
        if self._displacement_enabled:
            self._set_displacement_dirty(True)
            self._set_displacement_status(f"Subdivision {clamped} pending — regenerate to apply")

    @Slot(float)
    def setDisplacementPreviewScale(self, scale: float) -> None:
        """Adjust the preview-only height scale multiplier."""

        try:
            value = float(scale)
        except (TypeError, ValueError):
            value = 1.0
        clamped = max(-10.0, min(10.0, value))
        if abs(clamped - self._displacement_preview_scale) < 1e-6:
            return

        self._displacement_preview_scale = clamped
        self.displacementPreviewScaleChanged.emit()
        self._append_log(f"Preview scale set to {clamped:g}")
        if self._displacement_enabled:
            self._set_displacement_dirty(True)
            self._set_displacement_status("Preview scale changed — regenerate to apply")

    @Slot()
    def regenerateDisplacement(self) -> None:
        """Rebuild the displacement preview with the current settings."""

        if not self._displacement_enabled:
            self._set_status_message("Enable displacement preview before regenerating.")
            return
        self._append_log(f"Regenerating displacement preview at level {self._displacement_level}")
        self._set_displacement_dirty(False)
        self._start_displacement_job()

    @Slot(float)
    def setLightAzimuth(self, azimuth: float) -> None:
        """Adjust the viewport key-light azimuth (yaw)."""

        try:
            value = float(azimuth)
        except (TypeError, ValueError):
            value = self._light_azimuth

        wrapped = max(0.0, min(360.0, value % 360.0))
        if abs(wrapped - self._light_azimuth) < 1e-6:
            return

        self._light_azimuth = wrapped
        self.lightAzimuthChanged.emit()

    @Slot(float)
    def setLightElevation(self, elevation: float) -> None:
        """Adjust the viewport key-light elevation (pitch)."""

        try:
            value = float(elevation)
        except (TypeError, ValueError):
            value = self._light_elevation

        clamped = max(-80.0, min(80.0, value))
        if abs(clamped - self._light_elevation) < 1e-6:
            return

        self._light_elevation = clamped
        self.lightElevationChanged.emit()

    @Slot(int)
    def selectTile(self, tile: int) -> None:
        """Select a UDIM tile for the normal preview."""

        if tile not in self._udim_tiles:
            self._append_log(f"Ignoring selection for unknown UDIM tile {tile}")
            return

        self._pending_tile_selection = None
        self._set_selected_tile(tile)

    @Slot(str)
    def setOutputDirectory(self, directory: str) -> None:
        """Set the output directory used for baking."""

        normalized = str(Path(directory).expanduser()) if directory else ""
        if self._output_directory == normalized:
            return

        self._output_directory = normalized
        self.outputDirectoryChanged.emit()

    @Slot("QVariant", result=bool)
    def runBake(self, options: Any) -> bool:
        """Launch the bake pipeline in the background."""

        if self._bake_running:
            self._append_log("Bake already running; ignoring new request.")
            return False

        if not self._mesh_path:
            self._set_status_message("Select a mesh before starting a bake.")
            self._append_log("Bake aborted: no mesh selected")
            return False

        if not self._normal_path:
            self._set_status_message("Select a normal map before starting a bake.")
            self._append_log("Bake aborted: no normal map selected")
            return False

        if not self._output_directory:
            self._set_status_message("Select an output directory before starting a bake.")
            self._append_log("Bake aborted: no output directory selected")
            return False

        if isinstance(options, dict):
            options_map = dict(options)
        else:
            options_map = {}

        output_pattern = self._build_output_pattern(Path(self._output_directory))
        if not output_pattern:
            self._set_status_message("Unable to determine an output pattern for the bake.")
            return False

        uv_set = options_map.get("uvSet")
        if isinstance(uv_set, str) and not uv_set.strip():
            uv_set = None

        def _to_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _to_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        normalization = str(options_map.get("normalization", "auto")).lower()
        amplitude = max(0.0, min(10.0, _to_float(options_map.get("amplitude"), 1.0)))
        max_slope = _to_float(options_map.get("maxSlope"), 10.0)
        cg_tol = _to_float(options_map.get("cgTol"), 1e-6)
        cg_maxiter = max(1, _to_int(options_map.get("cgMaxIter"), 10000))

        processes_value = options_map.get("processes")
        if isinstance(processes_value, (int, float)):
            processes = int(processes_value)
        else:
            try:
                processes = int(str(processes_value)) if processes_value not in (None, "") else None
            except (TypeError, ValueError):
                processes = None

        material_overrides: Dict[str, str] = {}
        raw_overrides = options_map.get("materialOverrides")
        if isinstance(raw_overrides, dict):
            for key, value in raw_overrides.items():
                if value:
                    material_overrides[str(key)] = str(value)

        options_payload: Dict[str, Any] = {
            "uv_set": uv_set,
            "y_is_down": bool(options_map.get("yIsDown")),
            "normalization": normalization,
            "max_slope": max_slope,
            "amplitude": amplitude,
            "cg_tol": cg_tol,
            "cg_maxiter": cg_maxiter,
            "deterministic": bool(options_map.get("deterministic")),
            "processes": processes,
            "export_sidecars": bool(options_map.get("exportSidecars")),
        }

        self._reset_progress_state()
        self._bake_running = True
        self.bakeRunningChanged.emit()
        self._set_status_message("Starting bake…")
        self._append_log("Starting bake job")
        self._latest_outputs = []
        self._latest_sidecars = []
        self._set_latest_output_path("")
        self._update_output_actions(False, False)

        self._bake_job = BakeJob(
            Path(self._mesh_path),
            self._normal_path or None,
            output_pattern,
            options_payload,
            material_overrides,
        )
        self._bake_job.start()
        self._ensure_polling()
        return True

    @Slot()
    def cancelBake(self) -> None:
        """Request cancellation of the active bake job."""

        if self._bake_job and not self._bake_job.is_finished():
            self._append_log("Cancel requested")
            self._set_status_message("Canceling bake…")
            self._bake_job.cancel()

    @Slot(result=bool)
    def openOutputFolder(self) -> bool:
        """Open the bake output directory in the system file browser."""

        if not self._output_directory:
            return False

        url = QUrl.fromLocalFile(self._output_directory)
        return bool(QDesktopServices.openUrl(url))

    @Slot(result=bool)
    def revealLatestOutput(self) -> bool:
        """Reveal the most recent EXR in the system file browser."""

        if not self._latest_output_path:
            return False

        url = QUrl.fromLocalFile(self._latest_output_path)
        return bool(QDesktopServices.openUrl(url))

    def register_image_provider(self, provider: "N2DImageProvider") -> None:
        """Register the shared image provider and refresh previews."""

        self._image_provider = provider
        self._update_normal_texture()

    def _ensure_polling(self) -> None:
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_progress_state(self) -> None:
        self._progress_total_charts = 0
        self._progress_finished_charts = 0
        self._progress_stage_frac = 0.0
        self._progress_current_tile = None
        self._progress_current_chart = None
        self._progress_stage_name = ""
        self._progress_last_iter = None
        self._set_progress_value(0.0)
        self._set_progress_detail("")

    def _set_progress_value(self, value: float) -> None:
        clamped = max(0.0, min(1.0, float(value)))
        if abs(clamped - self._progress_value) > 1e-6:
            self._progress_value = clamped
            self.progressValueChanged.emit()

    def _set_progress_detail(self, detail: str) -> None:
        if self._progress_detail != detail:
            self._progress_detail = detail
            self.progressDetailChanged.emit()

    def _recalculate_progress(self) -> None:
        if self._progress_total_charts <= 0:
            value = 1.0 if self._progress_finished_charts > 0 else 0.0
        else:
            value = (self._progress_finished_charts + self._progress_stage_frac) / float(
                self._progress_total_charts
            )
        self._set_progress_value(value)

    def _update_progress_detail(self) -> None:
        parts: List[str] = []
        if self._progress_current_tile is not None:
            parts.append(f"Tile {self._progress_current_tile}")
        if self._progress_current_chart is not None:
            parts.append(f"Chart {self._progress_current_chart}")
        if self._progress_stage_name:
            stage_label = self._progress_stage_name.capitalize()
            if self._progress_stage_name == "solve" and self._progress_last_iter:
                iterations, residual = self._progress_last_iter
                stage_label += f" (iter {iterations}, residual {residual:.2e})"
            parts.append(stage_label)
        detail = " • ".join(parts)
        self._set_progress_detail(detail)

    def _set_latest_output_path(self, path: str) -> None:
        normalized = str(path) if path else ""
        if self._latest_output_path != normalized:
            self._latest_output_path = normalized
            self.latestOutputPathChanged.emit()

    def _update_output_actions(self, open_enabled: bool, reveal_enabled: bool) -> None:
        if self._can_open_output != open_enabled:
            self._can_open_output = open_enabled
            self.canOpenOutputChanged.emit()
        if self._can_reveal_output != reveal_enabled:
            self._can_reveal_output = reveal_enabled
            self.canRevealOutputChanged.emit()

    def _build_output_pattern(self, directory: Path) -> str:
        if not directory:
            return ""

        base_name = Path(self._mesh_path).stem or "displacement"
        multi_tile = len(self._udim_tiles) > 1
        filename = f"{base_name}_disp_<UDIM>.exr" if multi_tile else f"{base_name}_disp.exr"
        return str(directory / filename)

    def _handle_bake_event(self, event: Dict[str, Any]) -> None:
        kind = event.get("kind")

        if kind == "log":
            level = str(event.get("level", "INFO")).upper()
            message = str(event.get("message", ""))
            prefix = f"[{level}] " if level else ""
            self._append_log(prefix + message)
            if level == "ERROR":
                self._set_status_message(message)
            return

        if kind == "begin_job":
            self._bake_running = True
            self.bakeRunningChanged.emit()
            self._progress_total_charts = max(0, int(event.get("charts_total", 0)))
            self._progress_finished_charts = 0
            self._progress_stage_frac = 0.0
            self._progress_current_tile = None
            self._progress_current_chart = None
            self._progress_stage_name = ""
            self._progress_last_iter = None
            self._set_progress_value(0.0)
            self._set_progress_detail("")
            self._update_output_actions(False, False)
            self._set_status_message("Baking displacement…")
            return

        if kind == "begin_tile":
            tile = int(event.get("tile", 0))
            self._progress_current_tile = tile
            self._progress_current_chart = None
            self._progress_stage_name = ""
            self._progress_last_iter = None
            self._update_progress_detail()
            self._append_log(f"Processing tile {tile}")
            self._set_status_message(f"Baking tile {tile}…")
            return

        if kind == "stage":
            self._progress_current_tile = int(event.get("tile", 0))
            chart = event.get("chart")
            self._progress_current_chart = int(chart) if chart is not None else None
            stage = str(event.get("name", ""))
            self._progress_stage_name = stage
            if stage != "solve":
                self._progress_last_iter = None
            self._progress_stage_frac = max(0.0, min(1.0, float(event.get("frac", 0.0))))
            self._update_progress_detail()
            self._recalculate_progress()
            if stage:
                if self._progress_detail:
                    self._set_status_message(f"Baking — {self._progress_detail}")
                else:
                    self._set_status_message(f"Baking — {stage.capitalize()}")
            return

        if kind == "cg_iter":
            chart = event.get("chart")
            self._progress_current_tile = int(event.get("tile", 0))
            self._progress_current_chart = int(chart) if chart is not None else None
            iterations = int(event.get("iter", 0))
            residual = float(event.get("residual", 0.0))
            self._progress_last_iter = (iterations, residual)
            self._update_progress_detail()
            self._recalculate_progress()
            return

        if kind == "end_chart":
            chart = event.get("chart")
            self._progress_current_chart = int(chart) if chart is not None else None
            self._progress_finished_charts += 1
            self._progress_stage_frac = 0.0
            self._progress_stage_name = ""
            self._progress_last_iter = None
            self._update_progress_detail()
            self._recalculate_progress()
            return

        if kind == "end_tile":
            tile = int(event.get("tile", 0))
            self._append_log(f"Finished tile {tile}")
            return

        if kind == "end_job":
            self._bake_running = False
            self.bakeRunningChanged.emit()
            self._progress_stage_name = ""
            self._progress_last_iter = None
            self._update_progress_detail()
            if not event.get("ok", False):
                self._update_output_actions(False, False)
            return

        if kind == "result":
            if self._bake_job:
                self._bake_job.cleanup()
                self._bake_job = None

            if self._bake_running:
                self._bake_running = False
                self.bakeRunningChanged.emit()

            ok = bool(event.get("ok"))
            logs = event.get("logs") or []
            for line in logs:
                self._append_log(str(line))

            if ok:
                outputs = [str(path) for path in event.get("outputs", [])]
                sidecars = [str(path) for path in event.get("sidecars", [])]
                latest = event.get("latest_output") or (outputs[-1] if outputs else "")
                self._latest_outputs = outputs
                self._latest_sidecars = sidecars
                self._set_latest_output_path(str(latest) if latest else "")

                self._reset_displacement_preview(clear_height=True, clear_cache=False)
                amplitude_text = ""
                if self._ensure_height_field():
                    amplitude_text = self.displacementAmplitude

                output_dir = event.get("output_dir")
                if output_dir:
                    normalized_dir = str(Path(output_dir))
                    if self._output_directory != normalized_dir:
                        self._output_directory = normalized_dir
                        self.outputDirectoryChanged.emit()

                count = len(outputs)
                summary = f"Bake complete — wrote {count} file{'s' if count != 1 else ''}"
                if amplitude_text:
                    summary += f" • {amplitude_text}"
                self._set_status_message(summary)
                self._progress_finished_charts = self._progress_total_charts
                self._progress_stage_frac = 0.0
                self._recalculate_progress()
                if self._progress_total_charts == 0:
                    self._set_progress_value(1.0)
                self._set_progress_detail("Bake complete")
                self._update_output_actions(
                    bool(self._output_directory), bool(self._latest_output_path)
                )
            else:
                error_message = str(event.get("error", "Bake failed"))
                self._set_status_message(error_message)
                self._append_log(f"Bake failed: {error_message}")
                self._update_output_actions(False, False)
                self._set_latest_output_path("")
                self._set_progress_detail("Bake failed")
                self._set_progress_value(0.0)

            return

    def _set_mesh_path(self, mesh_path: str) -> None:
        normalised = str(Path(mesh_path)) if mesh_path else ""
        if self._mesh_path != normalised:
            self._reset_displacement_preview()
            self._mesh_buffers = None
            self._subdivision_cache = None
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

        default_tile: Optional[int]
        if (
            self._pending_tile_selection is not None
            and self._pending_tile_selection in self._udim_tiles
        ):
            default_tile = self._pending_tile_selection
        elif self._selected_tile in self._udim_tiles:
            default_tile = self._selected_tile
        else:
            default_tile = self._udim_tiles[0] if self._udim_tiles else None

        self._pending_tile_selection = None

        mesh_path_value = payload.get("path")
        if mesh_path_value:
            mesh_key = str(Path(mesh_path_value))
        else:
            mesh_key = self._mesh_path

        preferred = self._mesh_tile_preferences.get(mesh_key or "")
        if preferred in self._udim_tiles:
            default_tile = preferred

        self._set_selected_tile(default_tile)

        self._update_validation_warnings(payload)

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

        mesh_path = payload.get("path") or self._mesh_path
        if mesh_path:
            self._load_viewport_mesh(Path(mesh_path))

    def _poll_job_queue(self) -> None:
        active = False

        if self._inspect_job:
            message = self._inspect_job.poll()
            if message is None:
                active = True
            else:
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
                        self._set_selected_tile(None)
                        self._set_mesh_source("")
                finally:
                    self._inspect_job.cleanup()
                    self._inspect_job = None

        if self._bake_job:
            events = self._bake_job.poll()
            if events:
                for event in events:
                    self._handle_bake_event(event)

            if self._bake_job:
                if self._bake_job.is_finished() and not events:
                    self._bake_job.cleanup()
                    self._bake_job = None
                else:
                    active = True

        if not active:
            self._poll_timer.stop()

    def _load_viewport_mesh(self, mesh_path: Path) -> None:
        """Export ``mesh_path`` to GLB for the Quick 3D viewport."""

        try:
            import trimesh

            scene = trimesh.load(str(mesh_path), force="scene")
            if scene is None or getattr(scene, "is_empty", False):
                raise ValueError("Mesh contains no geometry to preview")

            self._temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = self._temp_dir / f"inspect_{uuid.uuid4().hex}.glb"
            scene.export(temp_path, file_type="glb")

            if isinstance(scene, trimesh.Scene):
                mesh = scene.to_mesh()
            else:
                mesh = scene

            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError("Unsupported mesh format for viewport preview")

            if mesh.faces is None or mesh.faces.size == 0:
                raise ValueError("Mesh contains no triangles to preview")

            if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
                mesh = mesh.triangulate()

            positions = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)

            if positions.size == 0 or faces.size == 0:
                raise ValueError("Mesh contains no data to preview")

            visual = getattr(mesh, "visual", None)
            uv_data = getattr(visual, "uv", None) if visual is not None else None
            if uv_data is None or len(uv_data) == 0:
                raise ValueError("Mesh is missing UVs required for displacement preview")

            uv_values = np.asarray(uv_data, dtype=np.float64)
            if uv_values.ndim != 2 or uv_values.shape[1] < 2:
                raise ValueError("Mesh UVs are malformed for displacement preview")

            if uv_values.shape[0] == positions.shape[0]:
                uv = uv_values[:, :2]
            elif uv_values.shape[0] == faces.shape[0] * 3:
                uv_faces = uv_values.reshape(-1, 3, uv_values.shape[1])[:, :, :2]
                positions = positions[faces].reshape(-1, 3)
                uv = uv_faces.reshape(-1, 2)
                faces = np.arange(positions.shape[0], dtype=np.int64).reshape(-1, 3)
            else:
                raise ValueError("Mesh UV layout is incompatible with displacement preview")

            if not np.isfinite(positions).all() or not np.isfinite(uv).all():
                raise ValueError("Mesh data contains NaN/Inf values")

            mesh_buffers = MeshBuffers(
                positions=np.ascontiguousarray(positions, dtype=np.float64),
                uv=np.ascontiguousarray(uv, dtype=np.float64),
                faces=np.ascontiguousarray(faces, dtype=np.int64),
            )
        except Exception as exc:  # pragma: no cover - depends on trimesh/asset
            self._mesh_buffers = None
            self._reset_displacement_preview(clear_height=False, clear_cache=True)
            self._set_status_message(f"Viewport load failed: {exc}")
            self._append_log(f"Viewport load failed: {exc}")
            self._set_mesh_source("")
            return

        self._mesh_buffers = mesh_buffers
        self._reset_displacement_preview(clear_height=False, clear_cache=True)

        face_count = int(mesh_buffers.faces.shape[0])
        vertex_count = int(mesh_buffers.positions.shape[0])
        self._append_log(
            f"Viewport mesh ready: {temp_path.name} • {face_count:,} tris / {vertex_count:,} verts"
        )
        self._set_status_message("Viewport mesh loaded")
        self._set_mesh_source(QUrl.fromLocalFile(str(temp_path)).toString())
        self._cleanup_temp_mesh(keep=temp_path)

    def _set_mesh_source(self, source: str) -> None:
        if not source:
            self._cleanup_temp_mesh()

        if self._mesh_source == source:
            return

        self._mesh_source = source
        self.meshSourceChanged.emit()

    def _cleanup_temp_mesh(self, keep: Optional[Path] = None) -> None:
        if self._temp_mesh_path and self._temp_mesh_path != keep:
            try:
                self._temp_mesh_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:  # pragma: no cover - filesystem dependent
                self._append_log(f"Failed to remove temporary mesh: {exc}")

        self._temp_mesh_path = keep

    def _prepare_normal_pattern(self, normal_path: str) -> tuple[str, Optional[int]]:
        if not normal_path:
            return "", None

        if "<UDIM>" in normal_path or "%04d" in normal_path:
            return normal_path, None

        name = Path(normal_path).name
        for match in re.finditer(r"(\d{4})", name):
            digits = int(match.group(1))
            if not self._udim_tiles or digits in self._udim_tiles:
                replaced = name[: match.start()] + "<UDIM>" + name[match.end() :]
                pattern = str(Path(normal_path).with_name(replaced))
                return pattern, digits

        return normal_path, None

    def _resolve_normal_tile_path(self, base_path: str, tile: Optional[int]) -> Optional[Path]:
        if not base_path:
            return None

        if "<UDIM>" in base_path:
            if tile is None:
                return None
            return Path(base_path.replace("<UDIM>", f"{tile:04d}")).expanduser()

        if "%04d" in base_path:
            if tile is None:
                return None
            try:
                return Path(base_path % tile).expanduser()
            except TypeError:
                return None

        return Path(base_path).expanduser()

    def _set_selected_tile(self, tile: Optional[int]) -> None:
        changed = tile != self._selected_tile
        self._selected_tile = tile
        if changed:
            self.selectedTileChanged.emit()
            if tile is not None and self._mesh_path:
                self._mesh_tile_preferences[str(Path(self._mesh_path))] = tile
        self._update_normal_texture()

    def _update_validation_warnings(self, payload: Dict[str, Any]) -> None:
        warnings: List[str] = []

        candidate_lists: List[Any] = []
        validation = payload.get("validation")
        if validation:
            candidate_lists.append(validation)
        alt_validation = payload.get("udim_validation")
        if alt_validation:
            candidate_lists.append(alt_validation)
        extra = payload.get("material_warnings")
        if extra:
            candidate_lists.append(extra)

        for entry in candidate_lists:
            if isinstance(entry, dict):
                materials = entry.get("materials")
                if isinstance(materials, dict):
                    for material_id, material_data in materials.items():
                        if isinstance(material_data, dict):
                            missing = material_data.get("missing_tiles") or material_data.get(
                                "missing"
                            )
                            if missing:
                                if isinstance(missing, (list, tuple, set)):
                                    missing_list = sorted({int(tile) for tile in missing})
                                else:
                                    missing_list = [missing]
                                name = (
                                    material_data.get("name")
                                    or material_data.get("material")
                                    or material_id
                                )
                                warnings.append(
                                    f"{name} missing {', '.join(str(tile) for tile in missing_list)}"
                                )
                elif isinstance(materials, list):
                    for material_data in materials:
                        if isinstance(material_data, dict):
                            missing = material_data.get("missing_tiles") or material_data.get(
                                "missing"
                            )
                            if missing:
                                if isinstance(missing, (list, tuple, set)):
                                    missing_list = sorted({int(tile) for tile in missing})
                                else:
                                    missing_list = [missing]
                                name = (
                                    material_data.get("name")
                                    or material_data.get("material")
                                    or "Material"
                                )
                                warnings.append(
                                    f"{name} missing {', '.join(str(tile) for tile in missing_list)}"
                                )
            elif isinstance(entry, (list, tuple, set)):
                for item in entry:
                    if item:
                        warnings.append(str(item))
            elif isinstance(entry, str):
                warnings.append(entry)

        if warnings != self._tile_warnings:
            self._tile_warnings = warnings
            self.advancedWarningsChanged.emit()

    def _update_normal_texture(self) -> None:
        provider = self._image_provider
        if provider is None:
            return

        resolved: Optional[Path] = None
        if self._normal_path:
            resolved = self._resolve_normal_tile_path(self._normal_path, self._selected_tile)

        preview_url = ""
        preview_path = ""

        if resolved is None:
            preview_url = provider.clear_normal_image()
            if self._normal_path:
                self._set_status_message("Select a UDIM tile to preview the normal map")
        else:
            if resolved.exists():
                try:
                    preview_url = provider.set_normal_image(resolved)
                    preview_path = str(resolved)
                    self._set_status_message(f"Normal map bound: {resolved.name}")
                    self._append_log(f"Normal map bound to viewport: {resolved}")
                except ValueError as exc:
                    self._append_log(str(exc))
                    self._set_status_message(str(exc))
                    preview_url = provider.clear_normal_image()
            else:
                message = f"Normal map tile not found: {resolved}"
                self._append_log(message)
                self._set_status_message(message)
                preview_url = provider.clear_normal_image()

        if self._normal_texture_url != preview_url:
            self._normal_texture_url = preview_url
            self.normalTextureChanged.emit()

        if self._normal_preview_path != preview_path:
            self._normal_preview_path = preview_path
            self.normalPreviewPathChanged.emit()

    def _set_displacement_busy(self, busy: bool) -> None:
        if self._displacement_busy != busy:
            self._displacement_busy = busy
            self.displacementBusyChanged.emit()

    def _set_displacement_dirty(self, dirty: bool) -> None:
        if self._displacement_dirty != dirty:
            self._displacement_dirty = dirty
            self.displacementDirtyChanged.emit()

    def _set_displacement_status(self, message: str) -> None:
        if self._displacement_status != message:
            self._displacement_status = message
            self.displacementStatusChanged.emit()

    def _set_displacement_geometry(self, geometry: Optional[Any]) -> None:
        if self._displacement_geometry is geometry:
            return
        self._displacement_geometry = geometry
        self.displacementGeometryChanged.emit()

    def _reset_displacement_preview(
        self,
        *,
        clear_height: bool = True,
        clear_cache: bool = True,
        clear_geometry: bool = True,
    ) -> None:
        self._displacement_request_id += 1
        self._displacement_thread = None
        if self._displacement_enabled:
            self._displacement_enabled = False
            self.displacementEnabledChanged.emit()
        self._set_displacement_busy(False)
        self._set_displacement_dirty(False)
        self._set_displacement_status("")

        if clear_geometry:
            self._set_displacement_geometry(None)

        if clear_height:
            self._height_field = None
            self._preview_amplitude = None
            self._preview_units = None
            self.displacementAmplitudeChanged.emit()
        if clear_cache:
            self._subdivision_cache = None
            self._height_cache.clear()

        if self._normal_preview_path and not self._normal_enabled:
            self._normal_enabled = True
            self.normalEnabledChanged.emit()

    def _disable_displacement(self) -> None:
        if not self._displacement_enabled and not self._displacement_busy:
            return
        self._append_log("Displacement preview disabled")
        self._set_status_message("Displacement preview disabled")
        self._reset_displacement_preview(clear_height=False, clear_cache=False)

    def _ensure_height_field(self) -> bool:
        if self._height_field is not None:
            return True
        if not self._latest_outputs:
            return False
        try:
            field = load_height_field([Path(path) for path in self._latest_outputs])
        except HeightFieldError as exc:
            message = str(exc)
            self._set_status_message(message)
            self._append_log(f"Displacement preview failed: {message}")
            return False

        self._height_field = field
        self._height_cache.clear()
        self._preview_amplitude = field.amplitude
        self._preview_units = field.units
        self.displacementAmplitudeChanged.emit()

        amplitude_label = self.displacementAmplitude
        if amplitude_label:
            self._set_status_message(f"Loaded displacement map — {amplitude_label}")
        else:
            self._set_status_message("Loaded displacement map for preview")
        self._append_log("Displacement height map ready")
        return True

    def _ensure_subdivision_cache(self) -> bool:
        if self._mesh_buffers is None:
            return False
        if self._subdivision_cache is None:
            try:
                self._subdivision_cache = LoopSubdivisionCache(self._mesh_buffers)
            except Exception as exc:  # pragma: no cover - depends on mesh data
                message = f"Subdivision preparation failed: {exc}"
                self._set_status_message(message)
                self._append_log(message)
                self._subdivision_cache = None
                return False
        return True

    def _start_displacement_job(self) -> None:
        if not self._displacement_enabled:
            return
        if self._displacement_busy:
            return
        if not self._ensure_height_field():
            return
        if not self._ensure_subdivision_cache():
            return

        cache = self._subdivision_cache
        field = self._height_field
        if cache is None or field is None:
            return

        level = int(self._displacement_level)
        scale = float(self._displacement_preview_scale)

        self._set_displacement_busy(True)
        self._set_displacement_status(f"Subdiv {level} — generating…")
        self._append_log(f"Generating displacement preview (level {level}, scale {scale:g})")

        request_id = self._displacement_request_id + 1
        self._displacement_request_id = request_id

        height_cache = self._height_cache

        def worker() -> None:
            try:
                result = generate_displacement(
                    cache,
                    field,
                    level,
                    scale,
                    precomputed_heights=height_cache,
                    height_cache=height_cache,
                )
            except Exception as exc:  # pragma: no cover - depends on mesh/exr
                self._invoke_on_main(lambda: self._finalize_displacement_result(request_id, exc))
            else:
                self._invoke_on_main(lambda: self._finalize_displacement_result(request_id, result))

        self._displacement_thread = threading.Thread(
            target=worker,
            name="n2d_displacement",
            daemon=True,
        )
        self._displacement_thread.start()

    def _finalize_displacement_result(self, request_id: int, result: Any) -> None:
        if request_id != self._displacement_request_id:
            return

        self._displacement_thread = None
        self._set_displacement_busy(False)

        if isinstance(result, DisplacementResult):
            geometry = self._build_displacement_geometry(result)
            self._set_displacement_geometry(geometry)
            summary = (
                f"Subdiv {result.level} • {result.triangle_count:,} tris "
                f"• {result.build_ms:.0f} ms build, {result.sample_ms:.0f} ms sample, "
                f"{result.displace_ms:.0f} ms displace"
            )
            self._set_displacement_status(summary)
            self._set_status_message(summary)
            self._append_log(summary)
        else:
            message = f"Displacement preview failed: {result}"
            self._set_displacement_status("Preview failed — normal map restored")
            self._set_status_message(message)
            self._append_log(message)
            self._set_displacement_geometry(None)
            self._reset_displacement_preview(clear_height=False, clear_cache=False)

    def _build_displacement_geometry(self, result: DisplacementResult) -> Optional[Any]:
        if QQuick3DGeometry is None:  # pragma: no cover - headless testing fallback
            return None

        geometry = QQuick3DGeometry()
        geometry.clear()
        geometry.setPrimitiveType(QQuick3DGeometry.PrimitiveType.Triangles)

        packed = np.concatenate((result.positions, result.normals), axis=1)
        vertex_bytes = QByteArray(packed.tobytes())
        geometry.setVertexData(vertex_bytes)
        stride = 6 * 4
        geometry.setStride(stride)
        geometry.setVertexCount(int(result.positions.shape[0]))
        geometry.addAttribute(
            QQuick3DGeometry.Attribute.Semantic.PositionSemantic,
            0,
            QQuick3DGeometry.Attribute.ComponentType.F32Type,
        )
        geometry.addAttribute(
            QQuick3DGeometry.Attribute.Semantic.NormalSemantic,
            3 * 4,
            QQuick3DGeometry.Attribute.ComponentType.F32Type,
        )

        indices = np.ascontiguousarray(result.faces.reshape(-1), dtype=np.uint32)
        geometry.setIndexData(QByteArray(indices.tobytes()))
        geometry.setIndexStride(4)
        geometry.setIndexCount(int(indices.size))
        geometry.setPrimitiveCount(int(result.faces.shape[0]))

        bounds_min = result.positions.min(axis=0)
        bounds_max = result.positions.max(axis=0)
        geometry.setBounds(
            QVector3D(float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2])),
            QVector3D(float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2])),
        )

        geometry.markAllDirty()
        return geometry

    def _invoke_on_main(self, func: Callable[[], None]) -> None:
        QMetaObject.invokeMethod(self, func, Qt.QueuedConnection)
