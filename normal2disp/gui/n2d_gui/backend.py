"""Qt bridge exposing normal2disp functionality to QML."""

from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from PySide6.QtCore import QObject, Property, QTimer, QUrl, Signal, Slot
from PySide6.QtWidgets import QFileDialog

from .jobs import InspectJob

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
        self._selected_tile: Optional[int] = None
        self._pending_tile_selection: Optional[int] = None

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

    @Slot(int)
    def selectTile(self, tile: int) -> None:
        """Select a UDIM tile for the normal preview."""

        if tile not in self._udim_tiles:
            self._append_log(f"Ignoring selection for unknown UDIM tile {tile}")
            return

        self._pending_tile_selection = None
        self._set_selected_tile(tile)

    def register_image_provider(self, provider: "N2DImageProvider") -> None:
        """Register the shared image provider and refresh previews."""

        self._image_provider = provider
        self._update_normal_texture()

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
        self._set_selected_tile(default_tile)

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
                self._set_selected_tile(None)
                self._set_mesh_source("")
        finally:
            self._inspect_job.cleanup()
            self._inspect_job = None

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
        except Exception as exc:  # pragma: no cover - depends on trimesh/asset
            self._set_status_message(f"Viewport load failed: {exc}")
            self._append_log(f"Viewport load failed: {exc}")
            self._set_mesh_source("")
            return

        self._append_log(f"Viewport mesh ready: {temp_path.name}")
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
        self._update_normal_texture()

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
