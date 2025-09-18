
from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["InspectJob", "BakeJob"]


def _inspect_worker(mesh_path: str, output: mp.Queue) -> None:
    """Worker entry point that runs :func:`normal2disp.n2d.inspect.run_inspect`."""

    try:
        from normal2disp.n2d.inspect import run_inspect

        payload = run_inspect(Path(mesh_path))
        output.put({"ok": True, "data": payload})
    except Exception as exc:  # pragma: no cover - depends on external assets
        output.put(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        try:
            output.close()
        except Exception:  # pragma: no cover - platform dependent cleanup
            pass


class InspectJob:
    """Run mesh inspection in a separate process and return JSON-safe data."""

    def __init__(self, mesh_path: Path) -> None:
        self._mesh_path = str(mesh_path)
        self._ctx = mp.get_context("spawn")
        self._queue: mp.Queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_inspect_worker,
            args=(self._mesh_path, self._queue),
            daemon=True,
        )
        self._finished = False

    def start(self) -> None:
        """Start the worker process."""

        if not self._process.is_alive():
            self._process.start()

    def poll(self) -> Optional[Dict[str, Any]]:
        """Return the worker result if available, otherwise ``None``."""

        if self._finished:
            return None

        try:
            message = self._queue.get_nowait()
        except queue.Empty:
            if not self._process.is_alive() and self._process.exitcode not in (0, None):
                self._finished = True
                return {
                    "ok": False,
                    "error": "Inspect process exited without a result",
                    "traceback": "",
                }
            return None
        except (EOFError, OSError):  # pragma: no cover - platform dependent
            self._finished = True
            return {
                "ok": False,
                "error": "Inspect worker failed before returning a result",
                "traceback": "",
            }

        self._finished = True
        return message

    def cleanup(self) -> None:
        """Ensure resources associated with the worker are released."""

        if self._process.is_alive():
            self._process.join(timeout=0.2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()

        try:
            self._queue.close()
        except Exception:  # pragma: no cover - platform dependent cleanup
            pass
        try:
            self._queue.join_thread()
        except Exception:  # pragma: no cover - platform dependent cleanup
            pass


def _bake_worker(
    payload: Dict[str, Any],
    output: mp.Queue,
    cancel_event: mp.Event,
) -> None:
    """Run the bake pipeline inside a worker process and stream progress."""

    try:
        from normal2disp.n2d import bake as bake_mod
        from normal2disp.n2d.bake import BakeOptions
        from normal2disp.n2d.core import (
            ImageIOError,
            SolverError,
            TextureAssignmentError,
            UDIMError,
        )
        from normal2disp.n2d.image_utils import read_texture_pixels
        from normal2disp.n2d.inspect import inspect_mesh
        from normal2disp.n2d.uv_raster import TileRasterResult
        from scipy.sparse.linalg import LinearOperator, cg
        import numpy as np
    except Exception as exc:  # pragma: no cover - import failures depend on env
        output.put(
            {
                "kind": "result",
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return

    class _CancelledError(Exception):
        """Internal sentinel raised when cancellation is requested."""

    def emit(message: Dict[str, Any]) -> None:
        output.put(message)

    def emit_log(level: str, message: str) -> None:
        emit({"kind": "log", "level": level, "message": message})

    def check_cancel() -> None:
        if cancel_event.is_set():
            raise _CancelledError()

    def solve_tile_with_progress(
        tile_result: "TileRasterResult",
        normal_path: Path,
        options: "BakeOptions",
        guard: float,
    ) -> "tuple[np.ndarray, List[str]]":
        pixels = read_texture_pixels(Path(normal_path))
        height = int(tile_result.height)
        width = int(tile_result.width)

        if pixels.shape[0] != height or pixels.shape[1] != width:
            raise ImageIOError(
                "Texture '%s' resolution %dx%d does not match expected %dx%d"
                % (normal_path, pixels.shape[1], pixels.shape[0], width, height)
            )
        if pixels.shape[2] < 2:
            raise TextureAssignmentError("Normal maps must have at least two channels")

        decode_mode, normalize_vectors = bake_mod._determine_normal_policy(
            pixels.shape[2], options.normalization
        )

        base_x = np.asarray(pixels[..., 0], dtype=np.float64) * 2.0 - 1.0
        base_y = np.asarray(pixels[..., 1], dtype=np.float64) * 2.0 - 1.0
        if decode_mode == "xyz":
            base_z = np.asarray(pixels[..., 2], dtype=np.float64) * 2.0 - 1.0
        else:
            squared = np.clip(1.0 - base_x * base_x - base_y * base_y, 0.0, None)
            base_z = np.sqrt(squared)

        if not (
            np.all(np.isfinite(base_x))
            and np.all(np.isfinite(base_y))
            and np.all(np.isfinite(base_z))
        ):
            raise ImageIOError(f"Texture '{normal_path}' contains invalid normal components")

        tile_heights = np.zeros((height, width), dtype=np.float64)
        chart_logs: List[str] = []

        for entry in sorted(tile_result.charts, key=lambda chart: chart.local_id):
            check_cancel()

            chart_id = int(entry.chart_id)
            mask = tile_result.chart_mask == entry.local_id
            mask_bool = mask.astype(bool)

            if not np.any(mask_bool):
                emit(
                    {
                        "kind": "stage",
                        "tile": int(tile_result.tile),
                        "chart": chart_id,
                        "name": "decode",
                        "frac": 1.0,
                    }
                )
                emit({"kind": "end_chart", "tile": int(tile_result.tile), "chart": chart_id})
                chart_logs.append(
                    f"Tile {tile_result.tile} chart {chart_id}: no chart pixels"
                )
                continue

            emit(
                {
                    "kind": "stage",
                    "tile": int(tile_result.tile),
                    "chart": chart_id,
                    "name": "decode",
                    "frac": 0.05,
                }
            )

            x = np.array(base_x[mask_bool], dtype=np.float64, copy=True)
            y = np.array(base_y[mask_bool], dtype=np.float64, copy=True)
            z = np.array(base_z[mask_bool], dtype=np.float64, copy=True)

            if options.y_is_down:
                y *= -1.0
            if entry.flip_u:
                x *= -1.0
            if entry.flip_v:
                y *= -1.0

            if normalize_vectors:
                length = np.sqrt(np.maximum(x * x + y * y + z * z, 1e-20))
                x /= length
                y /= length
                z /= length

            z = np.maximum(z, guard)

            du_values = -x / z
            dv_values = -y / z

            if not (
                np.all(np.isfinite(du_values))
                and np.all(np.isfinite(dv_values))
                and np.all(np.isfinite(z))
            ):
                raise SolverError(
                    f"Invalid slope values for tile {tile_result.tile} chart {chart_id}"
                )

            emit(
                {
                    "kind": "stage",
                    "tile": int(tile_result.tile),
                    "chart": chart_id,
                    "name": "slopes",
                    "frac": 0.25,
                }
            )

            du = np.zeros((height, width), dtype=np.float64)
            dv = np.zeros((height, width), dtype=np.float64)
            z_full = np.zeros((height, width), dtype=np.float64)
            du[mask_bool] = du_values
            dv[mask_bool] = dv_values
            z_full[mask_bool] = z

            emit(
                {
                    "kind": "stage",
                    "tile": int(tile_result.tile),
                    "chart": chart_id,
                    "name": "divergence",
                    "frac": 0.45,
                }
            )

            divergence = bake_mod._compute_divergence(du, dv, mask_bool)
            index_map, positions = bake_mod._build_index_map(mask_bool)
            laplacian = bake_mod._build_laplacian(mask_bool, index_map, positions)
            rhs = divergence[mask_bool]

            anchor = bake_mod._select_anchor_index(z_full, positions, width, index_map)
            bake_mod._apply_anchor(laplacian, rhs, anchor)

            matrix = laplacian.tocsr()
            diag = matrix.diagonal()
            inv_diag = np.where(diag != 0.0, 1.0 / diag, 1.0)
            preconditioner = LinearOperator(matrix.shape, matvec=lambda vec: inv_diag * vec)

            iterations = 0
            residual = 0.0

            emit(
                {
                    "kind": "stage",
                    "tile": int(tile_result.tile),
                    "chart": chart_id,
                    "name": "solve",
                    "frac": 0.5,
                }
            )

            def callback(vector):
                nonlocal iterations, residual
                iterations += 1
                residual = float(np.linalg.norm(matrix @ vector - rhs))
                emit(
                    {
                        "kind": "cg_iter",
                        "tile": int(tile_result.tile),
                        "chart": chart_id,
                        "iter": iterations,
                        "residual": residual,
                    }
                )
                max_iter = max(int(options.cg_maxiter), 1)
                frac = 0.5 + min(iterations / max_iter, 1.0) * 0.4
                emit(
                    {
                        "kind": "stage",
                        "tile": int(tile_result.tile),
                        "chart": chart_id,
                        "name": "solve",
                        "frac": frac,
                    }
                )
                check_cancel()

            solution, info = cg(
                matrix,
                rhs,
                tol=float(options.cg_tol),
                maxiter=int(options.cg_maxiter),
                M=preconditioner,
                callback=callback,
            )

            if info != 0:
                residual = float(np.linalg.norm(matrix @ solution - rhs))
                raise SolverError(
                    f"CG solver failed for tile {tile_result.tile} chart {chart_id}: info={info}, residual={residual:.3e}"
                )

            residual = float(np.linalg.norm(matrix @ solution - rhs))
            tile_heights[mask_bool] = solution.astype(np.float64, copy=False)

            emit(
                {
                    "kind": "stage",
                    "tile": int(tile_result.tile),
                    "chart": chart_id,
                    "name": "write",
                    "frac": 1.0,
                }
            )
            emit({"kind": "end_chart", "tile": int(tile_result.tile), "chart": chart_id})
            chart_logs.append(
                f"Tile {tile_result.tile} chart {chart_id}: iter={iterations}, residual={residual:.3e}"
            )

        if not chart_logs:
            chart_logs.append(f"Tile {tile_result.tile}: no chart pixels")

        return tile_heights, chart_logs

    try:
        mesh_path = Path(payload.get("mesh_path", ""))
        normal_pattern = payload.get("normal_pattern") or None
        output_pattern = payload.get("output_pattern") or ""
        options_payload: Dict[str, Any] = dict(payload.get("options", {}))
        overrides: Dict[str, str] = dict(payload.get("material_overrides", {}))

        amplitude = float(options_payload.get("amplitude", 1.0))
        amplitude = max(0.0, min(10.0, amplitude))
        options_payload["amplitude"] = amplitude

        processes = options_payload.get("processes")
        if processes is not None:
            try:
                processes_int = int(processes)
            except (TypeError, ValueError):
                processes_int = None
            else:
                if processes_int <= 0:
                    processes_int = None
            options_payload["processes"] = processes_int

        cg_tol = options_payload.get("cg_tol", 1e-6)
        try:
            options_payload["cg_tol"] = float(cg_tol)
        except (TypeError, ValueError):
            options_payload["cg_tol"] = 1e-6

        cg_maxiter = options_payload.get("cg_maxiter", 10000)
        try:
            options_payload["cg_maxiter"] = int(cg_maxiter)
        except (TypeError, ValueError):
            options_payload["cg_maxiter"] = 10000

        normalization = options_payload.get("normalization", "auto")
        if isinstance(normalization, str):
            options_payload["normalization"] = normalization.lower()

        bake_options = BakeOptions(**options_payload)
        bake_options.material_overrides = overrides

        bake_mod._configure_deterministic_env(bake_options.deterministic)

        emit_log("INFO", f"Inspecting mesh '{mesh_path}' for bake")
        mesh_info = inspect_mesh(mesh_path, loader=bake_options.loader)
        check_cancel()

        uv_set = bake_mod._resolve_uv_set(mesh_info, bake_options.uv_set)
        uv_info = mesh_info.uv_sets[uv_set]

        emit_log("INFO", f"Resolving textures for UV set '{uv_set}'")
        assignments = bake_mod.resolve_material_textures(
            mesh_info,
            uv_set,
            normal_pattern,
            bake_options.material_overrides,
        )

        missing_summary = bake_mod._format_missing_summary(assignments)
        if missing_summary:
            raise TextureAssignmentError(
                f"Missing UDIM tiles detected: {missing_summary}"
            )

        tile_resolutions = bake_mod._collect_tile_resolutions(assignments)
        if not tile_resolutions:
            if uv_info.charts:
                raise TextureAssignmentError(
                    "Cannot bake because no texture tiles were resolved."
                )
            emit_log("INFO", "UV set contains no charts; nothing to bake.")
            emit({"kind": "begin_job", "tiles": 0, "charts_total": 0})
            emit({"kind": "end_job", "ok": True})
            emit(
                {
                    "kind": "result",
                    "ok": True,
                    "outputs": [],
                    "sidecars": [],
                    "logs": ["UV set contains no charts; nothing to bake."],
                    "output_dir": str(Path(output_pattern).parent),
                    "latest_output": "",
                }
            )
            return

        emit_log("INFO", "Rasterising UV charts")
        raster_results = bake_mod.rasterize_uv_charts(
            uv_info, tile_resolutions, deterministic=bake_options.deterministic
        )
        check_cancel()

        tile_paths = bake_mod._collect_tile_paths(assignments)
        guard = bake_mod._max_slope_guard(float(bake_options.max_slope))

        if bake_options.export_sidecars:
            emit_log("INFO", "Exporting sidecars")
            sidecar_paths = bake_mod.export_sidecars(
                mesh_info,
                uv_set,
                assignments,
                deterministic=bake_options.deterministic,
                y_is_down=bake_options.y_is_down,
                precomputed=raster_results,
            )
        else:
            sidecar_paths = []

        total_charts = int(sum(len(result.charts) for result in raster_results))
        emit({"kind": "begin_job", "tiles": len(raster_results), "charts_total": total_charts})

        output_paths: List[Path] = []
        log_lines: List[str] = []

        multi_tile = len(raster_results) > 1
        for tile_result in raster_results:
            check_cancel()
            tile = int(tile_result.tile)
            emit({"kind": "begin_tile", "tile": tile, "charts_in_tile": len(tile_result.charts)})
            normal_path = tile_paths.get(tile)
            if normal_path is None:
                raise TextureAssignmentError(f"No normal map found for UDIM {tile}")

            tile_heights, chart_logs = solve_tile_with_progress(
                tile_result,
                Path(normal_path),
                bake_options,
                guard,
            )
            log_lines.extend(chart_logs)

            check_cancel()

            output_path = bake_mod._resolve_output_path(output_pattern, tile, multi_tile)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            tile_heights *= float(bake_options.amplitude)
            height_data = np.ascontiguousarray(tile_heights.astype(np.float32))

            metadata = {
                "n2d:space": "tangent",
                "n2d:units": "texel",
                "n2d:amplitude": float(bake_options.amplitude),
                "Software": "normal2disp",
                "SourceMesh": str(mesh_info.path),
            }

            bake_mod.write_exr_channels(output_path, {"height": height_data}, metadata=metadata)
            emit_log("INFO", f"Wrote tile {tile} to {output_path}")
            log_lines.append(f"Wrote tile {tile} to {output_path}")
            output_paths.append(output_path)
            emit({"kind": "end_tile", "tile": tile})

        emit({"kind": "end_job", "ok": True})
        emit(
            {
                "kind": "result",
                "ok": True,
                "outputs": [str(path) for path in output_paths],
                "sidecars": [str(path) for path in sidecar_paths],
                "logs": log_lines,
                "output_dir": str(Path(output_pattern).parent),
                "latest_output": str(output_paths[-1]) if output_paths else "",
            }
        )
    except _CancelledError:
        emit_log("WARN", "Canceled by user")
        emit({"kind": "end_job", "ok": False})
        emit(
            {
                "kind": "result",
                "ok": False,
                "error": "Bake canceled",
                "traceback": "",
            }
        )
    except (TextureAssignmentError, ImageIOError, SolverError, UDIMError) as exc:
        emit_log("ERROR", str(exc))
        emit({"kind": "end_job", "ok": False})
        emit(
            {
                "kind": "result",
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    except Exception as exc:  # pragma: no cover - depends on bake assets
        emit_log("ERROR", str(exc))
        emit({"kind": "end_job", "ok": False})
        emit(
            {
                "kind": "result",
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        try:
            output.close()
        except Exception:  # pragma: no cover - platform dependent cleanup
            pass


class BakeJob:
    """Spawn a worker process that executes the bake pipeline."""

    def __init__(
        self,
        mesh_path: Path,
        normal_pattern: Optional[str],
        output_pattern: str,
        options: Dict[str, Any],
        material_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self._ctx = mp.get_context("spawn")
        self._queue: mp.Queue = self._ctx.Queue()
        self._cancel_event: mp.Event = self._ctx.Event()
        self._payload = {
            "mesh_path": str(mesh_path),
            "normal_pattern": normal_pattern,
            "output_pattern": output_pattern,
            "options": dict(options),
            "material_overrides": dict(material_overrides or {}),
        }
        self._process = self._ctx.Process(
            target=_bake_worker,
            args=(self._payload, self._queue, self._cancel_event),
            daemon=True,
        )
        self._finished = False

    def start(self) -> None:
        if not self._process.is_alive():
            self._process.start()

    def poll(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self._finished:
            return messages

        while True:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError):  # pragma: no cover - platform dependent
                self._finished = True
                messages.append(
                    {
                        "kind": "result",
                        "ok": False,
                        "error": "Bake worker failed before returning a result",
                        "traceback": "",
                    }
                )
                return messages
            else:
                messages.append(message)
                if message.get("kind") == "result":
                    self._finished = True

        if not messages and not self._finished and not self._process.is_alive():
            self._finished = True
            messages.append(
                {
                    "kind": "result",
                    "ok": False,
                    "error": "Bake process exited without a result",
                    "traceback": "",
                }
            )

        return messages

    def cancel(self) -> None:
        if not self._finished:
            self._cancel_event.set()

    def is_finished(self) -> bool:
        return self._finished

    def cleanup(self) -> None:
        if self._process.is_alive():
            self._process.join(timeout=0.2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()


        try:
            self._queue.close()
        except Exception:  # pragma: no cover - platform dependent cleanup
            pass
        try:
            self._queue.join_thread()
        except Exception:  # pragma: no cover - platform dependent cleanup
            pass
