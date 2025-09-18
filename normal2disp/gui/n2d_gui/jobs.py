"""Background job helpers for the GUI."""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["InspectJob"]


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
