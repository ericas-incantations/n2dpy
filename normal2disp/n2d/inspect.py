"""Mesh inspection helpers for the ``n2d inspect`` CLI command."""

from __future__ import annotations

import ctypes
import importlib
import logging
import sys
import sysconfig
import types
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Set, Tuple
from contextlib import contextmanager


import numpy as np

from .core import MeshLoadError
from .mesh_utils import compute_triangle_tangent_frames

__all__ = [
    "run_inspect",
    "inspect_mesh",
    "MeshInfo",
    "UVSetInfo",
    "ChartInfo",
    "UVSetMeshData",
]


_LOGGER = logging.getLogger(__name__)

_UV_EPSILON = 1e-5
_VECTOR_EPSILON = 1e-6

# ``aiComponent`` bitflags copied from ``include/assimp/config.h`` to avoid importing
# the assimp shared library simply to query constants. Only the bits we currently
# need for inspection are defined here.
_AI_COMPONENT_BONEWEIGHTS = 0x20
_AI_COMPONENT_ANIMATIONS = 0x40


@dataclass(frozen=True)
class ChartInfo:
    """Information about a single UV chart."""

    id: int
    face_count: int
    bbox_uv: Tuple[float, float, float, float]
    mirrored: bool
    flip_u: bool
    flip_v: bool
    material_id: int
    material_name: str
    udims: Set[int] = field(default_factory=set)


@dataclass(frozen=True, eq=False)
class UVSetMeshData:
    """Raw mesh data for a UV set used during rasterisation."""

    faces: np.ndarray
    uv: np.ndarray
    face_materials: np.ndarray
    face_chart_ids: np.ndarray
    chart_faces: Mapping[int, np.ndarray]
    face_udims: Tuple[Set[int], ...]


@dataclass(frozen=True, eq=False)
class UVSetInfo:
    """Aggregated information about a UV set."""

    name: str
    charts: List[ChartInfo]
    udims: Set[int]
    per_material_udims: Mapping[int, Set[int]]
    mesh_data: UVSetMeshData


@dataclass(frozen=True, eq=False)
class MeshInfo:
    """Summary of mesh UV information used by inspection and baking."""

    path: Path
    materials: Mapping[int, str]
    uv_sets: Mapping[str, UVSetInfo]


def inspect_mesh(mesh_path: Path, loader: str = "auto") -> MeshInfo:
    """Load ``mesh_path`` and compute inspection metadata."""

    resolved_path = mesh_path.expanduser().resolve()
    loader_choice = loader.lower()
    if loader_choice not in {"auto", "pyassimp"}:
        raise MeshLoadError(
            f"Unsupported loader '{loader}'. Only 'auto' and 'pyassimp' are available in this phase."
        )

    use_pyassimp = loader_choice in {"auto", "pyassimp"}
    assimp_load = None
    pyassimp_errors = None

    if use_pyassimp:
        try:
            _ensure_pyassimp_dependencies()
            import pyassimp
            from pyassimp import errors as pyassimp_errors_mod
            from pyassimp import load as assimp_load_mod

            _apply_pyassimp_workarounds(pyassimp)

            assimp_load = assimp_load_mod
            pyassimp_errors = pyassimp_errors_mod
        except ImportError:
            if loader_choice == "pyassimp":
                raise MeshLoadError("pyassimp is required for mesh inspection")
            use_pyassimp = False

    if use_pyassimp and assimp_load is not None and pyassimp_errors is not None:
        try:
            processing_flags = getattr(
                pyassimp.postprocess,
                "aiProcess_Triangulate",
                0,
            )
            remove_component_step = getattr(pyassimp.postprocess, "aiProcess_RemoveComponent", 0)
            remove_mask = _AI_COMPONENT_ANIMATIONS | _AI_COMPONENT_BONEWEIGHTS

            if remove_component_step:
                processing_flags |= remove_component_step

            with _assimp_component_removal(pyassimp, remove_mask):
                with assimp_load(str(resolved_path), processing=processing_flags) as scene:
                    if scene is None or not scene.meshes:
                        raise MeshLoadError(f"Mesh '{resolved_path}' does not contain any geometry")

                    materials = _extract_materials(scene)
                    vertices, faces, uv_sets, face_materials = _collect_scene_geometry(scene)

                uv_infos: Dict[str, UVSetInfo] = {}
                for name, uv_coords in uv_sets.items():
                    uv_infos[name] = _analyse_uv_set(
                        vertices,
                        faces,
                        uv_coords,
                        face_materials,
                        materials,
                        uv_set_name=name,
                    )

                mesh_info = MeshInfo(path=resolved_path, materials=materials, uv_sets=uv_infos)
                _LOGGER.debug("Mesh info generated: %s", mesh_info)
                return mesh_info
        except pyassimp_errors.AssimpError as exc:  # pragma: no cover - depends on asset
            if loader_choice == "pyassimp":
                raise MeshLoadError(f"Failed to load mesh '{resolved_path}': {exc}") from exc
            _LOGGER.warning("pyassimp failed to load mesh %s: %s", resolved_path, exc)
            use_pyassimp = False

    if not use_pyassimp:
        return _inspect_with_obj_fallback(resolved_path)


def run_inspect(mesh_path: Path, loader: str = "auto") -> Dict[str, Any]:
    """Inspect ``mesh_path`` and return a serialisable report."""

    mesh_info = inspect_mesh(mesh_path, loader=loader)
    report = _mesh_info_to_report(mesh_info)
    _LOGGER.debug("Inspection report generated: %s", report)
    return report


def _mesh_info_to_report(mesh_info: MeshInfo) -> Dict[str, Any]:
    materials = [
        {"id": int(material_id), "name": name}
        for material_id, name in sorted(mesh_info.materials.items())
    ]

    uv_sets: Dict[str, Dict[str, Any]] = {}
    for uv_name, uv_info in sorted(mesh_info.uv_sets.items()):
        charts = [
            {
                "id": chart.id,
                "face_count": chart.face_count,
                "bbox_uv": list(chart.bbox_uv),
                "mirrored": chart.mirrored,
                "flip_u": chart.flip_u,
                "flip_v": chart.flip_v,
                "material_id": chart.material_id,
                "material_name": chart.material_name,
            }
            for chart in uv_info.charts
        ]

        uv_sets[uv_name] = {
            "chart_count": len(charts),
            "udims": sorted(int(tile) for tile in uv_info.udims),
            "charts": charts,
        }

    return {"path": str(mesh_info.path), "materials": materials, "uv_sets": uv_sets}


def _collect_scene_geometry(
    scene: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    vertex_parts: List[np.ndarray] = []
    face_parts: List[np.ndarray] = []
    face_material_parts: List[np.ndarray] = []
    uv_segments: Dict[str, List[Tuple[int, np.ndarray]]] = {}

    vertex_offset = 0
    for mesh in getattr(scene, "meshes", []):
        mesh_vertices = np.asarray(getattr(mesh, "vertices", ()), dtype=np.float64)
        if mesh_vertices.ndim != 2 or mesh_vertices.shape[0] == 0 or mesh_vertices.shape[1] < 3:
            continue

        faces_seq = getattr(mesh, "faces", ())
        faces = np.array(
            [np.asarray(face, dtype=np.int64) for face in faces_seq if len(face) == 3],
            dtype=np.int64,
        )
        if faces.size == 0:
            continue

        vertex_count = mesh_vertices.shape[0]
        vertex_parts.append(mesh_vertices[:, :3])
        face_parts.append(faces + vertex_offset)

        material_index = int(getattr(mesh, "materialindex", 0) or 0)
        face_material_parts.append(np.full(faces.shape[0], material_index, dtype=np.int32))

        coords_sequences = getattr(mesh, "texturecoords", [])
        component_counts = getattr(mesh, "numuvcomponents", [])
        for uv_index, coords in enumerate(coords_sequences):
            if coords is None:
                continue
            components = int(component_counts[uv_index]) if uv_index < len(component_counts) else 0
            if components < 2:
                continue
            uv_array = np.asarray(coords, dtype=np.float64)
            if uv_array.ndim != 2 or uv_array.shape[0] != vertex_count:
                continue
            uv_segments.setdefault(f"UV{uv_index}", []).append((vertex_offset, uv_array[:, :2]))

        vertex_offset += vertex_count

    if not vertex_parts or not face_parts:
        raise MeshLoadError("Mesh does not contain any triangular faces")

    vertices = np.vstack(vertex_parts)
    faces = np.vstack(face_parts)
    face_materials = np.concatenate(face_material_parts)

    total_vertices = vertices.shape[0]
    uv_sets: Dict[str, np.ndarray] = {}
    for name, segments in uv_segments.items():
        combined = np.full((total_vertices, 2), np.nan, dtype=np.float64)
        for offset, uv_array in segments:
            count = uv_array.shape[0]
            combined[offset : offset + count] = uv_array
        uv_sets[name] = combined

    return vertices, faces, uv_sets, face_materials


def _extract_materials(scene: Any) -> Dict[int, str]:
    materials: Dict[int, str] = {}

    for index, material in enumerate(getattr(scene, "materials", [])):
        name: str | None = None
        prop_source = getattr(material, "properties", None)
        if prop_source is not None:
            try:
                for key, value in prop_source.items():
                    if key == "name" and isinstance(value, str):
                        candidate = value.strip()
                        if candidate:
                            name = candidate
                            break
            except AttributeError:
                pass

        if name is None:
            props = None
            if hasattr(material, "properties") and hasattr(material, "contents"):
                try:
                    props = list(material.properties)
                except TypeError:
                    props = None
            if props is None and hasattr(material, "contents"):
                props = getattr(material.contents, "mProperties", None)

            if props is None:
                props = []

            for prop in props:
                key = getattr(prop, "key", "")
                data = getattr(prop, "data", b"")

                if hasattr(prop, "contents"):
                    key_bytes = getattr(prop.contents.mKey, "data", b"")
                    if isinstance(key_bytes, bytes):
                        key = key_bytes.decode("utf-8", errors="ignore")
                    data = bytes(prop.contents.mData[: prop.contents.mDataLength])
                else:
                    if isinstance(key, bytes):
                        key = key.decode("utf-8", errors="ignore")
                    if isinstance(data, bytes):
                        data = data

                if key not in {"?mat.name", "name"}:
                    continue

                decoded = _decode_material_name_bytes(data)
                if decoded:
                    name = decoded
                    break

        materials[index] = name or f"Material{index}"

    if not materials:
        materials[0] = "Material0"

    return materials


def _decode_material_name_bytes(data: Any) -> str | None:
    if isinstance(data, bytes):
        if len(data) >= 4:
            length = int.from_bytes(data[:4], "little", signed=False)
            if length > 0:
                name_bytes = data[4 : 4 + length]
                return name_bytes.decode("utf-8", errors="ignore").strip()
        decoded = data.decode("utf-8", errors="ignore").replace("\x00", "").strip()
        return decoded or None
    if isinstance(data, str):
        decoded = data.replace("\x00", "").strip()
        return decoded or None
    return None


def _analyse_uv_set(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    face_materials: np.ndarray,
    materials: Mapping[int, str],
    *,
    uv_set_name: str,
) -> UVSetInfo:

    if uv_coords.shape[0] != vertices.shape[0]:
        raise MeshLoadError(
            f"UV set '{uv_set_name}' has {uv_coords.shape[0]} coordinates for {vertices.shape[0]} vertices"
        )

    uv = np.asarray(uv_coords[:, :2], dtype=np.float64)
    valid_vertex_mask = np.isfinite(uv).all(axis=1)
    valid_faces_mask = np.all(valid_vertex_mask[faces], axis=1)

    if not valid_faces_mask.any():
        empty_faces = np.zeros((0, 3), dtype=np.int64)
        empty_materials = np.zeros(0, dtype=np.int32)
        empty_chart_ids = np.zeros(0, dtype=np.int32)
        mesh_data = UVSetMeshData(
            faces=empty_faces,
            uv=uv,
            face_materials=empty_materials,
            face_chart_ids=empty_chart_ids,
            chart_faces={},
            face_udims=tuple(),
        )
        return UVSetInfo(
            name=uv_set_name,
            charts=[],
            udims=set(),
            per_material_udims={},
            mesh_data=mesh_data,
        )

    faces_subset = faces[valid_faces_mask]
    face_materials_subset = face_materials[valid_faces_mask]

    adjacency = _build_uv_adjacency(faces_subset, uv, eps=_UV_EPSILON)
    islands = _connected_components(adjacency)

    tangents, bitangents, _normals, orientation = compute_triangle_tangent_frames(
        vertices, faces_subset, uv
    )

    used_vertices = np.unique(faces_subset.reshape(-1))
    udim_list = _compute_udims(uv, used_vertices)

    vertex_udims = _compute_vertex_udims(uv)
    face_udims = [_collect_face_udims(face, vertex_udims) for face in faces_subset]

    charts, chart_faces = _build_charts(
        faces_subset,
        islands,
        uv,
        tangents,
        bitangents,
        orientation,
        adjacency,
        face_materials_subset,
        face_udims,
        materials,
    )

    per_material_udims: Dict[int, Set[int]] = {}
    for chart in charts:
        per_material_udims.setdefault(chart.material_id, set()).update(chart.udims)

    face_chart_ids = np.zeros(len(faces_subset), dtype=np.int32)
    for chart_id, indices in chart_faces.items():
        face_chart_ids[indices] = int(chart_id)

    mesh_data = UVSetMeshData(
        faces=faces_subset,
        uv=uv,
        face_materials=face_materials_subset,
        face_chart_ids=face_chart_ids,
        chart_faces=chart_faces,
        face_udims=tuple(face_udims),
    )

    return UVSetInfo(
        name=uv_set_name,
        charts=charts,
        udims=set(udim_list),
        per_material_udims=per_material_udims,
        mesh_data=mesh_data,
    )


@contextmanager
def _assimp_component_removal(pyassimp_module: types.ModuleType, mask: int) -> Iterator[None]:
    """Temporarily configure Assimp to drop selected components during import."""

    if mask <= 0:
        yield
        return

    try:
        core = getattr(pyassimp_module, "core")
        assimp_lib = getattr(core, "_assimp_lib", None)
        dll = getattr(assimp_lib, "dll", None)
    except AttributeError:
        dll = None

    if dll is None:
        yield
        return

    setter = getattr(dll, "aiSetImportPropertyInteger", None)
    if setter is None:
        yield
        return

    property_name = b"AI_CONFIG_PP_RVC_FLAGS"

    try:
        setter.argtypes = (ctypes.c_char_p, ctypes.c_int)
    except AttributeError:
        pass
    setter.restype = None

    getter = getattr(dll, "aiGetImportPropertyInteger", None)
    previous: int | None = None
    if getter is not None:
        try:
            getter.argtypes = (ctypes.c_char_p, ctypes.c_int)
            getter.restype = ctypes.c_int
            previous = int(getter(property_name, 0))
        except AttributeError:
            previous = None
        except Exception:  # pragma: no cover - depends on ctypes implementation
            previous = None

    setter(property_name, int(mask))
    try:
        yield
    finally:
        restore_value = 0 if previous is None else int(previous)
        try:
            setter(property_name, restore_value)
        except Exception:  # pragma: no cover - best effort cleanup
            pass


def _apply_pyassimp_workarounds(pyassimp_module: types.ModuleType) -> None:
    """Install runtime patches that make pyassimp more tolerant of bad FBX data."""

    core = getattr(pyassimp_module, "core", None)
    if core is None:
        return

    if getattr(core, "_n2d_safe_init", False):  # type: ignore[attr-defined]
        return

    original_init = getattr(core, "_init", None)
    if not callable(original_init):  # pragma: no cover - depends on pyassimp internals
        return

    def _safe_init(self: Any, target: Any | None = None, parent: Any | None = None) -> Any:
        try:
            return original_init(self, target=target, parent=parent)
        except ValueError as exc:
            message = str(exc)
            if "NULL pointer access" not in message:
                raise

            _LOGGER.warning(
                "pyassimp reported NULL pointer access while initialising %s; "
                "treating missing data as empty to continue",
                self,
            )
            _zero_null_pointer_counters(self)
            return original_init(self, target=target, parent=parent)

    setattr(core, "_init", _safe_init)
    setattr(core, "_n2d_safe_init", True)


def _zero_null_pointer_counters(struct: Any) -> None:
    """Reset counter fields whose pointer data is unexpectedly NULL."""

    for attr in dir(struct):
        if not attr.startswith("mNum"):
            continue

        suffix = attr[4:]
        pointer_name = f"m{suffix}"
        if not hasattr(struct, pointer_name):
            continue

        try:
            count = getattr(struct, attr)
        except Exception:  # pragma: no cover - defensive
            continue

        if isinstance(count, np.ndarray):
            # ctypes values sometimes surface as tiny numpy arrays; convert to scalar.
            if count.size != 1:
                continue
            count = int(count[0])

        if not isinstance(count, (int, np.integer)) or count <= 0:
            continue

        pointer_value: Any
        try:
            pointer_value = getattr(struct, pointer_name)
        except Exception:  # pragma: no cover - defensive
            continue

        try:
            is_null = not bool(pointer_value)
        except Exception:  # pragma: no cover - defensive
            is_null = False

        if not is_null:
            continue

        try:
            setattr(struct, attr, 0)
        except Exception:  # pragma: no cover - ctypes structures may forbid assignment
            pass


def _ensure_pyassimp_dependencies() -> None:
    try:
        import distutils  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        try:
            distutils_module = importlib.import_module("setuptools._distutils")
        except ModuleNotFoundError:
            _install_distutils_stub()
            distutils_module = sys.modules["setuptools._distutils"]

        sys.modules.setdefault("distutils", distutils_module)
        if "distutils.sysconfig" not in sys.modules:
            try:
                sys.modules["distutils.sysconfig"] = importlib.import_module(
                    "setuptools._distutils.sysconfig"
                )
            except ModuleNotFoundError:
                sys.modules["distutils.sysconfig"] = sys.modules["setuptools._distutils.sysconfig"]

    system_site = Path("/usr/lib/python3/dist-packages")
    system_site_str = str(system_site)
    if system_site.exists() and system_site_str not in sys.path:
        sys.path.append(system_site_str)


def _install_distutils_stub() -> None:
    """Install a lightweight distutils replacement when setuptools is unavailable."""

    setuptools_module = types.ModuleType("setuptools")
    setuptools_module.__path__ = []  # type: ignore[attr-defined]

    distutils_module = types.ModuleType("setuptools._distutils")
    distutils_module.__path__ = []  # type: ignore[attr-defined]

    sysconfig_module = types.ModuleType("setuptools._distutils.sysconfig")
    sysconfig_module.get_config_var = sysconfig.get_config_var  # type: ignore[attr-defined]
    sysconfig_module.get_config_vars = sysconfig.get_config_vars  # type: ignore[attr-defined]
    sysconfig_module.get_paths = sysconfig.get_paths  # type: ignore[attr-defined]

    def _get_python_lib(plat_specific: bool = False, standard_lib: bool = False) -> str:
        """Minimal ``distutils.sysconfig.get_python_lib`` replacement."""

        if standard_lib:
            key = "platstdlib" if plat_specific else "stdlib"
        else:
            key = "platlib" if plat_specific else "purelib"

        path = sysconfig.get_path(key)
        if not isinstance(path, str):
            raise ImportError(f"Unable to resolve python lib path for key '{key}'")
        return path

    sysconfig_module.get_python_lib = _get_python_lib  # type: ignore[attr-defined]

    distutils_module.sysconfig = sysconfig_module  # type: ignore[attr-defined]
    setuptools_module._distutils = distutils_module  # type: ignore[attr-defined]

    sys.modules.setdefault("setuptools", setuptools_module)
    sys.modules.setdefault("setuptools._distutils", distutils_module)
    sys.modules.setdefault("setuptools._distutils.sysconfig", sysconfig_module)
    sys.modules.setdefault("distutils", distutils_module)
    sys.modules.setdefault("distutils.sysconfig", sysconfig_module)


def _inspect_with_obj_fallback(resolved_path: Path) -> MeshInfo:
    if resolved_path.suffix.lower() != ".obj":
        raise MeshLoadError("pyassimp is required for mesh inspection")

    try:
        (
            vertices,
            faces,
            uv_coords,
            face_materials,
            materials,
        ) = _load_obj_geometry(resolved_path)
    except OSError as exc:  # pragma: no cover - depends on filesystem
        raise MeshLoadError(f"Failed to read mesh '{resolved_path}': {exc}") from exc

    uv_set_name = "UV0"
    uv_info = _analyse_uv_set(
        vertices,
        faces,
        uv_coords,
        face_materials,
        materials,
        uv_set_name=uv_set_name,
    )

    return MeshInfo(path=resolved_path, materials=materials, uv_sets={uv_set_name: uv_info})


def _load_obj_geometry(
    mesh_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    vertices_raw: List[Tuple[float, float, float]] = []
    texcoords_raw: List[Tuple[float, float]] = []
    new_vertices: List[Tuple[float, float, float]] = []
    new_uvs: List[Tuple[float, float]] = []
    faces: List[List[int]] = []
    face_materials: List[int] = []
    vertex_map: Dict[Tuple[int, int], int] = {}

    material_lookup: Dict[str, int] = {}
    materials: Dict[int, str] = {}

    def _material_id(name: str) -> int:
        if name not in material_lookup:
            idx = len(material_lookup)
            material_lookup[name] = idx
            materials[idx] = name
        return material_lookup[name]

    current_material = _material_id("default")

    with mesh_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            prefix = parts[0]
            values = parts[1:]

            if prefix == "v" and len(values) >= 3:
                vertices_raw.append((float(values[0]), float(values[1]), float(values[2])))
            elif prefix == "vt" and len(values) >= 2:
                texcoords_raw.append((float(values[0]), float(values[1])))
            elif prefix.lower() == "usemtl":
                material_name = " ".join(values).strip() or "default"
                current_material = _material_id(material_name)
            elif prefix == "f" and values:
                if len(values) < 3:
                    continue
                indices: List[int] = []
                for token in values:
                    v_index, vt_index = _parse_obj_indices(
                        token, len(vertices_raw), len(texcoords_raw)
                    )
                    key = (v_index, vt_index)
                    mapped = vertex_map.get(key)
                    if mapped is None:
                        position = vertices_raw[v_index - 1]
                        uv = texcoords_raw[vt_index - 1] if vt_index > 0 else (0.0, 0.0)
                        mapped = len(new_vertices)
                        vertex_map[key] = mapped
                        new_vertices.append(position)
                        new_uvs.append(uv)
                    indices.append(mapped)

                if len(indices) < 3:
                    continue

                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])
                    face_materials.append(current_material)

    if not new_vertices or not faces:
        raise MeshLoadError(f"Mesh '{mesh_path}' does not contain triangulated geometry")

    vertices = np.asarray(new_vertices, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    uv_coords = np.asarray(new_uvs, dtype=np.float64)
    face_materials_array = np.asarray(face_materials, dtype=np.int32)

    return vertices, faces_array, uv_coords, face_materials_array, materials


def _parse_obj_indices(token: str, vertex_count: int, texcoord_count: int) -> Tuple[int, int]:
    parts = token.split("/")
    if not parts or not parts[0]:
        raise MeshLoadError(f"Invalid face index token '{token}' in OBJ file")

    vertex_index = int(parts[0])
    if vertex_index < 0:
        vertex_index = vertex_count + vertex_index + 1

    texcoord_index = 0
    if len(parts) > 1 and parts[1]:
        texcoord_index = int(parts[1])
        if texcoord_index < 0:
            texcoord_index = texcoord_count + texcoord_index + 1

    return vertex_index, texcoord_index


def _build_uv_adjacency(
    faces: np.ndarray,
    uv: np.ndarray,
    *,
    eps: float,
) -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(len(faces))]
    edge_to_faces: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}

    for face_index, face in enumerate(faces):
        for offset in range(3):
            vi = int(face[offset])
            vj = int(face[(offset + 1) % 3])

            if not _uv_vertex_finite(uv, vi) or not _uv_vertex_finite(uv, vj):
                continue

            key = (vi, vj) if vi < vj else (vj, vi)
            edge_to_faces.setdefault(key, []).append((face_index, vi, vj))

    for uses in edge_to_faces.values():
        if len(uses) < 2:
            continue

        for idx_a in range(len(uses)):
            face_a, a_v0, a_v1 = uses[idx_a]
            uv_a0 = uv[a_v0]
            uv_a1 = uv[a_v1]

            for idx_b in range(idx_a + 1, len(uses)):
                face_b, b_v0, b_v1 = uses[idx_b]
                uv_b0 = uv[b_v0]
                uv_b1 = uv[b_v1]

                if _uv_edge_matches(uv_a0, uv_a1, uv_b0, uv_b1, eps):
                    adjacency[face_a].append(face_b)
                    adjacency[face_b].append(face_a)

    return [sorted(set(neighbours)) for neighbours in adjacency]


def _connected_components(adjacency: Sequence[Sequence[int]]) -> List[List[int]]:
    if not adjacency:
        return []

    visited = np.zeros(len(adjacency), dtype=bool)
    components: List[List[int]] = []

    for start in range(len(adjacency)):
        if visited[start]:
            continue
        stack = [start]
        component: List[int] = []
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            visited[current] = True
            component.append(current)
            for neighbour in adjacency[current]:
                if not visited[neighbour]:
                    stack.append(neighbour)
        components.append(component)

    return components


def _build_charts(
    faces: np.ndarray,
    islands: Sequence[Sequence[int]],
    uv: np.ndarray,
    tangents: np.ndarray,
    bitangents: np.ndarray,
    orientation: np.ndarray,
    adjacency: Sequence[Sequence[int]],
    face_materials: np.ndarray,
    face_udims: Sequence[Set[int]],
    materials: Mapping[int, str],
) -> Tuple[List[ChartInfo], Dict[int, np.ndarray]]:
    charts: List[ChartInfo] = []
    chart_faces: Dict[int, np.ndarray] = {}

    for chart_id, island in enumerate(islands, start=1):
        face_indices = np.array(island, dtype=int)
        chart_faces[chart_id] = face_indices

        island_faces = faces[face_indices]
        vertex_indices = np.unique(island_faces.reshape(-1))
        uv_coords = uv[vertex_indices]
        finite_mask = np.isfinite(uv_coords).all(axis=1)

        if finite_mask.any():
            uv_valid = uv_coords[finite_mask]
            bbox_min = uv_valid.min(axis=0)
            bbox_max = uv_valid.max(axis=0)
            bbox_uv = (
                float(bbox_min[0]),
                float(bbox_min[1]),
                float(bbox_max[0]),
                float(bbox_max[1]),
            )
        else:
            bbox_uv = (0.0, 0.0, 0.0, 0.0)

        orientation_values = orientation[face_indices]
        neg_count = int(np.sum(orientation_values < 0))
        pos_count = int(np.sum(orientation_values > 0))
        mirrored = neg_count > pos_count or (neg_count > 0 and neg_count == pos_count)

        flip_u = False
        flip_v = False
        if mirrored:
            flip_u, flip_v = _determine_mirror_axis(adjacency, island, tangents, bitangents)

        material_counter = Counter(int(face_materials[idx]) for idx in face_indices)
        if material_counter:
            dominant_material_id = min(
                material_counter.items(), key=lambda item: (-item[1], item[0])
            )[0]
        else:
            dominant_material_id = -1

        if dominant_material_id >= 0:
            material_name = materials.get(dominant_material_id, f"Material{dominant_material_id}")
        else:
            material_name = "<unknown>"

        chart_udims: Set[int] = set()
        for idx in face_indices:
            chart_udims.update(face_udims[idx])

        charts.append(
            ChartInfo(
                id=chart_id,
                face_count=int(len(face_indices)),
                bbox_uv=bbox_uv,
                mirrored=mirrored,
                flip_u=flip_u,
                flip_v=flip_v,
                material_id=int(dominant_material_id),
                material_name=material_name,
                udims=chart_udims,
            )
        )

    return charts, chart_faces


def _determine_mirror_axis(
    adjacency: Sequence[Sequence[int]],
    island: Sequence[int],
    tangents: np.ndarray,
    bitangents: np.ndarray,
) -> Tuple[bool, bool]:
    island_set = set(int(face_index) for face_index in island)
    processed_pairs = set()
    t_inversions = 0
    t_pairs = 0
    b_inversions = 0
    b_pairs = 0

    for face_index in island_set:
        for neighbour in adjacency[face_index]:
            if neighbour not in island_set:
                continue
            pair = (min(face_index, neighbour), max(face_index, neighbour))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            tangent_a = tangents[face_index]
            tangent_b = tangents[neighbour]
            bitangent_a = bitangents[face_index]
            bitangent_b = bitangents[neighbour]

            if _vector_valid(tangent_a) and _vector_valid(tangent_b):
                t_pairs += 1
                if float(np.dot(tangent_a, tangent_b)) < 0.0:
                    t_inversions += 1

            if _vector_valid(bitangent_a) and _vector_valid(bitangent_b):
                b_pairs += 1
                if float(np.dot(bitangent_a, bitangent_b)) < 0.0:
                    b_inversions += 1

    if t_pairs == 0 and b_pairs == 0:
        return True, False

    t_ratio = t_inversions / t_pairs if t_pairs else 0.0
    b_ratio = b_inversions / b_pairs if b_pairs else 0.0

    if t_ratio > b_ratio:
        return True, False
    if b_ratio > t_ratio:
        return False, True
    return True, False


def _compute_vertex_udims(uv: np.ndarray) -> np.ndarray:
    vertex_udims = np.full(uv.shape[0], -1, dtype=np.int32)
    finite_mask = np.isfinite(uv).all(axis=1)
    if not finite_mask.any():
        return vertex_udims

    finite_uv = uv[finite_mask]
    tiles_u = np.floor(finite_uv[:, 0]).astype(int)
    tiles_v = np.floor(finite_uv[:, 1]).astype(int)
    vertex_udims[finite_mask] = 1001 + tiles_u + 10 * tiles_v
    return vertex_udims


def _collect_face_udims(face: np.ndarray, vertex_udims: np.ndarray) -> Set[int]:
    tiles: Set[int] = set()
    for vertex_index in face:
        tile = int(vertex_udims[int(vertex_index)])
        if tile >= 0:
            tiles.add(tile)
    return tiles


def _compute_udims(uv: np.ndarray, vertex_indices: np.ndarray) -> List[int]:
    if vertex_indices.size == 0:
        return []

    coords = uv[vertex_indices]
    finite_mask = np.isfinite(coords).all(axis=1)
    coords = coords[finite_mask]
    if coords.size == 0:
        return []

    tiles_u = np.floor(coords[:, 0]).astype(int)
    tiles_v = np.floor(coords[:, 1]).astype(int)
    tiles = 1001 + tiles_u + 10 * tiles_v
    return sorted(int(tile) for tile in set(tiles))


def _uv_vertex_finite(uv: np.ndarray, index: int) -> bool:
    coord = uv[index]
    return bool(np.isfinite(coord).all())


def _uv_edge_matches(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    eps: float,
) -> bool:
    return (_uv_points_close(a0, b0, eps) and _uv_points_close(a1, b1, eps)) or (
        _uv_points_close(a0, b1, eps) and _uv_points_close(a1, b0, eps)
    )


def _uv_points_close(p: np.ndarray, q: np.ndarray, eps: float) -> bool:
    return float(np.max(np.abs(p - q))) <= eps


def _vector_valid(vec: np.ndarray, eps: float = _VECTOR_EPSILON) -> bool:
    return bool(np.linalg.norm(vec) > eps)
