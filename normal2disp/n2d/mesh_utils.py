"""Mesh loading and analysis helpers used by ``n2d.inspect``."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import trimesh

__all__ = ["assimp_mesh_to_trimesh", "compute_triangle_tangent_frames"]


def assimp_mesh_to_trimesh(mesh: Any) -> trimesh.Trimesh:
    """Convert a pyassimp mesh to a :class:`trimesh.Trimesh` instance."""

    vertices = np.asarray(getattr(mesh, "vertices", ()), dtype=np.float64)
    faces_seq = getattr(mesh, "faces", ())
    faces = np.array(
        [np.asarray(face, dtype=np.int64) for face in faces_seq if len(face) == 3],
        dtype=np.int64,
    )

    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise ValueError("Mesh vertices are missing or malformed")
    if faces.size == 0:
        raise ValueError("Mesh does not contain any triangular faces")

    return trimesh.Trimesh(
        vertices=vertices[:, :3],
        faces=faces,
        process=False,
        maintain_order=True,
    )


def compute_triangle_tangent_frames(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    *,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-triangle tangent frames.

    Returns ``(tangent, bitangent, normal, orientation_sign)`` arrays.
    """

    face_count = int(faces.shape[0])
    tangents = np.zeros((face_count, 3), dtype=np.float64)
    bitangents = np.zeros((face_count, 3), dtype=np.float64)
    normals = np.zeros((face_count, 3), dtype=np.float64)
    orientation = np.zeros(face_count, dtype=np.float64)

    for face_index, face in enumerate(faces):
        idx0, idx1, idx2 = (int(face[0]), int(face[1]), int(face[2]))
        p0, p1, p2 = vertices[idx0], vertices[idx1], vertices[idx2]
        uv0, uv1, uv2 = uv[idx0], uv[idx1], uv[idx2]

        if not (np.all(np.isfinite(uv0)) and np.all(np.isfinite(uv1)) and np.all(np.isfinite(uv2))):
            continue

        edge1 = p1 - p0
        edge2 = p2 - p0
        normal = np.cross(edge1, edge2)
        normal_length = float(np.linalg.norm(normal))
        if normal_length < eps:
            continue

        du1, dv1 = uv1 - uv0
        du2, dv2 = uv2 - uv0
        denom = du1 * dv2 - dv1 * du2
        if abs(float(denom)) < eps:
            continue

        factor = 1.0 / float(denom)
        tangent = (edge1 * dv2 - edge2 * dv1) * factor
        bitangent = (edge2 * du1 - edge1 * du2) * factor

        tangent_length = float(np.linalg.norm(tangent))
        bitangent_length = float(np.linalg.norm(bitangent))
        if tangent_length < eps or bitangent_length < eps:
            continue

        tangent /= tangent_length
        bitangent /= bitangent_length
        normal_unit = normal / normal_length

        tangents[face_index] = tangent
        bitangents[face_index] = bitangent
        normals[face_index] = normal_unit

        handedness = float(np.dot(np.cross(tangent, bitangent), normal_unit))
        if handedness > eps:
            orientation[face_index] = 1.0
        elif handedness < -eps:
            orientation[face_index] = -1.0

    return tangents, bitangents, normals, orientation
