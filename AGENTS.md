# Agents.md — Normal→Displacement Converter (Python)

**Audience**: Code‑generating/implementing agents (e.g., Codex, Gemini)

**Goal**: Build a CPU‑only tool that converts tangent‑space normal maps to scalar displacement via Poisson integration, with robust handling of UV islands, UDIMs, multiple UV sets, and complex FBX files.

**Target OS/Toolchain**: Windows 11, macOS, Linux. Python 3.10+ installed via **pip** and/or **conda**.

**Scope**: CLI first (no GPU). Design a clean internal API that a future GUI or DCC plugin can call.

---

## Primary Output

**EXR displacement map(s)**

* **Default**: single‑channel 32‑bit float EXR (channel name `height`).
* **Optional**: RGBA EXR with `R=height`, `G/B/A=0`.
* **Metadata** (EXR attributes):

  * `n2d:space = "tangent"`
  * `n2d:units = "texel"` (or `"mm"` if a metric map is applied later)
  * `n2d:amplitude = <float>` (post‑scale applied before write)
  * `Software = "normal2disp"`
  * `SourceMesh = <path>`

---

## Key Features

* **Robust Mesh I/O**: Primary support for **FBX** (multi‑material, multi‑UV) plus OBJ and glTF.
* **UDIM Handling**: Expand `<UDIM>` / `%04d` patterns, validate coverage vs mesh UVs.
* **Correct Normal Decoding**: 2‑channel (BC5‑style) and 3/4‑channel maps with a precise normalization policy (see Algorithm §6.3).
* **UV Island Mirroring Detection**: Detect mirrored shells and flip slope components correctly.
* **Seamless Poisson Solve**: Per‑island masked solve with **Neumann** (zero‑flux) boundaries—no bevel at seams.
* **Sidecars**: Optional export of chart masks + per‑tile chart table JSON.
* **Determinism**: `--deterministic` flag for reproducible output (per‑platform bit‑for‑bit where feasible).

> **Primary workflow requirement**: FBX + PNG inputs must work flawlessly. Prefer reliability over purity; include fallbacks where necessary.

---

## 1) High‑Level Overview

**Problem**
Normal maps encode a tangent‑space direction **n** = (x,y,z). We seek a height field **h(u,v)** whose slopes match the normal‑implied gradients.

**Process**

1. **Decode** normals → slopes in UV space:  $\partial h/\partial u = -x/\max(z,\varepsilon)$, $\partial h/\partial v = -y/\max(z,\varepsilon)$.
2. **Compute** divergence of the slope field within each UV island.
3. **Solve** Poisson $\Delta h = \nabla\cdot g$ per island with **Neumann** boundaries.
4. **Export** height as float EXR with metadata.

**Mesh‑aware**
We load the mesh to extract UV charts, mirrored flags, UDIM tile mapping, and (later) optional metric. The integration is performed in image space but guided by mesh‑derived assets.

**Out of scope (v1)**
GPU compute, vector displacement, direct DCC plugins (API prepares for them).

---

## 2) Tech Stack & Libraries

* **Language**: Python 3.10+
* **Packaging**: `pyproject.toml` with a `console_scripts` entry point (creates the `n2d` command). Keep a `requirements.txt` for CI pins.

**Core libraries**

* **NumPy** — array math and vectorization.
* **SciPy** — sparse linear algebra (`scipy.sparse`, `scipy.sparse.linalg.cg`).
* **pyassimp** — primary FBX/mesh loader (materials, multiple UV sets).
* **trimesh** — post‑load analysis (UV charts, adjacency) and fallback for OBJ/glTF.
* **OpenImageIO (oiio‑python)** — primary image I/O (EXR/PNG, UDIM). *Installation via conda‑forge is recommended.*
* **OpenEXR + Imath** — fallback EXR read/write if OIIO unavailable.
* **Click** — CLI.
* **Rich** — terminal UI (tables/progress).
* **scikit‑image** *(or `opencv-python`)* — efficient triangle rasterization for UV masks.
* *(Optional)* **pyamg** — algebraic multigrid preconditioner for faster Poisson.
* *(Optional external)* **Blender (headless)** — FBX fallback exporter (see §6.1).

**Install guidance**

* Prefer **conda‑forge** for OIIO to avoid wheel gaps. Pip‑only installs may fall back to OpenEXR for EXR and a reduced UDIM convenience layer.

---

## 3) Project Layout

```
├─ n2d/                    # installable package
│  ├─ __init__.py
│  ├─ cli.py               # Click CLI
│  ├─ core.py              # math, datatypes, errors
│  ├─ inspect.py           # 'inspect' command logic
│  ├─ bake.py              # bake pipeline orchestration
│  ├─ mesh_utils.py        # loaders (pyassimp, Blender fallback) + UV analysis (trimesh)
│  ├─ image_utils.py       # image I/O (OIIO primary, OpenEXR fallback)
│  └─ uv_raster.py         # chart mask rasterization (skimage/opencv)
├─ tests/
│  ├─ test_inspect.py
│  ├─ test_bake.py
│  └─ data/                # small sample assets (text‑only by default)
├─ pyproject.toml
├─ requirements.txt
└─ .gitignore
```

---

## 4) Coding Conventions & Error Handling

* **Style**: `black` (format), `ruff` (lint). Enforced in CI.
* **Typing**: Full type hints for all public functions.
* **Errors**: Custom exceptions (`MeshLoadError`, `ImageIOError`, `SolverError`, `UDIMError`). CLI catches and prints actionable one‑liners.
* **Logging**: `logging` module with `--verbose` levels.
* **Version/Capabilities**: `n2d --version` should report: OIIO present? OpenEXR fallback? pyamg present? BLAS vendor? multiprocessing start method.

---

## 5) CLI Design

**`n2d inspect <mesh_path>`**

* `--inspect-json <file>`: write JSON report (materials, UV sets, UDIM tiles, per‑chart `flip_u/flip_v`).
* `--loader {auto,pyassimp,blender}`: force a loader path (useful for debugging FBX).

**`n2d bake <mesh_path>`**

* `--normal <pattern>` **(required)**: path or UDIM pattern for normal map(s).
* `--out <pattern>` **(required)**: path or UDIM pattern for output EXR(s).
* `--uv-set <name>`: pick UV set (default: material‑bound set or first channel).
* `--y-is-down`: flip green channel (−Y/OpenGL style). Default: +Y.
* `--normalization {auto,xyz,xy,none}` (default `auto`): 2‑ch→`xy`, 3/4‑ch→`xyz`.
* `--max-slope <float>` (default `10.0`): slope guard S.
* `--amplitude <float>` (default `1.0`): post‑scale heights before write.
* `--cg-tol <float>` (default `1e-6`), `--cg-maxiter <int>` (default `10000`).
* `--deterministic`: set BLAS/OMP threads=1, spawn start method, sorted processing.
* `--export-sidecars`: write chart masks and chart table JSON per tile.
* `--loader {auto,pyassimp,blender}`: force loader.

**UDIM behavior**

* `--normal` may include `<UDIM>` (or `%04d`). Expand tiles present on disk.
* Compute mesh tiles as `tile = 1001 + floor(u) + 10*floor(v)`.
* If mesh spans tiles with missing textures: error unless `--replicate-udims <list|all>` is supplied to reuse a single map.
* If multiple tiles are processed, `--out` **must** include `<UDIM>`.

**PNG normals are linear**

* Always read PNG/TIF normals as **linear** (no sRGB transform). If sRGB tag present, warn and proceed in linear.

---

## 6) Algorithm Details

### 6.1 Mesh Loading & Analysis

1. Load FBX/mesh via **pyassimp** (primary). Extract materials, UV sets, per‑face material IDs.
2. Convert to a `trimesh.Trimesh` for analysis.
3. Identify UV islands: build UV adjacency (face‑adjacent where UV edges coincide) and label connected components.
4. **Mirroring detection**:

   * For each triangle, compute signed UV area `A_uv`. Construct tangent/bitangent per triangle; use `sign(det(T,B,N))` to detect handedness. A sign flip indicates a mirrored parameterization.
   * Aggregate per island (majority vote). Decide axis: if tangent along +U flips sign → `flip_u=True`; if bitangent along +V flips → `flip_v=True`.
5. **Loader fallback (FBX hard‑mode)**: If pyassimp fails or yields inconsistent UV/material data, invoke a headless Blender script to export a compact mesh+UV+material JSON (and/or OBJ/GLTF) that we then load. Expose `--loader blender` to force this.

### 6.2 UV Chart Rasterization

* Rasterize each island into a per‑tile **chart ID mask** (image coords). Use `scikit-image` (`draw.polygon`) or `opencv` (`fillConvexPoly`) to avoid Python loops.
* Optionally produce a `boundary` mask (uint8) marking pixels whose 4‑neighbors cross chart IDs.

### 6.3 Normal Processing — normalization & slope policy (order matters)

For each masked texel (island pixel):

1. **Decode channels** to \[-1,1]: `x = 2R−1`, `y = 2G−1`, `z = 2B−1` (if present).
2. **Apply flips** to `(x,y)`:

   * Green channel orientation: if `--y-is-down`, set `y := -y`.
   * Mirroring: `flip_u ⇒ x := -x`; `flip_v ⇒ y := -y`.
3. **(Re)construct `z` if needed**:

   * 2‑channel: `z = +sqrt(max(0, 1 − x*x − y*y))`.
   * 3/4‑channel: keep decoded `z`.
4. **Normalize** the vector `(x,y,z)` to unit length.
5. **Max‑slope guard**: let `S = --max-slope`. Enforce `z := max(z, 1/sqrt(1 + S*S))`.
6. **Slopes**: `du = −x / z`, `dv = −y / z`.

### 6.4 Divergence & Laplacian (Neumann zero‑flux)

* Use a **staggered‑flux divergence** for stability:

  * `div[i,j] = (p[i+½,j] − p[i−½,j]) + (q[i,j+½] − q[i,j−½])`, with face fluxes
    `p[i+½,j] ≈ 0.5*(du[i+1,j] + du[i,j])`, `q[i,j+½] ≈ 0.5*(dv[i,j+1] + dv[i,j])`.
* At island boundaries, set the **exterior face flux equal to the interior** (ghost copy) to enforce zero normal flux rather than dropping neighbors.
* Assemble a masked 5‑point Laplacian `A`: `A[p,p]=degree_in_mask(p)`; `A[p,q]=−1` for 4‑neighbors within the island. The RHS `b` is `div`.

### 6.5 Nullspace Fix (Neumann singularity)

* Anchor one pixel per island: choose the largest‑`z` pixel (flattest normal); tie‑break by smallest linear index `(y*W + x)`. Overwrite its row/col to impose `h[p0]=0`.
* (Future) Replace with a weak mean‑zero constraint if desired.

### 6.6 Solver

* Baseline: `scipy.sparse.linalg.cg` with **Jacobi** (diagonal) preconditioner. Expose `--cg-tol`, `--cg-maxiter`.
* Optional: if `pyamg` is present, build a smoothed‑aggregation preconditioner and use with CG.
* After solving, subtract mean if no anchor was applied; apply `--amplitude`; write EXR.

---

## 7) Development Plan & Testing

**Phases**

* **P0 — Bootstrap**: `pyproject.toml`, CLI skeleton, `--version`, CI formatting/lint hooks.
* **P1 — Inspect**: pyassimp + trimesh; materials, UV sets, UDIM tiles, mirroring report; `--inspect-json`.
* **P2 — Texture I/O**: OIIO path (conda), EXR PNG linear reads, UDIM expansion, sRGB warnings.
* **P3 — Rasterization**: chart masks via skimage/opencv; optional boundary mask.
* **P4 — Single‑Island Solve**: normalization, divergence, Poisson, CG; nullspace pin.
* **P5 — Full Bake & Parallelism**: multiprocessing per island/UDIM; deterministic ordering.

**Testing strategy**

* **Integration (FBX+PNG)**: Use repository `testdata/` when present. Cases: two materials & UV sets; y‑flip behavior; UDIM presence/absence; bake on flat normals → near‑zero heights.
* **Golden/edge (generated)**: Create tiny runtime fixtures (≤16×16) in a temp dir and clean up. Include:

  * Procedural height → normals → pipeline → `RMS < 1e-3` on 512².
  * Mirrored islands (`flip_u`/`flip_v`) sign correctness.
  * Donut island (inner/outer Neumann) border flatness.
  * 2‑ch vs 3‑ch normalization consistency.
  * UDIM 1001/1002 expansion.
  * Near‑horizon normals hitting the `z_min` guard (no NaNs/Inf).

**Determinism test**

* `--deterministic` must yield identical hashes across runs on the same platform; `--processes 1` must match default within tolerance.

**Performance note**

* Expect seconds→tens of seconds per 4K tile with \~10 islands on an 8‑core CPU depending on BLAS/pyamg. Vectorize inner loops; avoid Python loops in per‑pixel ops.

---

## 8) API Surface (GUI‑ready)

```python
# n2d/bake.py (shape, not an implementation)
from pathlib import Path
from typing import Optional, Literal, Callable, Sequence, Tuple, List, Dict, Any
from dataclasses import dataclass

NormalizationMode = Literal["auto", "xyz", "xy", "none"]
LoaderMode = Literal["auto", "pyassimp", "blender"]

@dataclass
class BakeOptions:
    uv_set: Optional[str] = None
    y_is_down: bool = False
    normalization: NormalizationMode = "auto"
    max_slope: float = 10.0
    amplitude: float = 1.0
    cg_tol: float = 1e-6
    cg_maxiter: int = 10000
    deterministic: bool = False
    processes: Optional[int] = None
    loader: LoaderMode = "auto"
    export_sidecars: bool = False
    on_progress: Optional[Callable[[str, float], None]] = None  # message, 0..1

# Returns (output_paths, log_lines, sidecar_paths)
def bake(
    mesh_path: Path,
    normal_pattern: str,
    output_pattern: str,
    options: Optional[BakeOptions] = None
) -> Tuple[List[Path], List[str], List[Path]]: ...
```

---

## 9) Sidecars & Schemas

**Chart ID mask** (per tile): `uint16` EXR, channel `chart_id`. `0` = outside; `1..N` = island id. If OpenEXR fallback cannot write `uint16`, use `FLOAT` and document.

**Boundary mask** (optional): `uint8` EXR, channel `boundary` (`1` where 4‑neighbor crosses chart id).

**Chart table JSON** (per tile):

```json
{
  "tile": 1001,
  "uv_set": "UVMap",
  "charts": [
    {"id": 1, "flip_u": false, "flip_v": false, "pixel_count": 12345, "bbox_uv": [u0,v0,u1,v1], "bbox_px": [x0,y0,x1,y1]},
    {"id": 2, "flip_u": true,  "flip_v": false, "pixel_count": 6789,  "bbox_uv": [..],      "bbox_px": [..]}
  ],
  "y_channel": "+Y"  // or "-Y"
}
```

---

## 10) Determinism & Reproducibility

`--deterministic` enforces:

* Environment: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, optionally `PYTHONHASHSEED=0`.
* Multiprocessing: use **spawn** start method; sort UDIMs and chart IDs; stable work partitioning.
* Math: avoid fast‑math; prefer ordered reductions.
* Contract: identical inputs+flags must produce identical outputs per platform; cross‑platform last‑bit differences are acceptable.

---

## 11) Packaging & Install

* Prefer **conda‑forge** for OIIO. Provide a `environment.yml` and `requirements.txt` pins.
* On import, probe capabilities (OIIO vs OpenEXR fallback, pyamg) and expose via `n2d --version`.
* PNG/TIF normals are treated as **linear** (no sRGB transform). Warn on sRGB‑tagged normals.

---

## 12) Acceptance Criteria

* **Correctness**: Golden test RMS < 1e‑3; no bevel at UV seams; boundary behavior matches Neumann.
* **Robustness**: FBX + PNG primary workflow passes (multi‑material, multi‑UV); OBJ/glTF supported; 2/3/4‑channel normals supported; UDIM 1001–1010 handled; sRGB warning path.
* **UDIM**: input expansion and output mapping validated; replication rules enforced.
* **Determinism**: hashes identical across runs with `--deterministic`; `--processes 1` parity.
* **Performance**: Meets “seconds→tens of seconds per 4K tile on 8‑core” expectation (document machine & libs).
* **Output**: EXR written with metadata (`space`, `units`, `amplitude`, `SourceMesh`).
* **API**: `bake()` signature and sidecar schemas stable for GUI adoption.

---

## 13) Future GUI Notes

* The GUI will call `n2d.bake()` and consume sidecars for overlays (chart masks, bbox). Keep options in `BakeOptions` stable. Provide `on_progress` callback for live status.
* Keep all state local to the function to stay thread/process safe. Avoid globals.
