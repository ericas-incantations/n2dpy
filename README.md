<p align="center">
  <img src="assets/icons/n2dbanner.png" alt="normal2disp — tangent normal → displacement" width="900">
</p>

## Overview
normal2disp converts tangent-space normal maps into scalar displacement textures by integrating the slopes implied by each pixel. The toolchain is built around a robust CPU-only pipeline that inspects meshes, validates UDIM coverage, integrates normal maps per UV chart, and exports OpenEXR height fields ready for look-dev or offline rendering workflows. Both a command line interface (CLI) and a Qt Quick 3D GUI are included so teams can automate large batches or drive the bake interactively.

* **Inputs:** FBX (primary), OBJ, and glTF meshes with multiple materials and UV sets; PNG or TIFF tangent-space normal maps (2-, 3-, or 4-channel) with optional `<UDIM>` or `%04d` filename patterns.
* **Outputs:** Single-channel 32-bit float OpenEXR displacement maps per UDIM tile (`height` channel) enriched with metadata. Optional sidecars include chart ID masks, boundary masks, and per-tile chart tables in JSON.
* **Use cases:** Reconstruct sculpted height detail from shipped normal maps, prepare assets for displacement-capable renderers, validate UDIM coverage before baking, or preview displacement inside the bundled GUI.

## Quick Start (CLI)
All commands share the `--verbose` flag on the root `n2d` group for detailed logging.

### Inspect a mesh and save the report
```bash
n2d inspect path/to/mesh.fbx --inspect-json inspect.json
```
* Prints a Rich-formatted summary of UV sets, charts, and UDIM tiles.
* Writes `inspect.json` containing the structured report for downstream tooling.

### Validate UDIM/material mapping without solving
```bash
n2d bake mesh.fbx --uv-set UV0 --normal "norm_<UDIM>.png" --validate-only
```
* Expands the UDIM pattern, verifies coverage against the mesh, and reports missing tiles.
* Emits a JSON summary to stdout (and `--inspect-json` if supplied). No EXRs are written.

### Bake a single-tile displacement map
```bash
n2d bake mesh.fbx --normal normal_1001.png --out disp_1001.exr --amplitude 1
```
* Solves the Poisson integration for the chart(s) covered by the 1001 tile.
* Writes `disp_1001.exr` with a `height` channel and metadata, plus optional sidecars when `--export-sidecars` is enabled.

### Bake a multi-UDIM set
```bash
n2d bake mesh.fbx --uv-set UV0 --normal "norm_<UDIM>.png" --out "disp_<UDIM>.exr" --amplitude 1
```
* Processes all tiles referenced by the selected UV set, writing one EXR per UDIM using the pattern.
* Validation fails fast if any required tile is missing or mixes resolutions across materials.

## Quick Start (GUI)
Launch the desktop application with:
```bash
python -m n2d_gui.app
```

* **Left panel:** Pick Mesh, Normal (single file or UDIM pattern), Material, Output directory, and options (UV set, amplitude slider clamped to 0–10, Y-is-down, normalization mode, max slope, CG tolerance/iterations, deterministic mode, process count, and sidecar export).
* **Center viewport:** Qt Quick 3D scene showing the mesh with an orbit/pan/zoom camera, wireframe toggle, and normal-map material bound to the same texture shown on the right.
* **Right pane:** Normal-map preview synchronized with the viewport. UDIM selectors stay hidden until multiple tiles are detected or the Advanced drawer is opened.
* **Bottom status bar:** Progress bar, status text, live log output, Cancel control, and buttons to open the output directory or reveal the latest EXR.
* **Displacement preview:** After a successful bake, toggle the displacement preview to swap in subdivided geometry (Loop subdivision levels 0–5) displaced by the baked EXR. The normal-map preview disables automatically while displacement shading is active.
* **Lighting:** Light azimuth slider controls the viewport directional light (elevation lives under Advanced controls).

## Inputs & Outputs (detailed)
### Meshes and materials
* FBX via pyassimp is the primary path; OBJ and glTF fall back to internal parsers when pyassimp is unavailable.
* All meshes may contain multiple materials and UV sets. UV charts are grouped per material to enforce texture validation per assignment.

### Normal maps
* PNG/TIF files are treated as linear data. The reader emits a warning when an sRGB colorspace tag is detected.
* 2-channel XY (BC5-style), 3-channel XYZ, and 4-channel XYZA normal maps are supported. Missing Z is reconstructed when needed.
* Mirrored islands are detected during inspection; slope components are flipped per chart using the `flip_u`/`flip_v` flags.

### UDIM handling
* Filename patterns accept `<UDIM>` or `%04d`. Non-pattern paths are treated as single-tile assets; the tile ID defaults to the first `1xxx` number present or `1001`.
* Mesh validation enumerates the UDIM tiles touched by every chart. Missing tiles raise `TextureAssignmentError` with a per-material summary.
* Tiles with mixed resolutions across materials trigger a descriptive error before solving.

### Displacement EXR output
* Each bake writes one EXR per tile using the resolved pattern (single path for single-tile jobs).
* Channels: single `height` float32 layer (additional channels may be appended by future workflows).
* Metadata embedded in every file:
  * `n2d:space = "tangent"`
  * `n2d:units = "texel"`
  * `n2d:amplitude = <float>`
  * `Software = "normal2disp"`
  * `SourceMesh = <mesh path>`

### Sidecar artifacts (`--export-sidecars`)
* **Chart ID mask:** `chart_id` channel, `uint16` (or float fallback) marking each chart within the tile.
* **Boundary mask:** `boundary` channel, `uint8` with 4-connected chart boundaries.
* **Chart table JSON:**
  ```json
  {
    "tile": 1001,
    "uv_set": "UVMap",
    "charts": [
      {"id": 1, "flip_u": false, "flip_v": false, "pixel_count": 12345,
       "bbox_uv": [u0, v0, u1, v1], "bbox_px": [x0, y0, x1, y1]},
      {"id": 2, "flip_u": true, "flip_v": false, ...}
    ],
    "y_channel": "+Y"
  }
  ```
* Sidecars live in `<mesh stem>_sidecars/` next to the input mesh.

## How it Works (math & algorithm)
1. **Normal decoding:** Pixels are converted from [0,1] to [-1,1]. Depending on the selected normalization mode, vectors are normalized and missing Z is reconstructed as `sqrt(max(1 - x^2 - y^2, 0))`. Y-is-down toggles invert Y, and mirrored charts flip X/Y as required.
2. **Max-slope guard:** Normals are clamped so `z ≥ 1/√(1 + S²)` where `S` is the configured max slope. This protects the solve from near-horizontal normals.
3. **Slope field:** Per-pixel slopes are derived as `du = -x/z` and `dv = -y/z` inside each chart mask.
4. **Divergence:** Staggered fluxes produce a divergence image using a masked 5-point stencil while preserving Neumann (zero-flux) boundaries.
5. **Poisson solve:** Each chart becomes a sparse linear system `Δh = div(g)` solved via conjugate gradient (CG). The flattest pixel (largest Z, tie-broken by UV order) anchors the nullspace. CG tolerance and iteration cap follow CLI options.
6. **Displacement assembly:** Chart solutions populate per-tile height fields, scaled by the amplitude and written to EXR with metadata.
7. **Determinism:** The `--deterministic` flag pins BLAS/OMP thread counts, forces deterministic ordering of UDIM tiles and chart traversal, and stabilizes reductions for reproducible outputs on the same platform.

## Workflows
### Material-centric pipeline
1. Run `n2d inspect` to enumerate UV sets, materials, and charts.
2. Provide a default normal map pattern with `--normal` and material overrides via repeated `--mat-normal Name=pattern` entries. The CLI de-emphasizes UDIM selection by mapping tiles automatically once the material is chosen.
3. Execute `n2d bake` with `--validate-only` to confirm coverage. Full bakes require `--out` pointing to a filename or `<UDIM>` pattern.

### Single-tile & multi-UDIM baking
* **Single tile:** Supply an absolute path to the normal map and a matching `--out` destination. No placeholder is required unless multiple tiles are active.
* **Multi-UDIM:** Patterns must contain `<UDIM>` or `%04d`. Missing tiles, mixed resolutions, or absent outputs trigger descriptive errors before solving.

### GUI flow
1. Launch the GUI, load a mesh, and run Inspect. Material and UV selectors populate automatically; UDIM selectors stay hidden until multiple tiles appear.
2. Pick a normal map or pattern, set options (including deterministic or multi-process choices), and start the bake.
3. Track bake progress via global progress, per-chart stage messages (`decode`, `slopes`, `divergence`, `solve`, `write`), and CG iteration logs.
4. When complete, toggle displacement preview, adjust subdivision level (0–5), and compare displaced and normal-shaded views. Advanced controls allow selecting alternate tiles and adjusting lighting.

### Large jobs
* Increase `--processes` (CLI) or the GUI Processes field to parallelize per-tile work. Deterministic mode overrides the parallelism to one worker while clamping BLAS thread counts.
* Monitor CG residuals and iteration counts emitted in the log to diagnose convergence. Cancellation is honored between charts and tiles.

## Performance & Limits
* CPU-only implementation. Expect seconds to tens of seconds per 4K tile on modern 8-core CPUs with optimized BLAS.
* Memory grows with subdivision level during GUI displacement preview; level 5 may consume gigabytes on dense meshes—use with caution.
* Linear normal maps are required. Avoid sRGB conversions, keep normals within reasonable slope ranges, and clamp `max-slope` if near-horizon artifacts appear.

## Troubleshooting
| Symptom | Resolution |
| --- | --- |
| `Missing UDIM tiles detected` | Ensure every tile touched by the selected UV set exists on disk or provide material-specific overrides. |
| `UDIM ... has mixed resolutions` | Normalize texture resolutions per tile (e.g., re-export mismatched maps). |
| EXR read/write failures | Confirm OpenImageIO is installed and the output directory is writable; rerun with `--verbose` for details. |
| `pyassimp is required` errors | Install pyassimp (and system dependencies) or convert the mesh to OBJ/glTF for the fallback loader. |
| Solver non-convergence | Reduce `--amplitude`, tighten `--max-slope`, or increase `--cg-maxiter`. Check for invalid normals in the source textures. |
| GUI preview shows no displacement | Ensure the bake succeeded, displacement preview is enabled, and the baked EXR resides in the selected output directory. |
| Collect diagnostics | Use `--inspect-json` or GUI log exports, archive generated sidecars, and capture CLI `--verbose` output for support. |

## CLI Reference
### Root command
```
n2d [--verbose] COMMAND
```
* `--verbose` – enable DEBUG logging across subcommands.

### `n2d version`
Displays the package version and probes for optional modules (OpenImageIO, pyassimp).

### `n2d inspect MESH`
* `--inspect-json PATH` – write the inspection report as JSON.
* `--loader {auto,pyassimp}` – choose the mesh loader (default `auto`).

### `n2d bake MESH`
* `--normal PATH_OR_PATTERN` – default normal map (single file or pattern).
* `--out PATH_OR_PATTERN` – output EXR path or UDIM pattern (required unless `--validate-only`).
* `--mat-normal KEY=PATTERN` – material-specific normal overrides (repeatable; `KEY` may be an index or name).
* `--uv-set NAME` – UV set to bake (defaults to the first detected set).
* `--y-is-down` – interpret +Y normals as pointing down.
* `--normalization {auto,xyz,xy,none}` – normal decoding policy (default `auto`).
* `--max-slope FLOAT` – clamp normals to the provided maximum slope (default `10.0`).
* `--amplitude FLOAT` – scale applied before writing displacement (default `1.0`).
* `--cg-tol FLOAT` – conjugate gradient tolerance (default `1e-6`).
* `--cg-maxiter INT` – maximum CG iterations (default `10000`).
* `--validate-only` – validate texture assignments without solving.
* `--export-sidecars` – write chart masks, boundary masks, and chart table JSON per tile.
* `--deterministic` – enforce deterministic ordering and single-threaded math.
* `--inspect-json PATH` – mirror of the inspection report emitted during validation/bake.
* `--loader {auto,pyassimp}` – select the mesh loader.

## Glossary & References
* **UDIM:** Tiled UV indexing scheme where tile numbers (1001, 1002, …) encode U/V offsets.
* **Chart / Island:** Connected set of triangles in UV space solved as an independent Poisson problem.
* **Neumann boundary:** Zero-flux boundary condition preserving continuity across seams.
* **CG (Conjugate Gradient):** Iterative solver for the sparse Poisson system.
* **Amplitude:** Scalar multiplier applied to the solved height field before writing.
* **Normalization:** Policy for reconstructing and renormalizing normals prior to integration.

Further design details and architectural notes are documented in [AGENTS.md](AGENTS.md) (core/CLI) and [GUI.md](GUI.md) (desktop application).

## License / Credits
Project licensing will be defined in a future update. Attribution and third-party notices will be added once the distribution model is finalized.
