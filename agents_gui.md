# Agents\_GUI.md — normal2disp GUI (PySide6 + Qt Quick 3D)

**Audience:** Code‑generating agents (e.g., Codex) implementing a modern desktop GUI for the existing `normal2disp` Python package.

**Goal:** Deliver a beautiful, responsive GUI that previews meshes and normal maps, orchestrates `n2d` bakes, shows accurate progress, and (optionally) previews displacement by subdividing and displacing the mesh using the baked EXR.

**Non‑goals:** Alternate frameworks, alternate dependency plans, web UI, or GPU compute paths. Stick to this single plan.

---

## 1) UX Layout

* **Left panel (Controls):**

  * File selectors: Mesh, Normal map (or UDIM pattern), Output directory
  * Material selector (populated from `inspect`)
  * Options:

    * UV Set (dropdown)
    * Y‑is‑down (checkbox)
    * Normalization (auto/xyz/xy/none)
    * **Amplitude slider \[0..10]** + numeric
    * Max slope (slider + numeric)
    * CG tol / max iters (numeric)
    * Deterministic (checkbox)
    * Processes (int; 0=auto)
    * Export sidecars (checkbox)

* **Center:** 3D Viewport (Qt Quick 3D). Orbit/pan/zoom; wireframe toggle.

* **Right:** Normal Map Preview pane (dominant tile by default; UDIM controls hidden unless >1 tile).

* **Bottom:** Status output panel with **progress bar**, single‑line status, streaming log, Cancel button, Open output folder / Reveal latest EXR buttons.

* **Advanced toggles:** Light azimuth slider \[0..360°] on bottom bar; Elevation optional later.

* **UDIMs de‑emphasized:** Default to Material‑centric flow; expose tiles only if multiple are detected or user opens an “Advanced” drawer.

---

## 2) Architecture

### 2.1 Process Model & IPC

* **UI process:** PySide6 + QML.
* **Worker process:** Runs `inspect` and `bake` to keep UI responsive and to isolate BLAS/OMP threads.
* **IPC:** `multiprocessing` queues (pickleable work packets + JSON progress events). Cancellation flag via a `multiprocessing.Event`.

### 2.2 Backend Modules (new)

* `gui/n2d_gui/app.py`: Entry point (`python -m n2d_gui.app`), sets up QML engine and QML file.
* `gui/n2d_gui/backend.py`: `QObject` bridge exposing invokables to QML (browse, runInspect, runBake, cancel, openOutput, etc.).
* `gui/n2d_gui/jobs.py`: Job orchestration; worker process target; progress streaming; cancellation.
* `gui/n2d_gui/image_provider.py`: `QQuickImageProvider` to feed the right‑pane previews (normal maps, thumbnails, and later height thumbnails).
* `gui/n2d_gui/viewport.py`: Light wrapper for feeding geometry and materials to Quick 3D from Python (optional; most logic can stay in QML).
* `gui/n2d_gui/subdivision.py`: CPU Loop subdivision (levels 0–5) + displacement application using EXR sampling.
* `gui/n2d_gui/models.py`: Qt models for UV sets, materials, tiles, options.
* `gui/qml/`: QML files (MainWindow\.qml, LeftPanel.qml, Viewport.qml, RightPane.qml, StatusBar.qml, Theme.qml).

### 2.3 3D Viewport Strategy

* Use **Qt Quick 3D**.
* **Normal preview:** Material uses the selected normal map (single tile; if multiple tiles, show a selector in Advanced drawer). This is for orientation & coverage sanity checks.
* **Displacement preview (post‑bake):**

  * Load baked EXR via **OpenImageIO** (already installed).
  * **Loop subdivision levels 0–5** (no artificial cap; warn if memory explodes but proceed if the user wants).
  * Sample height at vertex UV (bilinear), displace along vertex normals (recomputed post‑subdiv), update a `QQuick3DGeometry` instance.
  * When displacement preview is **ON**, disable normal‑map preview to avoid double shading.
  * Regenerate geometry if subdiv level changes.

### 2.4 Performance Approach

* Favor **vectorized NumPy** for subdivision connectivity transforms and displacement sampling.
* Use worker **threads** inside the UI process for geometry generation to keep the GUI responsive; the heavy baking still occurs in the separate worker **process**.
* Do not hard‑limit performance; log triangle counts and timings. Users can push to 5 subdivs at their own risk.

---

## 3) Progress Protocol (wire format)

Progress events are JSON‑serializable dicts sent from the worker process:

```
{kind:"begin_job", tiles:int, charts_total:int}
{kind:"begin_tile", tile:int, charts_in_tile:int}
{kind:"stage", tile:int, chart:int, name:str, frac:float}  # name∈{"decode","slopes","divergence","solve","write"}
{kind:"cg_iter", tile:int, chart:int, iter:int, residual:float}
{kind:"end_chart", tile:int, chart:int}
{kind:"end_tile", tile:int}
{kind:"end_job", ok:bool}
{kind:"log", level:"INFO|WARN|ERROR", message:str}
```

UI computes global progress as `(finished_charts + current_stage.frac)/charts_total`. Show per‑tile/CG details in the status line; also append raw log lines to the console.

Cancellation: worker checks an Event between charts and at safe stage boundaries; emits `{kind:"log", level:"WARN", message:"Canceled by user"}` and `{kind:"end_job", ok:false}` on cancel.

---

## 4) User Flows

**Inspect Flow**

1. User picks Mesh → `InspectJob` runs → returns `MeshInfo` (materials, UV sets, charts, UDIMs, mirroring).
2. UI populates Material + UV set selectors; UDIM controls hidden unless multiple tiles.
3. Right pane shows normal map placeholder; warnings surface if coverage gaps exist.

**Bake Flow**

1. User picks Normal map / pattern and Output directory; sets options.
2. Submit `BakeJob` (mesh path, patterns, UV set, options). Deterministic mode sets BLAS/OMP env vars in the worker.
3. Progress stream updates bar and log; on success, bottom bar offers “Open output folder” and “Reveal latest EXR”.
4. **Displacement Preview:** After success, user toggles Displacement Preview, chooses Subdiv \[0..5], geometry regenerates from EXR.

---

## 5) Milestones (agent tasks)

### GUI‑P0 — App Skeleton & Theme

* Create `gui/requirements.txt` (above), `gui/n2d_gui/` package, and `gui/qml/`.
* Implement `app.py` launching QML `MainWindow.qml` with left/center/right/bottom layout and a basic theme.
* Add minimal actions (file pickers update labels; no backend yet).

### GUI‑P1 — Inspect Integration

* Implement `backend.py` QObject with invokables: browseMesh(), runInspect(meshPath), getMaterials(), getUVSets().
* Implement `jobs.py` `InspectJob` in a worker process, calling existing `inspect` path from `normal2disp`.
* Populate Material + UV set lists in the UI; show counts and any warnings.

### GUI‑P2 — Viewport MVP & Normal Preview

* Quick 3D viewport with orbit/pan/zoom; wireframe toggle.
* Bind selected normal map (single tile) to a preview material; show sRGB warning if any.
* Implement `image_provider.py` to expose preview tiles as `image://n2d/…` URLs in QML.

### GUI‑P3 — Bake Orchestration + Progress

* Implement `BakeJob` in `jobs.py`, wiring to `normal2disp.bake()` with all relevant options.
* Implement the **Progress Protocol** and UI aggregation → bottom progress bar + status line + log console; Cancel button.
* Enforce amplitude slider \[0..10] in UI and pass to options.

### GUI‑P4 — Displacement Preview (Subdiv 0–5)

* Implement `subdivision.py`:

  * Loop subdivision (tri meshes), levels 0–5.
  * Recompute vertex normals.
  * EXR sampling via OpenImageIO, bilinear in UV.
  * Build `QQuick3DGeometry` buffers and present in viewport.
* Toggle logic: when ON, disable normal material; show "Regenerate" when level changes.
* Performance: no hard caps; just show triangle counts and timings.

### GUI‑P5 — UDIM UX & Advanced Controls

* Keep UDIM controls hidden unless multi‑tile; provide an Advanced drawer to choose tile.
* Add bottom **Light azimuth slider \[0..360°]**; apply to a DirectionalLight in the scene.
* Optional: Elevation slider in Advanced.

### GUI‑P6 — Packaging & Installers

* Add `pyside6-deploy` config; produce per‑OS bundles.
* Windows: create Inno Setup script; macOS: codesign/notarize hooks + DMG; Linux: AppImage via linuxdeploy.
* CI: GitHub Actions matrix builds on tag push; publish artifacts.

### GUI‑P7 — Polish & Presets

* Preset save/load (JSON of all options + material mapping).
* Session restore (last files, window layout).
* Error banners & accessibility (contrast/keyboard nav).

---

## 6) File/Module Layout

```
normal2disp/
  n2d/                      # existing core package (unchanged)
  gui/
    requirements.txt
    n2d_gui/
      __init__.py
      app.py               # python -m n2d_gui.app
      backend.py           # QObject bridge
      jobs.py              # worker orchestration, progress, cancel
      image_provider.py    # QQuickImageProvider for previews
      subdivision.py       # Loop subdiv + displacement (0..5)
      models.py            # Qt list models, data classes
      viewport.py          # helpers to feed Quick3D geometry/materials
    qml/
      MainWindow.qml
      LeftPanel.qml
      Viewport.qml
      RightPane.qml
      StatusBar.qml
      Theme.qml
```

---

## 8) Acceptance Criteria (per milestone)

* **P0:** App launches, panels render, theme applied.
* **P1:** Inspect reads mesh; UI lists Materials + UV sets; UDIMs hidden unless >1.
* **P2:** Viewport shows mesh; normal-map preview visible; orbit/zoom works.
* **P3:** Bake runs; progress bar advances smoothly; Cancel works; logs stream; amplitude \[0..10] honored.
* **P4:** After a successful bake, displacement preview ON regenerates geometry; subdivision 0–5 selectable; normal-map preview disabled while disp ON; timings and tri counts shown.
* **P5:** Advanced drawer exposes tiles when multi‑tile; bottom light azimuth slider controls directional light.
* **P6:** Reproducible bundles produced for Win/macOS/Linux (unsigned OK in CI); installer assets exist.
* **P7:** Presets, session restore, polished errors, basic accessibility checks.

---

## 9) Notes & Constraints

* Keep **one** framework: PySide6 + Qt Quick 3D.
* Respect existing `normal2disp` APIs; do not fork logic into the GUI.
* No arbitrary runtime caps: allow up to **5 subdivision levels**; surface performance warnings but do not block.
* Progress events must adhere to the protocol above for a smooth progress bar.
* Deterministic mode: set env and disable parallelism in the bake worker; the GUI should present this option but not force it.

---
