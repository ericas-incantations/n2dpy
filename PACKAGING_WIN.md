# Windows Packaging Guide — normal2disp GUI

This document describes how to build the Windows folder bundle and installer for the **normal2disp GUI** on a local Windows 10/11 workstation using **Python 3.12**. The same steps are reflected in the Windows GitHub Actions workflow (`.github/workflows/windows-build.yml`).

> **Important:** All build artifacts (`dist/`, installers, signed binaries, etc.) must stay out of Git. Clean them up or add them to your local `.gitignore` before committing.

---

## 1. Prerequisites

* Windows 10 or 11 with the Desktop Experience.
* [Python 3.12.x](https://www.python.org/downloads/windows/) installed and available as `py -3.12`.
* Visual C++ build tools (part of the **Build Tools for Visual Studio 2022**) – required by some Python wheels.
* Local clone of this repository (e.g., `git clone https://github.com/<org>/normal2disp`).
* Internet access for installing Python packages and Inno Setup.

Optional (for signing):

* A code-signing certificate in **PFX** format and its password.
* Timestamp service URL (e.g., DigiCert or Sectigo).

Set the following environment variables before running any signing step:

* `N2D_CODESIGN_PFX_PATH` – absolute path to the `.pfx` file.
* `N2D_CODESIGN_PFX_PASSWORD` – password for the PFX.
* `N2D_CODESIGN_TIMESTAMP_URL` – RFC 3161 timestamp server URL (optional but recommended).

These variables are consumed by the Inno Setup script and any optional `signtool` invocations. **Do not** store certificates or passwords in the repository.

---

## 2. Create and Activate a Virtual Environment

```powershell
cd <repo>\normal2disp
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade packaging helpers inside the environment:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

---

## 3. Install Runtime and Development Dependencies

Install the normal2disp CLI + GUI requirements and the development tooling used by the smoke tests:

```powershell
python -m pip install -e .
python -m pip install -r normal2disp/gui/requirements.txt
python -m pip install ruff pytest
```

Install **PySide6** (matching the GUI requirements) and the deployment tool:

```powershell
python -m pip install PySide6==6.7.2 pyside6-deploy==6.7.2
```

> Depending on your Python installation you may also need `python -m pip install build` if you plan to create wheels separately.

Confirm the deploy tool version:

```powershell
pyside6-deploy --version
```

---

## 4. Headless Smoke Test

Set Qt to run headless during validation and execute the linters/tests:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m ruff check .
python -m pytest
```

Finally, ensure the GUI can start in offscreen mode. The build script runs a short helper that opens the event loop for a split second and exits. To approximate that manually, run:

```powershell
python -c "from PySide6.QtCore import QTimer; from n2d_gui.app import main; QTimer.singleShot(150, lambda: __import__('PySide6.QtWidgets').QtWidgets.QApplication.instance().quit()); __import__('sys').exit(main())"
```

If you only need to verify imports, use the lighter check: `python -c "import n2d_gui.app; print('GUI import ok')"`.

Clear the environment variable when you are done testing:

```powershell
Remove-Item Env:QT_QPA_PLATFORM
```

---

## 5. Build the Windows Folder Bundle

The PowerShell helper script orchestrates dependency installation, validation, and the `pyside6-deploy` run. Invoke it with the desired semantic version:

```powershell
pwsh packaging/windows/build.ps1 -Configuration Release -Version 1.2.3
```

This script performs the following:

1. Ensures `.venv/` exists (creates it with `py -3.12 -m venv` if necessary) and installs required Python packages (`pip`, `setuptools`, `wheel`, `PySide6==6.7.2`, `pyside6-deploy`, `normal2disp`, GUI extras, `ruff`, `pytest`).
2. Runs the headless smoke tests (`ruff`, `pytest`, and a short offscreen GUI launch) with `QT_QPA_PLATFORM=offscreen`.
3. Executes `pyside6-deploy` using `pysidedeploy.json`, targeting the entry point `normal2disp/gui/entrypoint.py`.
4. Writes the resulting folder bundle to `dist/windows/<version>/bundle/` and preserves the temporary work files under `dist/windows/<version>/deploy_work/`.

After completion you should find a structure similar to:

```
dist/
  windows/
    1.2.3/
      bundle/
        normal2disp-gui.exe
        PySide6/...
        qml/...
      deploy_work/
        ... (build intermediates)
```

> **Reminder:** Do not commit `dist/` or any generated EXE/DLL files.

If you need to rerun the build, delete the corresponding `dist/windows/<version>/` folder first.

---

## 6. Install Inno Setup

Install Inno Setup 6 via your preferred package manager:

```powershell
winget install --id JRSoftware.InnoSetup -e
# or
choco install innosetup -y
```

Ensure `ISCC.exe` is on your `PATH` or note the installation directory (usually `C:\Program Files (x86)\Inno Setup 6\`).

---

## 7. Build the Installer

Run the Inno Setup compiler against the provided script. Pass the same version number you used for the folder bundle:

```powershell
$version = "1.2.3"
$iss = "packaging/windows/installer.iss"
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" $iss /DMyAppVersion=$version
```

Optional signing: If the environment variables `N2D_CODESIGN_PFX_PATH` and `N2D_CODESIGN_PFX_PASSWORD` are populated, append `/DSign=1` to enable the scripted `SignTool` command inside `installer.iss`. You can also export `N2D_CODESIGN_TIMESTAMP_URL` to append ` /tr <url> /td SHA256` to the signing command.

The installer will read files from `dist/windows/<version>/bundle/` and emit `normal2disp-GUI-<version>.exe` (default name) to `dist/windows/<version>/installer/`.

---

## 8. Outputs and Cleanup

* **Folder bundle:** `dist/windows/<version>/bundle/`
* **Installer:** `dist/windows/<version>/installer/`
* **Temporary deploy work dir:** `dist/windows/<version>/deploy_work/` (safe to delete after confirming the installer)

After you validate the installer locally:

1. Optionally sign the installer manually if you skipped automatic signing.
2. Upload the artifacts to your release pipeline or storage.
3. Delete `dist/` if you no longer need the outputs locally.

Remember, none of these files should be committed.

---

## 9. CI Notes

The GitHub Actions workflow `.github/workflows/windows-build.yml` mirrors the steps above on `windows-latest` runners. On tag builds (`v*`) it:

1. Restores cached pip downloads to speed up installation.
2. Installs dependencies, runs `ruff` and `pytest` with `QT_QPA_PLATFORM=offscreen`.
3. Invokes `packaging/windows/build.ps1` to produce `dist/windows/<version>/bundle/`.
4. Installs Inno Setup and runs `ISCC.exe` with `/DMyAppVersion=<tag-without-v>` (and `/DSign=1` if signing environment variables are defined).
5. Uploads both the folder bundle and installer as build artifacts.

To enable signing in CI, add encrypted repository secrets that populate `N2D_CODESIGN_PFX_PATH`, `N2D_CODESIGN_PFX_PASSWORD`, and optionally `N2D_CODESIGN_TIMESTAMP_URL`. The workflow automatically toggles signing when these variables are present.

---

## 10. Providing Icon Assets

The installer references a Windows `.ico` file for shortcuts. Place your icon at `assets/icons/normal2disp.ico` (or adjust the path in `installer.iss`) **before** running the bundle/installer steps. The repository includes `assets/icons/PLACEHOLDER.txt` as a reminder; no binary assets ship with the repo.

---

## 11. Troubleshooting

* **PySide6 deployment fails:** Delete the `dist/windows/<version>/deploy_work/` folder and rerun the build script. Verify that `pyside6-deploy --version` matches the pinned PySide6 version.
* **Missing DLLs at runtime:** Ensure that `pysidedeploy.json` lists all required resources (QML, translations, etc.) and that the entry point is correct.
* **Installer cannot find bundle files:** Confirm that the version number passed to `ISCC.exe` matches the bundle directory under `dist/windows/`.
* **Signing errors:** Double-check the environment variables and confirm that `signtool.exe` exists (part of Windows 10 SDK or Visual Studio). Use `signtool sign /?` for syntax details.

---

You are now ready to produce Windows installers for the normal2disp GUI locally or via CI.
