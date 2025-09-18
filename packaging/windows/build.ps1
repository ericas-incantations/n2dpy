[CmdletBinding()]
param(
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [Parameter(Mandatory = $true)]
    [string]$Version
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Version)) {
    throw "-Version must be supplied (e.g., 1.2.3)."
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
Set-Location -LiteralPath $repoRoot

Write-Host "Repository root:" $repoRoot
Write-Host "Configuration:" $Configuration
Write-Host "Version:" $Version

$venvPath = Join-Path $repoRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at" $venvPath
    & py -3.12 -m venv $venvPath
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$deployExe = Join-Path $venvPath "Scripts\pyside6-deploy.exe"

Write-Host "Upgrading pip/setuptools/wheel"
& $pythonExe -m pip install --upgrade pip setuptools wheel

Write-Host "Installing project and runtime dependencies"
& $pythonExe -m pip install -e .
& $pythonExe -m pip install -r normal2disp/gui/requirements.txt
& $pythonExe -m pip install PySide6==6.7.2 pyside6-deploy==6.7.2

Write-Host "Installing development tools"
& $pythonExe -m pip install ruff pytest

Write-Host "Running headless smoke tests"
$env:QT_QPA_PLATFORM = "offscreen"
& $pythonExe -m ruff check .
& $pythonExe -m pytest

$smokeScript = @'
import sys
from PySide6.QtCore import QTimer
from n2d_gui.app import main as gui_main


def quit_app() -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.quit()


if __name__ == "__main__":
    QTimer.singleShot(200, quit_app)
    sys.exit(gui_main())
'@

$smokePath = Join-Path $scriptDir "smoke_test.py"
Set-Content -Path $smokePath -Value $smokeScript -Encoding UTF8
try {
    & $pythonExe $smokePath
}
finally {
    Remove-Item $smokePath -ErrorAction SilentlyContinue
}

Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue

$distRoot = Join-Path $repoRoot (Join-Path "dist\windows" $Version)
$bundleDir = Join-Path $distRoot "bundle"
$workDir = Join-Path $distRoot "deploy_work"

Write-Host "Preparing output directories under" $distRoot
Remove-Item $bundleDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $workDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null
New-Item -ItemType Directory -Path $workDir -Force | Out-Null

$pysideConfig = Join-Path $repoRoot "pysidedeploy.json"
if (-not (Test-Path $pysideConfig)) {
    throw "pysidedeploy.json not found at $pysideConfig"
}

$pysideArgs = @(
    "--config-file", $pysideConfig,
    "--workpath", $workDir,
    "--distpath", $bundleDir,
    "--name", "normal2disp-gui"
)

if ($Configuration -eq "Debug") {
    $pysideArgs += "--debug"
}

Write-Host "Running pyside6-deploy"
& $deployExe @pysideArgs

Write-Host "Bundle created at" $bundleDir
Write-Host "Temporary work files at" $workDir
