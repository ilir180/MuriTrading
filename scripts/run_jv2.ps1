# JV Boting v2 Runner - Windows
# Replaces scripts/run_jv2.sh (macOS). Used by Windows Task Scheduler and manual launches.

$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'

$ProjectRoot = 'C:\Users\ilir_\MuriTrading'
Set-Location -Path $ProjectRoot

$LogDir = Join-Path $ProjectRoot 'data\bot\jv2'
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# pyenv-win shim resolves Python via .python-version (3.11.9).
# cmd /c handles stdout+stderr redirection cleanly (avoids PS 5.1 NativeCommandError wrapping).
cmd /c "python src\jv2\runner.py >> data\bot\jv2\jv2_output.log 2>&1"
