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

# Absoluter Pfad statt PATH-Lookup: C:\Python314 schattet seit 10.06. den pyenv-Shim ab.
# cmd /c handles stdout+stderr redirection cleanly (avoids PS 5.1 NativeCommandError wrapping).
$Python = 'C:\Users\ilir_\.pyenv\pyenv-win\versions\3.11.9\python.exe'
cmd /c "`"$Python`" src\jv2\runner.py >> data\bot\jv2\jv2_output.log 2>&1"
