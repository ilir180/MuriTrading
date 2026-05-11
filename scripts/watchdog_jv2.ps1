# JV Boting v2 Watchdog
# Stellt sicher, dass MuriTrading-JV2 läuft UND das Log frisch ist.
# Wird vom Task MuriTrading-JV2-Watchdog alle 5 Minuten ausgeführt.

$ErrorActionPreference = 'Continue'

$TaskName         = 'MuriTrading-JV2'
$DataDir          = 'C:\Users\ilir_\MuriTrading\data\bot\jv2'
$LogFile          = Join-Path $DataDir 'jv2_output.log'
$HeartbeatFile    = Join-Path $DataDir 'heartbeat.txt'    # vom runner.py pro Iteration getoucht (~60s)
$WatchdogLog      = Join-Path $DataDir 'watchdog.log'
$LogStaleMin      = 15   # log darf bei wenig Aktivität (zwischen 4H-Kerzen) länger still sein
$HeartbeatStaleMin = 3   # heartbeat.txt muss jede ~60s frisch sein → 3 Min = großzügiger Slack
$BootGraceMin     = 3    # Bot darf nach Start 3 Min ohne Heartbeat brauchen (Cold Start, Imports, Binance-Init)

function Write-WdLog([string]$msg) {
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $WatchdogLog -Value "$ts  $msg" -Encoding utf8
}

try {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
} catch {
    Write-WdLog "ERROR: Task '$TaskName' nicht gefunden. Watchdog kann nichts tun."
    exit 1
}

$state = $task.State.ToString()
$running = $state -eq 'Running'

$logFresh = $false
$logAgeMin = -1
if (Test-Path $LogFile) {
    $logAgeMin = [math]::Round(((Get-Date) - (Get-Item $LogFile).LastWriteTime).TotalMinutes, 1)
    $logFresh = $logAgeMin -lt $LogStaleMin
}

# Semantischer Heartbeat: heartbeat.txt wird vom runner.py jede ~60s getoucht.
# Wenn Bot fehlerfrei loggt aber heartbeat nicht updated → Loop hängt.
$hbFresh = $false
$hbAgeMin = -1
if (Test-Path $HeartbeatFile) {
    $hbAgeMin = [math]::Round(((Get-Date) - (Get-Item $HeartbeatFile).LastWriteTime).TotalMinutes, 1)
    $hbFresh = $hbAgeMin -lt $HeartbeatStaleMin
}

# Python-Prozess: existiert ein runner.py-Prozess und wie alt ist er?
$pythonProc = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    try {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine
        $cmd -match 'jv2[\\/]runner\.py'
    } catch { $false }
} | Select-Object -First 1
$pythonRunning = $null -ne $pythonProc
$pyAgeMin = if ($pythonRunning) { [math]::Round(((Get-Date) - $pythonProc.StartTime).TotalMinutes, 1) } else { -1 }

# Cold-Start Grace: Bot darf jung sein und noch keinen Heartbeat haben (Imports + Binance-Init).
$inGracePeriod = $pythonRunning -and $pyAgeMin -ge 0 -and $pyAgeMin -lt $BootGraceMin

$healthy = $running -and $pythonRunning -and ($inGracePeriod -or ($logFresh -and $hbFresh))
$status  = "state=$state pyAge=${pyAgeMin}min logAge=${logAgeMin}min hbAge=${hbAgeMin}min grace=$inGracePeriod"

if ($healthy) {
    # Heartbeat: nur einmal pro Stunde loggen (im ersten 5min-Slot)
    if ((Get-Date).Minute -lt 5) {
        Write-WdLog "OK  $status"
    }
    exit 0
}

Write-WdLog "RESTART trigger: $status"

# Falls der Task laut Scheduler läuft, aber unhealthy → erst sauber stoppen
if ($running) {
    try {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction Stop
        Start-Sleep -Seconds 3
    } catch {
        Write-WdLog "WARN  Stop-ScheduledTask fehlgeschlagen: $($_.Exception.Message)"
    }
}

# Hängende python-Prozesse zum runner.py killen
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine
        if ($cmd -match 'jv2[\\/]runner\.py') {
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
            Write-WdLog "       killed stale python PID=$($_.Id)"
        }
    } catch {}
}

Start-Sleep -Seconds 2

try {
    Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    Write-WdLog "       Start-ScheduledTask ok"
} catch {
    Write-WdLog "ERROR Start-ScheduledTask failed: $($_.Exception.Message)"
    exit 1
}
