<#
.SYNOPSIS
    Runs indexer_multimodal.py as a resilient background process that survives
    Windows lock and prevents sleep/standby until ingestion completes.

.DESCRIPTION
    - Prevents Windows sleep using powercfg (restores original settings on exit)
    - Redirects all output (stdout + stderr) to a timestamped log file
    - Auto-retries up to 3 times on failure with a 30-second cooldown
    - Sets PYTHONIOENCODING=utf-8 to avoid cp1252 logging errors on Windows

.USAGE
    To launch in the background (survives terminal close + Windows lock):
        Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File "run_multimodal_indexer.ps1"' -WindowStyle Hidden -WorkingDirectory (Get-Location)
#>

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# --- Configuration ---
$PythonExe    = "C:\Python314\python.exe"
$Script       = "indexer_multimodal.py"
$MaxRetries   = 3
$RetryCooldown = 30  # seconds

# --- Log file ---
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile   = Join-Path $ScriptDir "logs\ingest_images_$Timestamp.log"
New-Item -ItemType Directory -Path (Join-Path $ScriptDir "logs") -Force | Out-Null

function Write-Log($msg) {
    $entry = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $msg"
    Add-Content -Path $LogFile -Value $entry -Encoding UTF8
}

# --- Prevent Windows sleep ---
Write-Log "Saving current power settings and disabling sleep/standby..."
$origAcStandby = (powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String "Current AC Power Setting Index" | ForEach-Object { ($_ -split ":\s*")[-1].Trim() })
$origDcStandby = (powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String "Current DC Power Setting Index" | ForEach-Object { ($_ -split ":\s*")[-1].Trim() })
powercfg /change standby-timeout-ac 0   # AC: never sleep
powercfg /change standby-timeout-dc 0   # DC: never sleep
Write-Log "Sleep disabled (AC was: $origAcStandby, DC was: $origDcStandby)"

function Restore-SleepSettings {
    # Convert hex seconds back to minutes for powercfg /change
    try {
        if ($origAcStandby) {
            $acMinutes = [Math]::Round([Convert]::ToInt32($origAcStandby, 16) / 60)
            powercfg /change standby-timeout-ac $acMinutes
        }
        if ($origDcStandby) {
            $dcMinutes = [Math]::Round([Convert]::ToInt32($origDcStandby, 16) / 60)
            powercfg /change standby-timeout-dc $dcMinutes
        }
        Write-Log "Restored original sleep settings (AC: $acMinutes min, DC: $dcMinutes min)"
    } catch {
        Write-Log "WARNING: Could not restore sleep settings: $_"
    }
}

# --- Main retry loop ---
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED  = "1"

$attempt = 0
$success = $false

while ($attempt -lt $MaxRetries -and -not $success) {
    $attempt++
    Write-Log "=== Attempt $attempt of $MaxRetries ==="

    try {
        $proc = Start-Process -FilePath $PythonExe `
            -ArgumentList "-u", $Script `
            -WorkingDirectory $ScriptDir `
            -RedirectStandardOutput (Join-Path $ScriptDir "logs\ingest_images_stdout_$Timestamp.log") `
            -RedirectStandardError  (Join-Path $ScriptDir "logs\ingest_images_stderr_$Timestamp.log") `
            -NoNewWindow -PassThru -Wait

        if ($proc.ExitCode -eq 0) {
            Write-Log "Ingestion completed successfully (exit code 0)"
            $success = $true
        } else {
            Write-Log "Ingestion exited with code $($proc.ExitCode)"
            if ($attempt -lt $MaxRetries) {
                Write-Log "Retrying in $RetryCooldown seconds..."
                Start-Sleep -Seconds $RetryCooldown
            }
        }
    } catch {
        Write-Log "ERROR: $_"
        if ($attempt -lt $MaxRetries) {
            Write-Log "Retrying in $RetryCooldown seconds..."
            Start-Sleep -Seconds $RetryCooldown
        }
    }
}

if (-not $success) {
    Write-Log "All $MaxRetries attempts failed."
}

# --- Restore sleep settings ---
Restore-SleepSettings
Write-Log "=== Runner finished ==="
