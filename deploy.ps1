# Deploy script for WikiSage to Azure App Service
# Usage: .\deploy.ps1
# Prerequisites: az CLI installed and logged in (az login)

$ErrorActionPreference = "Stop"

# Config: replace with your own resource group and globally unique app name
$ResourceGroup = "<your-resource-group>"
$AppName = "<your-app-name>"
$ZipPath = "app.zip"
$Files = @(
    "app.py",
    "retriever.py",
    "search_store.py",
    "common.py",
    "indexer.py",
    "startup.sh",
    "requirements.txt",
    "runtime.txt"
)

# Ensure az is in PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "=== WikiSage Deploy ===" -ForegroundColor Cyan

# Step 1: Fix startup.sh line endings (CRLF -> LF)
Write-Host "[1/4] Fixing startup.sh line endings..." -ForegroundColor Yellow
$startupPath = Join-Path $PSScriptRoot "startup.sh"
$content = [System.IO.File]::ReadAllText($startupPath)
$content = $content.Replace("`r`n", "`n")
[System.IO.File]::WriteAllText($startupPath, $content, [System.Text.UTF8Encoding]::new($false))
Write-Host "  Done (LF line endings)" -ForegroundColor Green

# Step 2: Build zip
Write-Host "[2/4] Building deployment zip..." -ForegroundColor Yellow
Remove-Item $ZipPath -Force -ErrorAction SilentlyContinue
Compress-Archive -Path $Files -DestinationPath $ZipPath -Force
$size = [math]::Round((Get-Item $ZipPath).Length / 1KB, 1)
Write-Host "  Zip created: $size KB" -ForegroundColor Green

# Step 3: Deploy
Write-Host "[3/4] Deploying to Azure ($AppName)..." -ForegroundColor Yellow
$fullZipPath = (Resolve-Path $ZipPath).Path
az webapp deploy -g $ResourceGroup -n $AppName --src-path $fullZipPath --type zip --async true 2>&1 | ForEach-Object { Write-Host "  $_" }
Write-Host "  Deploy initiated" -ForegroundColor Green

# Step 4: Monitor
Write-Host "[4/4] Waiting for app to start (up to 5 min)..." -ForegroundColor Yellow
$maxWait = 300
$elapsed = 0
$interval = 15
while ($elapsed -lt $maxWait) {
    Start-Sleep -Seconds $interval
    $elapsed += $interval
    $state = az webapp show -g $ResourceGroup -n $AppName --query state -o tsv 2>&1
    Write-Host "  [$elapsed`s] App state: $state"
    if ($state -eq "Running") {
        Write-Host ""
        Write-Host "=== DEPLOY COMPLETE ===" -ForegroundColor Green
        Write-Host "URL: https://$AppName.azurewebsites.net" -ForegroundColor Cyan
        exit 0
    }
}
Write-Host "  Timeout waiting for app to start. Check Azure Portal logs." -ForegroundColor Red
exit 1
