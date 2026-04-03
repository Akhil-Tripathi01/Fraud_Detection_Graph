$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$port = 8000
$hostName = "127.0.0.1"
$url = "http://$hostName`:$port/"
$healthUrl = "http://$hostName`:$port/api/health"

# Stop stale listeners on the same port.
$listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($listeners) {
    $ownerIds = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($ownerId in $ownerIds) {
        try { Stop-Process -Id $ownerId -Force } catch {}
    }
    Start-Sleep -Milliseconds 700
}

Write-Host "Starting Fraud_Detection_Graph server on $url ..."
$proc = Start-Process -FilePath python `
    -ArgumentList '-m','uvicorn','backend.app.main:app','--host',$hostName,'--port',"$port",'--app-dir',$projectRoot `
    -WorkingDirectory $projectRoot `
    -PassThru

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 3
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
}

if (-not $ready) {
    throw "Server failed to become ready on $url (PID=$($proc.Id))."
}

Write-Host "Server is ready (PID=$($proc.Id)). Opening browser..."
Start-Process $url
Write-Host "Also available:"
Write-Host "  Dashboard: $url"
Write-Host "  Notebook : http://$hostName`:$port/notebook"
Write-Host "  Docs     : http://$hostName`:$port/docs"
