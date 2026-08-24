param(
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$healthUrl = "http://127.0.0.1:8766/api/health"
$logDir = Join-Path $projectRoot "data\logs"

function Test-LocalMeetScribeHealth {
    try {
        $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        return $response.status -eq "ok"
    }
    catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "LocalMeetScribe Python environment is missing: $python"
}

if (-not (Test-LocalMeetScribeHealth)) {
    $listener = Get-NetTCPConnection -LocalPort 8766 -State Listen -ErrorAction SilentlyContinue
    if ($listener) {
        throw "Port 8766 is already used by another process."
    }

    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    Start-Process `
        -FilePath $python `
        -ArgumentList @(
            "-m",
            "local_meetscribe.cli",
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "8766"
        ) `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir "server.stdout.log") `
        -RedirectStandardError (Join-Path $logDir "server.stderr.log")

    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline -and -not (Test-LocalMeetScribeHealth)) {
        Start-Sleep -Milliseconds 250
    }
    if (-not (Test-LocalMeetScribeHealth)) {
        throw "LocalMeetScribe did not become healthy. Check data\logs\server.stderr.log."
    }
}

if (-not $Quiet) {
    Write-Output "LocalMeetScribe is running:"
    Write-Output "  This PC: http://127.0.0.1:8766/"
    Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object {
            $_.AddressState -eq "Preferred" -and
            $_.IPAddress -notlike "127.*" -and
            $_.InterfaceAlias -notlike "vEthernet*"
        } |
        Sort-Object InterfaceAlias, IPAddress |
        ForEach-Object {
            Write-Output ("  {0}: http://{1}:8766/" -f $_.InterfaceAlias, $_.IPAddress)
        }
}
