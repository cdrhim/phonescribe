param(
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$healthUrl = "http://127.0.0.1:8766/api/health"
$logDir = Join-Path $projectRoot "data\logs"
$phoneScribeDownloads = Join-Path ([Environment]::GetFolderPath("UserProfile")) "Downloads\PhoneScribe"

# Remote requests are protected by a passcode-issued bearer session. These values
# contain no secrets and are inherited only by the background server process.
$env:LOCAL_MEETSCRIBE_REMOTE_ACCESS = "true"
$env:LOCAL_MEETSCRIBE_REMOTE_SESSION_TTL_SEC = "7200"
$env:LOCAL_MEETSCRIBE_CORS_ORIGINS = "https://phonescribe.vercel.app,http://127.0.0.1:5173,http://localhost:5173"
$env:LOCAL_MEETSCRIBE_AUTO_EXPORT_DIR = $phoneScribeDownloads

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
    Write-Output "  Public frontend: https://phonescribe.vercel.app/"
}
