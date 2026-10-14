param(
    [switch]$SetupOnly,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$composeProcess = $null
$uiProcess = $null
$startedPostgres = $false
$startedObjectStorage = $false

function Invoke-Checked {
    param([string]$Description, [scriptblock]$Action)
    Write-Host $Description
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "$Description failed (exit code $LASTEXITCODE)." }
}

function Test-PostgresReady {
    & docker compose exec -T postgres pg_isready -U agent -d agent *> $null
    return $LASTEXITCODE -eq 0
}

function Test-ObjectStorageReady {
    try {
        $response = Invoke-WebRequest -Uri 'http://127.0.0.1:9000/minio/health/live' -TimeoutSec 2 -UseBasicParsing
        return $response.StatusCode -eq 200
    } catch { return $false }
}

function Get-AvailableChatPort {
    foreach ($port in 8501..8510) {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $port)
        try {
            $listener.Start()
            return $port
        } catch {
            # Keep the existing listener untouched and try the next local port.
        } finally {
            $listener.Stop()
        }
    }
    throw 'No free appv2 port is available in the range 8501-8510.'
}

Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.env')) {
        Copy-Item -LiteralPath '.env.example' -Destination '.env'
        throw 'Created .env from .env.example. Configure it, then rerun start.ps1.'
    }
    if (-not (Test-Path -LiteralPath 'runtime\local-auth.json')) {
        throw 'Missing runtime/local-auth.json. Configure the local development identity first.'
    }
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if (-not $launcher) { throw 'Python 3.11+ is required.' }
        Invoke-Checked 'Creating virtual environment' { & py -3 -m venv .venv }
    }
    if (-not $SkipInstall) {
        & $pythonExe -c 'import streamlit, alembic, fastapi, sqlalchemy, psycopg, pgvector, jsonschema, langchain, langchain_google_genai, langchain_ollama, langfuse, google.genai' 2>$null
        if ($LASTEXITCODE -ne 0) {
            $uv = Get-Command uv -ErrorAction SilentlyContinue
            if ($uv) {
                Invoke-Checked 'Installing Python dependencies' { & uv pip install --python $pythonExe -e '.[ui]' }
            } else {
                Invoke-Checked 'Installing Python dependencies' { & $pythonExe -m pip install -e '.[ui]' }
            }
        }
    }
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker is required. Install or start Docker Desktop, then rerun start.ps1.'
    }
    Invoke-Checked 'Checking Docker' { & docker info --format '{{.ServerVersion}}' | Out-Null }
    $before = @(& docker compose ps --status running --services)
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect local Docker services.' }

    if ($SetupOnly) {
        Invoke-Checked 'Starting PostgreSQL and document storage' { & docker compose up -d postgres object-storage }
        for ($attempt = 0; $attempt -lt 30 -and -not (Test-PostgresReady); $attempt++) {
            Start-Sleep -Seconds 2
        }
        if (-not (Test-PostgresReady)) { throw 'PostgreSQL did not become ready.' }
        $storageReady = $false
        for ($attempt = 0; $attempt -lt 30 -and -not $storageReady; $attempt++) {
            $storageReady = Test-ObjectStorageReady
            if (-not $storageReady) { Start-Sleep -Seconds 2 }
        }
        if (-not $storageReady) { throw 'Document storage did not become ready.' }
        Invoke-Checked 'Applying application migrations' { & $pythonExe -m alembic upgrade head }
        Invoke-Checked 'Preparing document storage' { & $pythonExe -m app.cli setup-storage }
        Write-Host 'appv2 database setup is complete.'
        return
    }

    $startedPostgres = 'postgres' -notin $before
    $startedObjectStorage = 'object-storage' -notin $before
    Write-Host 'Starting PostgreSQL, document storage, and the appv2 chat in this terminal.'
    $composeProcess = Start-Process -FilePath 'docker' `
        -ArgumentList @('compose', 'up', 'postgres', 'object-storage') `
        -WorkingDirectory $projectRoot -NoNewWindow -PassThru
    $databaseReady = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        $composeProcess.Refresh()
        if ($composeProcess.HasExited) {
            throw "PostgreSQL exited unexpectedly (exit code $($composeProcess.ExitCode))."
        }
        if (Test-PostgresReady) { $databaseReady = $true; break }
        Start-Sleep -Seconds 2
    }
    if (-not $databaseReady) { throw 'PostgreSQL did not become ready.' }
    $storageReady = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        $composeProcess.Refresh()
        if ($composeProcess.HasExited) {
            throw "Document storage exited unexpectedly (exit code $($composeProcess.ExitCode))."
        }
        if (Test-ObjectStorageReady) { $storageReady = $true; break }
        Start-Sleep -Seconds 2
    }
    if (-not $storageReady) { throw 'Document storage did not become ready.' }
    Invoke-Checked 'Applying application migrations' { & $pythonExe -m alembic upgrade head }
    Invoke-Checked 'Preparing document storage' { & $pythonExe -m app.cli setup-storage }

    $chatPort = Get-AvailableChatPort
    $uiProcess = Start-Process -FilePath $pythonExe `
        -ArgumentList @('-u', '-m', 'streamlit', 'run', 'appv2\ui.py', `
            '--server.address', '127.0.0.1', '--server.port', [string]$chatPort, '--server.headless', 'true') `
        -WorkingDirectory $projectRoot -NoNewWindow -PassThru
    Write-Host "Open http://127.0.0.1:$chatPort for chat. Keep this terminal open; closing it stops the UI and services started here."
    while ($true) {
        $uiProcess.Refresh()
        $composeProcess.Refresh()
        if ($uiProcess.HasExited) { throw "appv2 exited (code $($uiProcess.ExitCode))." }
        if ($composeProcess.HasExited) { throw "PostgreSQL exited (code $($composeProcess.ExitCode))." }
        Start-Sleep -Seconds 1
    }
} finally {
    foreach ($process in @($uiProcess, $composeProcess)) {
        if ($null -ne $process) {
            $process.Refresh()
            if (-not $process.HasExited) {
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        }
    }
    if ($startedPostgres -or $startedObjectStorage) {
        Write-Host 'Stopping local infrastructure started by this terminal.'
        $toStop = @()
        if ($startedPostgres) { $toStop += 'postgres' }
        if ($startedObjectStorage) { $toStop += 'object-storage' }
        & docker compose stop @toStop
    }
    Pop-Location
}
