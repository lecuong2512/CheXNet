$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$runtime = Join-Path $projectRoot '.runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$node = (Get-Command node.exe).Source
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create .venv and install the Python requirements first.' }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'Backend\dist\server.js'))) { throw 'Run yarn build in Backend first.' }

$services = @(
    @{ Name = 'backend'; Exe = $python; Args = @('-u', '-m', 'uvicorn', 'Backend.ai_service.main:app', '--host', '127.0.0.1', '--port', '8000'); Directory = $projectRoot; Url = 'http://127.0.0.1:8000/health' },
    @{ Name = 'gateway'; Exe = $node; Args = @('dist/server.js'); Directory = (Join-Path $projectRoot 'Backend'); Url = 'http://127.0.0.1:3006/health' },
    @{ Name = 'frontend'; Exe = $node; Args = @('node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5173', '--strictPort'); Directory = (Join-Path $projectRoot 'Frontend'); Url = 'http://127.0.0.1:5173/' }
)

foreach ($service in $services) {
    $running = $false
    try { $running = (Invoke-WebRequest -Uri $service.Url -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200 } catch {}
    if ($running) { Write-Host "$($service.Name): already running"; continue }
    $process = Start-Process -FilePath $service.Exe -ArgumentList $service.Args -WorkingDirectory $service.Directory -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime "$($service.Name).log") -RedirectStandardError (Join-Path $runtime "$($service.Name).error.log") -PassThru
    $process.Id | Set-Content -LiteralPath (Join-Path $runtime "$($service.Name).pid")
    Write-Host "$($service.Name): started (PID $($process.Id))"
}

Write-Host 'Frontend: http://127.0.0.1:5173/'
Write-Host 'API:      http://127.0.0.1:3006/docs'
Write-Host 'AI:       http://127.0.0.1:8000/health'
Write-Host 'Model loading can take a minute. Logs are in .runtime/.'
