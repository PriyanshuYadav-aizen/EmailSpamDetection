$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$modelPath = Join-Path $root "model"
$serverPath = Join-Path $root "server"
$clientPath = Join-Path $root "client"

function Test-PortInUse {
	param([int]$Port)
	$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
	return $null -ne $listener
}

if (Test-PortInUse 5001) {
	Write-Host "ML service already running on port 5001. Skipping launch."
} else {
	Write-Host "Starting ML service on port 5001..."
	Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$modelPath'; py -3.11 -m uvicorn api:app --host 0.0.0.0 --port 5001"
}

if (Test-PortInUse 5000) {
	Write-Host "Backend already running on port 5000. Skipping launch."
} else {
	Write-Host "Starting backend on port 5000..."
	Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$serverPath'; npm run dev"
}

if (Test-PortInUse 5173) {
	Write-Host "Frontend already running on port 5173. Skipping launch."
} else {
	Write-Host "Starting frontend on port 5173..."
	Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$clientPath'; npm run dev"
}

Write-Host "Services launched."
