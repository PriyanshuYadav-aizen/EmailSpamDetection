$port = 5001
$connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue

if (-not $connections) {
    Write-Host "No ML service found on port $port."
    exit 0
}

$processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique

foreach ($processId in $processIds) {
    try {
        Stop-Process -Id $processId -Force
        Write-Host "Stopped ML service process $processId on port $port."
    }
    catch {
        Write-Host "Failed to stop process $($processId): $($_.Exception.Message)"
    }
}
