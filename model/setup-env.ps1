# 🚀 Setup Environment

# 1. Create virtual environment
Write-Host "Creating virtual environment in .venv..." -ForegroundColor Cyan
py -3.13 -m venv .venv

# 2. Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip

# 3. Install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
& ".\.venv\Scripts\pip.exe" install -r requirements.txt

Write-Host "`n✅ Environment setup complete!" -ForegroundColor Green
Write-Host "To activate, run: .\.venv\Scripts\Activate.ps1"
