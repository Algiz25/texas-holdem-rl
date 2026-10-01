# Skrypt nocny SAC dla Windows (Faza 2 - Self-Play). 

$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location -Path $ProjectDir

$Python = ".venv\Scripts\python.exe"
$RunName = "phase2_selfplay_sac_seed_22002"
$Seed = 22002

# Faza 2 wymaga wiecej czasu na nauke GTO. Ustawiamy 10 milionow decyzji (1000 epok).
$TotalDecisions = 10000000

# Ewaluacja co 50 000 decyzji (czyli co 5 epok). 
$EvaluationInterval = 50000

# Wskazujemy najlepszy model z Fazy 1
$BaseModel = "checkpoints\sac\REAL_BEST\phase_1\step_000750000.pth"

$NightId = Get-Date -Format "yyyyMMdd_HHmmss"
$TerminalLogDir = "logs\sac\overnight_$NightId"
New-Item -ItemType Directory -Force -Path $TerminalLogDir | Out-Null

# Szybkie testy przed startem
$env:PYTHONPATH = "src"
Write-Host "[TESTY] Uruchamianie testow jednostkowych..."
& $Python -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "[BLAD] Testy nie przeszly. Trening zatrzymany." -ForegroundColor Red
    exit 1
}

$RunDir = "checkpoints\sac\$RunName"
$FinalState = "$RunDir\training_state_final.pth"
$LatestState = "$RunDir\training_state_latest.pth"
$TerminalLog = "$TerminalLogDir\$RunName.log"

if (Test-Path $FinalState) {
    Write-Host "[GOTOWE] $RunName osiagnal juz limit $TotalDecisions decyzji." -ForegroundColor Green
    exit 0
}

$Decisions = $TotalDecisions
$ExtraArgs = @()

if (Test-Path $LatestState) {
    $CompletedStr = & $Python -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["completed_env_steps"]))' $LatestState
    $Completed = [int]$CompletedStr
    $Decisions = $TotalDecisions - $Completed
    
    if ($Decisions -le 0) {
        Write-Host "[BLAD] Stan ma $Completed decyzji, ale brak pliku koncowego." -ForegroundColor Red
        exit 1
    }
    
    $ExtraArgs += "--resume", $LatestState
    $Decisions = [math]::Floor($Decisions / 10000) * 10000
    
    if ($Decisions -le 0) {
        Write-Host "[GOTOWE] $RunName jest juz przy technicznym limicie." -ForegroundColor Green
        exit 0
    }
    Write-Host "[WZNOWIENIE] ${RunName}: $Completed/$TotalDecisions decyzji." -ForegroundColor Yellow
}

Write-Host "[START] Trening SAC Faza 2 (Self-Play): $RunName." -ForegroundColor Cyan
Write-Host "[STOP] Aby przerwac, nacisnij Ctrl+C jeden raz." -ForegroundColor Yellow

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "src"

# Uruchomienie treningu SAC
& $Python src\training\train_sac.py `
    --run-name $RunName `
    --seed $Seed `
    --decisions $Decisions `
    --evaluation-interval $EvaluationInterval `
    --base-model $BaseModel `
    @ExtraArgs | Tee-Object -FilePath $TerminalLog -Append

Write-Host "`nTrening zakonczony albo bezpiecznie zatrzymany." -ForegroundColor Green
Write-Host "Model: $RunDir"