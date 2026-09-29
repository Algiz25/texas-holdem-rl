# Skrypt nocny IQN dla Windows (Faza 2 - Self-Play). 
# UWAGA: Aby komputer nie zasnal, zmien ustawienia zasilania Windows 
# (Ustawienia -> System -> Zasilanie i uspienie -> Uspienie: Nigdy).

$ErrorActionPreference = "Stop"

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location -Path $ProjectDir

$Python = ".venv\Scripts\python.exe"
$RunName = "phase2_selfplay_iqn_seed_22002"
$Seed = 22002

# Faza 2 wymaga więcej czasu, np. 20 milionów decyzji
$TotalDecisions = 20000000

# WSKAŻ TUTAJ ŚCIEŻKĘ DO NAJLEPSZEGO MODELU Z FAZY 1
$BaseModel = "checkpoints\iqn\REAL_BEST\phase_1\step_000900000.pth"

# Ewaluacja co 50 000 decyzji
$EvaluationInterval = 50000

# W fazie 2 chcemy, żeby agent eksplorował, ale mniej agresywnie niż w fazie 1.
# Możemy ustawić mniejsze tau, żeby szybciej zszedł do minimalnego epsilona.
$EpsilonTau = 1000000

$NightId = Get-Date -Format "yyyyMMdd_HHmmss"
$TerminalLogDir = "logs\iqn\overnight_$NightId"
New-Item -ItemType Directory -Force -Path $TerminalLogDir | Out-Null

# Szybkie testy przed startem
$env:PYTHONPATH = "src"
Write-Host "[TESTY] Uruchamianie testow jednostkowych..."
& $Python -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "[BLAD] Testy nie przeszly. Trening zatrzymany." -ForegroundColor Red
    exit 1
}

$RunDir = "checkpoints\iqn\$RunName"
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

Write-Host "[START] Jeden dlugi trening IQN (Faza 2): $RunName." -ForegroundColor Cyan
Write-Host "[STOP] Aby przerwac, nacisnij Ctrl+C jeden raz. Stan zostanie zapisany automatycznie." -ForegroundColor Yellow

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "src"

# Uruchomienie treningu IQN
& $Python src\training\train_iqn.py `
    --run-name $RunName `
    --seed $Seed `
    --decisions $Decisions `
    --evaluation-interval $EvaluationInterval `
    --epsilon-tau $EpsilonTau `
    --base-model $BaseModel `
    @ExtraArgs | Tee-Object -FilePath $TerminalLog -Append

Write-Host "`nTrening zakonczony albo bezpiecznie zatrzymany." -ForegroundColor Green
Write-Host "Model: $RunDir"
Write-Host "TensorBoard: .venv\Scripts\tensorboard.exe --logdir logs\iqn"