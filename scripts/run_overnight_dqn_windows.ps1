# Skrypt nocny dla Windows. 
# UWAGA: Aby komputer nie zasnął, zmień ustawienia zasilania Windows 
# (Ustawienia -> System -> Zasilanie i uśpienie -> Uśpienie: Nigdy).

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location -Path $ProjectDir

$Python = ".venv\Scripts\python.exe"
$RunName = "overnight_long_seed_11001"
$Seed = 11001
$TotalDecisions = 100000000
$EpsilonDecayDecisions = 700000
$EvaluationInterval = 250000

$NightId = Get-Date -Format "yyyyMMdd_HHmmss"
$TerminalLogDir = "logs\dqn\overnight_$NightId"
New-Item -ItemType Directory -Force -Path $TerminalLogDir | Out-Null

# Szybkie testy przed startem
$env:PYTHONPATH = "src"
Write-Host "[TESTY] Uruchamianie testów jednostkowych..."
& $Python -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "[BŁĄD] Testy nie przeszły. Trening zatrzymany." -ForegroundColor Red
    exit 1
}

$RunDir = "checkpoints\dqn\$RunName"
$FinalState = "$RunDir\training_state_final.pth"
$LatestState = "$RunDir\training_state_latest.pth"
$TerminalLog = "$TerminalLogDir\$RunName.log"

if (Test-Path $FinalState) {
    Write-Host "[GOTOWE] $RunName osiągnął już limit $TotalDecisions decyzji." -ForegroundColor Green
    exit 0
}

$Decisions = $TotalDecisions
$ExtraArgs = @()

if (Test-Path $LatestState) {
    $CompletedStr = & $Python -c "import sys, torch; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['completed_env_steps']))" $LatestState
    $Completed = [int]$CompletedStr
    $Decisions = $TotalDecisions - $Completed
    
    if ($Decisions -le 0) {
        Write-Host "[BŁĄD] Stan ma $Completed decyzji, ale brak pliku końcowego." -ForegroundColor Red
        exit 1
    }
    
    $ExtraArgs += "--resume", $LatestState
    $Decisions = [math]::Floor($Decisions / 10000) * 10000
    
    if ($Decisions -le 0) {
        Write-Host "[GOTOWE] $RunName jest już przy technicznym limicie." -ForegroundColor Green
        exit 0
    }
    Write-Host "[WZNOWIENIE] $RunName: $Completed/$TotalDecisions decyzji." -ForegroundColor Yellow
}

Write-Host "[START] Jeden długi trening: $RunName." -ForegroundColor Cyan
Write-Host "[STOP] Rano naciśnij Ctrl+C jeden raz. Stan zostanie zapisany automatycznie." -ForegroundColor Yellow

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "src"

# Uruchomienie treningu z przekierowaniem logów do pliku i na ekran
& $Python src\training\train_dqn.py `
    --run-name $RunName `
    --seed $Seed `
    --decisions $Decisions `
    --epsilon-decay-decisions $EpsilonDecayDecisions `
    --evaluation-interval $EvaluationInterval `
    @ExtraArgs | Tee-Object -FilePath $TerminalLog -Append

Write-Host "`nTrening zakończony albo bezpiecznie zatrzymany." -ForegroundColor Green
Write-Host "Model: $RunDir"
Write-Host "TensorBoard: .venv\Scripts\tensorboard.exe --logdir logs\dqn"