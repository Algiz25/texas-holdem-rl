# Skrypt nocny PPO dla Windows. 
# UWAGA: Aby komputer nie zasnął, zmień ustawienia zasilania Windows 
# (Ustawienia -> System -> Zasilanie i uśpienie -> Uśpienie: Nigdy).

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location -Path $ProjectDir

$Python = ".venv\Scripts\python.exe"
$RunName = "overnight_ppo_seed_11001"
$Seed = 11001
# 5 milionów decyzji to około 305 epok (przy buforze 16384). 
# Idealne na kilkugodzinny trening.
$TotalDecisions = 5000000 
$EvaluationInterval = 245760

$NightId = Get-Date -Format "yyyyMMdd_HHmmss"
$TerminalLogDir = "logs\ppo\overnight_$NightId"
New-Item -ItemType Directory -Force -Path $TerminalLogDir | Out-Null

# Szybkie testy przed startem
$env:PYTHONPATH = "src"
Write-Host "[TESTY] Uruchamianie testów jednostkowych..."
& $Python -m unittest discover -s tests -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "[BŁĄD] Testy nie przeszły. Trening zatrzymany." -ForegroundColor Red
    exit 1
}

$RunDir = "checkpoints\ppo\$RunName"
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
    # Zaokrąglamy do pełnych epok (16384)
    $Decisions = [math]::Floor($Decisions / 16384) * 16384
    
    if ($Decisions -le 0) {
        Write-Host "[GOTOWE] $RunName jest już przy technicznym limicie." -ForegroundColor Green
        exit 0
    }
    Write-Host "[WZNOWIENIE] ${RunName}: $Completed/$TotalDecisions decyzji." -ForegroundColor Yellow
}

Write-Host "[START] Jeden długi trening PPO: $RunName." -ForegroundColor Cyan
Write-Host "[STOP] Aby przerwać, naciśnij Ctrl+C jeden raz. Stan zostanie zapisany automatycznie." -ForegroundColor Yellow

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = "src"

# Uruchomienie treningu PPO
& $Python src\training\train_ppo.py `
    --run-name $RunName `
    --seed $Seed `
    --decisions $Decisions `
    --evaluation-interval $EvaluationInterval `
    @ExtraArgs | Tee-Object -FilePath $TerminalLog -Append

Write-Host "`nTrening zakończony albo bezpiecznie zatrzymany." -ForegroundColor Green
Write-Host "Model: $RunDir"
Write-Host "TensorBoard: .venv\Scripts\tensorboard.exe --logdir logs\ppo"