# Skrypt nocny IQN dla Windows (Faza 1). 
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
$RunName = "overnight_iqn_seed_11001"
$Seed = 11001

# Bardzo wysoki limit - rano zatrzymujemy recznie przez Ctrl+C
$TotalDecisions = 100000000

# Ewaluacja co 50 000 decyzji (czyli co 5 epok). 
$EvaluationInterval = 50000

# Stala czasowa tau dla epsilona - wydłużona do 500 000 dla długiego treningu nocnego,
# aby agent eksplorował przez pierwsze ~2 miliony kroków.
$EpsilonTau = 500000

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
    # Uzywamy pojedynczych cudzyslowow na zewnatrz, aby PowerShell nie ingerowal w kod Pythona
    $CompletedStr = & $Python -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["completed_env_steps"]))' $LatestState
    $Completed = [int]$CompletedStr
    $Decisions = $TotalDecisions - $Completed
    
    if ($Decisions -le 0) {
        Write-Host "[BLAD] Stan ma $Completed decyzji, ale brak pliku koncowego." -ForegroundColor Red
        exit 1
    }
    
    $ExtraArgs += "--resume", $LatestState
    # Zaokraglamy do pelnych epok (10 000 dla IQN)
    $Decisions = [math]::Floor($Decisions / 10000) * 10000
    
    if ($Decisions -le 0) {
        Write-Host "[GOTOWE] $RunName jest juz przy technicznym limicie." -ForegroundColor Green
        exit 0
    }
    Write-Host "[WZNOWIENIE] ${RunName}: $Completed/$TotalDecisions decyzji." -ForegroundColor Yellow
}

Write-Host "[START] Jeden dlugi trening IQN: $RunName." -ForegroundColor Cyan
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
    @ExtraArgs | Tee-Object -FilePath $TerminalLog -Append

Write-Host "`nTrening zakonczony albo bezpiecznie zatrzymany." -ForegroundColor Green
Write-Host "Model: $RunDir"
Write-Host "TensorBoard: .venv\Scripts\tensorboard.exe --logdir logs\iqn"