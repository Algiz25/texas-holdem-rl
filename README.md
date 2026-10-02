# Texas Hold'em Reinforcement Learning

Niniejsze repozytorium zawiera projekt końcowy zrealizowany w ramach inicjatywy **Wakacyjne Wyzwanie 2026**, organizowanej przez **Koło Naukowe Solvro** z **Politechniki Wrocławskiej**.

Projekt stanowi kompleksowe środowisko turniejowe No-Limit Texas Hold'em zintegrowane z systemem do trenowania zaawansowanych agentów opartych na uczeniu ze wzmocnieniem (Reinforcement Learning). Architektura została zbudowana z wykorzystaniem bibliotek PyTorch oraz Tianshou, natomiast logika pokera i interfejs środowiska opierają się na silnikach RLCard, PettingZoo oraz standardzie Gymnasium.

System obsługuje cztery algorytmy RL (DQN, PPO, SAC, IQN) oraz przeprowadza trening w dwóch niezależnych fazach: uczenie podstaw gry przeciwko botom heurystycznym (Faza 1) oraz zaawansowany Self-Play (Faza 2).

## Spis treści
1. [Architektura Systemu](#architektura-systemu)
2. [Przestrzeń Obserwacji i Akcji](#przestrzeń-obserwacji-i-akcji)
3. [Trening Modelu](#trening-modelu)
4. [Ewaluacja](#ewaluacja)
5. [Generator Powtórek i Interfejs UI](#generator-powtórek-i-interfejs-ui)
6. [Osiągnięte Wyniki](#osiągnięte-wyniki)
7. [Struktura Projektu](#struktura-projektu)

## Architektura Systemu

System integruje kilka warstw abstrakcji, aby zapewnić stabilny trening algorytmów RL w środowisku o ukrytej informacji:

- **Baza pokera (RLCard):** Odpowiada za mechanikę gry, tasowanie kart, ewaluację rąk (showdown) i pule.
- **Wieloagentowe środowisko (PettingZoo):** Klasa `TexasHoldemTournament` realizuje cykl AEC (Agent-Environment-Cycle). Śledzi statystyki VPIP/PFR, oblicza pulę, zarządza bankructwami oraz przyznaje nagrody za zajęte miejsca w turnieju.
- **Nakładka Jednoagentowa (Gymnasium):** Kluczowy element treningowy. Ukrywa przed agentem wieloagentową naturę środowiska. Oblicza ruchy stołu w tle i zwraca sterowanie do algorytmu RL (Tianshou) tylko w momencie, gdy uczeń (learner) musi podjąć decyzję. Zapewnia to prawidłową strukturę próbek w buforze pamięci (Replay Buffer).
- **Silnik Uczenia ze Wzmocnieniem (Tianshou):** Stanowi główny framework RL orkiestrujący cały proces treningowy. Odpowiada za zarządzanie pamięcią doświadczeń (Replay Buffer), wydajne i równoległe zbieranie danych z wielu środowisk (Collectors) oraz kontrolowanie głównych pętli uczących (Trainers). Tianshou dostarcza zoptymalizowane implementacje algorytmów takich jak DQN, PPO, SAC czy IQN, pełniąc rolę pomostu, który płynnie łączy sieci neuronowe zdefiniowane w PyTorch ze zintegrowanym środowiskiem zgodnym ze standardem Gymnasium.
  <img width="1326" height="798" alt="image" src="https://github.com/user-attachments/assets/839e79a6-1d31-46bf-bbac-576399fbfa18" />

- **Sieci Neuronowe:** Dwugałęziowa architektura (Dual-Branch). Pierwsza gałąź przetwarza 104 cechy kart (one-hot), a druga gałąź pozostałe metadane (stacki, statystyki, oddsy). Cechy są łączone w warstwach decyzyjnych Actory i Critica.
<img width="1624" height="913" alt="image" src="https://github.com/user-attachments/assets/b330a7a8-fcb4-4fb4-98ae-493b2b3efdef" />

## Przestrzeń Obserwacji i Akcji

### Przestrzeń Akcji (Discrete 5)
Każdy algorytm porusza się w stałej przestrzeni 5 dyskretnych akcji. System dynamicznie maskuje niedozwolone akcje (np. brak możliwości czekania przy przebiciu).
0. Fold
1. Check / Call (wybór zależy od tego, czy koszt sprawdzenia wynosi 0)
2. Raise Half Pot (podbicie o połowę aktualnej puli)
3. Raise Pot (podbicie o pełną pulę)
4. All-in

### Przestrzeń Obserwacji (222 cechy)
Agent ma dostęp do bogatego wektora stanu gry, który naśladuje informacje dostępne dla ludzkiego gracza:
- **Karty (0-103):** 52 karty prywatne + 52 karty wspólne (zakodowane jako one-hot).
- **Stan finansowy (104-115):** Znormalizowane stacki, wkłady na ulicy oraz wkłady w całym rozdaniu.
- **Statusy (116-135):** Flagi aktywności, pasów, all-inów, pozycja buttona i aktualna faza licytacji (preflop, flop, turn, river).
- **Historia Akcji (138-185):** Poprzednie ruchy każdego gracza oraz statystyki z poprzednich ulic (liczba przebić i sprawdzeń).
- **Statystyki Przeciwników (186-203):** VPIP, PFR, wskaźnik agresji, Fold to Raise oraz Showdown Win Rate dla każdego z 3 przeciwników.
- **Cechy Zaawansowane (204-221):** Pot odds, kategoria aktualnej ręki (np. para, strit), wykryte drawy (open-ended, gutshot, flush draw) oraz tekstura stołu.
<img width="984" height="658" alt="image" src="https://github.com/user-attachments/assets/904e2062-231d-4296-a7d0-968e48556478" />

## Funkcja Nagrody

System pozwala łączyć gęste sygnały (dense rewards) po każdym rozdaniu z rzadkimi sygnałami (sparse rewards) na koniec gry. W stabilnym profilu DQN fazy 1 premia za miejsce jest wyłączona (`placement_reward_weight=0`), aby cel treningu odpowiadał metryce żetonowej bb/100. Pozostałe treningi zachowują dotychczasową wartość domyślną `1`.

Nagroda dla agenta składa się z dwóch elementów:

1. **Nagroda za rozdanie (Znormalizowany Payoff):** 
   Po każdym zakończonym rozdaniu agent otrzymuje nagrodę równą liczbie wygranych (lub przegranych) żetonów, podzieloną przez wartość początkowego stacka (`STARTING_CHIPS = 200`). Dzięki temu agent ma natychmiastowy feedback (np. wygrana puli wielkości 100 żetonów daje nagrodę `+0.5`, a strata wpisowego blinda `-0.01`).

2. **Nagroda za pozycję w turnieju (Placement Reward):** 
   Kiedy gracz bankrutuje (lub wygrywa stół), otrzymuje dodatkową nagrodę za zajęte miejsce. Została ona zaprojektowana w oparciu o wzór `(2.5 - pozycja) * 2.0`, co faworyzuje przetrwanie i agresywną grę o pierwsze miejsce, wprowadzając następujące wartości:
   - **1. miejsce:** `+3.0`
   - **2. miejsce:** `+1.0`
   - **3. miejsce:** `-1.0`
   - **4. miejsce:** `-3.0`

W przypadku jednoczesnej eliminacji kilku graczy w tym samym rozdaniu, otrzymują oni średnią z miejsc ex aequo (np. obaj odpadający na 3. i 4. miejscu zajmują pozycję 3.5, co daje karę `-2.0`).
<img width="1624" height="913" alt="image" src="https://github.com/user-attachments/assets/605336a1-ee99-46e1-aae7-293b76cdf3f7" />


## Trening Modelu

Wszystkie parametry (rozmiary buforów, współczynniki uczenia, częstotliwości zapisu) znajdują się w pliku `src/config.py`. System używa biblioteki Tianshou do orkiestracji treningu.

Z poziomu terminala można uruchomić jeden z czterech obsługiwanych algorytmów:

```bash
PYTHONPATH=src python src/training/train_dqn.py --run-name dqn_phase1 --seed 11001
PYTHONPATH=src python src/training/train_ppo.py --run-name ppo_phase1 --seed 11002
PYTHONPATH=src python src/training/train_sac.py --run-name sac_phase1 --seed 11003
PYTHONPATH=src python src/training/train_iqn.py --run-name iqn_phase1 --seed 11004
```

### Fazy treningu
- **Faza 1:** Rozgrzewka. Agent uczy się grać przeciwko mieszance prostych, algorytmicznych botów (Random, Passive, Mixed, Aggressive).
- **Faza 2:** Self-Play. Przejście na grę przeciwko własnym historycznym wagom (`historical_self`) oraz najnowszej polityce (`latest_self`). Trening ten ma na celu aproksymację równowagi Nasha (GTO). Przejście do tej fazy wymaga podania w parametrach ścieżki do modelu bazowego z Fazy 1.
<img width="1904" height="1065" alt="image" src="https://github.com/user-attachments/assets/e7ff8615-6112-4558-ab8e-508f73b8550f" />



### Treningi nocne
W folderze `scripts/` znajdują się skrypty `.sh` i `.ps1` stworzone z myślą o wielomilionowych treningach trwających kilkanaście godzin. Zapewniają one regularne zapisywanie stanu `training_state_latest.pth` i pozwalają na przerwanie treningu standardowym sygnałem przerwania (Ctrl+C), po którym następuje bezpieczny zapis wszystkich wag. Parametry skryptów zostały zoptymalizowane pod kątem maksymalnego wykorzystania naszych stacji roboczych. W przypadku uruchamiania treningu na innym sprzęcie, może być konieczne zmniejszenie liczby równoległych środowisk (np. NUM_TRAIN_ENVS) lub innych parametrów specyficznych dla algorytmów (np. <algorytm>_BATCH_SIZE) w pliku config.py, aby uniknąć przeciążenia systemu.

Wgląd w metryki treningowe odbywa się przez TensorBoard:
```bash
tensorboard --logdir logs/
```

Dla DQN dostępne są osobne, bezpiecznie wznawialne profile na macOS:

```bash
./scripts/run_dqn_phase1_chip_only_macbook.sh
./scripts/run_dqn_selfplay_macbook.sh
./scripts/run_dqn_selfplay_ppo_style_macbook.sh
```

Skrypty self-play korzystają z istniejącego modelu
`checkpoints/dqn/REAL_BEST/phase_2/step_010500000.pth`; nie wymagają drugiej
kopii wag w repozytorium. `Ctrl+C` zapisuje stan runu, a ponowne wywołanie tej
samej komendy kontynuuje trening.

## Ewaluacja

Mechanizm walidacyjny wyodrębniono do osobnych procesów. Końcowy raport nie zależy od funkcji uczących, a wskaźniki (np. `bb/100`) są wiarygodne, ponieważ sprawdzane były na wyizolowanych seedach.
Walidacja odbywa się podczas treningu w zależności od parametrów w `config.py`.

Raporty generują szczegółowe pliki JSON z małą próbką każdego turnieju oraz plik zbiorczy `evaluations_v3.csv` wewnątrz folderu `checkpoints/<algorytm>/evaluations`. Próbki pozwalają po treningu policzyć przedziały ufności bez zapisywania pełnego przebiegu każdej gry. Dla zgodności z dotychczasowymi narzędziami tworzony jest również indeks `evaluations_v2.csv`.

## Generator Powtórek i Interfejs UI

Wizualna analiza wyuczonych modeli możliwa jest dzięki lokalnej aplikacji napisanej w Streamlit. Umożliwia ona rozegranie pokazowego turnieju pomiędzy dowolnymi algorytmami lub heurystykami.

1. **Konfiguracja stołu:** W pliku `src/app/app_config.py` zdefiniuj, kto usiądzie przy stole (np. DQN przeciwko PPO, SAC i botowi mieszanemu).
2. **Generacja historii:**
```bash
PYTHONPATH=src python src/app/replay_generator.py
```
3. **Uruchomienie stołu graficznego:**
```bash
PYTHONPATH=src streamlit run src/app/poker_ui.py
```
Aplikacja otwiera się w przeglądarce, odtwarza klatka po klatce wygenerowany plik JSON (animacje lotu żetonów, ukryte i odkryte karty, powiadomienia o akcjach).
<img width="1883" height="928" alt="image" src="https://github.com/user-attachments/assets/f0e555f9-2c1a-435f-994e-a1bec62ede01" />

## Osiągnięte Wyniki

Najlepsze wyuczone wagi modeli z poszczególnych eksperymentów zostały zarchiwizowane i znajdują się w podkatalogach `checkpoints/<algorytm>/REAL_BEST`. 

Poniżej przedstawiono wyniki ewaluacji poszczególnych algorytmów. Główną metryką oceniającą siłę agenta w pojedynczych rozdaniach jest zysk wyrażony w wielkich ciemnych na 100 rozdań (`bb/100`), natomiast `Win Rate` określa procent wygranych całych turniejów w danym zestawie walidacyjnym.

Dalszy trening DQN w Fazie 2 (Self-Play) nie przyniósł oczekiwanych rezultatów (brak poprawy względem polityki bazowej), dlatego głównym i docelowym modelem DQN pozostaje wersja z Fazy 1.

Cross-ewaluacja odbywała się na przestrzeni 1000 turniejów. Każdy algorytm zagrał 250 razy na każdym miejscu przy stole.
<img width="1624" height="913" alt="image" src="https://github.com/user-attachments/assets/8c39806e-1e5d-45a9-8a76-9ac96505ef20" />


## Struktura Projektu

```text
├── checkpoints/                - Automatycznie tworzone; tutaj trafiają wagi modeli (.pth) oraz raporty z ewaluacji (.csv/.json).
├── logs/                       - Eventy TensorBoard.
├── scripts/                    - Skrypty .sh i .ps1 do ciągłych, długich treningów (Faza 1 i Faza 2).
├── src/
│   ├── app/                    - Logika generatora powtórek i webowe UI wizualizujące rozgrywkę (Streamlit).
│   ├── evaluation/             - Niezależne skrypty weryfikujące siłę gry (bb/100, win rate) w wieloprocesowych środowiskach.
│   ├── observation/            - Kod transformacji gry do wektora obserwacji, inżynieria cech (karty, pot odds, historia).
│   ├── training/               - Logika pętli treningowych dla algorytmów Tianshou (base_trainer, dqn, ppo, sac, iqn).
│   ├── config.py               - Główne hiperparametry hiperparametry algorytmów, bufory, epsilony i konfiguracje środowiska.
│   ├── environment.py          - Główne środowisko gry Texas Hold'em (PettingZoo + RLCard).
│   ├── models.py               - Architektura sieci neuronowych PyTorch (MaskedActor, Critic).
│   └── paths.py                - Zunifikowane ścieżki do plików.
└── tests/                      - Testy jednostkowe potwierdzające integralność mechaniki gry i tensorów.
```
