"""Końcowe, dokładne porównanie wybranych checkpointów DQN.

Każdy model gra na tych samych seedach i na każdym z czterech miejsc przy
stole. Dzięki temu różnice między modelami nie wynikają ze szczęśliwszych kart
albo uprzywilejowanej pozycji. Szczegółowe próbki turniejów pozostają w JSON,
a kompaktowy ranking z bootstrapowymi przedziałami ufności trafia do CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import config
from evaluation.evaluator import EVALUATION_SUITES
from evaluation.evaluator_dqn import DQNEvaluator
from paths import dqn_run_dir


DEFAULT_RUN_NAME = "dqn_phase1_chip_only_stable_seed_14001_retry"
# Dwa wczesne checkpointy są ważną grupą kontrolną. Ich krótka walidacja była
# zaskakująco mocna, więc nie zakładamy z góry, że dłuższy trening jest lepszy.
DEFAULT_STEPS = (
    750_000,
    1_250_000,
    10_250_000,
    10_500_000,
    10_750_000,
    11_500_000,
    13_250_000,
)
STAGE = "final_comparison_1000"


def report_is_complete(
    path: Path,
    *,
    tournaments: int,
    suites: tuple[str, ...] = EVALUATION_SUITES,
) -> bool:
    """Rozpoznaj kompletny raport, aby bezpiecznie wznowić długi test."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return set(report) == set(suites) and all(
        int(report[suite].get("tournaments", -1)) == tournaments
        and len(report[suite].get("tournament_samples", ())) == tournaments
        for suite in suites
    )


def bootstrap_suite(
    result: dict,
    *,
    rng: np.random.Generator,
    repetitions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrapuj bb/100 i win rate, zachowując całe turnieje jako klastry."""
    samples = result["tournament_samples"]
    chips = np.asarray([sample["chip_delta"] for sample in samples], dtype=float)
    hands = np.asarray([sample["hands"] for sample in samples], dtype=float)
    indices = rng.integers(0, len(samples), size=(repetitions, len(samples)))
    bb_per_100 = (
        chips[indices].sum(axis=1)
        / hands[indices].sum(axis=1)
        * (100.0 / config.BIG_BLIND)
    )

    # Turnieje zatrzymane limitem rozdań nadal mają prawidłowy wynik żetonowy,
    # ale nie mają zwycięzcy. Dlatego nie włączamy ich do turniejowego win rate.
    resolved = [sample for sample in samples if sample["resolved"]]
    wins = np.asarray([sample["won"] for sample in resolved], dtype=float)
    win_indices = rng.integers(
        0,
        len(wins),
        size=(repetitions, len(wins)),
    )
    win_rate = wins[win_indices].mean(axis=1)
    return bb_per_100, win_rate


def build_summary(
    report_dir: Path,
    *,
    steps: tuple[int, ...],
    tournaments: int,
    bootstrap_repetitions: int,
) -> list[dict[str, float | int | str]]:
    """Zbuduj ranking tylko z ukończonych raportów bieżącego porównania."""
    summary_rows: list[dict[str, float | int | str]] = []
    for step in steps:
        report_path = report_dir / f"{STAGE}_step_{step:09d}.json"
        if not report_is_complete(report_path, tournaments=tournaments):
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        suite_bootstraps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        row: dict[str, float | int | str] = {
            "step": step,
            "checkpoint": f"step_{step:09d}.pth",
        }

        for suite_index, suite in enumerate(EVALUATION_SUITES):
            # Stały seed dotyczy wyłącznie analizy statystycznej. Nie wpływa na
            # turnieje i gwarantuje identyczny ranking przy każdym wznowieniu.
            rng = np.random.default_rng(
                config.EVAL_FINAL_SEED + step + suite_index * 10_000_000
            )
            bb_samples, win_samples = bootstrap_suite(
                report[suite],
                rng=rng,
                repetitions=bootstrap_repetitions,
            )
            suite_bootstraps[suite] = (bb_samples, win_samples)
            bb_low, bb_high = np.percentile(bb_samples, (2.5, 97.5))
            win_low, win_high = np.percentile(win_samples, (2.5, 97.5))
            row[f"{suite}_bb_per_100"] = float(report[suite]["bb_per_100"])
            row[f"{suite}_bb_ci95_low"] = float(bb_low)
            row[f"{suite}_bb_ci95_high"] = float(bb_high)
            row[f"{suite}_win_rate"] = float(report[suite]["tournament_win_rate"])
            row[f"{suite}_win_ci95_low"] = float(win_low)
            row[f"{suite}_win_ci95_high"] = float(win_high)

        weighted_samples = sum(
            config.PHASE1_OPPONENT_WEIGHTS[suite]
            * suite_bootstraps[suite][0]
            for suite in config.PHASE1_OPPONENT_WEIGHTS
        )
        weighted_low, weighted_high = np.percentile(
            weighted_samples,
            (2.5, 97.5),
        )
        row["weighted_bb_per_100"] = float(
            sum(
                config.PHASE1_OPPONENT_WEIGHTS[suite]
                * float(report[suite]["bb_per_100"])
                for suite in config.PHASE1_OPPONENT_WEIGHTS
            )
        )
        row["weighted_bb_ci95_low"] = float(weighted_low)
        row["weighted_bb_ci95_high"] = float(weighted_high)
        summary_rows.append(row)

    summary_rows.sort(key=lambda item: float(item["weighted_bb_per_100"]), reverse=True)
    return summary_rows


def save_summary(report_dir: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Zapisz ranking atomowo, aby przerwanie nie pozostawiło uciętego CSV."""
    if not rows:
        return
    path = report_dir / "comparison_summary.csv"
    temporary_path = path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def print_ranking(rows: list[dict[str, float | int | str]]) -> None:
    """Pokaż czytelny ranking ukończonych checkpointów."""
    print("\n=== RANKING UKOŃCZONYCH CHECKPOINTÓW ===")
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>2}. krok {int(row['step']):>9,} | "
            f"{float(row['weighted_bb_per_100']):8.2f} bb/100 | "
            f"95% CI "
            f"[{float(row['weighted_bb_ci95_low']):.2f}, "
            f"{float(row['weighted_bb_ci95_high']):.2f}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Porównaj checkpointy DQN na 1000 turniejach na zestaw."
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--tournaments", type=int, default=1_000)
    parser.add_argument("--seed-base", type=int, default=config.EVAL_FINAL_SEED)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = tuple(dict.fromkeys(args.steps))
    if args.tournaments <= 0:
        raise ValueError("Liczba turniejów musi być dodatnia")
    if args.bootstrap_repetitions <= 0:
        raise ValueError("Liczba powtórzeń bootstrapu musi być dodatnia")

    run_dir = dqn_run_dir(args.run_name)
    report_dir = run_dir / f"final_comparison_{args.tournaments}"
    report_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        run_dir / f"step_{step:09d}.pth"
        for step in steps
        if not (run_dir / f"step_{step:09d}.pth").is_file()
    ]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Brakuje checkpointów:\n{formatted}")

    for index, step in enumerate(steps, start=1):
        checkpoint = run_dir / f"step_{step:09d}.pth"
        report_path = report_dir / f"{STAGE}_step_{step:09d}.json"
        if report_is_complete(report_path, tournaments=args.tournaments):
            print(f"[POMINIĘTO {index}/{len(steps)}] Krok {step:,} jest już ukończony.")
        else:
            print(f"\n[TEST {index}/{len(steps)}] Checkpoint {step:,} decyzji.")
            evaluator = DQNEvaluator(
                num_tournaments=args.tournaments,
                model_path=checkpoint,
                training_phase=1,
                report_dir=report_dir,
            )
            evaluator.evaluate(
                stage=STAGE,
                step=step,
                seed_base=args.seed_base,
            )

        # Aktualizujemy ranking po każdym modelu. Jeśli użytkownik przerwie
        # kolejne testy, wyniki ukończonych checkpointów nadal są gotowe.
        rows = build_summary(
            report_dir,
            steps=steps,
            tournaments=args.tournaments,
            bootstrap_repetitions=args.bootstrap_repetitions,
        )
        save_summary(report_dir, rows)
        print_ranking(rows)

    print(f"\n[GOTOWE] Raporty i ranking: {report_dir}")


if __name__ == "__main__":
    main()
