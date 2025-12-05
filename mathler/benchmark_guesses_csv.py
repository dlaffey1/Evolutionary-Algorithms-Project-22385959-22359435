#!/usr/bin/env python3
"""
Train the GA+GP Mathler solver once (using evolve_mathler),
then evaluate it on a number of random Mathler games and
write a CSV for guess distribution analysis.

Usage:
    python benchmark_guesses_csv.py --games 100 --out gp_guess_bench.csv
"""

import argparse
import csv
from pathlib import Path

from config import CONFIG
from mathler_main import evolve_mathler, _random_secret_in_value_range
from mathler_fitness import play_game_with_individual, FAIL_PENALTY


def run_guess_benchmark(num_games: int, out_path: Path) -> None:
    # 1) Train the GP solver with current CONFIG
    print(f"[bench] training GP solver with generations={CONFIG['generations']}, "
          f"pop_size={CONFIG['pop_size']}, train_sample_size={CONFIG['train_sample_size']}")
    best = evolve_mathler()

    # 2) Evaluate on random games
    rows = []
    solved_count = 0
    total_guesses = 0.0

    for i in range(num_games):
        secret_expr, target_value = _random_secret_in_value_range()
        g = play_game_with_individual(best, secret_expr, target_value, verbose=False)

        solved = 0
        guesses_used = g
        if g >= FAIL_PENALTY:
            # Treat as failure; cap guesses_used at max_guesses for reporting
            guesses_used = float(CONFIG["max_guesses"])
            solved = 0
        else:
            solved = 1
            solved_count += 1

        total_guesses += guesses_used

        rows.append({
            "game_idx": i,
            "secret_expr": secret_expr,
            "target_value": target_value,
            "guesses_used": guesses_used,
            "solved": solved,
        })

        if (i + 1) % 10 == 0:
            print(f"[bench] finished {i+1}/{num_games} games")

    # 3) Write CSV
    fieldnames = ["game_idx", "secret_expr", "target_value", "guesses_used", "solved"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    avg_guesses = total_guesses / num_games
    solve_rate = solved_count / num_games if num_games > 0 else 0.0

    print(f"[bench] wrote {out_path}")
    print(f"[bench] avg_guesses={avg_guesses:.3f}, solve_rate={solve_rate*100:.1f}% "
          f"over {num_games} games")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GP solver guess distribution on random Mathler games."
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to play with the trained GP solver.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="gp_guess_bench.csv",
        help="Output CSV filename.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    run_guess_benchmark(args.games, out_path)


if __name__ == "__main__":
    main()
