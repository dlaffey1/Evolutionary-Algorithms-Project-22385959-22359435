#!/usr/bin/env python3


import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt

from config import CONFIG
from mathler_main import evolve_mathler


def approx_total_games(train_sample_size: int) -> int:
    """
    Approximate total number of Mathler games simulated in one GP run.

    For each training run we:
      - evaluate the initial population once
      - then evaluate once per generation

    Each evaluation plays `train_sample_size` games for each individual.
    """
    generations = CONFIG["generations"]
    pop_size = CONFIG["pop_size"]
    return (generations + 1) * pop_size * train_sample_size


def run_benchmark(
    train_sizes,
    runs_per_size: int,
    csv_path: Path,
    fig_path: Path,
) -> None:
    rows = []
    x_games = []
    y_runtime = []

    for ts in train_sizes:
        CONFIG["train_sample_size"] = ts

        games = approx_total_games(ts)

        runtimes = []
        for r in range(runs_per_size):
            print(f"[bench] train_sample_size={ts}, run {r+1}/{runs_per_size}")
            t0 = time.perf_counter()
            _ = evolve_mathler()  # uses CONFIG values internally
            t1 = time.perf_counter()
            elapsed = t1 - t0
            runtimes.append(elapsed)
            rows.append((ts, games, r + 1, elapsed))

        avg_rt = sum(runtimes) / len(runtimes)
        x_games.append(games)
        y_runtime.append(avg_rt)
        print(f"[bench] ts={ts}: games≈{games}, avg_runtime={avg_rt:.3f}s")

    # Write CSV
    csv_lines = ["train_sample_size,total_games,run_idx,runtime_seconds\n"]
    for ts, games, run_idx, rt in rows:
        csv_lines.append(f"{ts},{games},{run_idx},{rt:.6f}\n")
    csv_path.write_text("".join(csv_lines), encoding="utf-8")
    print(f"[bench] wrote CSV to {csv_path}")

    # Plot
    plt.figure()
    plt.plot(x_games, y_runtime, marker="o")
    plt.xlabel("Approx. total simulated games per GP run")
    plt.ylabel("Runtime per run (seconds)")
    plt.title("GA+GP Mathler solver: runtime vs simulated games")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    print(f"[bench] wrote plot to {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GA+GP runtime vs training sample size."
    )
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 24, 32],
        help="List of train_sample_size values to test.",
    )
    parser.add_argument(
        "--runs-per-size",
        type=int,
        default=3,
        help="Number of repeated GP runs per training size.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="gp_runtime_bench.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--png",
        type=str,
        default="gp_runtime_vs_games_gp.png",
        help="Output PNG filename.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    fig_path = Path(args.png)

    run_benchmark(args.train_sizes, args.runs_per_size, csv_path, fig_path)


if __name__ == "__main__":
    main()
