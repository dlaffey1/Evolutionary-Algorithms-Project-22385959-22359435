#!/usr/bin/env python3
"""
Parse a Mathler GP training log (output from mathler_main.py)
and plot best + average fitness (avg guesses) vs generation
for multiple runs in a single log file.

Usage:
    python benchmark_gp_fitness_curve.py --log gp_train_log.txt --out gp_fitness_curve.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

GEN_LINE_RE = re.compile(
    r"Generation\s+(\d+)\s*\|\s*Best fitness \(avg guesses\):\s*([0-9.]+)\s*\|\s*Avg fitness:\s*([0-9.]+)"
)


def parse_log_multi(path: Path):
    """
    Parse a log file that may contain multiple GP runs, each ending with
    '=== Finished GP training (Mathler) ==='.

    Returns:
        runs: list of (gens, best, avg) tuples, one per run.
    """
    runs = []
    gens = []
    best = []
    avg = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = GEN_LINE_RE.search(line)
            if m:
                g = int(m.group(1))
                b = float(m.group(2))
                a = float(m.group(3))
                gens.append(g)
                best.append(b)
                avg.append(a)
                continue

            if "=== Finished GP training (Mathler) ===" in line:
                if gens:
                    runs.append((gens, best, avg))
                    gens, best, avg = [], [], []

    # In case the last run doesn't have a Finished line (shouldn't happen but safe)
    if gens:
        runs.append((gens, best, avg))

    if not runs:
        raise RuntimeError(f"No generation lines found in log {path}")

    return runs

def plot_fitness_curve_multi(runs, out_path: Path):
    """
    Plot multiple runs on the same figure.

    runs: list of (gens, best, avg)
    """
    plt.figure()

    # One colour per run, same colour for best+avg of that run
    colors = ["tab:blue", "tab:orange", "tab:green",
              "tab:red", "tab:purple", "tab:brown"]

    for idx, (gens, best, avg) in enumerate(runs, start=1):
        c = colors[(idx - 1) % len(colors)]

        # Best = solid line
        plt.plot(
            gens,
            best,
            marker="o",
            linestyle="-",
            color=c,
            label=f"Run {idx} best",
        )

        # Avg = dashed line, same colour
        plt.plot(
            gens,
            avg,
            marker="s",
            linestyle="--",
            color=c,
            label=f"Run {idx} avg",
        )

    plt.xlabel("Generation")
    plt.ylabel("Fitness (average guesses, lower is better)")
    plt.title("GA+GP Mathler solver: fitness over generations (3 runs)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def main():
    parser = argparse.ArgumentParser(
        description="Plot GP fitness (best + avg) vs generation from training log (multi-run)."
    )
    parser.add_argument(
        "--log",
        type=str,
        default="gp_train_log.txt",
        help="Path to training log file (output of mathler_main.py).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="gp_fitness_curve.png",
        help="Output PNG filename.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    runs = parse_log_multi(log_path)

    print(f"Parsed {len(runs)} runs from {log_path}")
    for i, (gens, best, avg) in enumerate(runs, start=1):
        print(f"  Run {i}: {len(gens)} generations, "
              f"best fitness range [{min(best):.3f}, {max(best):.3f}]")

    out_path = Path(args.out)
    plot_fitness_curve_multi(runs, out_path)
    print(f"Wrote fitness curve to {out_path}")


if __name__ == "__main__":
    main()
