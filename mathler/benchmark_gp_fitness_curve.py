#!/usr/bin/env python3
"""

Parse a Mathler GP training log (output from mathler_main.py)
and plot best + average fitness (avg guesses) vs generation.

Usage:
    python  benchmark_gp_fitness_curve.py --log gp_train_log.txt --out gp_fitness_curve.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


GEN_LINE_RE = re.compile(
    r"Generation\s+(\d+)\s*\|\s*Best fitness \(avg guesses\):\s*([0-9.]+)\s*\|\s*Avg fitness:\s*([0-9.]+)"
)


def parse_log(path: Path):
    gens = []
    best = []
    avg = []

    finished_seen = False

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
                finished_seen = True
                # stop after the first full run
                break

    if not gens:
        raise RuntimeError(f"No generation lines found in log {path}")

    return gens, best, avg, finished_seen


def plot_fitness_curve(gens, best, avg, out_path: Path):
    plt.figure()
    plt.plot(gens, best, marker="o", label="Best fitness (avg guesses)")
    plt.plot(gens, avg, marker="s", label="Average fitness (population)")

    plt.xlabel("Generation")
    plt.ylabel("Fitness (average guesses, lower is better)")
    plt.title("GA+GP Mathler solver: fitness over generations")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def main():
    parser = argparse.ArgumentParser(
        description="Plot GP fitness (best + avg) vs generation from training log."
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
    gens, best, avg, finished = parse_log(log_path)

    print(f"Parsed {len(gens)} generations from {log_path} (finished={finished})")

    out_path = Path(args.out)
    plot_fitness_curve(gens, best, avg, out_path)
    print(f"Wrote fitness curve to {out_path}")


if __name__ == "__main__":
    main()
