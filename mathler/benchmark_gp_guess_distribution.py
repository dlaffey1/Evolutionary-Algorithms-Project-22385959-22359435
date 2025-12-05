#!/usr/bin/env python3
"""

Make a bar chart of how many games were solved in 1,2,...,6 guesses
(or failed) for the GP solver, using gp_guess_bench.csv from gp_benchmark.py.
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


def load_guess_csv(path):
    rows = []
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["guesses_used"] = float(row["guesses_used"])
            row["solved"] = int(row["solved"])
            rows.append(row)
    return rows


def plot_distribution(csv_path="gp_guess_bench.csv", out_path="gp_guess_distribution.png"):
    rows = load_guess_csv(csv_path)

    counts = Counter()
    for r in rows:
        if r["solved"]:
            # guesses_used should be integer 1..6 when solved
            g = int(round(r["guesses_used"]))
            counts[g] += 1
        else:
            counts["fail"] += 1

    # Ensure we have bins in order 1..6 + "fail"
    labels = [1, 2, 3, 4, 5, 6, "fail"]
    values = [counts[l] for l in labels]

    plt.figure()
    plt.bar([str(l) for l in labels], values)
    plt.xlabel("Guesses used (fail = not solved in max_guesses)")
    plt.ylabel("Number of games")
    plt.title("GP solver: distribution of guesses per game")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    plot_distribution()
