#!/usr/bin/env python3
"""
mathler_lineage_tuner.py

Fine tuner for the lineage-based Mathler solver in mathler_lineage_evolve.py.

- Runs multiple worker processes in parallel.
- Each worker:
    * Applies a slightly different set of lineage_* hyperparameters.
    * Plays N games (default 30) using MathlerLineageSolver.
    * Returns the average number of guesses (lower is better).
- Across 10 rounds:
    * Start with relatively large random changes to hyperparameters.
    * Gradually reduce perturbation magnitude as rounds progress (coarse → fine).
    * Track the best hyperparameters seen so far.
    * On each round, all workers perturb around the current best.
- At the end:
    * Save the best hyperparameters into best_lineage_hyperparams.json.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Tuple

from config import CONFIG
from mathler_env import (
    EXPR_LEN,
    safe_eval,
    generate_initial_candidates,
    mathler_feedback,
)

# ---------------------------------------------------------------------------
# Global tuner configuration
# ---------------------------------------------------------------------------

NUM_ROUNDS = 10
GAMES_PER_EVAL = 30
OUTPUT_JSON = "best_lineage_hyperparams.json"

# If None, we’ll use min(cpu_count(), 8)
NUM_WORKERS = None

# Range for target values when sampling games
MIN_TARGET = CONFIG.get("min_target_value", 100)
MAX_TARGET = CONFIG.get("max_target_value", 300)

MAX_GUESSES = CONFIG["max_guesses"]
FAIL_PENALTY = CONFIG["fail_penalty"]


# ---------------------------------------------------------------------------
# Hyperparameter representation
# ---------------------------------------------------------------------------

@dataclass
class LineageHyperParams:
    # Core GA/EA controls
    lineage_pop_size: int
    lineage_gens_first: int
    lineage_gens_per_step: int
    lineage_crossover_rate: float
    lineage_mutation_rate: float
    lineage_elite_fraction: float
    lineage_tournament_k: int
    lineage_mutation_attempts: int

    # Fitness weights / behaviour
    lineage_w_closeness: float
    lineage_closeness_max_score: float
    lineage_closeness_diff_clip: float
    lineage_inconsistency_penalty: float
    lineage_w_feedback_match: float

    # Feature weights (subset of those used in mathler_lineage_evolve)
    lineage_w_positional_freq_sum: float
    lineage_w_unique_letters: float
    lineage_w_distinct_digits: float
    lineage_w_symbol_entropy: float

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Convert to a dict of CONFIG keys → values that mathler_lineage_evolve expects.
        """
        return {
            "lineage_pop_size": self.lineage_pop_size,
            "lineage_gens_first": self.lineage_gens_first,
            "lineage_gens_per_step": self.lineage_gens_per_step,
            "lineage_crossover_rate": self.lineage_crossover_rate,
            "lineage_mutation_rate": self.lineage_mutation_rate,
            "lineage_elite_fraction": self.lineage_elite_fraction,
            "lineage_tournament_k": self.lineage_tournament_k,
            "lineage_mutation_attempts": self.lineage_mutation_attempts,
            "lineage_w_closeness": self.lineage_w_closeness,
            "lineage_closeness_max_score": self.lineage_closeness_max_score,
            "lineage_closeness_diff_clip": self.lineage_closeness_diff_clip,
            "lineage_inconsistency_penalty": self.lineage_inconsistency_penalty,
            "lineage_w_feedback_match": self.lineage_w_feedback_match,
            "lineage_w_positional_freq_sum": self.lineage_w_positional_freq_sum,
            "lineage_w_unique_letters": self.lineage_w_unique_letters,
            "lineage_w_distinct_digits": self.lineage_w_distinct_digits,
            "lineage_w_symbol_entropy": self.lineage_w_symbol_entropy,
        }

    @classmethod
    def from_config(cls) -> "LineageHyperParams":
        """
        Build initial hyperparameters from CONFIG (with reasonable defaults).
        """
        return cls(
            lineage_pop_size=CONFIG.get("lineage_pop_size", 60),
            lineage_gens_first=CONFIG.get("lineage_gens_first", 10),
            lineage_gens_per_step=CONFIG.get("lineage_gens_per_step", 10),
            lineage_crossover_rate=CONFIG.get("lineage_crossover_rate", 0.7),
            lineage_mutation_rate=CONFIG.get("lineage_mutation_rate", 0.4),
            lineage_elite_fraction=CONFIG.get("lineage_elite_fraction", 0.25),
            lineage_tournament_k=CONFIG.get("lineage_tournament_k", 3),
            lineage_mutation_attempts=CONFIG.get("lineage_mutation_attempts", 40),
            lineage_w_closeness=CONFIG.get("lineage_w_closeness", 2.0),
            lineage_closeness_max_score=CONFIG.get("lineage_closeness_max_score", 10.0),
            lineage_closeness_diff_clip=CONFIG.get("lineage_closeness_diff_clip", 200.0),
            lineage_inconsistency_penalty=CONFIG.get("lineage_inconsistency_penalty", 20.0),
            lineage_w_feedback_match=CONFIG.get("lineage_w_feedback_match", 1.5),
            lineage_w_positional_freq_sum=CONFIG.get("lineage_w_positional_freq_sum", 1.0),
            lineage_w_unique_letters=CONFIG.get("lineage_w_unique_letters", 0.1),
            lineage_w_distinct_digits=CONFIG.get("lineage_w_distinct_digits", 0.1),
            lineage_w_symbol_entropy=CONFIG.get("lineage_w_symbol_entropy", 0.5),
        )


# ---------------------------------------------------------------------------
# Hyperparameter perturbation
# ---------------------------------------------------------------------------

def perturb_hyperparams(
    base: LineageHyperParams,
    scale: float,
    rng: random.Random,
) -> LineageHyperParams:
    """
    Create a new hyperparameter set by perturbing the base values.
    'scale' controls how aggressive the perturbation is (0.0–1.0+ roughly).
    """
    def perturb_int(value: int, min_val: int, max_val: int) -> int:
        # +/- up to scale * value, but at least 1
        span = max(1, int(abs(value) * scale))
        delta = rng.randint(-span, span)
        new_val = max(min_val, min(max_val, value + delta))
        return new_val

    def perturb_float(value: float, min_val: float, max_val: float) -> float:
        # multiply by (1 + noise), noise ~ N(0, scale)
        noise = rng.gauss(0.0, scale)
        new_val = value * (1.0 + noise)
        new_val = max(min_val, min(max_val, new_val))
        return new_val

    b = base  # shorthand

    return LineageHyperParams(
        lineage_pop_size=perturb_int(b.lineage_pop_size, 20, 200),
        lineage_gens_first=perturb_int(b.lineage_gens_first, 3, 40),
        lineage_gens_per_step=perturb_int(b.lineage_gens_per_step, 3, 40),
        lineage_crossover_rate=perturb_float(b.lineage_crossover_rate, 0.2, 0.95),
        lineage_mutation_rate=perturb_float(b.lineage_mutation_rate, 0.05, 0.9),
        lineage_elite_fraction=perturb_float(b.lineage_elite_fraction, 0.05, 0.6),
        lineage_tournament_k=perturb_int(b.lineage_tournament_k, 2, 8),
        lineage_mutation_attempts=perturb_int(b.lineage_mutation_attempts, 10, 100),
        lineage_w_closeness=perturb_float(b.lineage_w_closeness, 0.5, 5.0),
        lineage_closeness_max_score=perturb_float(b.lineage_closeness_max_score, 5.0, 30.0),
        lineage_closeness_diff_clip=perturb_float(b.lineage_closeness_diff_clip, 50.0, 1000.0),
        lineage_inconsistency_penalty=perturb_float(b.lineage_inconsistency_penalty, 5.0, 80.0),
        lineage_w_feedback_match=perturb_float(b.lineage_w_feedback_match, 0.1, 5.0),
        lineage_w_positional_freq_sum=perturb_float(b.lineage_w_positional_freq_sum, 0.0, 5.0),
        lineage_w_unique_letters=perturb_float(b.lineage_w_unique_letters, 0.0, 3.0),
        lineage_w_distinct_digits=perturb_float(b.lineage_w_distinct_digits, 0.0, 3.0),
        lineage_w_symbol_entropy=perturb_float(b.lineage_w_symbol_entropy, 0.0, 5.0),
    )


# ---------------------------------------------------------------------------
# Worker: evaluate one hyperparameter set
# ---------------------------------------------------------------------------

def _evaluate_hyperparams_worker(
    args: Tuple[LineageHyperParams, int, int]
) -> float:
    """
    Worker function run in a separate process.

    Args:
        args: (hyperparams, games_per_eval, worker_seed)

    Returns:
        Average guesses over 'games_per_eval' games.
    """
    hyperparams, games_per_eval, worker_seed = args

    # Local imports inside the worker process
    import random as _random
    import importlib

    from config import CONFIG as _CONFIG
    from mathler_env import generate_initial_candidates, mathler_feedback
    from mathler_lineage_evolve import MathlerLineageSolver

    # Seed RNG for this worker
    if "random_seed" in _CONFIG:
        _random.seed(_CONFIG["random_seed"] + worker_seed)
    else:
        _random.seed(worker_seed)

    # Apply hyperparams to CONFIG
    hp_dict = hyperparams.to_config_dict()
    _CONFIG.update(hp_dict)

    # Reload mathler_lineage_evolve so it picks up new CONFIG values
    import mathler_lineage_evolve as _lineage_mod
    importlib.reload(_lineage_mod)
    from mathler_lineage_evolve import MathlerLineageSolver as WorkerSolver

    total_guesses = 0.0
    games_played = 0

    # We use CONFIG's MIN_TARGET/MAX_TARGET (they may have changed elsewhere,
    # but usually they don't depend on lineage_* params).
    min_t = _CONFIG.get("min_target_value", MIN_TARGET)
    max_t = _CONFIG.get("max_target_value", MAX_TARGET)

    for _ in range(games_per_eval):
        # Sample a random target and secret
        secret_expr = None
        target_value = None

        # Try a few times to get a valid (secret, target)
        for _tries in range(100):
            candidate_target = _random.randint(min_t, max_t)
            cands = generate_initial_candidates(candidate_target)
            if not cands:
                continue
            secret_expr = _random.choice(cands)
            try:
                val = safe_eval(secret_expr)
            except Exception:
                continue
            if val != candidate_target:
                continue
            target_value = candidate_target
            break

        if secret_expr is None or target_value is None:
            # failed to find any secret for this worker/game; treat as failure
            total_guesses += FAIL_PENALTY
            games_played += 1
            continue

        # Play one game with MathlerLineageSolver
        solver = WorkerSolver(target_value=target_value, verbose=False)

        # Initialise population around an auto-chosen seed
        solver.prepare_initial_guess(initial_guess=None)

        solved = False
        for attempt in range(1, MAX_GUESSES + 1):
            guess = solver.next_guess(first=(attempt == 1))
            fb = mathler_feedback(guess, secret_expr)
            if guess == secret_expr:
                total_guesses += float(attempt)
                solved = True
                break
            solver.register_feedback(guess, fb)

        if not solved:
            total_guesses += FAIL_PENALTY

        games_played += 1

    if games_played == 0:
        return float("inf")

    avg_guesses = total_guesses / games_played
    return avg_guesses


# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------

def tune_lineage_hyperparams(
    num_rounds: int = NUM_ROUNDS,
    games_per_eval: int = GAMES_PER_EVAL,
    num_workers: int | None = None,
    output_path: str = OUTPUT_JSON,
) -> None:
    """
    Run the lineage hyperparameter tuner.

    - num_rounds: number of perturbation rounds (default 10).
    - games_per_eval: number of games per worker evaluation (default 30).
    - num_workers: number of worker processes (default=min(cpu_count(), 8)).
    - output_path: where to write the best hyperparameters JSON.
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    print(
        f"[tune] Starting lineage tuner with {num_rounds} rounds, "
        f"{games_per_eval} games per eval, {num_workers} workers."
    )

    rng = random.Random(CONFIG.get("random_seed", 42))

    # Initial hyperparams from CONFIG
    best_hp = LineageHyperParams.from_config()

    # Evaluate initial hyperparams once to get a baseline
    print("[tune] Evaluating initial CONFIG-based hyperparameters...")
    baseline_score = _evaluate_hyperparams_worker((best_hp, games_per_eval, 0))
    best_score = baseline_score
    print(f"[tune] Baseline avg guesses = {best_score:.3f}")

    for round_idx in range(num_rounds):
        # scale goes from ~0.5 down to ~0.05 over rounds (roughly)
        frac = round_idx / max(1, num_rounds - 1)
        scale = (1.0 - frac) * 0.5 + frac * 0.05  # large → small

        print(
            f"\n[tune] Round {round_idx+1}/{num_rounds} "
            f"(perturbation scale={scale:.3f}, current best={best_score:.3f})"
        )

        # Build candidate hyperparams for this round
        jobs = []
        for worker_id in range(num_workers):
            if worker_id == 0:
                # First worker re-evaluates the current best (sanity check / noise smoothing)
                hp = best_hp
            else:
                hp = perturb_hyperparams(best_hp, scale, rng)
            jobs.append((hp, games_per_eval, round_idx * 1000 + worker_id))

        # Run workers in parallel
        with Pool(processes=num_workers) as pool:
            results = pool.map(_evaluate_hyperparams_worker, jobs)

        # Analyse results
        round_best_score = float("inf")
        round_best_hp = None

        for idx, (hp_job, avg_guesses) in enumerate(zip(jobs, results)):
            hp = hp_job[0]
            print(
                f"[tune]   Worker {idx}: avg guesses={avg_guesses:.3f} "
                f"({'BEST' if avg_guesses < best_score else ''})"
            )
            if avg_guesses < round_best_score:
                round_best_score = avg_guesses
                round_best_hp = hp

        # Update global best if improved
        if round_best_hp is not None and round_best_score < best_score:
            best_score = round_best_score
            best_hp = round_best_hp
            print(
                f"[tune]   >>> New global best avg guesses={best_score:.3f} "
                f"(updated from round {round_idx+1})"
            )
        else:
            print(
                f"[tune]   No improvement this round. "
                f"Global best still {best_score:.3f}"
            )

    # Save best hyperparameters to JSON
    best_dict = best_hp.to_config_dict()
    out = {
        "best_avg_guesses": best_score,
        "hyperparams": best_dict,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n[tune] Done. Best avg guesses={best_score:.3f}")
    print(f"[tune] Best hyperparameters written to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tune_lineage_hyperparams()
