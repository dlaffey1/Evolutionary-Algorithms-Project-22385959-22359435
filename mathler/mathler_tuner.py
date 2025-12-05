#!/usr/bin/env python3
"""
mathler_tuner.py

Hyperparameter tuner for mathler_evolve.py.

This script:
  - Imports mathler_evolve as `me`.
  - Monkey-patches me.fitness_expr() so that its weights are configurable.
  - Runs many simulated Mathler games with random secrets for each candidate
    hyperparameter set.
  - Measures performance as "average guesses to solve" (with penalties for fails).
  - Records the best hyperparameters found and saves them to best_hyperparams.json.

You do NOT need to modify mathler_evolve.py for this script to work,
as long as it exports:
  - MathlerGenomeSolver
  - generate_single_expression_for_target
  - safe_eval
  - mathler_feedback
"""

import json
import random
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple

import mathler_evolver as me  # your solver module


# ----------------------------- Hyperparameters ----------------------------- #

@dataclass
class HyperParams:
    # Fitness weights
    w_closeness: float = 3.0           # weight for numeric closeness term
    closeness_max_score: float = 10.0  # max closeness score when diff == 0
    closeness_diff_clip: float = 100.0 # max diff before score bottoms out

    w_unsatisfied: float = 2.0         # weight for unsatisfied_digits
    w_banned: float = -3.0             # weight for banned chars
    w_diversity: float = 0.3           # weight for structural diversity
    w_feedback: float = 1.5            # weight for feedback compatibility

    # GA runtime parameters (we’ll use these when constructing the solver)
    pop_size: int = 60
    gens_per_guess: int = 25


# --------------------- Monkey-patch fitness function ---------------------- #

def apply_hyperparams(params: HyperParams):
    """
    Monkey-patch me.fitness_expr to use the given weights.

    We keep the *structure* of the original fitness function in mathler_evolve,
    but replace the hard-coded constants with HyperParams fields.
    """
    def fitness_expr_with_params(
        expr: str,
        knowledge: "me.Knowledge",
        history: List[Tuple[str, List[int]]],
        target_value: int,
    ) -> float:
        score = 0.0

        # 1) Numeric closeness
        try:
            val = me.safe_eval(expr)
            diff = abs(target_value - val)
            # Map diff to [0, closeness_max_score]
            d_clipped = min(diff, params.closeness_diff_clip)
            closeness = max(
                0.0,
                params.closeness_max_score - d_clipped / (params.closeness_diff_clip / params.closeness_max_score)
            )
            score += params.w_closeness * closeness
        except ValueError:
            score -= 10.0  # harshly penalise invalid evals
            val = None

        # 2) Unsatisfied digits bonus
        unsat = knowledge.unsatisfied_digits()
        present_bonus = sum(1 for ch in expr if ch in unsat)
        score += params.w_unsatisfied * present_bonus

        # 3) Banned chars penalty
        banned_penalty = sum(1 for ch in expr if ch in knowledge.banned_chars)
        score += params.w_banned * banned_penalty

        # 4) Diversity bonus
        diversity = len(set(expr))
        score += params.w_diversity * diversity

        # 5) Feedback compatibility
        comp_bonus = 0.0
        for past_guess, true_fb in history:
            hypothetical_fb = me.mathler_feedback(past_guess, expr)
            match = sum(1 for a, b in zip(true_fb, hypothetical_fb) if a == b)
            comp_bonus += match / me.EXPR_LEN
        score += params.w_feedback * comp_bonus

        return score

    # Actually patch the module-level function
    me.fitness_expr = fitness_expr_with_params


# ----------------------------- Evaluation --------------------------------- #

def simulate_single_game(secret_expr: str, target_value: int, params: HyperParams, max_guesses: int = 6) -> int:
    """
    Simulate a single game of Mathler using the solver from mathler_evolve.

    Returns:
      - number of guesses used if solved within max_guesses,
      - max_guesses + 2 as a penalty score if failed.
    """
    solver = me.MathlerGenomeSolver(
        target_value=target_value,
        max_guesses=max_guesses,
        inner_pop_size=params.pop_size,
        inner_gens_per_guess=params.gens_per_guess,
        verbose=False,  # keep evaluation quiet
    )

    guess = solver.first_guess()

    for attempt in range(1, max_guesses + 1):
        fb = me.mathler_feedback(guess, secret_expr)
        solver.register_feedback(guess, fb)
        if all(v == 1 for v in fb):
            return attempt
        guess = solver.next_guess(guess)

    # If we get here, we failed to solve in max_guesses
    return max_guesses + 2


def evaluate_params(params: HyperParams, num_games: int = 30, max_guesses: int = 6, seed: int = 0) -> float:
    """
    Evaluate a given hyperparameter set by running num_games simulated games
    on random secrets.

    Score = average guesses, with penalties for failures.

    Lower is better.
    """
    random.seed(seed)
    apply_hyperparams(params)

    total_score = 0.0

    for game_idx in range(num_games):
        # Sample a random target by sampling a secret expression first
        # This guarantees the target is reachable
        # We choose random integer targets in some range, via the generator:
        target_val = random.randint(5, 200)
        secret_expr = me.generate_single_expression_for_target(target_val)
        # In case generator does something fancy, recompute target from secret
        target_value = me.safe_eval(secret_expr)

        guesses_used = simulate_single_game(secret_expr, target_value, params, max_guesses=max_guesses)
        total_score += guesses_used

    avg_score = total_score / num_games
    return avg_score


# ----------------------------- Search loop -------------------------------- #

def random_hyperparams() -> HyperParams:
    """
    Sample a random HyperParams configuration from reasonable ranges.
    Adjust these ranges as you learn more about what works.
    """
    return HyperParams(
        w_closeness=random.uniform(1.0, 6.0),
        closeness_max_score=random.uniform(5.0, 15.0),
        closeness_diff_clip=random.uniform(50.0, 200.0),

        w_unsatisfied=random.uniform(0.5, 4.0),
        w_banned=random.uniform(-6.0, -1.0),
        w_diversity=random.uniform(0.0, 1.5),
        w_feedback=random.uniform(0.0, 3.0),

        pop_size=random.randint(40, 100),
        gens_per_guess=random.randint(10, 40),
    )


def tune_hyperparams(
    iterations: int = 50,
    num_games_per_eval: int = 20,
    max_guesses: int = 6,
    seed: int = 0,
) -> Tuple[HyperParams, float]:
    """
    Simple random search over hyperparameters.

    For each iteration:
      - sample a random HyperParams,
      - evaluate it on num_games_per_eval random games,
      - keep track of the best (lowest avg guesses).

    Returns:
      (best_params, best_score)
    """
    random.seed(seed)
    best_params = None
    best_score = float("inf")

    start_time = time.time()

    for i in range(iterations):
        params = random_hyperparams()
        score = evaluate_params(params, num_games=num_games_per_eval, max_guesses=max_guesses, seed=seed + i)

        if score < best_score:
            best_score = score
            best_params = params

        print(
            f"[tune] Iteration {i+1}/{iterations}: "
            f"score={score:.3f}, best={best_score:.3f}"
        )

    elapsed = time.time() - start_time
    print(f"[tune] Done in {elapsed:.1f}s. Best avg guesses = {best_score:.3f}")

    return best_params, best_score


# ----------------------------- Main entry --------------------------------- #

def save_best_params(params: HyperParams, score: float, filename: str = "best_hyperparams.json") -> None:
    """
    Save best hyperparameters and their score to a JSON file.
    """
    data = asdict(params)
    data["avg_guesses"] = score
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[tune] Saved best hyperparams to {filename}")


def main():
    iterations = 50           # increase for bigger search
    num_games_per_eval = 20   # increase for more reliable scores
    max_guesses = 6
    seed = 42

    best_params, best_score = tune_hyperparams(
        iterations=iterations,
        num_games_per_eval=num_games_per_eval,
        max_guesses=max_guesses,
        seed=seed,
    )

    if best_params is not None:
        print("\nBest hyperparameters found:")
        print(best_params)
        print(f"Average guesses: {best_score:.3f}")
        save_best_params(best_params, best_score)
    else:
        print("[tune] No best parameters found (this should not happen).")


if __name__ == "__main__":
    main()
