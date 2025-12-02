#!/usr/bin/env python3
import math
from multiprocessing import Pool, cpu_count

from config import CONFIG
from mathler_env import (
    EXPR_LEN,
    safe_eval,
    mathler_feedback,
    compute_symbol_frequencies,
    compute_positional_symbol_frequencies,
    expression_features_precomputed,
    generate_initial_candidates,
    top_up_candidates,
)

MAX_GUESSES = CONFIG["max_guesses"]
FAIL_PENALTY = CONFIG["fail_penalty"]


def play_game_with_individual(individual, secret_expr: str, target_value: int, verbose: bool = False):
    """
    Play a single Mathler game with a given GP individual as the strategy.
    Returns the number of guesses used, or FAIL_PENALTY if it fails.
    """
    # Initial candidate sample for this target (on the fly)
    candidates = generate_initial_candidates(target_value)
    # Ensure the true secret is in the candidate set if it matches the target
    try:
        if safe_eval(secret_expr) == target_value and secret_expr not in candidates:
            candidates.append(secret_expr)
    except Exception:
        pass

    history = []  # list of (guess, feedback)

    for guess_num in range(1, MAX_GUESSES + 1):
        if not candidates:
            # If we somehow run out, try topping up once
            candidates = generate_initial_candidates(target_value)
            if not candidates:
                return FAIL_PENALTY

        # Precompute frequencies once per turn
        sym_freqs = compute_symbol_frequencies(candidates)
        pos_freqs = compute_positional_symbol_frequencies(candidates)
        remaining = len(candidates)

        best_expr = None
        best_score = -math.inf

        for expr in candidates:
            feats = expression_features_precomputed(expr, sym_freqs, pos_freqs, remaining)
            ctx = {"features": feats}
            score = individual.tree.eval(ctx)
            if score > best_score:
                best_score = score
                best_expr = expr

        guess = best_expr
        fb = mathler_feedback(guess, secret_expr)
        history.append((guess, fb))

        if verbose:
            print(f"Guess {guess_num}: {guess}  Feedback: {fb}")

        if guess == secret_expr:
            return float(guess_num)

        # Filter candidates consistent with this feedback
        new_candidates = [
            expr for expr in candidates
            if mathler_feedback(guess, expr) == fb
        ]
        candidates = new_candidates

        # Top up if we fell below some threshold (still on-the-fly sampling)
        if len(candidates) < 50:
            candidates = top_up_candidates(candidates, target_value, history)

    # failed in MAX_GUESSES
    return FAIL_PENALTY


# ===== Helpers for parallel evaluation =====

def _individual_fitness_value(individual, secrets):
    """
    Pure function: compute and return fitness value for an individual,
    without mutating it. Safe to use in multiprocessing.
    """
    total_guesses = 0.0
    for secret_expr, target_value in secrets:
        g = play_game_with_individual(individual, secret_expr, target_value, verbose=False)
        total_guesses += g
    return total_guesses / len(secrets)


def evaluate_population_mathler(pop, secrets):
    """
    Evaluate the given population on the given list of (secret_expr, target_value)
    USING MULTIPLE CORES via multiprocessing.Pool.

    This keeps the algorithm the same (same fitness definition),
    but runs individuals in parallel.
    """
    # Prepare arguments: same secrets for each individual
    args = [(ind, secrets) for ind in pop]

    # Use all available CPU cores
    with Pool(cpu_count()) as pool:
        fitnesses = pool.starmap(_individual_fitness_value, args)

    # Assign fitness back to individuals
    for ind, fit in zip(pop, fitnesses):
        ind.fitness = fit
