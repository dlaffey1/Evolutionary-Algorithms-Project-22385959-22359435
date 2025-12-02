#!/usr/bin/env python3
import math

from config import CONFIG
from online_vocab import ALLOWED_GUESSES
from wordle_env import (
    wordle_feedback,
    filter_candidates,
    compute_letter_frequencies,
    compute_positional_frequencies,
    word_features_precomputed,
)

# =========================
# Fitness evaluation
# =========================

MAX_GUESSES = CONFIG["max_guesses"]
FAIL_PENALTY = CONFIG["fail_penalty"]


def play_game_with_individual(individual, secret, verbose=False):
    """
    Play a single Wordle game with a given GP individual as the strategy.
    Returns the number of guesses used, or FAIL_PENALTY if it fails.
    """
    candidates = ALLOWED_GUESSES[:]
    for guess_num in range(1, MAX_GUESSES + 1):
        # precompute frequencies ONCE for this guess
        letter_freqs = compute_letter_frequencies(candidates)
        pos_freqs = compute_positional_frequencies(candidates)
        remaining = len(candidates)

        best_word = None
        best_score = -math.inf
        for w in candidates:
            feats = word_features_precomputed(w, letter_freqs, pos_freqs, remaining)
            ctx = {"features": feats}
            score = individual.tree.eval(ctx)
            if score > best_score:
                best_score = score
                best_word = w

        guess = best_word
        fb = wordle_feedback(guess, secret)
        if verbose:
            print(f"Guess {guess_num}: {guess}  Feedback: {fb}")

        if guess == secret:
            return float(guess_num)

        candidates = filter_candidates(candidates, guess, fb)

        if not candidates:
            # no options left, fail hard
            return FAIL_PENALTY

    # failed in MAX_GUESSES
    return FAIL_PENALTY


def evaluate_individual(individual, secrets):
    total_guesses = 0.0
    for s in secrets:
        g = play_game_with_individual(individual, s, verbose=False)
        total_guesses += g
    individual.fitness = total_guesses / len(secrets)


def evaluate_population(pop, secrets):
    for ind in pop:
        if ind.fitness is None:
            evaluate_individual(ind, secrets)
