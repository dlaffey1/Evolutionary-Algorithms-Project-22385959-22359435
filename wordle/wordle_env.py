#!/usr/bin/env python3
from collections import Counter, defaultdict

# =========================
# Wordle environment + features
# =========================


def wordle_feedback(guess: str, secret: str) -> str:
    """
    Return a simple feedback pattern:
    'G' = green (correct letter, correct position)
    'Y' = yellow (letter in word, wrong position)
    'B' = black/grey (letter not in word)
    NOTE: simplified duplicate-handling, sufficient for this project.
    """
    result = ["B"] * len(secret)
    # first pass: greens
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            result[i] = "G"
    # second pass: yellows (simplified)
    for i, g in enumerate(guess):
        if result[i] == "G":
            continue
        if g in secret:
            result[i] = "Y"
    return "".join(result)


def filter_candidates(candidates, guess, feedback):
    """Keep only candidates that would produce the same feedback."""
    return [
        w for w in candidates
        if wordle_feedback(guess, w) == feedback
    ]


def compute_letter_frequencies(candidates):
    counts = Counter("".join(candidates))
    total = sum(counts.values()) or 1
    freqs = {ch: c / total for ch, c in counts.items()}
    return freqs


def compute_positional_frequencies(candidates):
    if not candidates:
        return defaultdict(lambda: [0.0] * 5)
    length = len(candidates[0])
    pos_counts = [Counter() for _ in range(length)]
    for w in candidates:
        for i, ch in enumerate(w):
            pos_counts[i][ch] += 1
    pos_freqs = []
    for i in range(length):
        total = sum(pos_counts[i].values()) or 1
        pos_freqs.append(
            {ch: c / total for ch, c in pos_counts[i].items()}
        )
    return pos_freqs


def word_features(word, candidates):
    """
    Compute the feature vector used by the GP tree for a candidate word.
    """
    letter_freqs = compute_letter_frequencies(candidates)
    pos_freqs = compute_positional_frequencies(candidates)
    # letter frequency sum
    lsum = sum(letter_freqs.get(ch, 0.0) for ch in word)
    # positional frequency sum
    psum = 0.0
    for i, ch in enumerate(word):
        psum += pos_freqs[i].get(ch, 0.0)
    unique_letters = len(set(word))
    remaining = len(candidates)
    return {
        "letter_freq_sum": lsum,
        "positional_freq_sum": psum,
        "unique_letters": float(unique_letters),
        "remaining_candidates": float(remaining),
    }
