#!/usr/bin/env python3
import requests

# =========================
# Online Wordle vocabulary
# =========================

# GitHub repo: ed-fish/wordle-vocab, vocab.json
# Contains a "vocab" object (~2k answer words) and an "other" object (~10k allowed guesses).
WORD_VOCAB_URL = "https://raw.githubusercontent.com/ed-fish/wordle-vocab/main/vocab.json"


def load_word_lists():
    """
    Download Wordle vocabulary from the online JSON.
    Returns (secret_words, allowed_guesses).
    If download fails, falls back to a tiny local list (so the script still runs).
    """
    try:
        print(f"Downloading Wordle vocab from {WORD_VOCAB_URL} ...")
        resp = requests.get(WORD_VOCAB_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        vocab = data["vocab"]          # main answer list
        other = data.get("other", [])  # extra acceptable guesses

        # Secrets = vocab (like real Wordle)
        secret_words = list(vocab)

        # Allowed guesses = vocab + other (dedup, keep order)
        all_words = list(dict.fromkeys(list(vocab) + list(other)))

        print(f"Loaded {len(secret_words)} secret words and {len(all_words)} allowed guesses from online source.")
        return secret_words, all_words
    except Exception as e:
        print("WARNING: failed to load online word list, using tiny fallback:", e)
        fallback = [
            "cigar", "rebut", "sissy", "humph", "awake",
            "blush", "focal", "evade", "naval", "serve",
        ]
        return fallback, fallback[:]


SECRET_WORDS, ALLOWED_GUESSES = load_word_lists()

# To keep fitness evaluation fast, we train on a random subset of secrets:
TRAIN_SAMPLE_SIZE = min(200, len(SECRET_WORDS))
