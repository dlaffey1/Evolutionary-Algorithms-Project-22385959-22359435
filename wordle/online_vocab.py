#!/usr/bin/env python3
import requests
from config import CONFIG

# =========================
# Online Wordle vocabulary
# =========================

# stuartpb/wordles:
WORD_ANSWERS_URL = "https://raw.githubusercontent.com/stuartpb/wordles/master/wordles.json"
WORD_GUESSES_URL = "https://raw.githubusercontent.com/stuartpb/wordles/master/nonwordles.json"


def load_word_lists():
    """
    Download Wordle vocabulary from online JSON.
    Returns (secret_words, allowed_guesses).
    If download fails, falls back to a tiny local list.
    """
    try:
        print("Downloading Wordle vocab from stuartpb/wordles ...")
        ans_resp = requests.get(WORD_ANSWERS_URL, timeout=10)
        ans_resp.raise_for_status()
        guess_resp = requests.get(WORD_GUESSES_URL, timeout=10)
        guess_resp.raise_for_status()

        # These are just lists of strings
        answers = ans_resp.json()
        other = guess_resp.json()

        secret_words = list(answers)
        all_words = list(dict.fromkeys(list(answers) + list(other)))

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

TRAIN_SAMPLE_SIZE = min(CONFIG["train_sample_size"], len(SECRET_WORDS))
