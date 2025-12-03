#!/usr/bin/env python3
import random
from collections import Counter, defaultdict
from config import CONFIG
# Mathler environment and feature computation

DIGITS = "0123456789"
OPS = "+-*/"
EXPR_LEN = 6  # normal Mathler length (6 tiles)


def safe_eval(expr: str):
    """
    Safely evaluate an arithmetic expression consisting only of digits and + - * /.
    Returns an int if the result is an integer, otherwise raises ValueError.
    """
    # basic sanity: allowed characters only
    for ch in expr:
        if ch not in DIGITS and ch not in OPS:
            raise ValueError("Invalid character in expression")
    try:
        # Use restricted eval environment
        result = eval(expr, {"__builtins__": None}, {})
    except Exception as e:
        raise ValueError(f"Eval error: {e}")
    # Only allow numeric results
    if not isinstance(result, (int, float)):
        raise ValueError("Non-numeric result")
    # Only allow integer results (no decimals)
    if isinstance(result, float):
        if not result.is_integer():
            raise ValueError("Non-integer result")
        result = int(result)
    return int(result)


def is_valid_expression(expr: str) -> bool:
    """
    Check basic syntactic + semantic validity for a Mathler expression:
    - length == EXPR_LEN
    - only digits and operators
    - first char is digit 1-9 (no leading 0)
    - no two operators in a row
    - last char is a digit
    - no digit chunk starting with 0 immediately after an operator
    - evaluates to an integer (no decimals), no division by zero, etc.
    - must have at least 2 operators
    """
    if len(expr) != EXPR_LEN:
        return False
    # allowed chars
    for ch in expr:
        if ch not in DIGITS and ch not in OPS:
            return False
    # first char: digit 1-9
    if expr[0] not in DIGITS[1:]:
        return False
    # last char must be digit
    if expr[-1] not in DIGITS:
        return False

    # at least 2 operators
    if sum(1 for ch in expr if ch in OPS) < 2:
        return False
    prev = expr[0]
    # first char is already a non-zero digit, so ok
    for i in range(1, EXPR_LEN):
        ch = expr[i]
        if prev in OPS:
            # cannot have two operators in a row
            if ch in OPS:
                return False
            # first digit after operator cannot be '0' (no leading zero)
            if ch == "0":
                return False
        # else prev is digit; ch can be digit or operator
        prev = ch

    # semantic: must evaluate and be integer
    try:
        _ = safe_eval(expr)
    except ValueError:
        return False
    return True


def generate_random_expression_string():
    """
    Generate a random syntactically valid 6-char expression:
    - first char non-zero digit
    - no two operators in a row
    - no leading zeros after operator
    - last char digit
    NOTE: does NOT guarantee any particular value; caller filters by safe_eval.
    """
    chars = []
    # first char: 1-9
    chars.append(random.choice(DIGITS[1:]))
    for i in range(1, EXPR_LEN):
        prev = chars[-1]
        if i == EXPR_LEN - 1:
            # last position must be digit (0-9), but avoid leading zero after operator
            if prev in OPS:
                chars.append(random.choice(DIGITS[1:]))
            else:
                chars.append(random.choice(DIGITS))
        else:
            if prev in OPS:
                # must be digit, and not '0' as first digit after op
                chars.append(random.choice(DIGITS[1:]))
            else:
                # choose digit or operator
                choice = random.choices(
                    population=["digit", "operator"],
                    weights=[0.6, 0.4],
                    k=1
                )[0]
                if choice == "digit":
                    chars.append(random.choice(DIGITS))
                else:
                    chars.append(random.choice(OPS))
    return "".join(chars)


def generate_random_valid_expression():
    """
    Generate a random expression that is syntactically valid according to is_valid_expression.
    """
    while True:
        expr = generate_random_expression_string()
        if is_valid_expression(expr):
            return expr


def generate_expression_for_target(target_value: int):
    """
    Generate a random valid expression whose evaluated value equals target_value.
    Only considers expressions that evaluate to an integer (safe_eval).
    NOTE: This is used when we already know a target that is reachable.
    """
    while True:
        expr = generate_random_valid_expression()
        try:
            val = safe_eval(expr)
        except ValueError:
            continue
        if val == target_value:
            return expr


def mathler_feedback(guess: str, secret: str):
    """
    Wordle-style feedback for Mathler:
    - 1 for correct char in correct position (green)
    - -1 for char present elsewhere in secret (orange)
    - 0 for char not present in secret (gray)
    NOTE: Simple version (like your earlier function); not full multiplicity logic.
    """
    feedback = []
    for i in range(EXPR_LEN):
        if guess[i] == secret[i]:
            feedback.append(1)
        elif guess[i] in secret:
            feedback.append(-1)
        else:
            feedback.append(0)
    return feedback


def compute_symbol_frequencies(candidates):
    counts = Counter("".join(candidates))
    total = sum(counts.values()) or 1
    return {ch: c / total for ch, c in counts.items()}


def compute_positional_symbol_frequencies(candidates):
    if not candidates:
        return defaultdict(lambda: [0.0] * EXPR_LEN)
    length = len(candidates[0])
    pos_counts = [Counter() for _ in range(length)]
    for expr in candidates:
        for i, ch in enumerate(expr):
            pos_counts[i][ch] += 1
    pos_freqs = []
    for i in range(length):
        total = sum(pos_counts[i].values()) or 1
        pos_freqs.append({ch: c / total for ch, c in pos_counts[i].items()})
    return pos_freqs


def expression_features_precomputed(expr, sym_freqs, pos_freqs, remaining):
    """
    Map Mathler expression features into the same feature names used by gp_core
    so we can reuse the GP engine:
    - letter_freq_sum      -> sum of symbol frequencies
    - positional_freq_sum  -> sum of positional symbol frequencies
    - unique_letters       -> number of unique symbols (digits+ops)
    - remaining_candidates -> size of candidate set
    """
    sym_sum = sum(sym_freqs.get(ch, 0.0) for ch in expr)
    pos_sum = 0.0
    for i, ch in enumerate(expr):
        pos_sum += pos_freqs[i].get(ch, 0.0)
    unique_symbols = len(set(expr))
    return {
        "letter_freq_sum": sym_sum,
        "positional_freq_sum": pos_sum,
        "unique_letters": float(unique_symbols),
        "remaining_candidates": float(remaining),
    }



def generate_initial_candidates(target_value: int):
    """
    Generate up to CONFIG['max_candidates'] expressions that:
    - are syntactically valid Mathler expressions, and
    - evaluate to target_value.
    """
    max_candidates = CONFIG["max_candidates"]
    search = ConstraintSearch(target_value, history=[])
    return search.generate_n(max_candidates)



def top_up_candidates(
    candidates,
    target_value: int,
    history,
):
    max_candidates = CONFIG["max_candidates"]
    if len(candidates) >= max_candidates:
        return candidates

    # Only top up if we dropped below a minimum threshold
    if len(candidates) >= CONFIG["min_candidates"]:
        return candidates

    existing = set(candidates)
    search = ConstraintSearch(target_value, history)
    need = max_candidates - len(candidates)
    new_exprs = search.generate_n(need, existing=existing)
    return candidates + new_exprs


class ConstraintSearch:
    """
    Grammar-guided, constraint-based generator for Mathler expressions.

    It generates only 6-character expressions that:
    - Respect the syntax rules:
        * first char 1-9
        * last char digit
        * no two operators in a row
        * no '0' immediately after an operator
        * at least 2 operators total
    - Evaluate to the given target_value
    - Are consistent with all (guess, feedback) pairs in history
    """

    def __init__(self, target_value: int, history):
        """
        history: list of (guess, feedback) pairs, where feedback is the
                 [1, -1, 0] list from mathler_feedback.
        """
        self.target_value = target_value
        self.history = history
        # DFS stack holds tuples: (prefix_string, next_position_index)
        # position index is the length of prefix (0..EXPR_LEN)
        self.stack = []
        # Initialise with all possible first characters (1-9)
        for d in DIGITS[1:]:
            self.stack.append((d, 1))

    def _consistent_with_history(self, expr: str) -> bool:
        """
        Check if a completed expression would yield the recorded feedback
        for every past guess.
        """
        for guess, fb in self.history:
            if mathler_feedback(guess, expr) != fb:
                return False
        return True

    def _extend_prefix(self, prefix: str, pos: int) -> None:
        """
        Given a partial prefix and its current length pos, push all valid
        one-character extensions onto the DFS stack.

        We enforce:
        - Last position must be a digit.
        - No two operators in a row.
        - No '0' immediately after an operator.
        """
        prev = prefix[-1]
        is_last = (pos == EXPR_LEN - 1)

        if is_last:
            # At the last position we MUST choose a digit.
            if prev in OPS:
                digits = DIGITS[1:]  # 1..9 (no leading 0 after operator)
            else:
                digits = DIGITS      # 0..9
            for d in digits:
                self.stack.append((prefix + d, pos + 1))
        else:
            if prev in OPS:
                # Immediately after an operator: must be digit 1..9
                for d in DIGITS[1:]:
                    self.stack.append((prefix + d, pos + 1))
            else:
                # prev is a digit: we can choose digit or operator
                # digits 0..9
                for d in DIGITS:
                    self.stack.append((prefix + d, pos + 1))
                # operators
                for op in OPS:
                    self.stack.append((prefix + op, pos + 1))

    def next_expression(self, seen=None):
        """
        Return the next new expression satisfying all constraints,
        or None if the search space is exhausted.

        `seen` is a set of expressions to avoid (already used).
        """
        if seen is None:
            seen = set()

        while self.stack:
            prefix, pos = self.stack.pop()

            if pos < EXPR_LEN:
                # Not full length yet – extend this prefix and continue.
                self._extend_prefix(prefix, pos)
                continue

            # pos == EXPR_LEN: full-length candidate
            expr = prefix

            # Skip duplicates
            if expr in seen:
                continue

            # Must have at least 2 operators
            if sum(1 for ch in expr if ch in OPS) < 2:
                continue

            # Semantic checks: evaluate and check target
            try:
                val = safe_eval(expr)
            except ValueError:
                continue
            if val != self.target_value:
                continue

            # Check feedback consistency
            if not self._consistent_with_history(expr):
                continue

            seen.add(expr)
            return expr

        # Exhausted search space
        return None

    def generate_n(self, n: int, existing=None):
        """
        Generate up to n NEW expressions, skipping those in `existing`.
        Returns a list of expressions.
        """
        if existing is None:
            existing = set()

        out = []
        while len(out) < n:
            expr = self.next_expression(seen=existing)
            if expr is None:
                break
            out.append(expr)
            # `next_expression` already added to seen, but we keep this
            # here for clarity that existing and seen are the same object.
        return out
