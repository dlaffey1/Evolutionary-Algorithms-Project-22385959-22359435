#!/usr/bin/env python3
"""
mathler_evolve.py

New generation of the Mathler "genome" solver.

Key design decisions:

- The GA is allowed to evolve expressions *freely*:
    * We DO NOT require intermediate individuals to evaluate to the target.
    * We only require syntactic validity + successful safe_eval.

- At the end of each per-guess GA run:
    * We pick the best expression according to a fitness function that combines:
        - closeness to the target value,
        - consistency with feedback,
        - knowledge about present / banned digits,
        - structural diversity.
    * Then we run a local REPAIR phase to find a *nearby* expression that:
        - still respects locked green positions,
        - evaluates exactly to target_value.
      This repaired expression is the actual Mathler guess.

- Mutation is BOTH:
    * character-level, feedback-aware, and knowledge-aware, AND
    * algebra-aware via token-level operations like swapping operands around
      commutative operators (+, *).

- Numeric distance to the target influences mutation:
    * When an expression is numerically close to the target, we increase the
      mutation pressure on digits that are in number tokens attached to '+'
      or '-' operators (since small +/- changes are an intuitive way to adjust
      the value).
"""

import random
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from mathler_env import (
    DIGITS,
    OPS,
    EXPR_LEN,
    safe_eval,
    mathler_feedback,
    generate_expression_for_target,
)

# ---------------------------------------------------------------------------
# Basic expression validity
# ---------------------------------------------------------------------------

def is_valid_expression_loose(expr: str) -> bool:
    """
    Check syntactic + semantic validity for a Mathler expression WITHOUT
    enforcing the "at least 2 operators" requirement.

    Requirements:
      - length == EXPR_LEN
      - chars in DIGITS or OPS
      - first char is digit 1–9 (no leading 0, no leading operator)
      - last char is digit
      - no two operators in a row
      - no "0" starting a new number right after an operator
      - safe_eval must succeed
    """
    if len(expr) != EXPR_LEN:
        return False

    for ch in expr:
        if ch not in DIGITS and ch not in OPS:
            return False

    first = expr[0]
    if first not in DIGITS or first == "0":
        return False

    if expr[-1] not in DIGITS:
        return False

    prev = expr[0]
    for i in range(1, len(expr)):
        ch = expr[i]

        if prev in OPS and ch in OPS:
            return False

        if prev in OPS:
            if ch not in DIGITS or ch == "0":
                return False

        prev = ch

    try:
        _ = safe_eval(expr)
    except ValueError:
        return False

    return True

# ---------------------------------------------------------------------------
# Random expression generators
# ---------------------------------------------------------------------------

def generate_random_valid_expression_loose(max_tries: int = 10000) -> str:
    """
    Generate a random expression that satisfies is_valid_expression_loose,
    WITHOUT caring about its numeric value.

    This is used to seed populations and as a generic fallback.
    """
    alphabet = DIGITS + OPS
    for _ in range(max_tries):
        expr = "".join(random.choice(alphabet) for _ in range(EXPR_LEN))
        if is_valid_expression_loose(expr):
            return expr
    # Extremely unlikely to fail if max_tries reasonable.
    raise RuntimeError("Failed to generate a random valid expression.")

def generate_single_expression_for_target(target_value: int) -> str:
    """
    Generate ONE valid expression whose value equals `target_value`.

    Wrapper around mathler_env.generate_expression_for_target.
    """
    return generate_expression_for_target(target_value)

# ---------------------------------------------------------------------------
# Tokenization (for algebra-aware mutations)
# ---------------------------------------------------------------------------

def tokenize_with_spans(expr: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize expression into (token, start_idx, end_idx) triples.

    - token: either a multi-digit string (e.g. "23") or a single operator (+,-,*,/).
    - start_idx, end_idx: character indices in the original expr, inclusive.
    """
    tokens: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch in DIGITS:
            start = i
            j = i + 1
            while j < len(expr) and expr[j] in DIGITS:
                j += 1
            tokens.append((expr[start:j], start, j - 1))
            i = j
        else:
            tokens.append((ch, i, i))
            i += 1
    return tokens

def tokens_to_expr(tokens: List[str]) -> str:
    """
    Join token strings back into an expression string.
    Caller should still check is_valid_expression_loose and length.
    """
    return "".join(tokens)

# ---------------------------------------------------------------------------
# Feedback convenience + knowledge
# ---------------------------------------------------------------------------

def feedback_to_string(fb: List[int]) -> str:
    mapping = {1: "G", -1: "O", 0: "."}
    return "".join(mapping.get(v, "?") for v in fb)

class Knowledge:
    """
    Accumulates what we know about the secret:

    - present_digits: digits observed as green or orange (1 or -1)
    - banned_chars: chars observed as gray (0) anywhere
    - banned_positions: for each character, positions it cannot occupy (orange)
    - locked_positions: positions that are definitely correct (green)
    """

    def __init__(self):
        self.present_digits: set[str] = set()
        self.banned_chars: set[str] = set()
        self.banned_positions: Dict[str, set[int]] = defaultdict(set)
        self.locked_positions: Dict[int, str] = {}

    def unsatisfied_digits(self) -> set[str]:
        """
        Digits that are known to be present but not locked green anywhere.
        These are especially interesting to place and probe.
        """
        locked_chars = set(self.locked_positions.values())
        return {d for d in self.present_digits if d not in locked_chars}

    def update_from_feedback(self, guess: str, fb: List[int]) -> None:
        """
        Update knowledge with a single guess + feedback.
        """
        for idx, state in enumerate(fb):
            ch = guess[idx]
            if state == 1:
                self.locked_positions[idx] = ch
                if ch in DIGITS:
                    self.present_digits.add(ch)
            elif state == 0:
                self.banned_chars.add(ch)
            elif state == -1:
                if ch in DIGITS:
                    self.present_digits.add(ch)
                self.banned_positions[ch].add(idx)

# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def per_position_mutation_base_probs(fb: List[int]) -> List[float]:
    """
    Base mutation probabilities from feedback alone.
    """
    probs: List[float] = []
    for state in fb:
        if state == 1:
            probs.append(0.0)   # green: avoid
        elif state == 0:
            probs.append(0.9)   # gray: very mutatey
        else:
            probs.append(0.4)   # orange: moderate
    return probs

def get_plusminus_digit_positions(expr: str) -> set[int]:
    """
    Return a set of char indices that belong to number tokens that are
    directly after a '+' or '-' operator.

    These positions are "sensitive" to numeric adjustments; when the
    expression is close to the target, we may want to mutate these digits
    more aggressively.
    """
    tokens = tokenize_with_spans(expr)
    special_positions: set[int] = set()

    for i, (tok, start, end) in enumerate(tokens):
        if tok in DIGITS:  # numeric token
            # if previous token is '+' or '-'
            if i > 0:
                prev_tok, _, _ = tokens[i - 1]
                if prev_tok in {"+", "-"}:
                    special_positions.update(range(start, end + 1))

    return special_positions

def mutate_expression(
    expr: str,
    fb: List[int],
    knowledge: Knowledge,
    target_value: int,
    max_attempts: int = 50,
    debug: bool = False,
) -> str:
    """
    Feedback-aware AND algebra-aware mutation.

    IMPORTANT:
    - We DO NOT require the child to evaluate to target_value.
    - We ONLY require:
        * is_valid_expression_loose(candidate)
        * safe_eval(candidate) succeeds

    Strategies (each attempt):
      1) Possibly perform a token-level commutative swap around '+' or '*':
           (num1 op num2) -> (num2 op num1)
         This preserves value but changes syntax / positions.

      2) Otherwise, perform char-level mutation guided by:
           - feedback (G/O/.),
           - knowledge (present/banned digits, locked positions),
           - numeric distance to target:
               * If close to target, boost mutation probabilities on digits
                 after '+' or '-'.

    If all max_attempts fail to produce a valid candidate, we fall back to a
    random valid expression (not necessarily close to the parent).
    """
    try:
        parent_val = safe_eval(expr)
        diff = target_value - parent_val
        abs_diff = abs(diff)
    except ValueError:
        parent_val = None
        diff = 0
        abs_diff = None

    # Base per-position weights from feedback
    base_probs = per_position_mutation_base_probs(fb)
    # Positions associated with digits after '+' or '-'
    plusminus_positions = get_plusminus_digit_positions(expr)

    tokens_with_spans = tokenize_with_spans(expr)
    tokens_only = [t for t, _, _ in tokens_with_spans]

    for _ in range(max_attempts):
        mutation_kind = "unknown"

        # Choice: token-level (commutative) or char-level
        do_commutative = random.random() < 0.3  # 30% chance

        if do_commutative:
            # Try a commutative swap around '+' or '*'
            # pattern: [num, op, num]
            indices = [i for i, tok in enumerate(tokens_only) if tok in {"+", "*"}]
            random.shuffle(indices)
            done = False
            for iop in indices:
                if iop == 0 or iop == len(tokens_only) - 1:
                    continue
                left_tok = tokens_only[iop - 1]
                right_tok = tokens_only[iop + 1]
                # must be numeric tokens to swap
                if not left_tok[0].isdigit() or not right_tok[0].isdigit():
                    continue
                new_tokens = tokens_only[:]
                new_tokens[iop - 1], new_tokens[iop + 1] = new_tokens[iop + 1], new_tokens[iop - 1]
                candidate = tokens_to_expr(new_tokens)
                if len(candidate) != EXPR_LEN:
                    continue
                if not is_valid_expression_loose(candidate):
                    continue
                try:
                    _ = safe_eval(candidate)
                except ValueError:
                    continue
                mutation_kind = f"commutative_swap(op={tokens_only[iop]}, idx={iop})"
                if debug:
                    print(f"[mutate] {mutation_kind}: {expr} -> {candidate}")
                return candidate
            # if no suitable op found, fall through to char-level
            # (we don't treat this as a failure, just try another strategy)

        # --- Char-level mutation ---
        candidate_list = list(expr)

        # Build per-position probabilities, adjusted by numeric closeness
        probs = base_probs[:]
        if abs_diff is not None and abs_diff <= 50:
            # close to target: emphasise digits in plus/minus tokens
            for idx in plusminus_positions:
                probs[idx] *= 2.0

        if all(p == 0 for p in probs):
            probs = [1.0] * EXPR_LEN

        idx_choices = list(range(EXPR_LEN))
        pos = random.choices(idx_choices, weights=probs, k=1)[0]

        if pos in knowledge.locked_positions:
            # don't touch locked greens; try again
            continue

        old_ch = candidate_list[pos]

        def choose_digit_for_position(position: int) -> str:
            """
            Choose a digit with bias:
            - Prefer unsatisfied_digits (present but not locked),
            - Otherwise any digit not banned.
            - If we know the sign of (target - value) and this position is
              in plusminus_positions, we can nudge digits up or down.
            """
            unsat = knowledge.unsatisfied_digits()

            pool = [
                d for d in unsat
                if d not in knowledge.banned_chars
                and position not in knowledge.banned_positions.get(d, set())
            ]
            if not pool:
                pool = [
                    d for d in DIGITS
                    if d not in knowledge.banned_chars
                    and position not in knowledge.banned_positions.get(d, set())
                ]
            if not pool:
                pool = list(DIGITS)

            # If we are close and this is a plusminus digit, bias by diff sign
            if abs_diff is not None and abs_diff <= 50 and position in plusminus_positions:
                # sort digits ascending or descending depending on needed direction
                if diff > 0:
                    # need value to increase -> prefer larger digits
                    pool_sorted = sorted(pool, key=int)
                else:
                    # need to decrease -> prefer smaller digits
                    pool_sorted = sorted(pool, key=int, reverse=True)
                # choose biased digit from top few
                top_k = min(3, len(pool_sorted))
                return random.choice(pool_sorted[:top_k])

            return random.choice(pool)

        # Decide new char at position
        if pos == 0:
            # first char: digit 1–9, not banned
            choices = [d for d in DIGITS if d != "0" and d not in knowledge.banned_chars]
            if not choices:
                choices = [d for d in DIGITS if d != "0"]
            new_ch = random.choice(choices)
            candidate_list[pos] = new_ch
            mutation_kind = f"digit_first(idx=0, {old_ch}->{new_ch})"

        elif pos == EXPR_LEN - 1:
            # last char must be digit
            new_ch = choose_digit_for_position(pos)
            candidate_list[pos] = new_ch
            mutation_kind = f"digit_last(idx={pos}, {old_ch}->{new_ch})"

        else:
            # interior position: digit or operator
            banned_here = {
                ch for ch, positions in knowledge.banned_positions.items()
                if pos in positions
            }
            # Choose digit vs operator
            if random.random() < 0.65:
                new_ch = choose_digit_for_position(pos)
                if new_ch in banned_here:
                    fallback = [
                        d for d in DIGITS
                        if d not in knowledge.banned_chars and d not in banned_here
                    ]
                    if fallback:
                        new_ch = random.choice(fallback)
                candidate_list[pos] = new_ch
                mutation_kind = f"digit(idx={pos}, {old_ch}->{new_ch})"
            else:
                ops = [op for op in OPS if op not in knowledge.banned_chars]
                if not ops:
                    ops = list(OPS)
                new_ch = random.choice(ops)
                candidate_list[pos] = new_ch
                mutation_kind = f"op(idx={pos}, {old_ch}->{new_ch})"

        candidate = "".join(candidate_list)
        if not is_valid_expression_loose(candidate):
            continue

        try:
            _ = safe_eval(candidate)
        except ValueError:
            continue

        if debug:
            print(f"[mutate] {mutation_kind}: {expr} -> {candidate}")
        return candidate

    # Fallback: random valid expression (value can be anything)
    candidate = generate_random_valid_expression_loose()
    if debug:
        print(f"[mutate] fallback_random: {expr} -> {candidate}")
    return candidate

# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def fitness_expr(
    expr: str,
    knowledge: Knowledge,
    history: List[Tuple[str, List[int]]],
    target_value: int,
) -> float:
    """
    Heuristic fitness to rank expressions during GA.

    Components:
      1) Numeric closeness to target:
         - smaller |target - value(expr)| is better.
      2) Reward unsatisfied_digits (present but not yet locked).
      3) Penalise banned_chars.
      4) Reward structural diversity (unique chars).
      5) Feedback compatibility with history:
         - For each past guess, compute hypothetical feedback if `expr`
           were the secret and see how similar it is to the actual feedback.

    This is NOT a perfect probability model; it's a heuristic scoring function
    to drive evolution.
    """
    score = 0.0

    # 1) Numeric closeness
    try:
        val = safe_eval(expr)
        diff = abs(target_value - val)
        # Map diff to a 0..10-ish closeness score
        #   diff = 0   -> closeness = 10
        #   diff >=100 -> closeness ~ 0
        closeness = max(0.0, 10.0 - min(diff, 100) / 10.0)
        score += 3.0 * closeness
    except ValueError:
        score -= 10.0
        val = None

    # 2) Unsatisfied digits bonus
    unsat = knowledge.unsatisfied_digits()
    present_bonus = sum(1 for ch in expr if ch in unsat)
    score += 2.0 * present_bonus

    # 3) Banned chars penalty
    banned_penalty = sum(1 for ch in expr if ch in knowledge.banned_chars)
    score -= 3.0 * banned_penalty

    # 4) Diversity bonus
    diversity = len(set(expr))
    score += 0.3 * diversity

    # 5) Feedback compatibility
    comp_bonus = 0.0
    for past_guess, true_fb in history:
        hypothetical_fb = mathler_feedback(past_guess, expr)
        match = sum(1 for a, b in zip(true_fb, hypothetical_fb) if a == b)
        comp_bonus += match / EXPR_LEN
    score += 1.5 * comp_bonus

    return score

# ---------------------------------------------------------------------------
# Repair: project GA winner to a target-correct guess
# ---------------------------------------------------------------------------

def repair_to_target(
    expr: str,
    knowledge: Knowledge,
    target_value: int,
    fb: List[int],
    max_steps: int = 200,
    debug: bool = False,
) -> str:
    """
    Starting from a GA-best expression `expr`, try to find a nearby expression
    that:
      - respects locked_positions (greens),
      - is syntactically valid,
      - evaluates exactly to target_value.

    We do this by performing a random walk in expression space using the same
    mutate_expression operator, but applying a *global* constraint that
    we only accept expressions whose value == target_value.

    If this fails after max_steps, we fall back to:
      1) generating target-correct expressions and trying to enforce green
         positions, then
      2) a last-ditch random target-correct expression if needed.
    """
    # Start from expr but make sure locked green chars are in place
    current_list = list(expr)
    for idx, ch in knowledge.locked_positions.items():
        current_list[idx] = ch
    current = "".join(current_list)

    # If it's already perfect, return immediately
    try:
        if is_valid_expression_loose(current) and safe_eval(current) == target_value:
            if debug:
                print(f"[repair] starting expr already target-correct: {current}")
            return current
    except ValueError:
        pass

    # Local search
    for step in range(max_steps):
        # Check current candidate
        try:
            if is_valid_expression_loose(current) and safe_eval(current) == target_value:
                if debug:
                    print(f"[repair] success after {step} steps: {expr} -> {current}")
                return current
        except ValueError:
            pass

        # Random-walk step: apply free mutation and move to that point
        candidate = mutate_expression(current, fb, knowledge, target_value, debug=debug)
        current = candidate

    # Fallback 1: generate target-correct expression and try to enforce locked greens
    for _ in range(100):
        base = generate_single_expression_for_target(target_value)
        base_list = list(base)
        for idx, ch in knowledge.locked_positions.items():
            base_list[idx] = ch
        candidate = "".join(base_list)
        try:
            if is_valid_expression_loose(candidate) and safe_eval(candidate) == target_value:
                if debug:
                    print(f"[repair] fallback with locked positions: {candidate}")
                return candidate
        except ValueError:
            continue

    # Fallback 2: final random target-correct expression
    final = generate_single_expression_for_target(target_value)
    if debug:
        print(f"[repair] ultimate fallback: {final}")
    return final

# ---------------------------------------------------------------------------
# GA per guess
# ---------------------------------------------------------------------------

def evolve_from_feedback(
    current_guess: str,
    fb: List[int],
    knowledge: Knowledge,
    target_value: int,
    history: List[Tuple[str, List[int]]],
    pop_size: int,
    gens_per_guess: int,
    verbose: bool = False,
) -> str:
    """
    Run a GA to evolve expressions given the latest feedback.

    Steps:
      - Seed population with:
          * current_guess (if valid),
          * random valid expressions.
      - For gens_per_guess generations:
          * compute fitness_expr for each.
          * keep top ~25% as elites.
          * refill population by mutating elites (no target constraint).
      - After GA, choose best-by-fitness expression `best_expr`.
      - Call repair_to_target(best_expr, ...) to obtain a target-correct guess.
    """
    population: List[str] = []

    # Ensure a valid starting individual
    if not is_valid_expression_loose(current_guess):
        if verbose:
            print("[evolve] current_guess invalid; replacing with random valid expression.")
        current_guess = generate_random_valid_expression_loose()

    population.append(current_guess)

    # Initial population
    while len(population) < pop_size:
        if random.random() < 0.7:
            child = mutate_expression(current_guess, fb, knowledge, target_value, debug=verbose)
        else:
            child = generate_random_valid_expression_loose()
        population.append(child)

    if verbose:
        print(f"[evolve] Initial population size: {len(population)}")

    # GA loop
    for gen in range(gens_per_guess):
        scored: List[Tuple[float, str]] = [
            (fitness_expr(expr, knowledge, history, target_value), expr)
            for expr in population
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        elites = [expr for _, expr in scored[: max(2, pop_size // 4)]]

        if verbose and (gen == 0 or gen == gens_per_guess - 1):
            best_score, best_expr = scored[0]
            print(
                f"[evolve] Gen {gen+1}/{gens_per_guess} | "
                f"best fitness = {best_score:.3f} | best expr = {best_expr}"
            )

        new_pop: List[str] = elites[:]
        while len(new_pop) < pop_size:
            parent = random.choice(elites)
            child = mutate_expression(parent, fb, knowledge, target_value, debug=verbose)
            new_pop.append(child)

        population = new_pop

    # After GA, pick best and repair it to a target-correct guess
    final_scored: List[Tuple[float, str]] = [
        (fitness_expr(expr, knowledge, history, target_value), expr)
        for expr in population
    ]
    final_scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best_expr = final_scored[0]

    if verbose:
        print(f"[evolve] GA best before repair: {best_expr} (fitness={best_score:.3f})")

    repaired = repair_to_target(best_expr, knowledge, target_value, fb, debug=verbose)
    if verbose:
        print(f"[evolve] Repaired GA best to target-correct guess: {repaired}")
    return repaired

# ---------------------------------------------------------------------------
# High-level solver class
# ---------------------------------------------------------------------------

class MathlerGenomeSolver:
    """
    Encapsulated solver that evolves a 6-char expression genome between guesses.
    """

    def __init__(
        self,
        target_value: int,
        max_guesses: int = 6,
        inner_pop_size: int = 80,
        inner_gens_per_guess: int = 30,
        verbose: bool = True,
    ):
        self.target_value = target_value
        self.max_guesses = max_guesses
        self.inner_pop_size = inner_pop_size
        self.inner_gens_per_guess = inner_gens_per_guess
        self.verbose = verbose

        self.history: List[Tuple[str, List[int]]] = []
        self.knowledge = Knowledge()

    def first_guess(self, initial_guess: Optional[str] = None) -> str:
        """
        Choose the first guess:

        - If user provides an initial_guess, we only accept it if:
            * len == EXPR_LEN
            * is_valid_expression_loose
            * safe_eval(initial_guess) == target_value
          Otherwise we generate a random target-correct expression.

        - If no initial_guess, we generate a random target-correct expression.
        """
        if initial_guess:
            if self.verbose:
                print(f"[solver] Using user-provided initial guess: {initial_guess}")
            if len(initial_guess) == EXPR_LEN and is_valid_expression_loose(initial_guess):
                try:
                    if safe_eval(initial_guess) == self.target_value:
                        return initial_guess
                except ValueError:
                    pass
            if self.verbose:
                print("[solver] Initial guess invalid for target; generating instead.")

        guess = generate_single_expression_for_target(self.target_value)
        if self.verbose:
            print(f"[solver] Generated first guess: {guess}")
        return guess

    def register_feedback(self, guess: str, fb: List[int]) -> None:
        self.history.append((guess, fb))
        self.knowledge.update_from_feedback(guess, fb)

        if self.verbose:
            print(
                f"[solver] Feedback for guess {guess}: {fb} ({feedback_to_string(fb)})"
            )
            print(
                f"[solver] Knowledge -> present_digits={sorted(self.knowledge.present_digits)}, "
                f"banned_chars={sorted(self.knowledge.banned_chars)}, "
                f"locked_positions={sorted(self.knowledge.locked_positions.items())}"
            )

    def next_guess(self, current_guess: str) -> str:
        """
        Use GA + repair to produce the next guess.
        """
        if not self.history:
            if self.verbose:
                print("[solver] No history yet; returning current_guess.")
            return current_guess

        last_fb = self.history[-1][1]
        if self.verbose:
            print(
                f"[solver] Evolving from {current_guess} with feedback "
                f"{last_fb} ({feedback_to_string(last_fb)})"
            )

        return evolve_from_feedback(
            current_guess=current_guess,
            fb=last_fb,
            knowledge=self.knowledge,
            target_value=self.target_value,
            history=self.history,
            pop_size=self.inner_pop_size,
            gens_per_guess=self.inner_gens_per_guess,
            verbose=self.verbose,
        )

# ---------------------------------------------------------------------------
# Manual test harness
# ---------------------------------------------------------------------------

def run_manual_test():
    """
    Interactive test mode.

    - You can fix a secret expression or just give a target value.
    - You can provide an initial guess (must evaluate to target), or let the
      solver pick one.
    - The solver is considered successful only when feedback is all greens.
    """
    print("=== Mathler Genome Solver: Manual Test Mode ===")
    print("All expressions must be 6 characters of digits and + - * /.\n")

    secret_expr = input("Enter a secret expression (6 chars), or press Enter to skip: ").strip()

    if secret_expr:
        try:
            if len(secret_expr) != EXPR_LEN:
                raise ValueError(f"Secret must be exactly {EXPR_LEN} characters.")
            target_value = safe_eval(secret_expr)
        except ValueError as e:
            print(f"Error: invalid secret expression: {e}")
            return
        print(f"Using secret {secret_expr} with target value {target_value}.")
    else:
        target_str = input("No secret given. Enter a target integer value: ").strip()
        try:
            target_value = int(target_str)
        except ValueError:
            print("Error: target value must be an integer.")
            return
        secret_expr = generate_single_expression_for_target(target_value)
        print(
            f"Generated random secret {secret_expr} for target value {target_value}.\n"
            "(Shown here for debugging; in a real game the player wouldn't see this.)"
        )

    initial_guess = input(
        "Enter an initial guess expression (or press Enter to let the solver choose): "
    ).strip()
    if not initial_guess:
        initial_guess = None

    solver = MathlerGenomeSolver(
        target_value=target_value,
        max_guesses=6,
        inner_pop_size=60,
        inner_gens_per_guess=25,
        verbose=False,
    )

    guess = solver.first_guess(initial_guess)

    print("\n=== Starting game ===")
    for attempt in range(1, solver.max_guesses + 1):
        try:
            fb = mathler_feedback(guess, secret_expr)
        except Exception as e:
            print(f"[main] Error computing feedback: {e}")
            return

        print(
            f"[main] Attempt {attempt}: guess = {guess}, "
            f"feedback = {fb} ({feedback_to_string(fb)})"
        )

        solver.register_feedback(guess, fb)

        if all(v == 1 for v in fb):
            print(f"[main] Solved! Secret {secret_expr} found in {attempt} guesses.\n")
            return

        if attempt == solver.max_guesses:
            break

        guess = solver.next_guess(guess)

    print(
        f"[main] Failed to find the exact secret {secret_expr} "
        f"within {solver.max_guesses} guesses."
    )

if __name__ == "__main__":
    run_manual_test()
