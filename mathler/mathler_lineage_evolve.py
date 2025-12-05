#!/usr/bin/env python3
"""
mathler_lineage_evolve.py

Lineage-based GA / EA Mathler solver.

Key ideas:
- We maintain a single evolving population of expressions across the whole game.
- The first guess is treated as a seed; the population is initialised around it.
- After each guess, we:
    * receive feedback,
    * update history,
    * filter / repopulate the population to be consistent with all feedback,
    * evolve the population for a number of generations,
    * pick the best feasible individual (evaluates to target_value) as the next guess.
- Evolution is grammar-aware (uses mathler_env.is_valid_expression), and
  feedback-aware (penalises inconsistency with history).
- Fitness combines:
    * numeric closeness to target_value,
    * statistical / structural features via expression_features_precomputed,
    * feedback compatibility,
    * strong penalty for violating feedback.

All hyperparameters are read from CONFIG in config.py, with fallback defaults.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Dict, Optional, Set

from config import CONFIG
from mathler_env import (
    DIGITS,
    OPS,
    EXPR_LEN,
    safe_eval,
    is_valid_expression,
    mathler_feedback,
    compute_symbol_frequencies,
    compute_positional_symbol_frequencies,
    expression_features_precomputed,
    generate_initial_candidates,
    top_up_candidates,
    _consistent_with_history,
)

# ---------------------------------------------------------------------------
# Random seed (optional, for reproducibility)
# ---------------------------------------------------------------------------

if "random_seed" in CONFIG:
    random.seed(CONFIG["random_seed"])

# ---------------------------------------------------------------------------
# Lineage GA configuration (tune in config.py)
# ---------------------------------------------------------------------------

LINEAGE_POP_SIZE: int = CONFIG.get("lineage_pop_size", 60)
LINEAGE_GENS_FIRST: int = CONFIG.get("lineage_gens_first", 10)
LINEAGE_GENS_PER_STEP: int = CONFIG.get("lineage_gens_per_step", 10)

LINEAGE_CROSSOVER_RATE: float = CONFIG.get("lineage_crossover_rate", 0.7)
LINEAGE_MUTATION_RATE: float = CONFIG.get("lineage_mutation_rate", 0.4)
LINEAGE_ELITE_FRACTION: float = CONFIG.get("lineage_elite_fraction", 0.25)
LINEAGE_TOURNAMENT_K: int = CONFIG.get("lineage_tournament_k", 3)
LINEAGE_MUTATION_ATTEMPTS: int = CONFIG.get("lineage_mutation_attempts", 40)

# Fitness weights for closeness
LINEAGE_W_CLOSENESS: float = CONFIG.get("lineage_w_closeness", 2.0)
LINEAGE_CLOSENESS_MAX_SCORE: float = CONFIG.get("lineage_closeness_max_score", 10.0)
LINEAGE_CLOSENESS_DIFF_CLIP: float = CONFIG.get("lineage_closeness_diff_clip", 200.0)

# Penalty for violating feedback history
LINEAGE_INCONSISTENCY_PENALTY: float = CONFIG.get("lineage_inconsistency_penalty", 20.0)

# Feedback compatibility weight
LINEAGE_W_FEEDBACK_MATCH: float = CONFIG.get("lineage_w_feedback_match", 1.5)

# Feature weights from expression_features_precomputed
# You can override any of these in CONFIG using keys like "lineage_w_letter_freq_sum", etc.
_default_feature_weights: Dict[str, float] = {
    "letter_freq_sum": 0.0,
    "positional_freq_sum": 1.0,
    "unique_letters": 0.1,
    "remaining_candidates": 0.0,
    "operator_count": 0.0,
    "digit_count": 0.0,
    "distinct_digits": 0.1,
    "plus_count": 0.0,
    "minus_count": 0.0,
    "mul_count": 0.0,
    "div_count": 0.0,
    "symbol_entropy": 0.5,
}

FEATURE_WEIGHTS: Dict[str, float] = {}
for k, default_val in _default_feature_weights.items():
    FEATURE_WEIGHTS[k] = CONFIG.get(f"lineage_w_{k}", default_val)

# Game-level config
MAX_GUESSES: int = CONFIG["max_guesses"]
FAIL_PENALTY: float = CONFIG["fail_penalty"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def feedback_to_string(fb: List[int]) -> str:
    m = {1: "G", -1: "O", 0: "."}
    return "".join(m.get(v, "?") for v in fb)


def closeness_score(val: int, target_value: int) -> float:
    """
    Map |val - target| into [0, LINEAGE_CLOSENESS_MAX_SCORE], then scaled.
    """
    diff = abs(target_value - val)
    d_clipped = min(diff, LINEAGE_CLOSENESS_DIFF_CLIP)
    # linear falloff
    base = max(
        0.0,
        LINEAGE_CLOSENESS_MAX_SCORE
        - d_clipped * (LINEAGE_CLOSENESS_MAX_SCORE / LINEAGE_CLOSENESS_DIFF_CLIP),
    )
    return base


def lineage_fitness(
    expr: str,
    target_value: int,
    history: List[Tuple[str, List[int]]],
    sym_freqs: Dict[str, float],
    pos_freqs: List[Dict[str, float]],
    remaining: int,
) -> float:
    """
    Fitness function for lineage GA.

    Components:
    - numeric closeness to target (via safe_eval),
    - expression_features_precomputed with configurable weights,
    - feedback-compatibility term,
    - strong penalty if inconsistent with history.
    """
    score = 0.0

    # 1) Numeric closeness
    try:
        val = safe_eval(expr)
        close = closeness_score(val, target_value)
        score += LINEAGE_W_CLOSENESS * close
    except Exception:
        # Can't evaluate? Very bad.
        return -1e9

    # 2) Structural / statistical features
    feats = expression_features_precomputed(expr, sym_freqs, pos_freqs, remaining)
    for name, w in FEATURE_WEIGHTS.items():
        if name in feats:
            score += w * feats[name]

    # 3) Feedback compatibility (soft term)
    fb_match_total = 0.0
    for guess, true_fb in history:
        hypothetical = mathler_feedback(guess, expr)
        matches = sum(1 for a, b in zip(true_fb, hypothetical) if a == b)
        fb_match_total += matches / EXPR_LEN

    if history and LINEAGE_W_FEEDBACK_MATCH != 0.0:
        score += LINEAGE_W_FEEDBACK_MATCH * fb_match_total

    # 4) Hard inconsistency penalty
    if history and not _consistent_with_history(expr, history):
        score -= LINEAGE_INCONSISTENCY_PENALTY * len(history)

    return score


def tournament_select(pop: List[str], fitnesses: List[float], k: int) -> str:
    """
    Basic tournament selection over the current population.
    """
    n = len(pop)
    best_idx = random.randrange(n)
    best_fit = fitnesses[best_idx]
    for _ in range(k - 1):
        idx = random.randrange(n)
        if fitnesses[idx] > best_fit:
            best_idx = idx
            best_fit = fitnesses[idx]
    return pop[best_idx]


# ---------------------------------------------------------------------------
# Crossover & mutation
# ---------------------------------------------------------------------------

def one_point_crossover(p1: str, p2: str) -> Tuple[str, str]:
    """
    1-point crossover on two 6-char expression strings.
    Returns two children.
    """
    if len(p1) != EXPR_LEN or len(p2) != EXPR_LEN:
        return p1, p2
    cut = random.randint(1, EXPR_LEN - 1)
    c1 = p1[:cut] + p2[cut:]
    c2 = p2[:cut] + p1[cut:]
    return c1, c2


def mutate_expr(
    expr: str,
    max_attempts: int = LINEAGE_MUTATION_ATTEMPTS,
) -> str:
    """
    Grammar-aware mutation:
    - Single-position mutation with local rules:
        * first char non-zero digit,
        * last char digit (no leading zero after operator),
        * no two operators in a row,
        * must remain is_valid_expression (which includes safe_eval etc).
    - Does NOT force target equality or feedback consistency; that's handled
      by fitness and population filters.

    If no valid mutation is found in max_attempts, returns the original expr.
    """
    if len(expr) != EXPR_LEN:
        return expr

    for _ in range(max_attempts):
        chars = list(expr)
        pos = random.randrange(EXPR_LEN)
        prev_ch = chars[pos - 1] if pos > 0 else None
        next_ch = chars[pos + 1] if pos < EXPR_LEN - 1 else None

        # Decide new character based on local grammar
        if pos == 0:
            # First char: non-zero digit
            new_ch = random.choice(DIGITS[1:])
        elif pos == EXPR_LEN - 1:
            # Last char: must be a digit; if previous is op, avoid '0'
            if prev_ch in OPS:
                new_ch = random.choice(DIGITS[1:])
            else:
                new_ch = random.choice(DIGITS)
        else:
            if prev_ch in OPS:
                # After an operator: must be digit 1–9
                new_ch = random.choice(DIGITS[1:])
            else:
                # prev is digit: digit or operator
                if random.random() < 0.6:
                    new_ch = random.choice(DIGITS)
                else:
                    new_ch = random.choice(OPS)

            # Optional: simple protection against two ops in a row
            if new_ch in OPS and next_ch in OPS:
                # pick a digit instead
                new_ch = random.choice(DIGITS)

        chars[pos] = new_ch
        candidate = "".join(chars)

        if not is_valid_expression(candidate):
            continue

        # If it parses and is syntactically valid, accept as a mutation.
        return candidate

    # Fallback: give up and return original
    return expr


# ---------------------------------------------------------------------------
# Lineage GA / EA core
# ---------------------------------------------------------------------------

class MathlerLineageSolver:
    """
    Lineage-based GA / EA Mathler solver.

    - Start from a seed guess (user-provided or auto-generated).
    - Build an initial population around that seed (plus other candidates).
    - After each feedback:
        * update history,
        * filter / top up population for consistency,
        * evolve for LINEAGE_GENS_PER_STEP generations,
        * pick the best feasible individual as next guess.
    """

    def __init__(self, target_value: int, verbose: bool = True) -> None:
        self.target_value = target_value
        self.verbose = verbose

        self.history: List[Tuple[str, List[int]]] = []
        self.population: List[str] = []
        self.used_guesses: Set[str] = set()

    # ------------------------ population init -------------------------------

    def _init_population_from_seed(self, seed_expr: str) -> None:
        """
        Initialise the population using:
        - the given seed_expr,
        - other target-correct candidates from generate_initial_candidates,
        - and mutations for diversity.
        """
        base_candidates = generate_initial_candidates(self.target_value)
        cand_set = set(base_candidates)

        if seed_expr not in cand_set:
            cand_set.add(seed_expr)
            base_candidates.append(seed_expr)

        if self.verbose:
            print(
                f"[lineage] Initialising population from seed={seed_expr}, "
                f"base_candidates={len(base_candidates)}, target={self.target_value}"
            )

        pop: List[str] = []

        # Start with seed
        pop.append(seed_expr)

        # Add some distinct base candidates (target-equal, grammar-valid)
        random.shuffle(base_candidates)
        for expr in base_candidates:
            if expr not in pop:
                pop.append(expr)
            if len(pop) >= min(LINEAGE_POP_SIZE // 2, LINEAGE_POP_SIZE):
                break

        # Fill remaining slots with mutations for diversity
        while len(pop) < LINEAGE_POP_SIZE:
            parent = random.choice(pop)
            child = mutate_expr(parent)
            if child not in pop and is_valid_expression(child):
                pop.append(child)

        self.population = pop

        if self.verbose:
            print(f"[lineage] Initial population size={len(self.population)}")

    def prepare_initial_guess(self, initial_guess: Optional[str]) -> str:
        """
        Prepare the first guess and initialise the population around it.

        - If initial_guess is provided and valid & target-correct, we use it.
        - Otherwise, we pick a seed from generate_initial_candidates(target_value).
        """
        # Try to use user-provided initial guess if possible
        if initial_guess:
            if self.verbose:
                print(f"[lineage] User-provided initial guess: {initial_guess}")

            try:
                if len(initial_guess) == EXPR_LEN and is_valid_expression(initial_guess):
                    val = safe_eval(initial_guess)
                    if val == self.target_value:
                        seed_expr = initial_guess
                    else:
                        if self.verbose:
                            print(
                                f"[lineage] Initial guess value {val} != target {self.target_value}; "
                                f"will auto-generate seed."
                            )
                        seed_expr = None
                else:
                    if self.verbose:
                        print("[lineage] Initial guess is not a valid Mathler expression; auto-generating seed.")
                    seed_expr = None
            except Exception as e:
                if self.verbose:
                    print(f"[lineage] Error validating initial guess: {e}; auto-generating seed.")
                seed_expr = None
        else:
            seed_expr = None

        if seed_expr is None:
            # Auto-generate: use initial candidates for the target
            base_candidates = generate_initial_candidates(self.target_value)
            if not base_candidates:
                raise RuntimeError(f"No candidates available for target={self.target_value}.")

            seed_expr = random.choice(base_candidates)
            if self.verbose:
                print(f"[lineage] Auto-chosen seed initial guess: {seed_expr}")

        # Initialise population around the seed
        self._init_population_from_seed(seed_expr)
        self.used_guesses.add(seed_expr)
        return seed_expr

    # --------------------- evolution after feedback -------------------------

    def register_feedback(self, guess: str, fb: List[int]) -> None:
        """
        Add (guess, feedback) to history and filter / top-up the population.
        """
        self.history.append((guess, fb))
        if self.verbose:
            print(
                f"[lineage] Registered feedback for guess={guess}: "
                f"{fb} ({feedback_to_string(fb)})"
            )

        # Filter existing population for consistency with all history
        survivors = [
            expr for expr in self.population
            if _consistent_with_history(expr, self.history)
        ]

        if self.verbose:
            print(
                f"[lineage] Population filtered by feedback: "
                f"{len(self.population)} -> {len(survivors)}"
            )

        # If survivors are too few, top up using environment helper
        if len(survivors) < max(10, LINEAGE_POP_SIZE // 3):
            # top_up_candidates will generate expressions that:
            # - respect grammar,
            # - evaluate to target_value,
            # - are consistent with history.
            survivors = top_up_candidates(survivors, self.target_value, self.history)
            if self.verbose:
                print(
                    f"[lineage] After top-up from environment: "
                    f"population size={len(survivors)}"
                )

        # If still empty, reinitialise from scratch (catastrophic reset)
        if not survivors:
            if self.verbose:
                print("[lineage] No survivors after feedback; reinitialising population.")
            # Use history when generating: top_up_candidates([], ...) will honour it
            survivors = top_up_candidates([], self.target_value, self.history)
            if not survivors:
                raise RuntimeError("Failed to reinitialise population from history / target.")

        # If we have more than LINEAGE_POP_SIZE, downsample
        if len(survivors) > LINEAGE_POP_SIZE:
            survivors = random.sample(survivors, LINEAGE_POP_SIZE)

        # If we have fewer, fill with mutations of survivors
        pop = list(survivors)
        while len(pop) < LINEAGE_POP_SIZE:
            parent = random.choice(pop)
            child = mutate_expr(parent)
            pop.append(child)

        self.population = pop

        if self.verbose:
            print(f"[lineage] Population after feedback processing: size={len(self.population)}")

    def _evolve_population(self, n_generations: int) -> None:
        """
        Run n_generations of GA / EA over the current population.
        Population is updated in place.
        """
        if not self.population:
            return

        for gen in range(n_generations):
            pop = self.population
            remaining = len(pop)

            # Compute global stats for this population
            sym_freqs = compute_symbol_frequencies(pop)
            pos_freqs = compute_positional_symbol_frequencies(pop)

            # Compute fitnesses
            fitnesses = [
                lineage_fitness(expr, self.target_value, self.history, sym_freqs, pos_freqs, remaining)
                for expr in pop
            ]

            # Rank
            ranked = sorted(zip(pop, fitnesses), key=lambda t: t[1], reverse=True)
            pop_sorted = [expr for expr, _ in ranked]
            fit_sorted = [fit for _, fit in ranked]

            best_expr = pop_sorted[0]
            best_fit = fit_sorted[0]

            if self.verbose and (gen == 0 or gen == n_generations - 1 or gen % 5 == 0):
                print(
                    f"[lineage] Gen {gen+1}/{n_generations}: "
                    f"best={best_expr}, fitness={best_fit:.3f}"
                )

            # Elitism
            elite_count = max(1, int(LINEAGE_ELITE_FRACTION * len(pop_sorted)))
            elites = pop_sorted[:elite_count]

            # Build new population
            new_pop: List[str] = elites[:]
            while len(new_pop) < len(pop_sorted):
                # Select parents
                p1 = tournament_select(pop_sorted, fit_sorted, LINEAGE_TOURNAMENT_K)
                p2 = tournament_select(pop_sorted, fit_sorted, LINEAGE_TOURNAMENT_K)

                c1, c2 = p1, p2

                # Crossover
                if random.random() < LINEAGE_CROSSOVER_RATE:
                    c1, c2 = one_point_crossover(p1, p2)

                # Mutation
                if random.random() < LINEAGE_MUTATION_RATE:
                    c1 = mutate_expr(c1)
                if len(new_pop) + 1 < len(pop_sorted) and random.random() < LINEAGE_MUTATION_RATE:
                    c2 = mutate_expr(c2)

                # Append children, ensuring length
                new_pop.append(c1)
                if len(new_pop) < len(pop_sorted):
                    new_pop.append(c2)

            self.population = new_pop

    # ------------------------------ guessing --------------------------------

    def _choose_best_feasible_guess(self) -> str:
        """
        Choose the best expression from the population that:
        - evaluates exactly to target_value,
        - is consistent with history,
        - has not been guessed before (if possible).

        If none exist, fall back to environment candidate generation.
        """
        pop = self.population
        if not pop:
            raise RuntimeError("Population is empty in _choose_best_feasible_guess.")

        remaining = len(pop)
        sym_freqs = compute_symbol_frequencies(pop)
        pos_freqs = compute_positional_symbol_frequencies(pop)

        scored: List[Tuple[str, float]] = []
        for expr in pop:
            fit = lineage_fitness(expr, self.target_value, self.history, sym_freqs, pos_freqs, remaining)
            scored.append((expr, fit))

        scored.sort(key=lambda t: t[1], reverse=True)

        # Prefer unused, feasible, consistent guesses
        for expr, fit in scored:
            if expr in self.used_guesses:
                continue
            try:
                val = safe_eval(expr)
            except Exception:
                continue
            if val != self.target_value:
                continue
            if self.history and not _consistent_with_history(expr, self.history):
                continue
            if self.verbose:
                print(f"[lineage] Chosen next guess from population: {expr} (fitness={fit:.3f})")
            return expr

        # If no feasible unused candidate, try environment-based generation
        if self.verbose:
            print("[lineage] No feasible unused guess in population; using environment top-up.")

        env_candidates = top_up_candidates([], self.target_value, self.history)
        if not env_candidates:
            # As last resort, just use the best expression even if reused
            expr, fit = scored[0]
            if self.verbose:
                print(
                    "[lineage] Environment could not produce any candidate; "
                    f"fallback to best in population: {expr} (fitness={fit:.3f})"
                )
            return expr

        guess = env_candidates[0]
        if self.verbose:
            print(f"[lineage] Chosen next guess from environment: {guess}")
        return guess

    def next_guess(self, first: bool = False) -> str:
        """
        Produce the next guess based on the current population and history.

        - If first=True, evolve for LINEAGE_GENS_FIRST generations.
        - Otherwise, evolve for LINEAGE_GENS_PER_STEP generations.
        """
        if not self.population:
            raise RuntimeError("Population is not initialised. Call prepare_initial_guess first.")

        gens = LINEAGE_GENS_FIRST if first else LINEAGE_GENS_PER_STEP
        if self.verbose:
            print(f"[lineage] Evolving population for {gens} generations (first={first}).")
        self._evolve_population(gens)

        guess = self._choose_best_feasible_guess()
        self.used_guesses.add(guess)
        return guess


# ---------------------------------------------------------------------------
# Manual test harness
# ---------------------------------------------------------------------------

def run_manual_test():
    """
    Interactively test the lineage-based solver on a single secret.
    """
    print("=== Mathler Lineage GA Solver: Manual Test Mode ===")
    print("All expressions must be 6 characters of digits and + - * /.\n")

    secret_expr = input("Enter a secret expression (6 chars), or press Enter to auto-generate: ").strip()

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
        target_str = input("Enter a target integer value: ").strip()
        try:
            target_value = int(target_str)
        except ValueError:
            print("Error: target value must be an integer.")
            return

        base_candidates = generate_initial_candidates(target_value)
        if not base_candidates:
            print(f"Could not find any expression for target={target_value}.")
            return
        secret_expr = random.choice(base_candidates)
        print(
            f"Generated random secret {secret_expr} for target value {target_value}.\n"
            "(Shown here for debugging; in a real game the player wouldn't see this.)"
        )

    initial_guess = input(
        "Enter an initial guess expression (or press Enter to let the solver choose): "
    ).strip()
    if not initial_guess:
        initial_guess = None

    solver = MathlerLineageSolver(target_value=target_value, verbose=True)

    print("\n=== Starting game ===")

    # Prepare population and first guess
    first_guess = solver.prepare_initial_guess(initial_guess)
    # Optionally evolve a bit before playing first guess
    first_guess = solver.next_guess(first=True)

    for attempt in range(1, MAX_GUESSES + 1):
        guess = first_guess if attempt == 1 else solver.next_guess(first=False)

        fb = mathler_feedback(guess, secret_expr)
        print(
            f"[main] Attempt {attempt}: guess={guess}, "
            f"feedback={fb} ({feedback_to_string(fb)})"
        )

        if guess == secret_expr:
            print(f"[main] Solved! Secret {secret_expr} found in {attempt} guesses.\n")
            return

        solver.register_feedback(guess, fb)

    print(
        f"[main] Failed to find exact secret {secret_expr} "
        f"within {MAX_GUESSES} guesses."
    )


if __name__ == "__main__":
    run_manual_test()
