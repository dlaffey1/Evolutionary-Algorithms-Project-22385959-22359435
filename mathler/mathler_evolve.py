#!/usr/bin/env python3
"""
mathler_evolve.py

Mathler solver using a free-roaming GA + repair-to-target approach.

Key ideas:
- Expressions are fixed-length 6-char genomes (digits + +-*/).
- GA evolves expressions without forcing them to match the target value.
- Fitness combines:
    * numeric closeness to target,
    * feedback compatibility,
    * digit presence/absence knowledge,
    * diversity.
- After each GA run, we "repair" the GA-best expression into one that:
    * respects green (locked) positions,
    * evaluates exactly to the target.
- All hyperparameters are loaded from a JSON file (mathler_config.json):
    * GA sizes and generations,
    * fitness weights,
    * mutation probabilities and feedback weights,
    * repair steps, etc.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict, field
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
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GAConfig:
    max_guesses: int = 6
    pop_size: int = 60
    gens_per_guess: int = 25


@dataclass
class FitnessConfig:
    w_closeness: float = 3.0
    closeness_max_score: float = 10.0
    closeness_diff_clip: float = 100.0

    w_unsatisfied: float = 2.0
    w_banned: float = -3.0
    w_diversity: float = 0.3
    w_feedback: float = 1.5


@dataclass
class MutationConfig:
    # Probability of applying commutative token swap (+/*) before char-level mutation
    prob_commutative_swap: float = 0.30

    # When doing char-level mutation, probability of choosing a digit vs operator
    prob_digit_mutation: float = 0.65

    # Feedback-based base mutation weights per position
    fb_weight_green: float = 0.0    # green (1)
    fb_weight_gray: float = 0.9     # gray (0)
    fb_weight_orange: float = 0.4   # orange (-1)

    # Extra mutation bias for digits in tokens after + or - if we're near target
    plusminus_close_diff: float = 50.0
    plusminus_boost_factor: float = 2.0

    # Max attempts to generate a valid child inside mutate_expression
    mutate_max_attempts: int = 50


@dataclass
class RepairConfig:
    # Steps in local random walk during repair_to_target
    repair_max_steps: int = 200


@dataclass
class Config:
    ga: GAConfig = field(default_factory=GAConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)

    @classmethod
    def load(cls, path: str = "mathler_config.json") -> "Config":
        """
        Load config from JSON, falling back to dataclass defaults for any
        missing fields or if the file doesn't exist.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return cls()

        def merged(dcls, key):
            default_inst = dcls()
            default_dict = asdict(default_inst)
            override = data.get(key, {})
            default_dict.update(override)
            return dcls(**default_dict)

        return cls(
            ga=merged(GAConfig, "ga"),
            fitness=merged(FitnessConfig, "fitness"),
            mutation=merged(MutationConfig, "mutation"),
            repair=merged(RepairConfig, "repair"),
        )


CONFIG = Config.load()

# ---------------------------------------------------------------------------
# Expression validation and helpers
# ---------------------------------------------------------------------------


def is_valid_expression(expr: str) -> bool:
    """Syntactic + semantic validity for a 6-char Mathler expression."""
    if len(expr) != EXPR_LEN:
        return False

    for ch in expr:
        if ch not in DIGITS and ch not in OPS:
            return False

    # first char: digit 1–9
    if expr[0] not in DIGITS or expr[0] == "0":
        return False

    # last char: digit
    if expr[-1] not in DIGITS:
        return False

    # no two operators in a row, no 0 after op
    prev = expr[0]
    for ch in expr[1:]:
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


def generate_random_valid_expression(max_tries: int = 10000) -> str:
    """Generate any valid expression (value does not matter)."""
    alphabet = DIGITS + OPS
    for _ in range(max_tries):
        expr = "".join(random.choice(alphabet) for _ in range(EXPR_LEN))
        if is_valid_expression(expr):
            return expr
    raise RuntimeError("Failed to generate random valid expression")


def generate_expression_for_target_value(target_value: int) -> str:
    """Wrapper over mathler_env.generate_expression_for_target."""
    return generate_expression_for_target(target_value)

# ---------------------------------------------------------------------------
# Tokenization (for algebra-aware mutations)
# ---------------------------------------------------------------------------


def tokenize_with_spans(expr: str) -> List[Tuple[str, int, int]]:
    """
    Return triples (token, start_idx, end_idx).
    Numeric tokens may be multi-digit; operators are single char.
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
    return "".join(tokens)


def plusminus_digit_positions(expr: str) -> set[int]:
    """
    Return char indices that belong to number tokens directly after '+' or '-'.
    """
    tokens = tokenize_with_spans(expr)
    positions: set[int] = set()
    for i, (tok, start, end) in enumerate(tokens):
        if tok[0].isdigit() and i > 0:
            prev_tok, _, _ = tokens[i - 1]
            if prev_tok in {"+", "-"}:
                positions.update(range(start, end + 1))
    return positions

# ---------------------------------------------------------------------------
# Knowledge & feedback
# ---------------------------------------------------------------------------


def feedback_to_string(fb: List[int]) -> str:
    m = {1: "G", -1: "O", 0: "."}
    return "".join(m.get(v, "?") for v in fb)


class Knowledge:
    def __init__(self) -> None:
        self.present_digits: set[str] = set()
        self.banned_chars: set[str] = set()
        self.banned_positions: Dict[str, set[int]] = {}
        self.locked_positions: Dict[int, str] = {}

    def unsatisfied_digits(self) -> set[str]:
        locked_chars = set(self.locked_positions.values())
        return {d for d in self.present_digits if d not in locked_chars}

    def update_from_feedback(self, guess: str, fb: List[int]) -> None:
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
                self.banned_positions.setdefault(ch, set()).add(idx)

# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def base_mutation_probs_from_feedback(fb: List[int]) -> List[float]:
    """
    Convert per-position feedback into base mutation weights.
    """
    m = CONFIG.mutation
    probs: List[float] = []
    for state in fb:
        if state == 1:
            probs.append(m.fb_weight_green)
        elif state == 0:
            probs.append(m.fb_weight_gray)
        else:
            probs.append(m.fb_weight_orange)
    if all(p == 0 for p in probs):
        probs = [1.0] * EXPR_LEN
    return probs


def mutate_expression(
    expr: str,
    fb: List[int],
    knowledge: Knowledge,
    target_value: int,
) -> str:
    """
    Feedback- and algebra-aware mutation. Does NOT require child value == target.
    """
    mut_cfg = CONFIG.mutation

    try:
        parent_val = safe_eval(expr)
        abs_diff = abs(target_value - parent_val)
    except ValueError:
        parent_val = None
        abs_diff = None

    plusminus_positions = plusminus_digit_positions(expr)
    tokens_with_spans = tokenize_with_spans(expr)
    tokens_only = [t for t, _, _ in tokens_with_spans]

    for _ in range(mut_cfg.mutate_max_attempts):
        # 1) Commutative swap
        if random.random() < mut_cfg.prob_commutative_swap:
            iops = [i for i, tok in enumerate(tokens_only) if tok in {"+", "*"}]
            random.shuffle(iops)
            for iop in iops:
                if iop == 0 or iop == len(tokens_only) - 1:
                    continue
                left = tokens_only[iop - 1]
                right = tokens_only[iop + 1]
                if not left[0].isdigit() or not right[0].isdigit():
                    continue
                new_tokens = tokens_only[:]
                new_tokens[iop - 1], new_tokens[iop + 1] = right, left
                candidate = tokens_to_expr(new_tokens)
                if len(candidate) != EXPR_LEN or not is_valid_expression(candidate):
                    continue
                try:
                    _ = safe_eval(candidate)
                except ValueError:
                    continue
                return candidate

        # 2) Char-level mutation
        candidate_list = list(expr)
        probs = base_mutation_probs_from_feedback(fb)

        if abs_diff is not None and abs_diff <= mut_cfg.plusminus_close_diff:
            for idx in plusminus_positions:
                probs[idx] *= mut_cfg.plusminus_boost_factor

        idx_choices = list(range(EXPR_LEN))
        pos = random.choices(idx_choices, weights=probs, k=1)[0]

        if pos in knowledge.locked_positions:
            continue

        def choose_digit(position: int) -> str:
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

            if (
                abs_diff is not None
                and abs_diff <= mut_cfg.plusminus_close_diff
                and position in plusminus_positions
                and parent_val is not None
            ):
                need_up = target_value > parent_val
                sorted_pool = sorted(pool, key=int, reverse=need_up)
                top_k = min(3, len(sorted_pool))
                return random.choice(sorted_pool[:top_k])

            return random.choice(pool)

        if pos == 0:
            choices = [d for d in DIGITS if d != "0" and d not in knowledge.banned_chars]
            if not choices:
                choices = [d for d in DIGITS if d != "0"]
            candidate_list[pos] = random.choice(choices)

        elif pos == EXPR_LEN - 1:
            candidate_list[pos] = choose_digit(pos)

        else:
            banned_here = {
                ch for ch, positions in knowledge.banned_positions.items()
                if pos in positions
            }
            if random.random() < mut_cfg.prob_digit_mutation:
                d = choose_digit(pos)
                if d in banned_here:
                    alt = [
                        x for x in DIGITS
                        if x not in knowledge.banned_chars and x not in banned_here
                    ]
                    if alt:
                        d = random.choice(alt)
                candidate_list[pos] = d
            else:
                ops = [op for op in OPS if op not in knowledge.banned_chars]
                if not ops:
                    ops = list(OPS)
                candidate_list[pos] = random.choice(ops)

        candidate = "".join(candidate_list)
        if not is_valid_expression(candidate):
            continue
        try:
            _ = safe_eval(candidate)
        except ValueError:
            continue
        return candidate

    return generate_random_valid_expression()

# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------


def fitness_expr(
    expr: str,
    knowledge: Knowledge,
    history: List[Tuple[str, List[int]]],
    target_value: int,
) -> float:
    """Heuristic fitness for ranking expressions in the GA. Higher is better."""
    cfg = CONFIG.fitness
    score = 0.0

    # 1) Numeric closeness
    try:
        val = safe_eval(expr)
        diff = abs(target_value - val)
        d_clipped = min(diff, cfg.closeness_diff_clip)
        closeness = max(
            0.0,
            cfg.closeness_max_score
            - d_clipped * (cfg.closeness_max_score / cfg.closeness_diff_clip),
        )
        score += cfg.w_closeness * closeness
    except ValueError:
        score -= 10.0
        val = None

    # 2) Unsatisfied digits
    unsat = knowledge.unsatisfied_digits()
    present_bonus = sum(1 for ch in expr if ch in unsat)
    score += cfg.w_unsatisfied * present_bonus

    # 3) Banned chars
    banned_penalty = sum(1 for ch in expr if ch in knowledge.banned_chars)
    score += cfg.w_banned * banned_penalty

    # 4) Diversity
    diversity = len(set(expr))
    score += cfg.w_diversity * diversity

    # 5) Feedback compatibility
    comp_bonus = 0.0
    for past_guess, true_fb in history:
        hypothetical_fb = mathler_feedback(past_guess, expr)
        match = sum(1 for a, b in zip(true_fb, hypothetical_fb) if a == b)
        comp_bonus += match / EXPR_LEN
    score += cfg.w_feedback * comp_bonus

    return score

# ---------------------------------------------------------------------------
# Repair to target
# ---------------------------------------------------------------------------


def repair_to_target(
    expr: str,
    knowledge: Knowledge,
    target_value: int,
    fb: List[int],
    debug: bool = False,
) -> str:
    """
    Local random walk to project expr onto the set of expressions that
    evaluate exactly to target_value while respecting locked positions.
    """
    max_steps = CONFIG.repair.repair_max_steps

    current_list = list(expr)
    for idx, ch in knowledge.locked_positions.items():
        current_list[idx] = ch
    current = "".join(current_list)

    for step in range(max_steps):
        try:
            if is_valid_expression(current) and safe_eval(current) == target_value:
                if debug:
                    print(f"[repair] success after {step} steps: {expr} -> {current}")
                return current
        except ValueError:
            pass

        current = mutate_expression(current, fb, knowledge, target_value)

    # Fallback: generate target-correct expression and enforce greens
    for _ in range(100):
        base = generate_expression_for_target_value(target_value)
        base_list = list(base)
        for idx, ch in knowledge.locked_positions.items():
            base_list[idx] = ch
        candidate = "".join(base_list)
        try:
            if is_valid_expression(candidate) and safe_eval(candidate) == target_value:
                if debug:
                    print(f"[repair] fallback with locked positions: {candidate}")
                return candidate
        except ValueError:
            continue

    final = generate_expression_for_target_value(target_value)
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
    verbose: bool = False,
) -> str:
    """Run the GA from current_guess using last feedback, then repair."""
    pop_size = CONFIG.ga.pop_size
    gens = CONFIG.ga.gens_per_guess

    if not is_valid_expression(current_guess):
        if verbose:
            print("[evolve] current_guess invalid; replacing with random valid expression.")
        current_guess = generate_random_valid_expression()

    population: List[str] = [current_guess]

    while len(population) < pop_size:
        if random.random() < 0.7:
            child = mutate_expression(current_guess, fb, knowledge, target_value)
        else:
            child = generate_random_valid_expression()
        population.append(child)

    if verbose:
        print(f"[evolve] Initial population size: {len(population)}")

    for gen in range(gens):
        scored = [
            (fitness_expr(expr, knowledge, history, target_value), expr)
            for expr in population
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        elites = [expr for _, expr in scored[: max(2, pop_size // 4)]]

        if verbose and (gen == 0 or gen == gens - 1):
            best_score, best_expr = scored[0]
            print(f"[evolve] Gen {gen+1}/{gens} | best={best_expr} (fitness={best_score:.3f})")

        new_pop: List[str] = elites[:]
        while len(new_pop) < pop_size:
            parent = random.choice(elites)
            child = mutate_expression(parent, fb, knowledge, target_value)
            new_pop.append(child)

        population = new_pop

    final_scored = [
        (fitness_expr(expr, knowledge, history, target_value), expr)
        for expr in population
    ]
    final_scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best_expr = final_scored[0]

    if verbose:
        print(f"[evolve] GA best before repair: {best_expr} (fitness={best_score:.3f})")

    repaired = repair_to_target(best_expr, knowledge, target_value, fb, debug=verbose)
    if verbose:
        print(f"[evolve] Repaired GA best to target-correct: {repaired}")
    return repaired

# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


class MathlerGenomeSolver:
    def __init__(
        self,
        target_value: int,
        verbose: bool = True,
    ) -> None:
        self.target_value = target_value
        self.max_guesses = CONFIG.ga.max_guesses
        self.verbose = verbose
        self.history: List[Tuple[str, List[int]]] = []
        self.knowledge = Knowledge()

    def first_guess(self, initial_guess: Optional[str] = None) -> str:
        if initial_guess:
            if self.verbose:
                print(f"[solver] Using user-provided initial guess: {initial_guess}")
            if len(initial_guess) == EXPR_LEN and is_valid_expression(initial_guess):
                try:
                    if safe_eval(initial_guess) == self.target_value:
                        return initial_guess
                except ValueError:
                    pass
            if self.verbose:
                print("[solver] Initial guess invalid for target; generating instead.")

        guess = generate_expression_for_target_value(self.target_value)
        if self.verbose:
            print(f"[solver] Generated first guess: {guess}")
        return guess

    def register_feedback(self, guess: str, fb: List[int]) -> None:
        self.history.append((guess, fb))
        self.knowledge.update_from_feedback(guess, fb)
        if self.verbose:
            print(
                f"[solver] Feedback {guess}: {fb} ({feedback_to_string(fb)}) | "
                f"present={sorted(self.knowledge.present_digits)}, "
                f"banned={sorted(self.knowledge.banned_chars)}, "
                f"locked={sorted(self.knowledge.locked_positions.items())}"
            )

    def next_guess(self, current_guess: str) -> str:
        if not self.history:
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
            verbose=self.verbose,
        )

# ---------------------------------------------------------------------------
# Manual test harness
# ---------------------------------------------------------------------------


def run_manual_test():
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
        secret_expr = generate_expression_for_target_value(target_value)
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
        verbose=True,
    )

    guess = solver.first_guess(initial_guess)

    print("\n=== Starting game ===")
    for attempt in range(1, solver.max_guesses + 1):
        fb = mathler_feedback(guess, secret_expr)
        print(
            f"[main] Attempt {attempt}: guess={guess}, "
            f"feedback={fb} ({feedback_to_string(fb)})"
        )
        solver.register_feedback(guess, fb)

        if all(v == 1 for v in fb):
            print(f"[main] Solved! Secret {secret_expr} found in {attempt} guesses.\n")
            return

        if attempt == solver.max_guesses:
            break

        guess = solver.next_guess(guess)

    print(
        f"[main] Failed to find exact secret {secret_expr} "
        f"within {solver.max_guesses} guesses."
    )


if __name__ == "__main__":
    run_manual_test()
