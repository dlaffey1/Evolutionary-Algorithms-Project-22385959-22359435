#!/usr/bin/env python3
"""
mathler_tuner_parallel.py

Parallel fine-tuner for mathler_evolve.py hyperparameters.

- Loads the best hyperparameters from best_hyperparams.json (produced by mathler_tuner.py).
- Uses multiple worker processes to explore the space around that best config.
- Each worker:
    * Starts from the current global best HyperParams,
    * Applies small noise to all parameters,
    * Applies larger noise to a single "focus" parameter (unique per worker).
- The main process:
    * Evaluates all candidates (via workers),
    * Updates the global best when a better candidate is found,
    * Tracks which focus parameters most often lead to improvements.

Lower average guesses = better.
"""

import json
import random
import time
from dataclasses import dataclass, asdict, fields
from typing import List, Tuple, Optional, Dict

import multiprocessing as mp

import mathler_evolver as me  # your solver module


# ----------------------------- HyperParams ----------------------------- #

@dataclass
class HyperParams:
    # Fitness weights
    w_closeness: float
    closeness_max_score: float
    closeness_diff_clip: float

    w_unsatisfied: float
    w_banned: float
    w_diversity: float
    w_feedback: float

    # GA runtime parameters
    pop_size: int
    gens_per_guess: int


PARAM_NAMES = [
    "w_closeness",
    "closeness_max_score",
    "closeness_diff_clip",
    "w_unsatisfied",
    "w_banned",
    "w_diversity",
    "w_feedback",
    "pop_size",
    "gens_per_guess",
]


# --------------------- Load / Save HyperParams -------------------------- #

def load_best_params(filename: str = "best_hyperparams.json") -> HyperParams:
    """
    Load best hyperparameters from JSON into a HyperParams instance.
    Ignores extra keys like 'avg_guesses'.
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    hp_fields = {f.name for f in fields(HyperParams)}
    filtered = {k: v for k, v in data.items() if k in hp_fields}
    return HyperParams(**filtered)


def save_best_params(params: HyperParams, score: float, filename: str = "best_hyperparams_refined.json") -> None:
    data = asdict(params)
    data["avg_guesses"] = score
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[refine] Saved refined hyperparams to {filename}")


# --------------------- Monkey-patch fitness_expr ------------------------ #

def apply_hyperparams(params: HyperParams):
    """
    Monkey-patch me.fitness_expr to use the given weights.
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

            d_clipped = min(diff, params.closeness_diff_clip)
            # map diff -> [0, closeness_max_score]
            closeness = max(
                0.0,
                params.closeness_max_score
                - d_clipped * (params.closeness_max_score / params.closeness_diff_clip)
            )
            score += params.w_closeness * closeness
        except ValueError:
            score -= 10.0
            val = None

        # 2) Unsatisfied digits
        unsat = knowledge.unsatisfied_digits()
        present_bonus = sum(1 for ch in expr if ch in unsat)
        score += params.w_unsatisfied * present_bonus

        # 3) Banned chars
        banned_penalty = sum(1 for ch in expr if ch in knowledge.banned_chars)
        score += params.w_banned * banned_penalty

        # 4) Diversity
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

    me.fitness_expr = fitness_expr_with_params


# ----------------------------- Evaluation --------------------------------- #

def simulate_single_game(
    secret_expr: str,
    target_value: int,
    params: HyperParams,
    max_guesses: int = 6,
) -> int:
    solver = me.MathlerGenomeSolver(
        target_value=target_value,
        max_guesses=max_guesses,
        inner_pop_size=params.pop_size,
        inner_gens_per_guess=params.gens_per_guess,
        verbose=False,
    )

    guess = solver.first_guess()

    for attempt in range(1, max_guesses + 1):
        fb = me.mathler_feedback(guess, secret_expr)
        solver.register_feedback(guess, fb)
        if all(v == 1 for v in fb):
            return attempt
        guess = solver.next_guess(guess)

    return max_guesses + 2  # failure penalty


def evaluate_params(
    params: HyperParams,
    num_games: int,
    max_guesses: int,
    seed: int,
) -> float:
    """
    Evaluate hyperparameters by running num_games random games.
    Lower score is better (avg guesses).
    """
    random.seed(seed)
    apply_hyperparams(params)

    total_score = 0.0
    for _ in range(num_games):
        target_val = random.randint(5, 200)
        secret_expr = me.generate_single_expression_for_target(target_val)
        target_value = me.safe_eval(secret_expr)
        guesses_used = simulate_single_game(
            secret_expr,
            target_value,
            params,
            max_guesses=max_guesses,
        )
        total_score += guesses_used

    return total_score / num_games


# ------------------------- Candidate generation --------------------------- #

def make_perturbed_candidate(
    base: HyperParams,
    focus_param: str,
    small_sigma_float: float = 0.4,
    big_sigma_float: float = 1.5,
    small_step_int: int = 4,
    big_step_int: int = 12,
) -> HyperParams:
    """
    Create a candidate HyperParams based on `base`, adding:
      - small noise to all parameters,
      - larger noise to the single `focus_param`.

    Floats get Gaussian noise; ints get +/- random step.
    """
    # Start from base
    hp_dict = asdict(base)

    # Helper to perturb float
    def perturb_float(name: str):
        sigma = small_sigma_float
        if name == focus_param:
            sigma = big_sigma_float
        hp_dict[name] = hp_dict[name] + random.gauss(0.0, sigma)

    # Helper to perturb int
    def perturb_int(name: str, min_val: int, max_val: int):
        step = small_step_int
        if name == focus_param:
            step = big_step_int
        delta = random.randint(-step, step)
        hp_dict[name] = max(min_val, min(max_val, hp_dict[name] + delta))

    # Floats
    perturb_float("w_closeness")
    perturb_float("closeness_max_score")
    perturb_float("closeness_diff_clip")

    perturb_float("w_unsatisfied")
    perturb_float("w_banned")
    perturb_float("w_diversity")
    perturb_float("w_feedback")

    # Ints
    perturb_int("pop_size", min_val=20, max_val=150)
    perturb_int("gens_per_guess", min_val=5, max_val=60)

    # Clamp some floats to sensible bounds
    hp_dict["closeness_max_score"] = max(1.0, hp_dict["closeness_max_score"])
    hp_dict["closeness_diff_clip"] = max(10.0, hp_dict["closeness_diff_clip"])
    hp_dict["w_closeness"] = max(0.1, hp_dict["w_closeness"])
    hp_dict["w_unsatisfied"] = max(0.0, hp_dict["w_unsatisfied"])
    hp_dict["w_diversity"] = max(0.0, hp_dict["w_diversity"])
    hp_dict["w_feedback"] = max(0.0, hp_dict["w_feedback"])

    return HyperParams(**hp_dict)


# ----------------------------- Worker ------------------------------------- #

def worker_eval(args):
    """
    Worker function for multiprocessing.Pool.

    Args tuple:
      (params, focus_param, num_games, max_guesses, seed)

    Returns:
      (params, focus_param, score)
    """
    params, focus_param, num_games, max_guesses, seed = args
    score = evaluate_params(params, num_games=num_games, max_guesses=max_guesses, seed=seed)
    return params, focus_param, score


# -------------------------- Parallel refinement --------------------------- #

def parallel_refine(
    start_params: HyperParams,
    rounds: int = 20,
    num_workers: Optional[int] = None,
    num_games_per_eval: int = 40,
    max_guesses: int = 6,
    seed: int = 1234,
) -> Tuple[HyperParams, float]:
    """
    Parallel hill-climbing refinement around `start_params`.

    Each "round":
      - For each worker, generate a candidate that:
          * starts from current best,
          * small noise on all params,
          * big noise on one focus parameter (unique per worker).
      - Evaluate all candidates in parallel.
      - Update global best if any candidate improves it.
      - Track which focus parameters produced improvements.
    """
    if num_workers is None:
        num_workers = max(2, mp.cpu_count() - 1)

    random.seed(seed)

    # Evaluate starting params to get baseline
    current_best = start_params
    current_score = evaluate_params(
        current_best,
        num_games=num_games_per_eval,
        max_guesses=max_guesses,
        seed=seed,
    )

    print(
        f"[refine] Starting from score={current_score:.3f} "
        f"with params={current_best}"
    )

    improvements_by_param: Dict[str, int] = {name: 0 for name in PARAM_NAMES}
    start_time = time.time()

    for round_idx in range(rounds):
        jobs = []
        # Assign each worker a focus param (cycle through PARAM_NAMES)
        for w in range(num_workers):
            focus_param = PARAM_NAMES[w % len(PARAM_NAMES)]
            candidate = make_perturbed_candidate(current_best, focus_param)
            job_seed = seed + 1000 + round_idx * num_workers + w
            jobs.append((candidate, focus_param, num_games_per_eval, max_guesses, job_seed))

        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(worker_eval, jobs)

        # Check all candidates, update global best
        for candidate, focus_param, score in results:
            if score < current_score:
                print(
                    f"[refine] Round {round_idx+1}: new best {score:.3f} "
                    f"(prev {current_score:.3f}), focus={focus_param}"
                )
                current_score = score
                current_best = candidate
                improvements_by_param[focus_param] += 1

        # Progress report
        print(
            f"[refine] Round {round_idx+1}/{rounds} done. "
            f"Current best avg guesses = {current_score:.3f}"
        )
        print(
            "[refine] Improvements by param so far: "
            + ", ".join(f"{k}:{v}" for k, v in improvements_by_param.items())
        )

    elapsed = time.time() - start_time
    print(
        f"[refine] Finished {rounds} rounds in {elapsed:.1f}s. "
        f"Best avg guesses = {current_score:.3f}"
    )
    print("[refine] Final improvements by param:")
    for k, v in improvements_by_param.items():
        print(f"  {k}: {v}")

    return current_best, current_score


# ----------------------------- Main entry --------------------------------- #

def main():
    # Load the global-best params from the previous random tuner
    base_params = load_best_params("best_hyperparams.json")

    # Refine around them in parallel
    refined_params, refined_score = parallel_refine(
        start_params=base_params,
        rounds=20,            # number of fine-tuning rounds
        num_workers=None,     # use CPU count - 1
        num_games_per_eval=40, # more games per eval for more reliable averages
        max_guesses=6,
        seed=999,
    )

    print("\n[refine] Refined hyperparameters:")
    print(refined_params)
    print(f"[refine] Average guesses: {refined_score:.3f}")

    save_best_params(refined_params, refined_score, filename="best_hyperparams_refined.json")


if __name__ == "__main__":
    main()
