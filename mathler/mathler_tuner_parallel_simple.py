#!/usr/bin/env python3
"""
mathler_tuner_parallel_simple.py

Parallel fine-tuner for mathler_evolve.py hyperparameters.

Changes vs previous parallel tuner:
- NO focus parameter per thread anymore.
- All workers perturb ALL parameters slightly, with a shrinking step size
  over rounds (annealing).
- Each worker prints detailed status:
    * when it starts
    * when each game starts/finishes (solved/failed)
    * when it finishes, with its average guesses.
- The main process:
    * collects all candidates,
    * updates the global best when a candidate is better,
    * tracks how much each hyperparameter changes across improvements.

Requires:
- mathler_evolve.py in the same directory.
- best_hyperparams.json from the previous tuning run.

The goal is PURE fine-tuning near the current best, without "one param gets
yeeted" behaviour.
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

def load_best_params(filename: str = "best_hyperparams_refined.json") -> HyperParams:
    """
    Load best hyperparameters from JSON into a HyperParams instance.
    Ignores extra keys like 'avg_guesses'.
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    hp_fields = {f.name for f in fields(HyperParams)}
    filtered = {k: v for k, v in data.items() if k in hp_fields}
    return HyperParams(**filtered)


def save_best_params(params: HyperParams, score: float, filename: str = "best_hyperparams_refined_simple.json") -> None:
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

def simulate_single_game_verbose(
    secret_expr: str,
    target_value: int,
    params: HyperParams,
    max_guesses: int,
    thread_id: int,
    game_index: int,
) -> int:
    """
    Run a single game of Mathler for a given thread, with verbose prints.
    Returns number of guesses used (or max_guesses + 2 on failure).
    """
    print(f"[thread {thread_id}] Game {game_index}: starting (secret={secret_expr}, target={target_value})")

    solver = me.MathlerGenomeSolver(
        target_value=target_value,
        max_guesses=max_guesses,
        inner_pop_size=params.pop_size,
        inner_gens_per_guess=params.gens_per_guess,
        verbose=False,  # suppress internal GA chatter here
    )

    guess = solver.first_guess()

    for attempt in range(1, max_guesses + 1):
        fb = me.mathler_feedback(guess, secret_expr)
        solver.register_feedback(guess, fb)
        if all(v == 1 for v in fb):
            print(f"[thread {thread_id}] Game {game_index}: solved in {attempt} guesses (guess={guess})")
            return attempt
        guess = solver.next_guess(guess)

    print(f"[thread {thread_id}] Game {game_index}: FAILED after {max_guesses} guesses (secret={secret_expr})")
    return max_guesses + 2


def evaluate_params_verbose_thread(
    params: HyperParams,
    num_games: int,
    max_guesses: int,
    seed: int,
    thread_id: int,
) -> float:
    """
    Evaluate hyperparameters in a given worker, with verbose logging.
    """
    random.seed(seed)
    apply_hyperparams(params)

    print(f"[thread {thread_id}] Starting evaluation of params: {params}")

    total_score = 0.0
    for game_index in range(num_games):
        target_val = random.randint(5, 200)
        secret_expr = me.generate_single_expression_for_target(target_val)
        target_value = me.safe_eval(secret_expr)

        guesses_used = simulate_single_game_verbose(
            secret_expr,
            target_value,
            params,
            max_guesses=max_guesses,
            thread_id=thread_id,
            game_index=game_index,
        )
        total_score += guesses_used

    avg_score = total_score / num_games
    print(f"[thread {thread_id}] Finished evaluation. Average guesses = {avg_score:.3f}")
    return avg_score


# ------------------------- Candidate generation --------------------------- #

def make_perturbed_candidate(
    base: HyperParams,
    scale: float = 1.0,
    small_sigma_float: float = 0.15,
    small_step_int: int = 1,
) -> HyperParams:
    """
    Create a candidate HyperParams based on `base`, adding small Gaussian noise
    to all parameters.

    `scale` shrinks the noise over rounds:
      - scale ~ 1.0 -> normal noise
      - scale ~ 0.2 -> very small noise (fine tuning)
    """
    hp_dict = asdict(base)

    # Helper to perturb float
    def perturb_float(name: str):
        sigma = small_sigma_float * scale
        hp_dict[name] = hp_dict[name] + random.gauss(0.0, sigma)

    # Helper to perturb int
    def perturb_int(name: str, min_val: int, max_val: int):
        base_step = int(round(small_step_int * scale))
        if base_step < 1:
            base_step = 1
        delta = random.randint(-base_step, base_step)
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

    # Clamp some floats
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
      (params, thread_id, num_games, max_guesses, seed)

    Returns:
      (params, score, thread_id)
    """
    params, thread_id, num_games, max_guesses, seed = args
    score = evaluate_params_verbose_thread(
        params=params,
        num_games=num_games,
        max_guesses=max_guesses,
        seed=seed,
        thread_id=thread_id,
    )
    return params, score, thread_id


# -------------------------- Parallel refinement --------------------------- #

def parallel_refine_simple(
    start_params: HyperParams,
    rounds: int = 10,
    num_workers: Optional[int] = None,
    num_games_per_eval: int = 20,
    max_guesses: int = 6,
    seed: int = 1234,
) -> Tuple[HyperParams, float]:
    """
    Parallel refinement around `start_params` WITHOUT any focus parameter.

    Each round:
      - Compute a `scale` in [0.2, 1.0] to shrink step size over time.
      - Spawn `num_workers` candidates, each:
          * starting from current best,
          * perturbed slightly (noise * scale).
      - Evaluate all candidates in parallel (each printing its status).
      - If any candidate has a lower avg guess score, adopt it as new best.
      - Track per-parameter absolute change when improvements occur.

    Returns:
      (best_params, best_score)
    """
    if num_workers is None:
        num_workers = max(2, mp.cpu_count() - 1)
    print("num_workers =", num_workers)
    random.seed(seed)

    # Initial evaluation of starting params
    print("[refine] Evaluating starting parameters once to get baseline...")
    baseline_seed = seed
    baseline_score = evaluate_params_verbose_thread(
        params=start_params,
        num_games=num_games_per_eval,
        max_guesses=max_guesses,
        seed=baseline_seed,
        thread_id=-1,  # special "baseline" thread
    )

    current_best = start_params
    current_score = baseline_score

    print(
        f"[refine] Starting refinement from avg guesses={current_score:.3f} "
        f"with params={current_best}"
    )

    # For tracking how much each param changed over accepted improvements
    total_param_change: Dict[str, float] = {name: 0.0 for name in PARAM_NAMES}
    improvements_count = 0

    start_time = time.time()

    for round_idx in range(rounds):
        frac = round_idx / max(1, rounds - 1)
        # Anneal from 1.0 down to 0.2
        scale = 0.4 - 0.2 * frac

        print(
            f"\n[refine] === Round {round_idx+1}/{rounds} (scale={scale:.3f}) ==="
        )
        print(
            f"[refine] Current best avg guesses = {current_score:.3f} "
            f"with params={current_best}"
        )

        jobs = []
        for w in range(num_workers):
            candidate = make_perturbed_candidate(current_best, scale=scale)
            job_seed = seed + 1000 + round_idx * num_workers + w
            jobs.append((candidate, w, num_games_per_eval, max_guesses, job_seed))

        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(worker_eval, jobs)

        # Check all candidates for improvements
        for candidate, score, thread_id in results:
            if score < current_score:
                print(
                    f"[refine] Round {round_idx+1}: new best {score:.3f} "
                    f"(prev {current_score:.3f}) from thread {thread_id}"
                )
                # Track parameter changes
                for name in PARAM_NAMES:
                    old_val = getattr(current_best, name)
                    new_val = getattr(candidate, name)
                    total_param_change[name] += abs(new_val - old_val)

                current_best = candidate
                current_score = score
                improvements_count += 1

        print(
            f"[refine] Round {round_idx+1} complete. "
            f"Current best avg guesses = {current_score:.3f}"
        )

    elapsed = time.time() - start_time
    print(
        f"\n[refine] Finished {rounds} rounds in {elapsed:.1f}s. "
        f"Best avg guesses = {current_score:.3f}"
    )
    print(f"[refine] Total accepted improvements: {improvements_count}")

    print("[refine] Total absolute parameter change across improvements:")
    for name in PARAM_NAMES:
        print(f"  {name}: {total_param_change[name]:.4f}")

    return current_best, current_score


# ----------------------------- Main entry --------------------------------- #

def main():
    # Load the best params from previous (global) tuning
    base_params = load_best_params("best_hyperparams_refined.json")

    # Run refinement
    refined_params, refined_score = parallel_refine_simple(
        start_params=base_params,
        rounds=10,             # number of fine-tuning rounds
        num_workers=None,      # use CPU count - 1
        num_games_per_eval=10, # lower this if logs are too spammy
        max_guesses=6,
        seed=999,
    )

    print("\n[refine] Refined hyperparameters:")
    print(refined_params)
    print(f"[refine] Average guesses: {refined_score:.3f}")

    save_best_params(refined_params, refined_score)


if __name__ == "__main__":
    main()
