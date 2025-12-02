#!/usr/bin/env python3
import random

from config import CONFIG
from gp_core import (
    RNG,
    Individual,
    init_population,
    tournament_select,
    subtree_crossover,
    subtree_mutation,
)
from mathler_env import generate_random_valid_expression, safe_eval
from mathler_fitness import (
    evaluate_population_mathler,
    play_game_with_individual,
    FAIL_PENALTY,
)


def generate_training_secrets(num_secrets: int):
    """
    Generate a list of (secret_expr, target_value) pairs for training.
    Each secret is a random valid expression; its target_value is its evaluated result.
    """
    secrets = []
    seen = set()
    while len(secrets) < num_secrets:
        expr = generate_random_valid_expression()
        try:
            val = safe_eval(expr)
        except Exception:
            continue
        key = (expr, val)
        if key in seen:
            continue
        seen.add(key)
        secrets.append(key)
    return secrets


def evolve_mathler(
    generations=CONFIG["generations"],
    pop_size=CONFIG["pop_size"],
    max_depth=CONFIG["max_depth"],
    crossover_rate=CONFIG["crossover_rate"],
    mutation_rate=CONFIG["mutation_rate"],
    tournament_k=CONFIG["tournament_k"],
):
    # training secrets: list of (expr, value)
    train_secrets = generate_training_secrets(CONFIG["train_sample_size"])

    pop = init_population(pop_size, max_depth)
    evaluate_population_mathler(pop, train_secrets)

    for gen in range(generations):
        pop.sort(key=lambda ind: ind.fitness)
        best = pop[0]
        avg_fitness = sum(ind.fitness for ind in pop) / len(pop)
        print(
            f"Generation {gen:02d} | "
            f"Best fitness (avg guesses): {best.fitness:.3f} | "
            f"Avg fitness: {avg_fitness:.3f}"
        )
        print(f"  Best individual: {best}")

        # create next generation
        new_pop = []
        # elitism: carry over the best
        new_pop.append(Individual(best.tree.clone()))
        while len(new_pop) < pop_size:
            if RNG.random() < crossover_rate:
                p1 = tournament_select(pop, tournament_k)
                p2 = tournament_select(pop, tournament_k)
                c1, c2 = subtree_crossover(p1, p2, max_depth)
                c1 = subtree_mutation(c1, max_depth, mutation_rate)
                if len(new_pop) < pop_size:
                    new_pop.append(c1)
                if len(new_pop) < pop_size:
                    c2 = subtree_mutation(c2, max_depth, mutation_rate)
                    new_pop.append(c2)
            else:
                p = tournament_select(pop, tournament_k)
                c = subtree_mutation(p, max_depth, mutation_rate)
                new_pop.append(c)

        pop = new_pop
        evaluate_population_mathler(pop, train_secrets)

    pop.sort(key=lambda ind: ind.fitness)
    best = pop[0]
    print("\n=== Finished GP training (Mathler) ===")
    print(f"Best fitness (avg guesses): {best.fitness:.3f}")
    print(f"Best individual: {best}")
    return best


def demo_mathler_game(best_individual):
    """
    Play one demo Mathler game against a random secret.
    """
    # Generate a random secret expression and its target value
    secret_expr = generate_random_valid_expression()
    target_value = safe_eval(secret_expr)

    print("\n=== Demo Mathler game with evolved solver ===")
    print(f"Target value is: {target_value}")
    print("(Secret expression is hidden until the end)")

    guesses_used = play_game_with_individual(
        best_individual,
        secret_expr,
        target_value,
        verbose=True,
    )

    if guesses_used >= FAIL_PENALTY:
        print(f"Failed to solve within {CONFIG['max_guesses']} guesses.")
    else:
        print(f"SOLVED in {int(guesses_used)} guesses!")
    print(f"Secret expression was: {secret_expr}")


if __name__ == "__main__":
    best = evolve_mathler()
    demo_mathler_game(best)
