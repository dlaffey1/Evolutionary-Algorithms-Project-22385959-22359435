#!/usr/bin/env python3
from config import CONFIG
import random
from gp_core import (
    RNG,
    Individual,
    init_population,
    tournament_select,
    subtree_crossover,
    subtree_mutation,
)
from online_vocab import SECRET_WORDS, TRAIN_SAMPLE_SIZE
from fitness import (
    evaluate_population,
    play_game_with_individual,
    FAIL_PENALTY,
)


def evolve(
    generations=CONFIG["generations"],
    pop_size=CONFIG["pop_size"],
    max_depth=CONFIG["max_depth"],
    crossover_rate=CONFIG["crossover_rate"],
    mutation_rate=CONFIG["mutation_rate"],
    tournament_k=CONFIG["tournament_k"],
):
    # random subset of secrets for training, from the ONLINE list
    train_secrets = RNG.sample(SECRET_WORDS, TRAIN_SAMPLE_SIZE)

    pop = init_population(pop_size, max_depth)
    evaluate_population(pop, train_secrets)

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
        evaluate_population(pop, train_secrets)

    pop.sort(key=lambda ind: ind.fitness)
    best = pop[0]
    print("\n=== Finished GP training ===")
    print(f"Best fitness (avg guesses): {best.fitness:.3f}")
    print(f"Best individual: {best}")
    return best


def demo_game(best_individual):
    """Play one demo game against a random secret from the ONLINE list."""
    secret = random.choice(SECRET_WORDS)  # uses system-random seed
    print("\n=== Demo game with evolved solver ===")
    print("(Secret word is hidden until the end)")

    guesses_used = play_game_with_individual(
        best_individual,
        secret,
        verbose=True,
    )

    if guesses_used >= FAIL_PENALTY:
        print(f"Failed to solve within {CONFIG['max_guesses']} guesses.")
    else:
        print(f"SOLVED in {int(guesses_used)} guesses!")
    print(f"Secret was: {secret}")


if __name__ == "__main__":
    best = evolve()
    demo_game(best)
