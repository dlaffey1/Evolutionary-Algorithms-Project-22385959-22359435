#!/usr/bin/env python3

CONFIG = {
    # Randomness / reproducibility
    "random_seed": 42,

    # GP hyperparameters
    "generations": 10,
    "pop_size": 30,
    "max_depth": 5,
    "crossover_rate": 0.8,
    "mutation_rate": 0.2,
    "tournament_k": 3,

    # GP numeric terminals
    "const_range": (-2.0, 2.0),

    # Wordle / fitness settings
    "max_guesses": 6,
    "fail_penalty": 10.0,

    # Training subset size (from online vocab)
    "train_sample_size": 200,
}
