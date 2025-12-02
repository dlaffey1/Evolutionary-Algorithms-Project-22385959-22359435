CONFIG = {
    "random_seed": 42,

    # GP hyperparameters (development mode)
    "generations": 2,
    "pop_size": 5,
    "max_depth": 4,
    "crossover_rate": 0.8,
    "mutation_rate": 0.2,
    "tournament_k": 3,

    "const_range": (-2.0, 2.0),

    "max_guesses": 6,
    "fail_penalty": 10.0,

    # small training subset of secrets
    "train_sample_size": 10,
}
