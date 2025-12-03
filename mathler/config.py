# CONFIG = {
#     "random_seed": 42,
#     "generations": 3,
#     "pop_size": 10,
#     "max_depth": 5,
#     "crossover_rate": 0.8,
#     "mutation_rate": 0.2,
#     "tournament_k": 3,
#     "const_range": (-2.0, 2.0),
#     "max_guesses": 6,
#     "fail_penalty": 10.0,
#     "train_sample_size": 20,
#     "max_candidates": 150,   # was effectively 500 before
#     "min_candidates": 40,    # when topping up
#     "debug": False,          # turn on to see logs
# }

CONFIG = {
    "random_seed": 42,

    # Make evolution as cheap as possible
    "generations": 1,          # only 1 generation
    "pop_size": 5,             # tiny population
    "max_depth": 3,            # shallow trees

    "crossover_rate": 0.8,
    "mutation_rate": 0.2,
    "tournament_k": 2,

    # GP terminals
    "const_range": (-2.0, 2.0),

    # Mathler game settings
    "max_guesses": 6,
    "fail_penalty": 10.0,

    # Training set size (how many secrets per fitness eval)
    "train_sample_size": 5,    # very small for speed

    # DFS candidate generation limits
    "max_candidates": 50,      # was 500; much faster
    "min_candidates": 10,      # only top up when it gets tiny

    # Logging
    "debug": False,            # set True if you want to see DFS stats
}
