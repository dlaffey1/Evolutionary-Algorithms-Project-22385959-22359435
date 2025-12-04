#Light setup for using with DFS candidate generation and GP
CONFIG = {
    "random_seed": 42,
    "generations": 3,          # was 1
    "pop_size": 8,             # was 5
    "max_depth": 4,            # was 3
    "crossover_rate": 0.8,
    "mutation_rate": 0.2,
    "tournament_k": 3,
    "const_range": (-2.0, 2.0),

    # Mathler game settings
    "max_guesses": 6,
    "fail_penalty": 10.0,

    # Training set size (how many secrets per fitness eval)
    "train_sample_size": 8,    # was 5, still small

    # DFS candidate generation limits
    "max_candidates": 60,      # was 50; tiny bump
    "min_candidates": 15,      # was 10

    # Logging
    "debug": False,            # set True if you want to see DFS stats

    # Restrict target values for secrets (both training + demo)
    # Set to None to disable the bound.
    "min_target_value": 100,   # only allow targets >= 100
    "max_target_value": 300,   # only allow targets <= 200
    "parallel_eval": True,      # Enable parallel fitness evaluation
    "use_constraint_search": True,
}

# #Heavier non-DFS setup
# CONFIG = {
#     "random_seed": 42,

#     # Evolution budget
#     "generations": 5,
#     "pop_size": 10,
#     "max_depth": 4,

#     "crossover_rate": 0.8,
#     "mutation_rate": 0.3,
#     "tournament_k": 3,

#     "const_range": (-2.0, 2.0),

#     "max_guesses": 6,
#     "fail_penalty": 10.0,

#     "train_sample_size": 10,

#     # Candidate generation limits (RANDOM)
#     "max_candidates": 120,        # was 80 – aim for a bigger pool
#     "min_candidates": 30,         # don't top up if we still have 30+

#     # Be much more persistent when hunting for candidates
#     "max_init_attempts": 40000,   # was 8000
#     "max_topup_attempts": 20000,  # was 4000

#     "debug": False,               # turn off spam for now

#     "min_target_value": 100,
#     "max_target_value": 200,

#     "parallel_eval": True,
#     "use_constraint_search": False,
# }
