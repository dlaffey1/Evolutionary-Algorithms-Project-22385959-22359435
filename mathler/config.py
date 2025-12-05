#Light setup for using with DFS candidate generation and GP
CONFIG = {
    "random_seed": 123,
    "generations": 3,
    "pop_size": 20,
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
    "max_candidates": 100,      # was 50; tiny bump
    "min_candidates": 20,      # was 10

    # Logging
    "debug": False,            # set True if you want to see DFS stats

    # Restrict target values for secrets (both training + demo)
    # Set to None to disable the bound.
    "min_target_value": 100,
    "max_target_value": 300,
    "parallel_eval": True,      # Enable parallel fitness evaluation
    "use_constraint_search": True,


    # Lineage Evolution Parameters
    "lineage_pop_size": 70,
    "lineage_gens_first": 8,
    "lineage_gens_per_step": 10,
    "lineage_crossover_rate": 0.4164031446449828,
    "lineage_mutation_rate": 0.2877572136914232,
    "lineage_elite_fraction": 0.13428203916266124,
    "lineage_tournament_k": 3,
    "lineage_mutation_attempts": 55,

    "lineage_w_closeness": 1.3632820383597766,
    "lineage_closeness_max_score": 10.487208984635947,
    "lineage_closeness_diff_clip": 188.33810837155823,
    "lineage_inconsistency_penalty": 21.54012620306195,
    "lineage_w_feedback_match": 0.38429479897497115,

    # feature weights
    "lineage_w_letter_freq_sum": 0.0,
    "lineage_w_positional_freq_sum": 0.616644567516661,
    "lineage_w_unique_letters": 0.09093127777477376,
    "lineage_w_remaining_candidates": 0.0,
    "lineage_w_operator_count": 0.0,
    "lineage_w_digit_count": 0.0,
    "lineage_w_distinct_digits": 0.08461222678064872,
    "lineage_w_plus_count": 0.0,
    "lineage_w_minus_count": 0.0,
    "lineage_w_mul_count": 0.0,
    "lineage_w_div_count": 0.0,
    "lineage_w_symbol_entropy": 0.4058441178368745,
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
