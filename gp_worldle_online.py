#!/usr/bin/env python3
import random
import math
from collections import Counter, defaultdict
import requests

# =========================
# Online Wordle vocabulary
# =========================

# GitHub repo: ed-fish/wordle-vocab, vocab.json
# Contains a "vocab" object (~2k answer words) and an "other" object (~10k allowed guesses). :contentReference[oaicite:1]{index=1}
WORD_VOCAB_URL = "https://raw.githubusercontent.com/ed-fish/wordle-vocab/main/vocab.json"


def load_word_lists():
    """
    Download Wordle vocabulary from the online JSON.
    Returns (secret_words, allowed_guesses).
    If download fails, falls back to a tiny local list (so the script still runs).
    """
    try:
        print(f"Downloading Wordle vocab from {WORD_VOCAB_URL} ...")
        resp = requests.get(WORD_VOCAB_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        vocab = data["vocab"]          # main answer list
        other = data.get("other", [])  # extra acceptable guesses

        # Secrets = vocab (like real Wordle)
        secret_words = list(vocab)

        # Allowed guesses = vocab + other (dedup, keep order)
        all_words = list(dict.fromkeys(list(vocab) + list(other)))

        print(f"Loaded {len(secret_words)} secret words and {len(all_words)} allowed guesses from online source.")
        return secret_words, all_words
    except Exception as e:
        print("WARNING: failed to load online word list, using tiny fallback:", e)
        fallback = [
            "cigar", "rebut", "sissy", "humph", "awake",
            "blush", "focal", "evade", "naval", "serve",
        ]
        return fallback, fallback[:]


SECRET_WORDS, ALLOWED_GUESSES = load_word_lists()

# To keep fitness evaluation fast, we train on a random subset of secrets:
TRAIN_SAMPLE_SIZE = min(200, len(SECRET_WORDS))

# =========================
# Wordle environment
# =========================

def wordle_feedback(guess: str, secret: str) -> str:
    """
    Return a simple feedback pattern:
    'G' = green (correct letter, correct position)
    'Y' = yellow (letter in word, wrong position)
    'B' = black/grey (letter not in word)
    NOTE: simplified duplicate-handling, sufficient for this project.
    """
    result = ["B"] * len(secret)
    # first pass: greens
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            result[i] = "G"
    # second pass: yellows (simplified)
    for i, g in enumerate(guess):
        if result[i] == "G":
            continue
        if g in secret:
            result[i] = "Y"
    return "".join(result)


def filter_candidates(candidates, guess, feedback):
    """Keep only candidates that would produce the same feedback."""
    return [
        w for w in candidates
        if wordle_feedback(guess, w) == feedback
    ]


# =========================
# GP tree representation
# =========================

class Node:
    def eval(self, context):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def depth(self):
        raise NotImplementedError


class FuncNode(Node):
    def __init__(self, name, func, arity, children):
        self.name = name
        self.func = func
        self.arity = arity
        self.children = children  # list of Node

    def eval(self, context):
        args = [child.eval(context) for child in self.children]
        return self.func(*args)

    def clone(self):
        return FuncNode(
            self.name,
            self.func,
            self.arity,
            [child.clone() for child in self.children],
        )

    def depth(self):
        return 1 + max(child.depth() for child in self.children)

    def __str__(self):
        if self.arity == 1:
            return f"{self.name}({self.children[0]})"
        return f"{self.name}(" + ", ".join(str(c) for c in self.children) + ")"


class TerminalNode(Node):
    def __init__(self, kind, value):
        """
        kind: "feature" or "const"
        value: feature_name or constant value
        """
        self.kind = kind
        self.value = value

    def eval(self, context):
        if self.kind == "feature":
            return context["features"][self.value]
        elif self.kind == "const":
            return self.value
        else:
            raise ValueError("Unknown terminal kind")

    def clone(self):
        return TerminalNode(self.kind, self.value)

    def depth(self):
        return 1

    def __str__(self):
        if self.kind == "feature":
            return self.value
        return f"{self.value:.2f}"


# =========================
# GP primitives
# =========================

def protected_div(x, y):
    if abs(y) < 1e-6:
        return x
    return x / y


def if_greater(a, b, x, y):
    return x if a > b else y


FUNCTION_SET = [
    ("add", lambda x, y: x + y, 2),
    ("sub", lambda x, y: x - y, 2),
    ("mul", lambda x, y: x * y, 2),
    ("pdiv", protected_div, 2),
    ("max", max, 2),
    ("min", min, 2),
    ("ifgt", if_greater, 4),
]

FEATURES = [
    "letter_freq_sum",
    "positional_freq_sum",
    "unique_letters",
    "remaining_candidates",
]

CONST_RANGE = (-2.0, 2.0)

RNG = random.Random(42)


def random_terminal():
    if RNG.random() < 0.7:
        # feature terminal
        feat = RNG.choice(FEATURES)
        return TerminalNode("feature", feat)
    else:
        # numeric constant
        val = RNG.uniform(*CONST_RANGE)
        return TerminalNode("const", val)


def random_tree(max_depth, grow=True):
    """Generate a random tree using 'grow' or 'full' style."""
    if max_depth == 1:
        return random_terminal()

    if (not grow) or (grow and RNG.random() < 0.5):
        # choose function node
        name, func, arity = RNG.choice(FUNCTION_SET)
        children = [random_tree(max_depth - 1, grow) for _ in range(arity)]
        return FuncNode(name, func, arity, children)
    else:
        return random_terminal()


# =========================
# GP operators
# =========================

def get_all_nodes(node, parent=None, index_in_parent=None, nodes=None):
    """Return list of (node, parent, index_in_parent)."""
    if nodes is None:
        nodes = []
    nodes.append((node, parent, index_in_parent))
    if isinstance(node, FuncNode):
        for i, child in enumerate(node.children):
            get_all_nodes(child, node, i, nodes)
    return nodes


class Individual:
    def __init__(self, tree):
        self.tree = tree
        self.fitness = None

    def __str__(self):
        return str(self.tree)


def subtree_crossover(parent1, parent2, max_depth):
    p1 = parent1.tree.clone()
    p2 = parent2.tree.clone()

    nodes1 = get_all_nodes(p1)
    nodes2 = get_all_nodes(p2)

    node1, parent_of_1, idx1 = RNG.choice(nodes1)
    node2, parent_of_2, idx2 = RNG.choice(nodes2)

    # swap
    if parent_of_1 is None:
        new1 = node2.clone()
    else:
        parent_of_1.children[idx1] = node2.clone()
        new1 = p1

    if parent_of_2 is None:
        new2 = node1.clone()
    else:
        parent_of_2.children[idx2] = node1.clone()
        new2 = p2

    # simple depth cap: if too deep, just regenerate random trees
    if new1.depth() > max_depth:
        new1 = random_tree(max_depth, grow=True)
    if new2.depth() > max_depth:
        new2 = random_tree(max_depth, grow=True)

    return Individual(new1), Individual(new2)


def subtree_mutation(individual, max_depth, mutation_rate=0.1):
    root = individual.tree.clone()
    if RNG.random() > mutation_rate:
        return Individual(root)

    nodes = get_all_nodes(root)
    node, parent, idx = RNG.choice(nodes)

    new_subtree = random_tree(max_depth=max_depth, grow=True)

    if parent is None:
        new_root = new_subtree
    else:
        parent.children[idx] = new_subtree
        new_root = root

    if new_root.depth() > max_depth:
        new_root = random_tree(max_depth, grow=True)

    return Individual(new_root)


def init_population(pop_size, max_depth):
    pop = []
    # ramped half-and-half flavour
    depths = list(range(2, max_depth + 1))
    for i in range(pop_size):
        depth = depths[i % len(depths)]
        grow = (i % 2 == 0)
        tree = random_tree(depth, grow=grow)
        pop.append(Individual(tree))
    return pop


def tournament_select(pop, k=3):
    competitors = RNG.sample(pop, k)
    return min(competitors, key=lambda ind: ind.fitness)


# =========================
# Features for Wordle state
# =========================

def compute_letter_frequencies(candidates):
    counts = Counter("".join(candidates))
    total = sum(counts.values()) or 1
    freqs = {ch: c / total for ch, c in counts.items()}
    return freqs


def compute_positional_frequencies(candidates):
    if not candidates:
        return defaultdict(lambda: [0.0] * 5)
    length = len(candidates[0])
    pos_counts = [Counter() for _ in range(length)]
    for w in candidates:
        for i, ch in enumerate(w):
            pos_counts[i][ch] += 1
    pos_freqs = []
    for i in range(length):
        total = sum(pos_counts[i].values()) or 1
        pos_freqs.append(
            {ch: c / total for ch, c in pos_counts[i].items()}
        )
    return pos_freqs


def word_features(word, candidates):
    letter_freqs = compute_letter_frequencies(candidates)
    pos_freqs = compute_positional_frequencies(candidates)
    # letter frequency sum
    lsum = sum(letter_freqs.get(ch, 0.0) for ch in word)
    # positional frequency sum
    psum = 0.0
    for i, ch in enumerate(word):
        psum += pos_freqs[i].get(ch, 0.0)
    unique_letters = len(set(word))
    remaining = len(candidates)
    return {
        "letter_freq_sum": lsum,
        "positional_freq_sum": psum,
        "unique_letters": float(unique_letters),
        "remaining_candidates": float(remaining),
    }


# =========================
# Fitness evaluation
# =========================

MAX_GUESSES = 6
FAIL_PENALTY = 10.0  # pseudo number of guesses if fail


def play_game_with_individual(individual, secret, verbose=False):
    candidates = ALLOWED_GUESSES[:]
    for guess_num in range(1, MAX_GUESSES + 1):
        # score each candidate using GP tree
        best_word = None
        best_score = -math.inf
        for w in candidates:
            feats = word_features(w, candidates)
            ctx = {"features": feats}
            score = individual.tree.eval(ctx)
            if score > best_score:
                best_score = score
                best_word = w

        guess = best_word
        fb = wordle_feedback(guess, secret)
        if verbose:
            print(f"Guess {guess_num}: {guess}  Feedback: {fb}")

        if guess == secret:
            return float(guess_num)

        candidates = filter_candidates(candidates, guess, fb)

        if not candidates:
            # no options left, fail hard
            return FAIL_PENALTY
    # failed in 6 guesses
    return FAIL_PENALTY


def evaluate_individual(individual, secrets):
    total_guesses = 0.0
    for s in secrets:
        g = play_game_with_individual(individual, s, verbose=False)
        total_guesses += g
    individual.fitness = total_guesses / len(secrets)


def evaluate_population(pop, secrets):
    for ind in pop:
        if ind.fitness is None:
            evaluate_individual(ind, secrets)


# =========================
# Main GP loop + demo
# =========================

def evolve(
    generations=10,
    pop_size=30,
    max_depth=5,
    crossover_rate=0.8,
    mutation_rate=0.2,
    tournament_k=3,
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
    secret = RNG.choice(SECRET_WORDS)
    print("\n=== Demo game with evolved solver ===")
    print("(Secret word is hidden until the end)")

    candidates = ALLOWED_GUESSES[:]
    for guess_num in range(1, MAX_GUESSES + 1):
        best_word = None
        best_score = -math.inf
        for w in candidates:
            feats = word_features(w, candidates)
            ctx = {"features": feats}
            score = best_individual.tree.eval(ctx)
            if score > best_score:
                best_score = score
                best_word = w

        guess = best_word
        fb = wordle_feedback(guess, secret)
        print(f"Guess {guess_num}: {guess}  Feedback: {fb}")

        if guess == secret:
            print(f"SOLVED in {guess_num} guesses!")
            print(f"Secret was: {secret}")
            return

        candidates = filter_candidates(candidates, guess, fb)
        if not candidates:
            print("No candidates left – solver failed.")
            print(f"Secret was: {secret}")
            return

    print("Failed to solve within 6 guesses.")
    print(f"Secret was: {secret}")


if __name__ == "__main__":
    best = evolve(
        generations=10,   # you can increase this later
        pop_size=30,
        max_depth=5,
    )
    demo_game(best)
