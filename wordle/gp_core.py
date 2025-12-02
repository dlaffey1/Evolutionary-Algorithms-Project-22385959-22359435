#!/usr/bin/env python3
import random
from config import CONFIG

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

CONST_RANGE = CONFIG["const_range"]

RNG = random.Random(CONFIG["random_seed"])


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
# GP operators + population
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
