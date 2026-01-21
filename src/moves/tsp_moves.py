from __future__ import annotations

from typing import Tuple

import numpy as np

from moves import Move
from problems.tsp import TSPInstance, TSPSolution


def two_opt_action_pairs(n: int) -> np.ndarray:
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return np.array(pairs, dtype=np.int32)


def two_opt_action_mask(solution: TSPSolution, action_pairs: np.ndarray) -> np.ndarray:
    next_nodes = solution.next
    mask = np.ones(len(action_pairs), dtype=np.bool_)
    for idx, (a, c) in enumerate(action_pairs):
        if a == c or next_nodes[a] == c or next_nodes[c] == a:
            mask[idx] = False
    if not mask.any():
        mask[:] = True
    return mask


def is_legal_two_opt(solution: TSPSolution, move: Move) -> bool:
    a, c = move.params
    if a == c:
        return False
    if solution.next[a] == c or solution.next[c] == a:
        return False
    return True


def delta_two_opt(instance: TSPInstance, solution: TSPSolution, move: Move) -> float:
    a, c = move.params
    b = solution.next[a]
    d = solution.next[c]
    if b == c or d == a:
        return 0.0
    dist = instance.dist
    return float(dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d])


def apply_two_opt(solution: TSPSolution, move: Move) -> None:
    a, c = move.params
    b = solution.next[a]
    d = solution.next[c]
    if b == c or d == a:
        return
    segment = []
    cur = b
    while True:
        segment.append(cur)
        if cur == c:
            break
        cur = solution.next[cur]
    reversed_segment = list(reversed(segment))
    solution.next[a] = reversed_segment[0]
    solution.prev[reversed_segment[0]] = a
    solution.next[reversed_segment[-1]] = d
    solution.prev[d] = reversed_segment[-1]
    for idx in range(len(reversed_segment) - 1):
        left = reversed_segment[idx]
        right = reversed_segment[idx + 1]
        solution.next[left] = right
        solution.prev[right] = left


def sample_two_opt_move(solution: TSPSolution, rng: np.random.Generator) -> Move:
    n = solution.instance.n
    for _ in range(50):
        a, c = rng.choice(n, size=2, replace=False)
        move = Move("tsp_2opt", (int(a), int(c)))
        if is_legal_two_opt(solution, move):
            return move
    a, c = 0, 1
    return Move("tsp_2opt", (int(a), int(c)))
