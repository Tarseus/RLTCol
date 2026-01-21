from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class TSPInstance:
    coords: np.ndarray  # shape (n, 2)
    dist: np.ndarray  # shape (n, n)

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])


def _compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1, dtype=np.float64)).astype(np.float32)


def generate_tsp_instance(n: int, seed: Optional[int] = None) -> TSPInstance:
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2)).astype(np.float32)
    dist = _compute_distance_matrix(coords)
    return TSPInstance(coords=coords, dist=dist)


class TSPSolution:
    def __init__(self, instance: TSPInstance, next_nodes: np.ndarray, cost: float):
        self.instance = instance
        self.next = next_nodes.astype(np.int32)
        self.prev = np.empty_like(self.next)
        for i, nxt in enumerate(self.next):
            self.prev[nxt] = i
        self.cost = float(cost)

    @classmethod
    def from_tour(cls, instance: TSPInstance, tour: Iterable[int]) -> "TSPSolution":
        tour_list = list(tour)
        n = len(tour_list)
        next_nodes = np.empty(n, dtype=np.int32)
        for idx, node in enumerate(tour_list):
            next_nodes[node] = tour_list[(idx + 1) % n]
        cost = float(np.sum(instance.dist[np.arange(n), next_nodes]))
        return cls(instance, next_nodes, cost)

    @classmethod
    def random(cls, instance: TSPInstance, seed: Optional[int] = None) -> "TSPSolution":
        rng = np.random.default_rng(seed)
        tour = rng.permutation(instance.n).tolist()
        return cls.from_tour(instance, tour)

    def copy(self) -> "TSPSolution":
        return TSPSolution(self.instance, self.next.copy(), self.cost)

    def tour(self, start: int = 0) -> List[int]:
        tour = [start]
        cur = self.next[start]
        while cur != start:
            tour.append(cur)
            cur = self.next[cur]
        return tour

    def recompute_cost(self) -> float:
        self.cost = float(np.sum(self.instance.dist[np.arange(self.instance.n), self.next]))
        return self.cost

    def verify(self) -> None:
        n = self.instance.n
        visited = set()
        cur = 0
        for _ in range(n):
            if cur in visited:
                raise ValueError("Tour has a cycle before visiting all nodes.")
            visited.add(cur)
            cur = self.next[cur]
        if cur != 0:
            raise ValueError("Tour is not a single cycle.")
        for i in range(n):
            if self.next[self.prev[i]] != i:
                raise ValueError("Next/prev mismatch.")
        cost = self.recompute_cost()
        if not np.isfinite(cost):
            raise ValueError("Invalid tour cost.")


def greedy_tsp_solution(instance: TSPInstance, start: int = 0) -> TSPSolution:
    n = instance.n
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    cur = start
    while unvisited:
        next_node = min(unvisited, key=lambda j: instance.dist[cur, j])
        tour.append(next_node)
        unvisited.remove(next_node)
        cur = next_node
    return TSPSolution.from_tour(instance, tour)
