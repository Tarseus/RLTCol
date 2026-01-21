from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CVRPInstance:
    coords: np.ndarray  # shape (n+1, 2), coords[0] is depot
    demand: np.ndarray  # shape (n+1,), demand[0] = 0
    capacity: float
    dist: np.ndarray  # shape (n+1, n+1)

    @property
    def n_customers(self) -> int:
        return int(self.coords.shape[0] - 1)


def _compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1, dtype=np.float64)).astype(np.float32)


def generate_cvrp_instance(
    n_customers: int,
    capacity: float,
    demand_range: Tuple[int, int] = (1, 10),
    seed: Optional[int] = None,
) -> CVRPInstance:
    rng = np.random.default_rng(seed)
    coords = rng.random((n_customers + 1, 2)).astype(np.float32)
    demand = np.zeros(n_customers + 1, dtype=np.float32)
    demand[1:] = rng.integers(demand_range[0], demand_range[1] + 1, size=n_customers)
    if np.any(demand[1:] > capacity):
        raise ValueError("Demand exceeds capacity. Increase capacity or adjust range.")
    dist = _compute_distance_matrix(coords)
    return CVRPInstance(coords=coords, demand=demand, capacity=float(capacity), dist=dist)


def _route_cost(instance: CVRPInstance, route: List[int]) -> float:
    if not route:
        return 0.0
    cost = instance.dist[0, route[0]]
    for i in range(len(route) - 1):
        cost += instance.dist[route[i], route[i + 1]]
    cost += instance.dist[route[-1], 0]
    return float(cost)


class CVRPSolution:
    def __init__(
        self,
        instance: CVRPInstance,
        routes: List[List[int]],
        max_routes: Optional[int] = None,
    ):
        self.instance = instance
        self.max_routes = max_routes or instance.n_customers
        if len(routes) < self.max_routes:
            routes = routes + [[] for _ in range(self.max_routes - len(routes))]
        if len(routes) != self.max_routes:
            raise ValueError("routes length must match max_routes.")
        self.routes = [list(route) for route in routes]
        self.route_loads = np.zeros(self.max_routes, dtype=np.float32)
        self.route_costs = np.zeros(self.max_routes, dtype=np.float32)
        self.customer_route = np.full(instance.n_customers + 1, -1, dtype=np.int32)
        self.customer_pos = np.full(instance.n_customers + 1, -1, dtype=np.int32)
        self.total_cost = 0.0
        self.recompute_costs()

    @classmethod
    def from_routes(
        cls, instance: CVRPInstance, routes: List[List[int]], max_routes: Optional[int] = None
    ) -> "CVRPSolution":
        return cls(instance, routes, max_routes=max_routes)

    @classmethod
    def random(cls, instance: CVRPInstance, seed: Optional[int] = None) -> "CVRPSolution":
        rng = np.random.default_rng(seed)
        customers = list(range(1, instance.n_customers + 1))
        rng.shuffle(customers)
        routes: List[List[int]] = [[]]
        loads = [0.0]
        for cust in customers:
            demand = instance.demand[cust]
            if loads[-1] + demand <= instance.capacity:
                routes[-1].append(cust)
                loads[-1] += demand
            else:
                routes.append([cust])
                loads.append(float(demand))
        return cls(instance, routes)

    def copy(self) -> "CVRPSolution":
        new = CVRPSolution(self.instance, [list(r) for r in self.routes], self.max_routes)
        new.route_loads = self.route_loads.copy()
        new.route_costs = self.route_costs.copy()
        new.customer_route = self.customer_route.copy()
        new.customer_pos = self.customer_pos.copy()
        new.total_cost = float(self.total_cost)
        return new

    @property
    def cost(self) -> float:
        return float(self.total_cost)

    @cost.setter
    def cost(self, value: float) -> None:
        self.total_cost = float(value)

    def recompute_costs(self) -> float:
        total = 0.0
        self.customer_route.fill(-1)
        self.customer_pos.fill(-1)
        for ridx in range(self.max_routes):
            route = self.routes[ridx]
            load = float(np.sum(self.instance.demand[route], dtype=np.float64))
            cost = _route_cost(self.instance, route)
            self.route_loads[ridx] = load
            self.route_costs[ridx] = cost
            total += cost
            for pidx, cust in enumerate(route):
                self.customer_route[cust] = ridx
                self.customer_pos[cust] = pidx
        self.total_cost = float(total)
        return self.total_cost

    def route_cost(self, route: List[int]) -> float:
        return _route_cost(self.instance, route)

    def is_feasible(self) -> bool:
        n = self.instance.n_customers
        seen = set()
        for ridx, route in enumerate(self.routes):
            load = float(np.sum(self.instance.demand[route], dtype=np.float64))
            if load - self.instance.capacity > 1e-6:
                return False
            for cust in route:
                if cust == 0 or cust > n:
                    return False
                if cust in seen:
                    return False
                seen.add(cust)
            if abs(load - self.route_loads[ridx]) > 1e-6:
                return False
        return len(seen) == n


def greedy_cvrp_solution(instance: CVRPInstance) -> CVRPSolution:
    customers = set(range(1, instance.n_customers + 1))
    routes: List[List[int]] = []
    while customers:
        route = []
        load = 0.0
        current = 0
        while True:
            candidates = [
                cust
                for cust in customers
                if load + instance.demand[cust] <= instance.capacity
            ]
            if not candidates:
                break
            next_cust = min(candidates, key=lambda j: instance.dist[current, j])
            route.append(next_cust)
            customers.remove(next_cust)
            load += instance.demand[next_cust]
            current = next_cust
        routes.append(route)
    return CVRPSolution(instance, routes)
