from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from moves import Move
from problems.cvrp import CVRPInstance, CVRPSolution


def _simulate_relocate(
    routes: list[list[int]],
    from_route: int,
    from_pos: int,
    to_route: int,
    insert_pos: int,
) -> tuple[list[int], list[int]]:
    if from_route == to_route:
        new_route = list(routes[from_route])
        customer = new_route.pop(from_pos)
        if insert_pos >= from_pos:
            insert_pos -= 1
        insert_at = insert_pos + 1
        new_route.insert(insert_at, customer)
        return new_route, new_route
    new_from = list(routes[from_route])
    customer = new_from.pop(from_pos)
    new_to = list(routes[to_route])
    insert_at = insert_pos + 1
    new_to.insert(insert_at, customer)
    return new_from, new_to


def is_legal_relocate(solution: CVRPSolution, move: Move) -> bool:
    customer, to_route, insert_pos = move.params
    if customer <= 0 or customer > solution.instance.n_customers:
        return False
    from_route = solution.customer_route[customer]
    from_pos = solution.customer_pos[customer]
    if from_route < 0:
        return False
    if to_route < 0 or to_route >= solution.max_routes:
        return False
    if insert_pos < -1:
        return False
    if insert_pos >= len(solution.routes[to_route]):
        if len(solution.routes[to_route]) == 0 and insert_pos == -1:
            pass
        elif insert_pos >= len(solution.routes[to_route]):
            return False
    if from_route == to_route:
        if insert_pos >= from_pos:
            insert_pos -= 1
        if insert_pos == from_pos - 1:
            return False
    if from_route != to_route:
        demand = solution.instance.demand[customer]
        if solution.route_loads[to_route] + demand - solution.instance.capacity > 1e-6:
            return False
    return True


def delta_relocate(
    instance: CVRPInstance, solution: CVRPSolution, move: Move
) -> float:
    customer, to_route, insert_pos = move.params
    from_route = solution.customer_route[customer]
    from_pos = solution.customer_pos[customer]
    if from_route < 0:
        return 0.0
    if not is_legal_relocate(solution, move):
        return 0.0
    old_cost = solution.route_costs[from_route]
    if to_route != from_route:
        old_cost += solution.route_costs[to_route]
    new_from, new_to = _simulate_relocate(
        solution.routes, from_route, from_pos, to_route, insert_pos
    )
    new_cost = solution.route_cost(new_from)
    if to_route != from_route:
        new_cost += solution.route_cost(new_to)
    return float(new_cost - old_cost)


def apply_relocate(solution: CVRPSolution, move: Move) -> None:
    if not is_legal_relocate(solution, move):
        return
    customer, to_route, insert_pos = move.params
    from_route = solution.customer_route[customer]
    from_pos = solution.customer_pos[customer]
    if from_route == to_route:
        route = solution.routes[from_route]
        route.pop(from_pos)
        if insert_pos >= from_pos:
            insert_pos -= 1
        insert_at = insert_pos + 1
        route.insert(insert_at, customer)
        solution.routes[from_route] = route
        solution.route_costs[from_route] = solution.route_cost(route)
        solution.route_loads[from_route] = float(
            np.sum(solution.instance.demand[route], dtype=np.float64)
        )
        for idx, cust in enumerate(route):
            solution.customer_route[cust] = from_route
            solution.customer_pos[cust] = idx
        return
    from_route_list = solution.routes[from_route]
    customer_value = from_route_list.pop(from_pos)
    to_route_list = solution.routes[to_route]
    insert_at = insert_pos + 1
    to_route_list.insert(insert_at, customer_value)
    solution.routes[from_route] = from_route_list
    solution.routes[to_route] = to_route_list
    solution.route_costs[from_route] = solution.route_cost(from_route_list)
    solution.route_costs[to_route] = solution.route_cost(to_route_list)
    solution.route_loads[from_route] = float(
        np.sum(solution.instance.demand[from_route_list], dtype=np.float64)
    )
    solution.route_loads[to_route] = float(
        np.sum(solution.instance.demand[to_route_list], dtype=np.float64)
    )
    for idx, cust in enumerate(from_route_list):
        solution.customer_route[cust] = from_route
        solution.customer_pos[cust] = idx
    for idx, cust in enumerate(to_route_list):
        solution.customer_route[cust] = to_route
        solution.customer_pos[cust] = idx


def is_legal_swap(solution: CVRPSolution, move: Move) -> bool:
    a, b = move.params
    if a == b:
        return False
    if a <= 0 or b <= 0:
        return False
    if a > solution.instance.n_customers or b > solution.instance.n_customers:
        return False
    ra = solution.customer_route[a]
    rb = solution.customer_route[b]
    if ra < 0 or rb < 0:
        return False
    if ra == rb:
        return True
    demand_a = solution.instance.demand[a]
    demand_b = solution.instance.demand[b]
    load_a = solution.route_loads[ra] - demand_a + demand_b
    load_b = solution.route_loads[rb] - demand_b + demand_a
    if load_a - solution.instance.capacity > 1e-6:
        return False
    if load_b - solution.instance.capacity > 1e-6:
        return False
    return True


def delta_swap(instance: CVRPInstance, solution: CVRPSolution, move: Move) -> float:
    a, b = move.params
    if not is_legal_swap(solution, move):
        return 0.0
    ra = solution.customer_route[a]
    rb = solution.customer_route[b]
    old_cost = solution.route_costs[ra]
    if rb != ra:
        old_cost += solution.route_costs[rb]
    if ra == rb:
        route = list(solution.routes[ra])
        pa = solution.customer_pos[a]
        pb = solution.customer_pos[b]
        route[pa], route[pb] = route[pb], route[pa]
        new_cost = solution.route_cost(route)
        return float(new_cost - old_cost)
    route_a = list(solution.routes[ra])
    route_b = list(solution.routes[rb])
    pa = solution.customer_pos[a]
    pb = solution.customer_pos[b]
    route_a[pa], route_b[pb] = route_b[pb], route_a[pa]
    new_cost = solution.route_cost(route_a) + solution.route_cost(route_b)
    return float(new_cost - old_cost)


def apply_swap(solution: CVRPSolution, move: Move) -> None:
    if not is_legal_swap(solution, move):
        return
    a, b = move.params
    ra = solution.customer_route[a]
    rb = solution.customer_route[b]
    pa = solution.customer_pos[a]
    pb = solution.customer_pos[b]
    if ra == rb:
        route = solution.routes[ra]
        route[pa], route[pb] = route[pb], route[pa]
        solution.routes[ra] = route
        solution.route_costs[ra] = solution.route_cost(route)
        solution.route_loads[ra] = float(
            np.sum(solution.instance.demand[route], dtype=np.float64)
        )
        for idx, cust in enumerate(route):
            solution.customer_route[cust] = ra
            solution.customer_pos[cust] = idx
        return
    route_a = solution.routes[ra]
    route_b = solution.routes[rb]
    route_a[pa], route_b[pb] = route_b[pb], route_a[pa]
    solution.routes[ra] = route_a
    solution.routes[rb] = route_b
    solution.route_costs[ra] = solution.route_cost(route_a)
    solution.route_costs[rb] = solution.route_cost(route_b)
    solution.route_loads[ra] = float(
        np.sum(solution.instance.demand[route_a], dtype=np.float64)
    )
    solution.route_loads[rb] = float(
        np.sum(solution.instance.demand[route_b], dtype=np.float64)
    )
    for idx, cust in enumerate(route_a):
        solution.customer_route[cust] = ra
        solution.customer_pos[cust] = idx
    for idx, cust in enumerate(route_b):
        solution.customer_route[cust] = rb
        solution.customer_pos[cust] = idx


def sample_relocate_move(solution: CVRPSolution, rng: np.random.Generator) -> Move:
    customers = np.arange(1, solution.instance.n_customers + 1)
    customer = int(rng.choice(customers))
    to_route = int(rng.integers(0, solution.max_routes))
    route_len = len(solution.routes[to_route])
    insert_pos = int(rng.integers(-1, max(route_len, 1)))
    if route_len == 0:
        insert_pos = -1
    return Move("cvrp_relocate", (customer, to_route, insert_pos))


def sample_swap_move(solution: CVRPSolution, rng: np.random.Generator) -> Move:
    customers = np.arange(1, solution.instance.n_customers + 1)
    a, b = rng.choice(customers, size=2, replace=False)
    return Move("cvrp_swap", (int(a), int(b)))
