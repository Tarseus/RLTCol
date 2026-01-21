from unittest import TestCase

import numpy as np
import os.path
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from moves.cvrp_moves import (
    apply_relocate,
    apply_swap,
    delta_relocate,
    delta_swap,
    is_legal_relocate,
    is_legal_swap,
    sample_relocate_move,
    sample_swap_move,
)
from problems.cvrp import CVRPSolution, generate_cvrp_instance


class TestCvrpMoves(TestCase):
    def test_relocate_delta(self):
        instance = generate_cvrp_instance(12, capacity=25, seed=7)
        solution = CVRPSolution.random(instance, seed=9)
        rng = np.random.default_rng(2)
        for _ in range(200):
            move = sample_relocate_move(solution, rng)
            if not is_legal_relocate(solution, move):
                continue
            before = solution.cost
            delta = delta_relocate(instance, solution, move)
            candidate = solution.copy()
            apply_relocate(candidate, move)
            candidate.cost = before + delta
            recomputed = candidate.recompute_costs()
            self.assertLess(abs(candidate.cost - recomputed), 1e-6)
            self.assertTrue(candidate.is_feasible())

    def test_swap_delta(self):
        instance = generate_cvrp_instance(12, capacity=25, seed=11)
        solution = CVRPSolution.random(instance, seed=13)
        rng = np.random.default_rng(4)
        for _ in range(200):
            move = sample_swap_move(solution, rng)
            if not is_legal_swap(solution, move):
                continue
            before = solution.cost
            delta = delta_swap(instance, solution, move)
            candidate = solution.copy()
            apply_swap(candidate, move)
            candidate.cost = before + delta
            recomputed = candidate.recompute_costs()
            self.assertLess(abs(candidate.cost - recomputed), 1e-6)
            self.assertTrue(candidate.is_feasible())
