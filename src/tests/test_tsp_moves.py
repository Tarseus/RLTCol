from unittest import TestCase

import numpy as np
import os.path
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from moves.tsp_moves import apply_two_opt, delta_two_opt, sample_two_opt_move
from problems.tsp import TSPSolution, generate_tsp_instance


class TestTspMoves(TestCase):
    def test_two_opt_delta(self):
        instance = generate_tsp_instance(12, seed=123)
        solution = TSPSolution.random(instance, seed=456)
        rng = np.random.default_rng(1)
        for _ in range(200):
            move = sample_two_opt_move(solution, rng)
            before = solution.cost
            delta = delta_two_opt(instance, solution, move)
            candidate = solution.copy()
            apply_two_opt(candidate, move)
            candidate.cost = before + delta
            recomputed = candidate.recompute_cost()
            self.assertLess(abs(candidate.cost - recomputed), 1e-6)
