from unittest import TestCase

import numpy as np
import os.path
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from ho.sa import SASchedule, run_sa
from moves import MoveOperator
from moves.tsp_moves import apply_two_opt, delta_two_opt, is_legal_two_opt, sample_two_opt_move
from problems.tsp import TSPSolution, generate_tsp_instance


class TestSA(TestCase):
    def test_acceptance_rate_temperature(self):
        instance = generate_tsp_instance(10, seed=5)
        solution = TSPSolution.random(instance, seed=6)
        move_operator = MoveOperator(
            sample=sample_two_opt_move,
            apply=apply_two_opt,
            delta=delta_two_opt,
            is_legal=is_legal_two_opt,
        )
        rng = np.random.default_rng(3)
        high_schedule = SASchedule(t0=10.0, alpha=0.99)
        low_schedule = SASchedule(t0=0.01, alpha=0.99)
        _, _, high_stats = run_sa(
            instance, solution, steps=200, schedule=high_schedule, move_operator=move_operator, rng=rng
        )
        rng = np.random.default_rng(3)
        _, _, low_stats = run_sa(
            instance, solution, steps=200, schedule=low_schedule, move_operator=move_operator, rng=rng
        )
        self.assertGreaterEqual(high_stats["accept_rate"], low_stats["accept_rate"])
