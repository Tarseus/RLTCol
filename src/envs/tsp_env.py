from __future__ import annotations

from typing import Optional
from time import perf_counter

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ho.sa import SASchedule, run_sa
from moves import Move, MoveOperator
from moves.tsp_moves import (
    apply_two_opt,
    delta_two_opt,
    is_legal_two_opt,
    sample_two_opt_move,
    two_opt_action_mask,
    two_opt_action_pairs,
)
from problems.tsp import (
    TSPInstance,
    TSPSolution,
    generate_tsp_instance,
    greedy_tsp_solution,
)


class TspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_nodes: int = 20,
        rl_steps: int = 50,
        sa_steps: int = 200,
        sa_schedule: Optional[SASchedule] = None,
        sa_converge: bool = False,
        sa_stall_steps: int = 0,
        sa_log_interval: int = 0,
        seed: Optional[int] = None,
        initial_solution: str = "random",
        tail_scale: float = 1.0,
        log_episode: bool = False,
    ):
        self.n_nodes = n_nodes
        self.rl_steps = rl_steps
        self.sa_steps = sa_steps
        self.sa_schedule = sa_schedule or SASchedule(t0=1.0, alpha=0.995)
        self.sa_converge = sa_converge
        self.sa_stall_steps = sa_stall_steps
        self.sa_log_interval = sa_log_interval
        self.initial_solution = initial_solution
        self.tail_scale = float(tail_scale)
        self.log_episode = log_episode
        self.rng = np.random.default_rng(seed)

        self.action_pairs = two_opt_action_pairs(n_nodes)
        self.num_actions = len(self.action_pairs)
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0.0, high=1.0, shape=(n_nodes, 6), dtype=np.float32
                ),
                "anchor_features": spaces.Box(
                    low=0.0, high=1.0, shape=(n_nodes, 6), dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
                ),
            }
        )

        self.move_operator = MoveOperator(
            sample=sample_two_opt_move,
            apply=apply_two_opt,
            delta=delta_two_opt,
            is_legal=is_legal_two_opt,
        )
        self.instance: Optional[TSPInstance] = None
        self.solution: Optional[TSPSolution] = None
        self.step_counter = 0
        self.sum_step_reward = 0.0
        self.cost_s0 = 0.0
        self.last_episode_info = None
        self.episode_start_time = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "instance" in options:
            self.instance = options["instance"]
        else:
            self.instance = generate_tsp_instance(self.n_nodes, seed=seed)
        if options and "initial_solution" in options:
            self.solution = options["initial_solution"]
        elif self.initial_solution == "greedy":
            self.solution = greedy_tsp_solution(self.instance)
        else:
            self.solution = TSPSolution.random(self.instance, seed=seed)
        self.step_counter = 0
        self.sum_step_reward = 0.0
        self.cost_s0 = float(self.solution.cost)
        self.last_episode_info = None
        self.episode_start_time = perf_counter()
        obs = self._get_obs()
        return obs, {"solution": self.solution}

    def step(self, action):
        assert self.instance is not None and self.solution is not None
        action_id = int(action)
        a, c = self.action_pairs[action_id]
        move = Move("tsp_2opt", (int(a), int(c)))
        reward = 0.0
        if self.move_operator.is_legal(self.solution, move):
            delta = self.move_operator.delta(self.instance, self.solution, move)
            self.move_operator.apply(self.solution, move)
            self.solution.cost += delta
            reward = -delta
        self.sum_step_reward += reward
        self.step_counter += 1
        terminated = self.step_counter >= self.rl_steps
        info = {"solution": self.solution}
        if terminated:
            cost_sx = float(self.solution.cost)
            tail_improve = 0.0
            sa_stats = {
                "accept_rate": 0.0,
                "avg_delta": 0.0,
                "best_improve": 0.0,
                "final_temp": 0.0,
                "time": 0.0,
            }
            if self.sa_steps > 0:
                best, best_cost, sa_stats = run_sa(
                    self.instance,
                    self.solution,
                    None if self.sa_converge else self.sa_steps,
                    self.sa_schedule,
                    self.move_operator,
                    rng=self.rng,
                    max_steps=self.sa_steps if self.sa_converge else None,
                    stall_steps=self.sa_stall_steps,
                    log_interval=self.sa_log_interval,
                    log_prefix="TSP",
                )
                self.solution = best
                tail_improve = float(cost_sx - best_cost)
            total_time = perf_counter() - self.episode_start_time
            rl_time = max(0.0, total_time - sa_stats["time"])
            sa_ratio = float(sa_stats["time"] / total_time) if total_time > 0 else 0.0
            # Tail improvement is injected only at terminal step (strict RLHO).
            reward += self.tail_scale * tail_improve
            episode_info = {
                "cost_s0": float(self.cost_s0),
                "cost_sx": cost_sx,
                "cost_sxy": float(self.solution.cost),
                "sum_step_reward": float(self.sum_step_reward),
                "tail_improve": float(tail_improve),
                "sa_accept_rate": float(sa_stats["accept_rate"]),
                "sa_best_improve": float(sa_stats["best_improve"]),
                "sa_time": float(sa_stats["time"]),
                "rl_time": float(rl_time),
                "total_time": float(total_time),
                "sa_time_ratio": float(sa_ratio),
            }
            info["episode_stats"] = episode_info
            self.last_episode_info = episode_info
            if self.log_episode:
                print(
                    "RLHO episode:",
                    episode_info,
                    flush=True,
                )
        obs = self._get_obs()
        return obs, float(reward), terminated, False, info

    def _get_obs(self):
        coords = self.instance.coords
        next_nodes = self.solution.next
        prev_nodes = self.solution.prev
        succ_coords = coords[next_nodes]
        prev_coords = coords[prev_nodes]
        node_features = np.concatenate([coords, succ_coords, prev_coords], axis=1)
        mask = two_opt_action_mask(self.solution, self.action_pairs).astype(np.float32)
        return {
            "node_features": node_features.astype(np.float32),
            "anchor_features": node_features.astype(np.float32),
            "action_mask": mask,
        }
