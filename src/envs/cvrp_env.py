from __future__ import annotations

from typing import Optional
from time import perf_counter

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ho.sa import SASchedule, run_sa
from moves import Move, MoveOperator
from moves.cvrp_moves import (
    apply_relocate,
    delta_relocate,
    is_legal_relocate,
    sample_relocate_move,
)
from problems.cvrp import (
    CVRPInstance,
    CVRPSolution,
    generate_cvrp_instance,
    greedy_cvrp_solution,
)


class CvrpEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_customers: int = 20,
        capacity: float = 30.0,
        rl_steps: int = 50,
        sa_steps: int = 200,
        sa_schedule: Optional[SASchedule] = None,
        sa_converge: bool = False,
        sa_stall_steps: int = 0,
        sa_log_interval: int = 0,
        env_id: int = 0,
        sa_log_only_env: Optional[int] = None,
        seed: Optional[int] = None,
        initial_solution: str = "random",
        tail_scale: float = 1.0,
        log_episode: bool = False,
    ):
        self.n_customers = n_customers
        self.capacity = float(capacity)
        self.max_routes = n_customers
        self.rl_steps = rl_steps
        self.sa_steps = sa_steps
        self.sa_schedule = sa_schedule or SASchedule(t0=1.0, alpha=0.995)
        self.sa_converge = sa_converge
        self.sa_stall_steps = sa_stall_steps
        self.sa_log_interval = sa_log_interval
        self.env_id = env_id
        self.sa_log_only_env = sa_log_only_env
        self.initial_solution = initial_solution
        self.tail_scale = float(tail_scale)
        self.log_episode = log_episode
        self.rng = np.random.default_rng(seed)

        self.action_pairs = self._build_action_pairs()
        self.num_actions = len(self.action_pairs)
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0.0, high=1.0, shape=(n_customers, 7), dtype=np.float32
                ),
                "anchor_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(n_customers + self.max_routes, 7),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
                ),
            }
        )

        self.move_operator = MoveOperator(
            sample=sample_relocate_move,
            apply=apply_relocate,
            delta=delta_relocate,
            is_legal=is_legal_relocate,
        )
        self.instance: Optional[CVRPInstance] = None
        self.solution: Optional[CVRPSolution] = None
        self.step_counter = 0
        self.sum_step_reward = 0.0
        self.cost_s0 = 0.0
        self.last_episode_info = None
        self.episode_start_time = 0.0

    def _build_action_pairs(self) -> np.ndarray:
        pairs = []
        for cust_idx in range(self.n_customers):
            for anchor_idx in range(self.n_customers + self.max_routes):
                pairs.append((cust_idx, anchor_idx))
        return np.array(pairs, dtype=np.int32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "instance" in options:
            self.instance = options["instance"]
        else:
            self.instance = generate_cvrp_instance(
                self.n_customers, self.capacity, seed=seed
            )
        if options and "initial_solution" in options:
            self.solution = options["initial_solution"]
        elif self.initial_solution == "greedy":
            self.solution = greedy_cvrp_solution(self.instance)
        else:
            self.solution = CVRPSolution.random(self.instance, seed=seed)
        self.step_counter = 0
        self.sum_step_reward = 0.0
        self.cost_s0 = float(self.solution.total_cost)
        self.last_episode_info = None
        self.episode_start_time = perf_counter()
        obs = self._get_obs()
        return obs, {"solution": self.solution}

    def step(self, action):
        assert self.instance is not None and self.solution is not None
        action_id = int(action)
        cust_idx, anchor_idx = self.action_pairs[action_id]
        customer = int(cust_idx + 1)
        if anchor_idx < self.n_customers:
            anchor_customer = int(anchor_idx + 1)
            to_route = int(self.solution.customer_route[anchor_customer])
            insert_pos = int(self.solution.customer_pos[anchor_customer])
        else:
            to_route = int(anchor_idx - self.n_customers)
            insert_pos = -1
        move = Move("cvrp_relocate", (customer, to_route, insert_pos))
        reward = 0.0
        if self.move_operator.is_legal(self.solution, move):
            delta = self.move_operator.delta(self.instance, self.solution, move)
            self.move_operator.apply(self.solution, move)
            self.solution.total_cost += delta
            reward = -delta
        self.sum_step_reward += reward
        self.step_counter += 1
        terminated = self.step_counter >= self.rl_steps
        info = {"solution": self.solution}
        if terminated:
            cost_sx = float(self.solution.total_cost)
            tail_improve = 0.0
            sa_stats = {
                "accept_rate": 0.0,
                "avg_delta": 0.0,
                "best_improve": 0.0,
                "final_temp": 0.0,
                "time": 0.0,
            }
            if self.sa_steps > 0:
                log_interval = self.sa_log_interval
                if self.sa_log_only_env is not None and self.env_id != self.sa_log_only_env:
                    log_interval = 0
                best, best_cost, sa_stats = run_sa(
                    self.instance,
                    self.solution,
                    None if self.sa_converge else self.sa_steps,
                    self.sa_schedule,
                    self.move_operator,
                    rng=self.rng,
                    max_steps=self.sa_steps if self.sa_converge else None,
                    stall_steps=self.sa_stall_steps,
                    log_interval=log_interval,
                    log_prefix=f"CVRP#{self.env_id}",
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
                "cost_sxy": float(self.solution.total_cost),
                "sum_step_reward": float(self.sum_step_reward),
                "tail_improve": float(tail_improve),
                "sa_accept_rate": float(sa_stats["accept_rate"]),
                "sa_best_improve": float(sa_stats["best_improve"]),
                "sa_time": float(sa_stats["time"]),
                "sa_steps": float(sa_stats.get("steps", 0.0)),
                "rl_time": float(rl_time),
                "total_time": float(total_time),
                "sa_time_ratio": float(sa_ratio),
            }
            info["episode_stats"] = episode_info
            self.last_episode_info = episode_info
            if self.log_episode:
                print("RLHO episode:", episode_info, flush=True)
        obs = self._get_obs()
        return obs, float(reward), terminated, False, info

    def _get_obs(self):
        coords = self.instance.coords
        depot = coords[0]
        node_features = np.zeros((self.n_customers, 7), dtype=np.float32)
        anchor_features = np.zeros((self.n_customers + self.max_routes, 7), dtype=np.float32)
        for cust in range(1, self.n_customers + 1):
            route_id = self.solution.customer_route[cust]
            route_load = self.solution.route_loads[route_id]
            remaining = self.instance.capacity - route_load
            route_size = len(self.solution.routes[route_id])
            feat = [
                coords[cust, 0],
                coords[cust, 1],
                self.instance.demand[cust] / max(1.0, self.instance.capacity),
                route_load / max(1.0, self.instance.capacity),
                remaining / max(1.0, self.instance.capacity),
                route_size / max(1, self.n_customers),
                0.0,
            ]
            node_features[cust - 1] = feat
            anchor_features[cust - 1] = feat
        for route_id in range(self.max_routes):
            route_load = self.solution.route_loads[route_id]
            remaining = self.instance.capacity - route_load
            route_size = len(self.solution.routes[route_id])
            anchor_features[self.n_customers + route_id] = [
                depot[0],
                depot[1],
                0.0,
                route_load / max(1.0, self.instance.capacity),
                remaining / max(1.0, self.instance.capacity),
                route_size / max(1, self.n_customers),
                1.0,
            ]
        mask = self._action_mask().astype(np.float32)
        return {
            "node_features": node_features,
            "anchor_features": anchor_features,
            "action_mask": mask,
        }

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.bool_)
        for idx, (cust_idx, anchor_idx) in enumerate(self.action_pairs):
            customer = int(cust_idx + 1)
            if anchor_idx < self.n_customers:
                anchor_customer = int(anchor_idx + 1)
                to_route = int(self.solution.customer_route[anchor_customer])
                insert_pos = int(self.solution.customer_pos[anchor_customer])
            else:
                to_route = int(anchor_idx - self.n_customers)
                insert_pos = -1
            move = Move("cvrp_relocate", (customer, to_route, insert_pos))
            mask[idx] = self.move_operator.is_legal(self.solution, move)
        if not mask.any():
            mask[:] = True
        return mask
