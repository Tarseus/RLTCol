import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional

import numpy as np
import torch
from tianshou.data import Batch, Collector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import BasePolicy, PPOPolicy
from tianshou.utils.net.common import ActorCritic

from envs.cvrp_env import CvrpEnv
from envs.tsp_env import TspEnv
from ho.sa import SASchedule, run_sa
from moves import MoveOperator
from moves.cvrp_moves import (
    apply_relocate,
    delta_relocate,
    is_legal_relocate,
    sample_relocate_move,
)
from moves.tsp_moves import (
    apply_two_opt,
    delta_two_opt,
    is_legal_two_opt,
    sample_two_opt_move,
)
from problems.cvrp import CVRPSolution, generate_cvrp_instance, greedy_cvrp_solution
from problems.tsp import TSPSolution, generate_tsp_instance, greedy_tsp_solution
from routing_network import PairActorNetwork, RoutingCriticNetwork


class RandomPolicy(BasePolicy):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def forward(self, batch, state=None, **kwargs):
        action = np.array([self.action_space.sample()])
        return Batch(act=action)

    def learn(self, batch: Batch, **kwargs):
        return {}


@dataclass
class EvalStats:
    episodes: int
    mean_cost_s0: float
    mean_cost_sx: float
    mean_cost_sxy: float
    mean_tail_improve: float
    mean_sa_accept_rate: float
    mean_sa_time: float
    mean_sa_steps: float
    mean_total_time: float
    mean_sa_time_ratio: float


def build_env(problem, args, env_id: int, log_episode: bool):
    schedule = SASchedule(t0=args.sa_t0, alpha=args.sa_alpha)
    log_only_env = None if args.sa_log_env < 0 else args.sa_log_env
    if problem == "tsp":
        return TspEnv(
            n_nodes=args.nodes,
            rl_steps=args.rl_steps,
            sa_steps=args.sa_steps,
            sa_schedule=schedule,
            sa_converge=args.sa_converge,
            sa_stall_steps=args.sa_stall_steps,
            sa_log_interval=args.sa_log_interval,
            env_id=env_id,
            sa_log_only_env=log_only_env,
            seed=args.seed + env_id,
            tail_scale=args.tail_scale,
            log_episode=log_episode,
        )
    return CvrpEnv(
        n_customers=args.customers,
        capacity=args.capacity,
        rl_steps=args.rl_steps,
        sa_steps=args.sa_steps,
        sa_schedule=schedule,
        sa_converge=args.sa_converge,
        sa_stall_steps=args.sa_stall_steps,
        sa_log_interval=args.sa_log_interval,
        env_id=env_id,
        sa_log_only_env=log_only_env,
        seed=args.seed + env_id,
        tail_scale=args.tail_scale,
        log_episode=log_episode,
    )


def build_policy(action_pairs, node_dim, anchor_dim, action_space, device, policy_path):
    actor = PairActorNetwork(
        node_input_dim=node_dim,
        anchor_input_dim=anchor_dim,
        action_pairs=action_pairs,
        device=device,
    ).to(device)
    critic = RoutingCriticNetwork(node_input_dim=node_dim, device=device).to(device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)
    dist = lambda logits: torch.distributions.Categorical(logits=logits)
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        action_space=action_space,
    )
    if policy_path:
        policy.load_state_dict(torch.load(policy_path, map_location=device))
    return policy


def summarize(stats: List[dict]) -> EvalStats:
    def mean(key):
        return float(np.mean([s[key] for s in stats])) if stats else 0.0

    return EvalStats(
        episodes=len(stats),
        mean_cost_s0=mean("cost_s0"),
        mean_cost_sx=mean("cost_sx"),
        mean_cost_sxy=mean("cost_sxy"),
        mean_tail_improve=mean("tail_improve"),
        mean_sa_accept_rate=mean("sa_accept_rate"),
        mean_sa_time=mean("sa_time"),
        mean_sa_steps=mean("sa_steps"),
        mean_total_time=mean("total_time"),
        mean_sa_time_ratio=mean("sa_time_ratio"),
    )


def print_summary(summary: EvalStats):
    print(
        {
            "episodes": summary.episodes,
            "mean_cost_s0": summary.mean_cost_s0,
            "mean_cost_sx": summary.mean_cost_sx,
            "mean_cost_sxy": summary.mean_cost_sxy,
            "mean_tail_improve": summary.mean_tail_improve,
            "mean_sa_accept_rate": summary.mean_sa_accept_rate,
            "mean_sa_time": summary.mean_sa_time,
            "mean_sa_steps": summary.mean_sa_steps,
            "mean_total_time": summary.mean_total_time,
            "mean_sa_time_ratio": summary.mean_sa_time_ratio,
        }
    )


def eval_rlho(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "rl-only":
        args.sa_steps = 0
        args.tail_scale = 0.0
        args.sa_converge = False
        args.sa_log_interval = 0

    env_fns = [
        (lambda env_id=env_id: build_env(args.problem, args, env_id=env_id, log_episode=False))
        for env_id in range(args.num_envs)
    ]
    vector_env = (
        SubprocVectorEnv(env_fns) if args.num_envs > 1 else DummyVectorEnv(env_fns)
    )

    sample_env = build_env(args.problem, args, env_id=0, log_episode=False)
    policy = build_policy(
        action_pairs=sample_env.action_pairs,
        node_dim=sample_env.observation_space["node_features"].shape[-1],
        anchor_dim=sample_env.observation_space["anchor_features"].shape[-1],
        action_space=sample_env.action_space,
        device=device,
        policy_path=args.policy if args.mode == "rlho" else None,
    )
    if args.policy is None and args.mode == "rlho":
        policy = RandomPolicy(action_space=sample_env.action_space)
    policy.eval()

    collector = Collector(policy, vector_env)
    collector.reset()

    stats: List[dict] = []
    episodes_done = 0
    start_time = perf_counter()
    while episodes_done < args.episodes:
        batch = args.episodes if args.log_interval <= 0 else args.log_interval
        n = min(batch, args.episodes - episodes_done)
        result = collector.collect(n_episode=n)
        infos = result.get("infos", [])
        flat_infos = []
        for info in infos:
            if isinstance(info, (list, tuple)):
                flat_infos.extend(info)
            else:
                flat_infos.append(info)
        stats.extend(
            [
                info["episode_stats"]
                for info in flat_infos
                if info and "episode_stats" in info
            ]
        )
        episodes_in_batch = 0
        for key in ("n/ep", "n_episode", "n_ep", "n_episodes"):
            if key in result:
                try:
                    episodes_in_batch = int(result[key])
                except (TypeError, ValueError):
                    episodes_in_batch = 0
                break
        if episodes_in_batch == 0:
            episodes_in_batch = sum(
                1 for info in flat_infos if info and "episode_stats" in info
            )
        episodes_done += episodes_in_batch
        if args.log_interval > 0:
            elapsed = perf_counter() - start_time
            rate = episodes_done / elapsed if elapsed > 0 else 0.0
            eta = (args.episodes - episodes_done) / rate if rate > 0 else 0.0
            if episodes_done > 0:
                mean_cost = float(np.mean([s["cost_sxy"] for s in stats]))
                mean_tail = float(np.mean([s["tail_improve"] for s in stats]))
                print(
                    f"[progress] {episodes_done}/{args.episodes} "
                    f"mean_cost_sxy={mean_cost:.4f} mean_tail={mean_tail:.4f} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

    summary = summarize(stats)
    print_summary(summary)


def eval_sa_only(args):
    rng = np.random.default_rng(args.seed)
    schedule = SASchedule(t0=args.sa_t0, alpha=args.sa_alpha)
    stats = []
    start_time = perf_counter()

    for ep in range(args.episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        if args.problem == "tsp":
            instance = generate_tsp_instance(args.nodes, seed=seed)
            solution = (
                greedy_tsp_solution(instance)
                if args.mode == "greedy-sa"
                else TSPSolution.random(instance, seed=seed)
            )
            move_operator = MoveOperator(
                sample=sample_two_opt_move,
                apply=apply_two_opt,
                delta=delta_two_opt,
                is_legal=is_legal_two_opt,
            )
            cost_s0 = float(solution.cost)
        else:
            instance = generate_cvrp_instance(args.customers, args.capacity, seed=seed)
            solution = (
                greedy_cvrp_solution(instance)
                if args.mode == "greedy-sa"
                else CVRPSolution.random(instance, seed=seed)
            )
            move_operator = MoveOperator(
                sample=sample_relocate_move,
                apply=apply_relocate,
                delta=delta_relocate,
                is_legal=is_legal_relocate,
            )
            cost_s0 = float(solution.total_cost)

        best, best_cost, sa_stats = run_sa(
            instance,
            solution,
            None if args.sa_converge else args.sa_steps,
            schedule,
            move_operator,
            max_steps=args.sa_steps if args.sa_converge else None,
            stall_steps=args.sa_stall_steps,
            log_interval=args.sa_log_interval,
            log_prefix="SA",
        )
        stats.append(
            {
                "cost_s0": cost_s0,
                "cost_sx": cost_s0,
                "cost_sxy": float(best_cost),
                "tail_improve": float(cost_s0 - best_cost),
                "sa_accept_rate": float(sa_stats["accept_rate"]),
                "sa_time": float(sa_stats["time"]),
                "sa_steps": float(sa_stats.get("steps", 0.0)),
                "total_time": float(sa_stats["time"]),
                "sa_time_ratio": 1.0,
            }
        )

        if args.log_interval > 0 and (ep + 1) % args.log_interval == 0:
            elapsed = perf_counter() - start_time
            mean_cost = float(np.mean([s["cost_sxy"] for s in stats]))
            print(
                f"[progress] {ep + 1}/{args.episodes} mean_cost_sxy={mean_cost:.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    summary = summarize(stats)
    print_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate strict RLHO routing variants")
    parser.add_argument("--problem", choices=["tsp", "cvrp"], default="tsp")
    parser.add_argument("--policy", type=str, default=None, help="Path to PPO policy")
    parser.add_argument(
        "--mode",
        choices=["rlho", "rl-only", "sa-only", "random-sa", "greedy-sa"],
        default="rlho",
    )
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--customers", type=int, default=20)
    parser.add_argument("--capacity", type=float, default=30.0)
    parser.add_argument("--rl-steps", type=int, default=50)
    parser.add_argument("--sa-steps", type=int, default=200)
    parser.add_argument("--sa-t0", type=float, default=1.0)
    parser.add_argument("--sa-alpha", type=float, default=0.995)
    parser.add_argument("--sa-converge", action="store_true")
    parser.add_argument("--sa-stall-steps", type=int, default=0)
    parser.add_argument(
        "--sa-log-interval",
        type=int,
        default=0,
        help="Print SA progress every N SA steps (0 disables)",
    )
    parser.add_argument(
        "--sa-log-env",
        type=int,
        default=0,
        help="Only log SA from this env id (-1 logs all envs)",
    )
    parser.add_argument("--tail-scale", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Parallel envs for evaluation"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Print progress every N episodes (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode in {"rlho", "rl-only"}:
        eval_rlho(args)
    else:
        eval_sa_only(args)
