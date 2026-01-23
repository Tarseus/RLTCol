import argparse
import time

import numpy as np
import torch
from tianshou.data import Batch, Collector
from tianshou.env import DummyVectorEnv
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


def build_env(problem, args, log_episode=False):
    schedule = SASchedule(t0=args.sa_t0, alpha=args.sa_alpha)
    if problem == "tsp":
        return TspEnv(
            n_nodes=args.nodes,
            rl_steps=args.rl_steps,
            sa_steps=args.sa_steps,
            sa_schedule=schedule,
            sa_converge=args.sa_converge,
            sa_stall_steps=args.sa_stall_steps,
            seed=args.seed,
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
        seed=args.seed,
        tail_scale=args.tail_scale,
        log_episode=log_episode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate strict RLHO routing variants")
    parser.add_argument("--problem", choices=["tsp", "cvrp"], default="tsp")
    parser.add_argument("--policy", type=str, default=None, help="Path to PPO policy")
    parser.add_argument("--mode", choices=["rlho", "rl-only", "sa-only", "random-sa", "greedy-sa"], default="rlho")
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--customers", type=int, default=20)
    parser.add_argument("--capacity", type=float, default=30.0)
    parser.add_argument("--rl-steps", type=int, default=50)
    parser.add_argument("--sa-steps", type=int, default=200)
    parser.add_argument("--sa-t0", type=float, default=1.0)
    parser.add_argument("--sa-alpha", type=float, default=0.995)
    parser.add_argument("--sa-converge", action="store_true")
    parser.add_argument("--sa-stall-steps", type=int, default=0)
    parser.add_argument("--tail-scale", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode in {"rlho", "rl-only"}:
        env = build_env(args.problem, args, log_episode=True)
        if args.mode == "rl-only":
            env.sa_steps = 0
            env.tail_scale = 0.0
        vector_env = DummyVectorEnv([lambda: env])
        action_pairs = env.action_pairs
        node_dim = env.observation_space["node_features"].shape[-1]
        anchor_dim = env.observation_space["anchor_features"].shape[-1]
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
            action_space=env.action_space,
        )
        if args.policy is None:
            policy = RandomPolicy(action_space=env.action_space)
        else:
            policy.load_state_dict(torch.load(args.policy, map_location=device))
        policy.eval()
        collector = Collector(policy, vector_env)
        collector.reset()
        for _ in range(args.episodes):
            collector.collect(n_episode=1)
        print("Last episode stats:", env.last_episode_info)
        raise SystemExit(0)

    if args.problem == "tsp":
        instance = generate_tsp_instance(args.nodes, seed=args.seed)
        if args.mode == "greedy-sa":
            solution = greedy_tsp_solution(instance)
        else:
            solution = TSPSolution.random(instance, seed=args.seed)
        move_operator = MoveOperator(
            sample=sample_two_opt_move,
            apply=apply_two_opt,
            delta=delta_two_opt,
            is_legal=is_legal_two_opt,
        )
    else:
        instance = generate_cvrp_instance(args.customers, args.capacity, seed=args.seed)
        if args.mode == "greedy-sa":
            solution = greedy_cvrp_solution(instance)
        else:
            solution = CVRPSolution.random(instance, seed=args.seed)
        move_operator = MoveOperator(
            sample=sample_relocate_move,
            apply=apply_relocate,
            delta=delta_relocate,
            is_legal=is_legal_relocate,
        )

    if args.mode == "random-sa":
        pass
    elif args.mode == "greedy-sa":
        pass
    elif args.mode == "sa-only":
        pass
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    schedule = SASchedule(t0=args.sa_t0, alpha=args.sa_alpha)
    start_cost = float(solution.cost if args.problem == "tsp" else solution.total_cost)
    start = time.time()
    best, best_cost, stats = run_sa(
        instance,
        solution,
        None if args.sa_converge else args.sa_steps,
        schedule,
        move_operator,
        max_steps=args.sa_steps if args.sa_converge else None,
        stall_steps=args.sa_stall_steps,
    )
    elapsed = time.time() - start
    print(
        {
            "mode": args.mode,
            "cost_s0": start_cost,
            "cost_sxy": float(best_cost),
            "sa_accept_rate": stats["accept_rate"],
            "sa_best_improve": stats["best_improve"],
            "sa_time": stats["time"],
            "total_time": elapsed,
        }
    )
