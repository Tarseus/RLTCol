import argparse
import os

import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic

from envs.cvrp_env import CvrpEnv
from envs.tsp_env import TspEnv
from ho.sa import SASchedule
from routing_network import PairActorNetwork, RoutingCriticNetwork


def make_env(problem, args, seed, log_episode=False):
    schedule = SASchedule(t0=args.sa_t0, alpha=args.sa_alpha)
    if problem == "tsp":
        return TspEnv(
            n_nodes=args.nodes,
            rl_steps=args.rl_steps,
            sa_steps=args.sa_steps,
            sa_schedule=schedule,
            seed=seed,
            tail_scale=args.tail_scale,
            log_episode=log_episode,
        )
    if problem == "cvrp":
        return CvrpEnv(
            n_customers=args.customers,
            capacity=args.capacity,
            rl_steps=args.rl_steps,
            sa_steps=args.sa_steps,
            sa_schedule=schedule,
            seed=seed,
            tail_scale=args.tail_scale,
            log_episode=log_episode,
        )
    raise ValueError(f"Unknown problem: {problem}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO with strict RLHO for routing problems"
    )
    parser.add_argument("output", type=str, help="Path to policy output file")
    parser.add_argument("--input", type=str, default=None, help="Path to policy input file")
    parser.add_argument("--problem", choices=["tsp", "cvrp"], default="tsp")
    parser.add_argument("--nodes", type=int, default=20, help="TSP node count")
    parser.add_argument("--customers", type=int, default=20, help="CVRP customer count")
    parser.add_argument("--capacity", type=float, default=30.0, help="CVRP capacity")
    parser.add_argument("--rl-steps", type=int, default=50, help="RL perturbation steps x")
    parser.add_argument("--sa-steps", type=int, default=200, help="SA steps y")
    parser.add_argument("--sa-t0", type=float, default=1.0, help="SA initial temperature")
    parser.add_argument("--sa-alpha", type=float, default=0.995, help="SA decay rate")
    parser.add_argument("--tail-scale", type=float, default=1.0, help="Tail reward scale")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-envs", type=int, default=8)
    parser.add_argument("--test-envs", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--step-per-epoch", type=int, default=50000)
    parser.add_argument("--step-per-collect", type=int, default=5000)
    parser.add_argument("--repeat-per-collect", type=int, default=5)
    parser.add_argument(
        "--pair-chunk-size",
        type=int,
        default=0,
        help="Split action-pair logits into chunks to reduce GPU memory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-episodes", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(args.problem, args, args.seed)
    action_pairs = env.action_pairs
    node_dim = env.observation_space["node_features"].shape[-1]
    anchor_dim = env.observation_space["anchor_features"].shape[-1]

    actor = PairActorNetwork(
        node_input_dim=node_dim,
        anchor_input_dim=anchor_dim,
        action_pairs=action_pairs,
        pair_chunk_size=args.pair_chunk_size or None,
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

    if args.input is not None:
        policy.load_state_dict(torch.load(args.input, map_location=device))

    train_envs = SubprocVectorEnv(
        [
            lambda s=seed: make_env(args.problem, args, s)
            for seed in range(args.seed, args.seed + args.train_envs)
        ]
    )
    test_envs = SubprocVectorEnv(
        [
            lambda s=seed: make_env(args.problem, args, s, log_episode=args.log_episodes)
            for seed in range(args.seed + 100, args.seed + 100 + args.test_envs)
        ]
    )

    replay_buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    train_collector = Collector(policy, train_envs, replay_buffer)
    test_collector = Collector(policy, test_envs)

    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.epochs,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=5,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(policy.state_dict(), args.output)
