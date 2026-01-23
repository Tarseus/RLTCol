import numpy as np
import torch
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector
from tianshou.policy import PPOPolicy, BasePolicy
from tianshou.utils.net.common import ActorCritic

from envs.tsp_env import TspEnv
from ho.sa import SASchedule
from problems.tsp import generate_tsp_instance
from routing_network import PairActorNetwork, RoutingCriticNetwork

NUM = 1000
NODES = 100
SEED = 42
RL_STEPS = 128
SA_T0 = 5.0
SA_ALPHA = 0.995
SA_STEPS = 50000
STALL = 2000
POLICY_PATH = "outputs/tsp100.pt"

schedule = SASchedule(t0=SA_T0, alpha=SA_ALPHA)
env = TspEnv(
    n_nodes=NODES,
    rl_steps=RL_STEPS,
    sa_steps=SA_STEPS,
    sa_schedule=schedule,
    sa_converge=True,
    sa_stall_steps=STALL,
    seed=SEED,
    tail_scale=1.0,
    log_episode=False,
)

vector_env = DummyVectorEnv([lambda: env])

action_pairs = env.action_pairs
node_dim = env.observation_space["node_features"].shape[-1]
anchor_dim = env.observation_space["anchor_features"].shape[-1]
actor = PairActorNetwork(node_dim, anchor_dim, action_pairs, device="cuda" if torch.cuda.is_available() else "cpu")
critic = RoutingCriticNetwork(node_dim, device="cuda" if torch.cuda.is_available() else "cpu")
optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=3e-4)
dist = lambda logits: torch.distributions.Categorical(logits=logits)
policy = PPOPolicy(actor=actor, critic=critic, optim=optim, dist_fn=dist, action_space=env.action_space)
policy.load_state_dict(torch.load(POLICY_PATH, map_location="cpu"))
policy.eval()

collector = Collector(policy, vector_env)
collector.reset()

stats = []
for i in range(NUM):
    instance = generate_tsp_instance(NODES, seed=SEED + i)
    env.reset(options={"instance": instance})
    collector.collect(n_episode=1)
    stats.append(env.last_episode_info)

mean_cost = float(np.mean([s["cost_sxy"] for s in stats]))
print("mean cost_sxy:", mean_cost)