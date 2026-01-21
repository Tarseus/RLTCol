from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


class PairActorNetwork(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        anchor_input_dim: int,
        action_pairs: np.ndarray,
        embed_dim: int = 64,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.anchor_mlp = nn.Sequential(
            nn.Linear(anchor_input_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        self.device = device
        self.register_buffer(
            "action_pairs",
            torch.as_tensor(action_pairs.astype(np.int64)),
        )

    def forward(self, obs, state=None, info=None):
        node_features = torch.as_tensor(
            obs["node_features"], device=self.device, dtype=torch.float32
        )
        anchor_features = torch.as_tensor(
            obs["anchor_features"], device=self.device, dtype=torch.float32
        )
        action_mask = torch.as_tensor(
            obs["action_mask"], device=self.device, dtype=torch.bool
        )
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        if anchor_features.dim() == 2:
            anchor_features = anchor_features.unsqueeze(0)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
        node_embed = self.node_mlp(node_features)
        anchor_embed = self.anchor_mlp(anchor_features)
        node_idx = self.action_pairs[:, 0]
        anchor_idx = self.action_pairs[:, 1]
        node_sel = node_embed[:, node_idx, :]
        anchor_sel = anchor_embed[:, anchor_idx, :]
        pair_feat = torch.cat([node_sel, anchor_sel, node_sel * anchor_sel], dim=-1)
        logits = self.pair_mlp(pair_feat).squeeze(-1)
        for b in range(action_mask.shape[0]):
            if not action_mask[b].any():
                action_mask[b] = True
        logits = logits.masked_fill(~action_mask, -1e9)
        return logits, state


class RoutingCriticNetwork(nn.Module):
    def __init__(self, node_input_dim: int, embed_dim: int = 64, device: str = "cpu"):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.value_head = nn.Linear(embed_dim, 1)
        self.device = device

    def forward(self, obs, **kwargs):
        node_features = torch.as_tensor(
            obs["node_features"], device=self.device, dtype=torch.float32
        )
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        node_embed = self.node_mlp(node_features)
        pooled = node_embed.mean(dim=1)
        return self.value_head(pooled)
