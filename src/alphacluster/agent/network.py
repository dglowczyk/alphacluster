"""Custom SB3 feature extractor for the TradingEnv Dict observation space.

The observation is a Dict with:
- "market": (batch, 576, 5) — OHLCV window normalized as ratios to current close
- "account": (batch, 7) — normalized account features

The extractor uses a 1D CNN branch for market data and an MLP branch for
account state, concatenating their outputs into a 192-dim feature vector.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class TradingFeatureExtractor(BaseFeaturesExtractor):
    """CNN backbone for market data + MLP for account state.

    Market branch: 1D CNN
        Conv1d(5, 32, kernel_size=5, stride=1) + ReLU + BatchNorm
        Conv1d(32, 64, kernel_size=5, stride=2) + ReLU + BatchNorm
        Conv1d(64, 128, kernel_size=3, stride=2) + ReLU + BatchNorm
        AdaptiveAvgPool1d(1) -> 128-dim

    Account branch: MLP
        Linear(7, 64) + ReLU

    Output: 128 + 64 = 192 dimensional feature vector
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 192) -> None:
        # Must call super with the final features_dim
        super().__init__(observation_space, features_dim)

        market_shape = observation_space["market"].shape  # (576, 5)
        account_shape = observation_space["account"].shape  # (7,)

        n_channels = market_shape[1]  # 5 (OHLCV)
        account_dim = account_shape[0]  # 7

        # ── Market branch: 1D CNN ─────────────────────────────────────────
        self.market_cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),  # -> (batch, 128, 1)
        )
        market_out_dim = 128

        # ── Account branch: MLP ──────────────────────────────────────────
        self.account_mlp = nn.Sequential(
            nn.Linear(account_dim, 64),
            nn.ReLU(),
        )
        account_out_dim = 64

        assert market_out_dim + account_out_dim == features_dim, (
            f"Expected features_dim={features_dim}, got "
            f"{market_out_dim} + {account_out_dim} = {market_out_dim + account_out_dim}"
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # Market data arrives as (batch, 576, 5); Conv1d needs (batch, channels, length)
        market = observations["market"]  # (batch, 576, 5)
        market = market.permute(0, 2, 1)  # -> (batch, 5, 576)

        market_features = self.market_cnn(market)  # -> (batch, 128, 1)
        market_features = market_features.squeeze(-1)  # -> (batch, 128)

        # Account data: (batch, 7)
        account = observations["account"]
        account_features = self.account_mlp(account)  # -> (batch, 32)

        # Concatenate
        return torch.cat([market_features, account_features], dim=1)  # -> (batch, 160)
