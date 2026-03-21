"""CNN + Transformer feature extractor for the TradingEnv Dict observation space.

The observation is a Dict with:
- "market": (batch, 576, 19) — OHLCV + technical indicators
- "account": (batch, 12) — normalized account features

The extractor uses a 1D CNN to compress the time series, a Transformer encoder
to capture temporal dependencies, and an MLP for account state.  The outputs
are concatenated into a 192-dim feature vector.
"""

from __future__ import annotations

import math

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class TradingFeatureExtractor(BaseFeaturesExtractor):
    """CNN + Transformer backbone for market data + MLP for account state.

    Market branch:
        Conv1d compression: (batch, 19, 576) -> (batch, 128, 144)
        Learnable positional encoding
        Transformer encoder: 3 layers, 4 heads, d_model=128
        Adaptive average pool -> (batch, 128)

    Account branch:
        Linear(12, 64) -> ReLU -> Dropout -> Linear(64, 64) -> ReLU

    Output: 128 + 64 = 192 dimensional feature vector
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 192) -> None:
        super().__init__(observation_space, features_dim)

        market_shape = observation_space["market"].shape  # (576, 19)
        account_shape = observation_space["account"].shape  # (12,)

        n_channels = market_shape[1]  # 19
        account_dim = account_shape[0]  # 12

        # ── Market branch: CNN compression ───────────────────────────────
        self.market_cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        # After CNN: (batch, 128, seq_len) where seq_len = 144

        # Compute CNN output sequence length
        seq_len = _conv_output_len(market_shape[0], 5, 2, 2)  # Conv1
        seq_len = _conv_output_len(seq_len, 3, 2, 1)  # Conv2
        self._seq_len = seq_len  # 144

        d_model = 128

        # ── Positional encoding ──────────────────────────────────────────
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # ── Transformer encoder ──────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # ── Pooling ──────────────────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool1d(1)

        market_out_dim = 128

        # ── Account branch: MLP ──────────────────────────────────────────
        self.account_mlp = nn.Sequential(
            nn.Linear(account_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        account_out_dim = 64

        assert market_out_dim + account_out_dim == features_dim, (
            f"Expected features_dim={features_dim}, got "
            f"{market_out_dim} + {account_out_dim} = {market_out_dim + account_out_dim}"
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # Market data: (batch, 576, 19) -> Conv1d needs (batch, channels, length)
        market = observations["market"]  # (batch, 576, 19)
        market = market.permute(0, 2, 1)  # -> (batch, 19, 576)

        # CNN compression -> (batch, 128, 144)
        market = self.market_cnn(market)

        # Permute for transformer: (batch, 144, 128)
        market = market.permute(0, 2, 1)

        # Add positional encoding
        market = market + self.pos_encoding

        # Transformer encoder -> (batch, 144, 128)
        market = self.transformer(market)

        # Pool over sequence dim: (batch, 128, 144) -> (batch, 128, 1) -> (batch, 128)
        market = market.permute(0, 2, 1)
        market_features = self.pool(market).squeeze(-1)

        # Account data: (batch, 12)
        account = observations["account"]
        account_features = self.account_mlp(account)  # -> (batch, 64)

        # Concatenate
        return torch.cat([market_features, account_features], dim=1)  # -> (batch, 192)


def _conv_output_len(input_len: int, kernel_size: int, stride: int, padding: int) -> int:
    """Compute the output length of a 1D convolution."""
    return math.floor((input_len + 2 * padding - kernel_size) / stride) + 1
