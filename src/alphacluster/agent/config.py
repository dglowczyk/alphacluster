"""Training hyperparameter configuration for the RL agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Hyperparameters and schedule for PPO training.

    Sensible defaults are provided for BTC perpetual trading on 5-min candles.
    """

    # ── Environment ───────────────────────────────────────────────────────
    window_size: int = 576
    episode_length: int = 2016  # 1 week of 5-min candles

    # ── PPO ───────────────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05  # Entropy bonus for exploration

    # ── Training schedule ────────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 10

    # ── Tournament ───────────────────────────────────────────────────────
    tournament_freq: int = 50_000  # Timesteps between tournaments
    tournament_episodes: int = 20
    promotion_threshold: float = 0.55  # Win rate to promote
