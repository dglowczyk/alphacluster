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

    # ── Simple action mode ─────────────────────────────────────────────
    simple_actions: bool = False  # True → Discrete(3), False → MultiDiscrete([3,4,3])
    fixed_size_pct: float = 0.10  # position size when simple_actions=True
    fixed_leverage: int = 10  # leverage when simple_actions=True

    # ── PPO ───────────────────────────────────────────────────────────────
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 128  # reduced for Transformer memory
    n_epochs: int = 10
    gamma: float = 0.995  # slightly higher for long-term credit assignment
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05  # moderate exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ── Training schedule ────────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 10

    # ── Parallel environments ────────────────────────────────────────────
    n_envs: int = 4  # SubprocVecEnv parallelism

    # ── Curriculum ───────────────────────────────────────────────────────
    curriculum_enabled: bool = True
    phase1_end: float = 0.3  # first 30% of training
    phase2_end: float = 0.6  # middle 30%
    # Phase 3: remaining 40%

    # ── Tournament ───────────────────────────────────────────────────────
    tournament_freq: int = 100_000
    tournament_episodes: int = 20
    promotion_threshold: float = 0.55  # win rate to promote
