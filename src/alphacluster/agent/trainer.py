"""Training orchestration: agent creation, training loop, save/load.

This module wires up SB3's PPO with the custom TradingFeatureExtractor
and provides helpers for checkpointing and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from alphacluster.agent.config import TrainingConfig
from alphacluster.agent.network import TradingFeatureExtractor

if TYPE_CHECKING:
    import gymnasium as gym

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_agent(env: gym.Env, config: TrainingConfig) -> PPO:
    """Instantiate an SB3 PPO agent with the custom feature extractor.

    Parameters
    ----------
    env:
        A TradingEnv (or compatible) with Dict observation space.
    config:
        Training hyperparameters.

    Returns
    -------
    PPO
        Ready-to-train PPO agent.
    """
    policy_kwargs = dict(
        features_extractor_class=TradingFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=160),
    )

    agent = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )
    logger.info("Created PPO agent: lr=%s, batch=%d", config.learning_rate, config.batch_size)
    return agent


# ---------------------------------------------------------------------------
# Tournament callback (stub — actual tournament logic is Phase 5)
# ---------------------------------------------------------------------------

class TournamentCallback(BaseCallback):
    """Periodically triggers a tournament evaluation.

    This is a stub: the real tournament logic (ELO rating, champion
    comparison) will be implemented in Phase 5.  For now it simply logs
    a message every ``tournament_freq`` timesteps.
    """

    def __init__(self, tournament_freq: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.tournament_freq = tournament_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.tournament_freq == 0:
            msg = (
                f"Tournament would run here (timestep {self.num_timesteps}). "
                "Actual tournament logic is in Phase 5."
            )
            logger.info(msg)
            if self.verbose:
                print(msg)
        return True


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    agent: PPO,
    config: TrainingConfig,
    eval_env: gym.Env | None = None,
    checkpoint_dir: str | Path | None = None,
    run_tournament: bool = False,
) -> PPO:
    """Train the agent with evaluation, checkpointing, and optional tournament hooks.

    Parameters
    ----------
    agent:
        PPO agent to train (already bound to a training env).
    config:
        Training configuration.
    eval_env:
        Optional separate environment for periodic evaluation.
    checkpoint_dir:
        Directory to save periodic checkpoints.  Defaults to ``models/checkpoints``.
    run_tournament:
        If True, include the tournament callback stub.

    Returns
    -------
    PPO
        The trained agent.
    """
    callbacks: list[BaseCallback] = []

    # ── Checkpoint callback ───────────────────────────────────────────
    if checkpoint_dir is None:
        checkpoint_dir = Path("models") / "checkpoints"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks.append(
        CheckpointCallback(
            save_freq=config.eval_freq,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_trading",
            verbose=1,
        )
    )

    # ── Eval callback ─────────────────────────────────────────────────
    if eval_env is not None:
        eval_log_dir = checkpoint_dir / "eval_logs"
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(checkpoint_dir / "best_model"),
                log_path=str(eval_log_dir),
                eval_freq=config.eval_freq,
                n_eval_episodes=config.n_eval_episodes,
                deterministic=True,
                verbose=1,
            )
        )

    # ── Tournament callback (stub) ───────────────────────────────────
    if run_tournament:
        callbacks.append(
            TournamentCallback(
                tournament_freq=config.tournament_freq,
                verbose=1,
            )
        )

    # ── Train ─────────────────────────────────────────────────────────
    logger.info(
        "Starting training: %d timesteps, eval_freq=%d",
        config.total_timesteps,
        config.eval_freq,
    )
    agent.learn(
        total_timesteps=config.total_timesteps,
        callback=CallbackList(callbacks) if callbacks else None,
        progress_bar=False,
    )
    logger.info("Training complete.")

    return agent


# ---------------------------------------------------------------------------
# Save / Load helpers
# ---------------------------------------------------------------------------

def save_agent(agent: PPO, path: str | Path) -> Path:
    """Save a trained PPO agent to disk.

    Parameters
    ----------
    agent:
        The trained agent.
    path:
        Destination file path (without extension; SB3 adds ``.zip``).

    Returns
    -------
    Path
        The resolved path (with extension).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path))
    logger.info("Agent saved to %s", path)
    return path


def load_agent(path: str | Path, env: gym.Env | None = None) -> PPO:
    """Load a trained PPO agent from disk.

    Parameters
    ----------
    path:
        Path to the saved model (with or without ``.zip`` extension).
    env:
        Optional environment to bind to the loaded agent.

    Returns
    -------
    PPO
        The loaded agent.
    """
    path = Path(path)
    custom_objects = {
        "policy_kwargs": dict(
            features_extractor_class=TradingFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=160),
        )
    }
    agent = PPO.load(str(path), env=env, custom_objects=custom_objects)
    logger.info("Agent loaded from %s", path)
    return agent
