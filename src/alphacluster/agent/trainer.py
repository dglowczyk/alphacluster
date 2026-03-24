"""Training orchestration: agent creation, training loop, save/load.

This module wires up SB3's PPO with the custom TradingFeatureExtractor
and provides helpers for checkpointing and evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from alphacluster.agent.config import TrainingConfig
from alphacluster.agent.network import TradingFeatureExtractor

if TYPE_CHECKING:
    import gymnasium as gym

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_agent(
    env: gym.Env,
    config: TrainingConfig,
    verbose: int = 1,
) -> PPO:
    """Instantiate an SB3 PPO agent with the custom feature extractor.

    Parameters
    ----------
    env:
        A TradingEnv (or VecEnv wrapping TradingEnvs) with Dict observation space.
    config:
        Training hyperparameters.
    verbose:
        Verbosity level for PPO logging (0=silent, 1=info, 2=debug).

    Returns
    -------
    PPO
        Ready-to-train PPO agent.
    """
    policy_kwargs = dict(
        features_extractor_class=TradingFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=192),
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
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
    )
    logger.info("Created PPO agent: lr=%s, batch=%d", config.learning_rate, config.batch_size)
    return agent


# ---------------------------------------------------------------------------
# Curriculum callback
# ---------------------------------------------------------------------------


class CurriculumCallback(BaseCallback):
    """Adjusts reward parameters and entropy coefficient across training phases.

    Phase 1 — "Learn to Trade" (0–30%):
        Moderate exploration, early fee/churn awareness.

    Phase 2 — "Learn Quality" (30–60%):
        Reduced exploration, full cost penalties.

    Phase 3 — "Refine & Exploit" (60–100%):
        Low exploration, strict fee/churn/drawdown penalties.
    """

    def __init__(self, config: TrainingConfig, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.config = config
        self._current_phase: int = 0

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.config.total_timesteps
        new_phase = self._get_phase(progress)

        if new_phase != self._current_phase:
            self._current_phase = new_phase
            self._apply_phase(new_phase)

        return True

    def _get_phase(self, progress: float) -> int:
        if progress < self.config.phase1_end:
            return 1
        if progress < self.config.phase2_end:
            return 2
        return 3

    def _apply_phase(self, phase: int) -> None:
        if phase == 1:
            ent_coef = 0.05
            reward_config = {
                "inactivity_penalty_scale": 0.0,
                "fee_scale": 0.5,
                "drawdown_penalty_scale": 0.3,
                "churn_penalty_scale": 0.5,
                "quality_scale": 1.0,
            }
        elif phase == 2:
            ent_coef = 0.03
            reward_config = {
                "inactivity_penalty_scale": 0.0,
                "fee_scale": 1.0,
                "drawdown_penalty_scale": 1.0,
                "churn_penalty_scale": 1.0,
                "quality_scale": 1.0,
            }
        else:  # phase 3
            ent_coef = 0.005
            reward_config = {
                "inactivity_penalty_scale": 0.0,
                "fee_scale": 2.0,
                "drawdown_penalty_scale": 1.5,
                "churn_penalty_scale": 2.0,
                "quality_scale": 0.5,
            }

        # Update agent entropy coefficient
        self.model.ent_coef = ent_coef

        # Update reward config on all environments
        self._set_env_reward_config(reward_config)

        msg = f"Curriculum: phase {phase} (ent_coef={ent_coef})"
        logger.info(msg)
        if self.verbose:
            print(msg)

    def _set_env_reward_config(self, reward_config: dict[str, float]) -> None:
        """Set reward_config on underlying TradingEnv(s)."""
        env = self.model.get_env()
        if env is None:
            return

        # VecEnv: set_attr broadcasts to all sub-environments
        try:
            env.set_attr("reward_config", reward_config)
        except AttributeError:
            # Fallback for single env
            unwrapped = getattr(env, "unwrapped", env)
            if hasattr(unwrapped, "reward_config"):
                unwrapped.reward_config = reward_config


# ---------------------------------------------------------------------------
# Tournament callback
# ---------------------------------------------------------------------------


class TournamentCallback(BaseCallback):
    """Periodically saves the current model as a new generation and runs
    a tournament against the reigning champion.

    On each trigger the callback:
    1. Saves the current model as ``gen_<N>`` via the versioning module.
    2. If no champion exists yet, automatically crowns this generation.
    3. Otherwise loads the champion, runs a tournament, and promotes the
       candidate if it wins enough episodes.

    Parameters
    ----------
    tournament_freq:
        Trigger interval in *environment steps* (calls to ``_on_step``).
    eval_env:
        A separate TradingEnv used for head-to-head evaluation.
    config:
        Training configuration (carries ``tournament_episodes`` and
        ``promotion_threshold``).
    base_dir:
        Root directory for model storage (passed to versioning helpers).
    verbose:
        Verbosity level.
    """

    def __init__(
        self,
        tournament_freq: int,
        eval_env: Any | None = None,
        config: TrainingConfig | None = None,
        base_dir: str | Path | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.tournament_freq = tournament_freq
        self.eval_env = eval_env
        self.config = config or TrainingConfig()
        self.base_dir = base_dir
        self._generation = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.tournament_freq != 0:
            return True

        from alphacluster.tournament.arena import run_tournament
        from alphacluster.tournament.elo import EloRating
        from alphacluster.tournament.versioning import (
            get_champion,
            load_generation,
            save_generation,
            set_champion,
        )

        gen = self._generation
        self._generation += 1

        # 1. Save the current model as a new generation
        agent = self.model  # SB3 agent attached by CallbackList
        save_generation(
            model=agent,
            generation=gen,
            metadata={"timestep": self.num_timesteps},
            base_dir=self.base_dir,
        )
        logger.info(
            "Tournament: saved generation %d (timestep %d)",
            gen,
            self.num_timesteps,
        )

        # 2. If no champion yet, crown this generation automatically
        champ_gen = get_champion(base_dir=self.base_dir)
        if champ_gen is None:
            set_champion(gen, base_dir=self.base_dir)
            msg = f"No champion found -- generation {gen} becomes champion by default."
            logger.info(msg)
            if self.verbose:
                print(msg)
            return True

        # 3. Load champion and run tournament
        if self.eval_env is None:
            logger.warning("Tournament: no eval_env provided; skipping match.")
            return True

        try:
            champion_model, _meta = load_generation(
                champ_gen,
                env=self.eval_env,
                base_dir=self.base_dir,
            )
        except Exception:
            logger.exception(
                "Failed to load champion gen %d; skipping tournament.",
                champ_gen,
            )
            return True

        elo = EloRating()
        result = run_tournament(
            candidate=agent,
            champion=champion_model,
            env=self.eval_env,
            candidate_id=f"gen_{gen}",
            champion_id=f"gen_{champ_gen}",
            n_episodes=self.config.tournament_episodes,
            promotion_threshold=self.config.promotion_threshold,
            elo=elo,
        )

        msg = (
            f"Tournament gen_{gen} vs gen_{champ_gen}: "
            f"winner={result.winner}, "
            f"candidate_wins={result.candidate_wins}, "
            f"promoted={result.candidate_promoted}"
        )
        logger.info(msg)
        if self.verbose:
            print(msg)

        # 4. Promote if the candidate beat the champion
        if result.candidate_promoted:
            set_champion(gen, base_dir=self.base_dir)
            logger.info("Generation %d promoted to champion.", gen)

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
    progress_bar: bool = True,
    verbose: int = 1,
    extra_callbacks: list[BaseCallback] | None = None,
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
        If True, include the tournament callback.
    progress_bar:
        If True, display a tqdm progress bar during training.  Disable when
        running under papermill to avoid IOPub timeouts.
    verbose:
        Verbosity level for callbacks (0=silent, 1=info).
    extra_callbacks:
        Additional callbacks to include in the training loop.

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
            verbose=verbose,
        )
    )

    # ── Eval callback ─────────────────────────────────────────────────
    if eval_env is not None:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

        eval_log_dir = checkpoint_dir / "eval_logs"
        eval_log_dir.mkdir(parents=True, exist_ok=True)

        # Wrap raw gym.Env → Monitor → DummyVecEnv
        if not isinstance(eval_env, VecEnv):
            eval_env = Monitor(eval_env, filename=str(eval_log_dir / "monitor"))
            eval_env = DummyVecEnv([lambda: eval_env])

        # Match training env's VecNormalize wrapper (no-op normalization)
        if not isinstance(eval_env, VecNormalize):
            eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)

        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(checkpoint_dir / "best_model"),
                log_path=str(eval_log_dir),
                eval_freq=config.eval_freq,
                n_eval_episodes=config.n_eval_episodes,
                deterministic=True,
                verbose=verbose,
            )
        )

    # ── Curriculum callback ──────────────────────────────────────────
    if config.curriculum_enabled:
        callbacks.append(CurriculumCallback(config, verbose=verbose))

    # ── Tournament callback ──────────────────────────────────────────
    if run_tournament:
        callbacks.append(
            TournamentCallback(
                tournament_freq=config.tournament_freq,
                eval_env=eval_env,
                config=config,
                verbose=verbose,
            )
        )

    # ── Extra callbacks ────────────────────────────────────────────────
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # ── Train ─────────────────────────────────────────────────────────
    logger.info(
        "Starting training: %d timesteps, eval_freq=%d",
        config.total_timesteps,
        config.eval_freq,
    )
    agent.learn(
        total_timesteps=config.total_timesteps,
        callback=CallbackList(callbacks) if callbacks else None,
        progress_bar=progress_bar,
    )
    logger.info("Training complete.")

    return agent


# ---------------------------------------------------------------------------
# Save / Load helpers
# ---------------------------------------------------------------------------


def save_agent(agent: PPO, path: str | Path) -> Path:
    """Save a trained PPO agent to disk as a ``.pt`` state dict.

    Parameters
    ----------
    agent:
        The trained agent.
    path:
        Destination file path (extension is forced to ``.pt``).

    Returns
    -------
    Path
        The resolved ``.pt`` path.
    """
    import torch

    path = Path(path).with_suffix(".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.policy.state_dict(), str(path))
    logger.info("Agent saved to %s", path)
    return path


def load_agent(path: str | Path, env: gym.Env | None = None) -> PPO:
    """Load a trained PPO agent from a ``.pt`` state dict.

    Parameters
    ----------
    path:
        Path to the saved ``.pt`` model.
    env:
        Environment to bind to the loaded agent (required).

    Returns
    -------
    PPO
        The loaded agent.
    """
    import torch

    path = Path(path)

    if env is None:
        raise ValueError("An environment is required when loading a .pt state dict.")
    config = TrainingConfig()
    agent = create_agent(env, config)
    state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    agent.policy.load_state_dict(state_dict)
    logger.info("Agent loaded from %s", path)
    return agent
