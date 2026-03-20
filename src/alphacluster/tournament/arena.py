"""Tournament arena: head-to-head model evaluation.

Runs matches between model generations on identical environment episodes
to determine the stronger model, then updates ELO ratings accordingly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alphacluster.tournament.elo import EloRating

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Detailed result of a single head-to-head match."""

    winner: str  # "model_a", "model_b", or "draw"
    model_a_pnl: float
    model_b_pnl: float
    model_a_sharpe: float
    model_b_sharpe: float
    n_episodes: int
    episode_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TournamentResult:
    """Result of a full tournament (candidate vs champion)."""

    winner: str  # model identifier
    candidate_id: str
    champion_id: str
    candidate_wins: int
    champion_wins: int
    draws: int
    candidate_promoted: bool
    elo_before: tuple[float, float]  # (candidate, champion)
    elo_after: tuple[float, float]
    match_result: MatchResult


def _compute_sharpe(returns: list[float]) -> float:
    """Compute the Sharpe ratio from a list of per-episode returns.

    Returns 0.0 when there are fewer than 2 returns or when std is zero.
    """
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    std = arr.std(ddof=1)
    if std == 0.0:
        return 0.0
    return float(arr.mean() / std)


def run_match(
    model_a: Any,
    model_b: Any,
    env: Any,
    n_episodes: int = 10,
) -> MatchResult:
    """Run a head-to-head match between two models on the same episodes.

    For each episode both models face the **exact same** starting conditions
    (achieved via identical seeds in ``env.reset``).  Each model trades
    through the episode and we record its total PnL.

    Parameters
    ----------
    model_a:
        First SB3 model (has ``.predict(obs, deterministic=True)``).
    model_b:
        Second SB3 model.
    env:
        A Gymnasium environment (TradingEnv) shared between both models.
    n_episodes:
        Number of episodes to play.

    Returns
    -------
    MatchResult
        Aggregated result with winner, PnL totals, and Sharpe ratios.
    """
    a_total_pnl = 0.0
    b_total_pnl = 0.0
    a_returns: list[float] = []
    b_returns: list[float] = []
    episode_details: list[dict[str, Any]] = []

    for ep_idx in range(n_episodes):
        seed = ep_idx  # deterministic seed per episode

        # ── Model A plays the episode ──────────────────────────────
        a_pnl = _play_episode(model_a, env, seed)
        a_total_pnl += a_pnl
        a_returns.append(a_pnl)

        # ── Model B plays the same episode ─────────────────────────
        b_pnl = _play_episode(model_b, env, seed)
        b_total_pnl += b_pnl
        b_returns.append(b_pnl)

        episode_details.append(
            {
                "episode": ep_idx,
                "seed": seed,
                "model_a_pnl": a_pnl,
                "model_b_pnl": b_pnl,
            }
        )

    a_sharpe = _compute_sharpe(a_returns)
    b_sharpe = _compute_sharpe(b_returns)

    # Determine winner: primary = total PnL, tie-break = Sharpe ratio
    if a_total_pnl > b_total_pnl:
        winner = "model_a"
    elif b_total_pnl > a_total_pnl:
        winner = "model_b"
    else:
        # Tie on PnL — use Sharpe as tiebreaker
        if a_sharpe > b_sharpe:
            winner = "model_a"
        elif b_sharpe > a_sharpe:
            winner = "model_b"
        else:
            winner = "draw"

    logger.info(
        "Match complete: A PnL=%.2f (Sharpe=%.3f), B PnL=%.2f (Sharpe=%.3f) -> %s",
        a_total_pnl, a_sharpe, b_total_pnl, b_sharpe, winner,
    )

    return MatchResult(
        winner=winner,
        model_a_pnl=a_total_pnl,
        model_b_pnl=b_total_pnl,
        model_a_sharpe=a_sharpe,
        model_b_sharpe=b_sharpe,
        n_episodes=n_episodes,
        episode_details=episode_details,
    )


def _play_episode(model: Any, env: Any, seed: int) -> float:
    """Play a single episode with *model* and return the total PnL.

    PnL is measured as the change in equity from start to finish.
    """
    obs, info = env.reset(seed=seed)
    initial_equity = info.get("equity", info.get("balance", 10_000.0))

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)

    final_equity = info.get("equity", info.get("balance", 0.0))
    return final_equity - initial_equity


def run_tournament(
    candidate: Any,
    champion: Any,
    env: Any,
    candidate_id: str = "candidate",
    champion_id: str = "champion",
    n_episodes: int = 10,
    promotion_threshold: float = 0.55,
    elo: EloRating | None = None,
) -> TournamentResult:
    """Run a full tournament: candidate vs champion.

    Runs a match, updates ELO ratings, and determines whether the
    candidate should be promoted to champion.

    Parameters
    ----------
    candidate:
        The challenger SB3 model.
    champion:
        The current champion SB3 model.
    env:
        Trading environment for evaluation.
    candidate_id:
        Identifier for the candidate (used for ELO tracking).
    champion_id:
        Identifier for the champion.
    n_episodes:
        Number of episodes in the match.
    promotion_threshold:
        Fraction of episodes the candidate must win to be promoted.
    elo:
        Optional EloRating instance. One is created if not provided.

    Returns
    -------
    TournamentResult
        Full tournament outcome including promotion decision.
    """
    if elo is None:
        elo = EloRating()

    elo_before = (elo.get_rating(candidate_id), elo.get_rating(champion_id))

    # Run the match: candidate = model_a, champion = model_b
    result = run_match(candidate, champion, env, n_episodes=n_episodes)

    # Count per-episode wins for promotion decision
    candidate_wins = 0
    champion_wins = 0
    draws = 0
    for ep in result.episode_details:
        if ep["model_a_pnl"] > ep["model_b_pnl"]:
            candidate_wins += 1
        elif ep["model_b_pnl"] > ep["model_a_pnl"]:
            champion_wins += 1
        else:
            draws += 1

    # Update ELO based on overall match result
    if result.winner == "model_a":
        elo.update_ratings(winner=candidate_id, loser=champion_id)
    elif result.winner == "model_b":
        elo.update_ratings(winner=champion_id, loser=candidate_id)
    # On draw: no ELO update

    elo_after = (elo.get_rating(candidate_id), elo.get_rating(champion_id))

    # Promotion: candidate must win at least promotion_threshold fraction
    win_rate = candidate_wins / n_episodes if n_episodes > 0 else 0.0
    promoted = win_rate >= promotion_threshold

    overall_winner = candidate_id if result.winner == "model_a" else (
        champion_id if result.winner == "model_b" else "draw"
    )

    logger.info(
        "Tournament: %s vs %s -> winner=%s, candidate_wins=%d/%d (%.1f%%), promoted=%s",
        candidate_id, champion_id, overall_winner,
        candidate_wins, n_episodes, win_rate * 100, promoted,
    )

    return TournamentResult(
        winner=overall_winner,
        candidate_id=candidate_id,
        champion_id=champion_id,
        candidate_wins=candidate_wins,
        champion_wins=champion_wins,
        draws=draws,
        candidate_promoted=promoted,
        elo_before=elo_before,
        elo_after=elo_after,
        match_result=result,
    )
