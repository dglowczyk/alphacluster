"""ELO rating system for model generation evaluation.

Tracks ratings for all model generations using the standard ELO formula,
with persistence to JSON for cross-session continuity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from alphacluster.config import ELO_K_FACTOR, INITIAL_ELO, MODELS_DIR

logger = logging.getLogger(__name__)


class EloRating:
    """Tracks ELO ratings for all model generations.

    Parameters
    ----------
    initial_elo:
        Default rating assigned to new models.
    k_factor:
        K-factor controlling rating volatility per update.
    """

    def __init__(
        self,
        initial_elo: float = INITIAL_ELO,
        k_factor: float = ELO_K_FACTOR,
    ) -> None:
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self._ratings: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def get_rating(self, player: str) -> float:
        """Return the current rating for *player*, creating it if needed."""
        if player not in self._ratings:
            self._ratings[player] = self.initial_elo
        return self._ratings[player]

    def set_rating(self, player: str, rating: float) -> None:
        """Manually set the rating for *player*."""
        self._ratings[player] = rating

    @property
    def ratings(self) -> dict[str, float]:
        """Return a copy of all current ratings."""
        return dict(self._ratings)

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Compute the expected score (win probability) for player A.

        Uses the standard ELO formula:
            E_A = 1 / (1 + 10^((R_B - R_A) / 400))

        Parameters
        ----------
        rating_a:
            Rating of player A.
        rating_b:
            Rating of player B.

        Returns
        -------
        float
            Expected score for player A in [0, 1].
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(
        self,
        winner: str,
        loser: str,
        k: float | None = None,
    ) -> tuple[float, float]:
        """Update ratings after a match where *winner* beat *loser*.

        Parameters
        ----------
        winner:
            Identifier of the winning model.
        loser:
            Identifier of the losing model.
        k:
            Optional override for the K-factor.

        Returns
        -------
        tuple[float, float]
            ``(new_winner_rating, new_loser_rating)``
        """
        k = k if k is not None else self.k_factor

        ra = self.get_rating(winner)
        rb = self.get_rating(loser)

        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea

        new_ra = ra + k * (1.0 - ea)
        new_rb = rb + k * (0.0 - eb)

        self._ratings[winner] = new_ra
        self._ratings[loser] = new_rb

        logger.info(
            "ELO update: %s %.1f -> %.1f, %s %.1f -> %.1f",
            winner, ra, new_ra,
            loser, rb, new_rb,
        )
        return new_ra, new_rb

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> Path:
        """Save ratings to a JSON file.

        Parameters
        ----------
        path:
            File path. Defaults to ``<MODELS_DIR>/elo_ratings.json``.

        Returns
        -------
        Path
            The resolved file path.
        """
        if path is None:
            path = MODELS_DIR / "elo_ratings.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "initial_elo": self.initial_elo,
            "k_factor": self.k_factor,
            "ratings": self._ratings,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("ELO ratings saved to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> EloRating:
        """Load ratings from a JSON file.

        Parameters
        ----------
        path:
            File path. Defaults to ``<MODELS_DIR>/elo_ratings.json``.

        Returns
        -------
        EloRating
            A new instance with the loaded state.
        """
        if path is None:
            path = MODELS_DIR / "elo_ratings.json"
        path = Path(path)

        data = json.loads(path.read_text())
        instance = cls(
            initial_elo=data.get("initial_elo", INITIAL_ELO),
            k_factor=data.get("k_factor", ELO_K_FACTOR),
        )
        instance._ratings = data.get("ratings", {})
        logger.info("ELO ratings loaded from %s (%d players)", path, len(instance._ratings))
        return instance
