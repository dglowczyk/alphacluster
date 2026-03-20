"""Tournament & ELO system for AlphaZero-inspired model evaluation."""

from alphacluster.tournament.arena import (
    MatchResult,
    TournamentResult,
    run_match,
    run_tournament,
)
from alphacluster.tournament.elo import EloRating
from alphacluster.tournament.versioning import (
    get_champion,
    list_generations,
    load_generation,
    save_generation,
    set_champion,
)

__all__ = [
    "EloRating",
    "MatchResult",
    "TournamentResult",
    "get_champion",
    "list_generations",
    "load_generation",
    "run_match",
    "run_tournament",
    "save_generation",
    "set_champion",
]
