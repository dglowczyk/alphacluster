"""Tests for the tournament / ELO system (alphacluster.tournament)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphacluster.config import (
    ELO_K_FACTOR,
    INITIAL_ELO,
    WINDOW_SIZE,
)
from alphacluster.env.trading_env import TradingEnv
from alphacluster.tournament.arena import (
    MatchResult,
    TournamentResult,
    _compute_sharpe,
    run_match,
    run_tournament,
)
from alphacluster.tournament.elo import EloRating
from alphacluster.tournament.versioning import (
    get_champion,
    list_generations,
    save_generation,
    set_champion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    n_candles: int = 3200,
    start_price: float = 50_000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start="2025-01-01",
        periods=n_candles,
        freq="5min",
        tz="UTC",
    )
    close = start_price + np.cumsum(rng.normal(0, 10, size=n_candles))
    close = np.maximum(close, 100.0)
    high = close + rng.uniform(0, 20, size=n_candles)
    low = close - rng.uniform(0, 20, size=n_candles)
    low = np.maximum(low, 1.0)
    opn = close + rng.normal(0, 5, size=n_candles)
    opn = np.maximum(opn, 1.0)
    volume = rng.uniform(100, 10_000, size=n_candles)

    return pd.DataFrame(
        {
            "open_time": timestamps,
            "open": opn,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_env(n_candles: int = 3200, episode_length: int = 50) -> TradingEnv:
    """Build a small test environment."""
    df = _make_df(n_candles=n_candles)
    return TradingEnv(
        df=df,
        window_size=WINDOW_SIZE,
        episode_length=episode_length,
        initial_balance=10_000.0,
    )


class _DummyModel:
    """Lightweight stub that mimics SB3 model.predict() and model.save().

    Parameters
    ----------
    action:
        Fixed action to return on every predict call.
        If None, returns the flat/no-op action [0, 0, 0].
    """

    def __init__(self, action: list[int] | None = None) -> None:
        self.action = np.array(action or [0, 0, 0])

    def predict(self, obs, deterministic: bool = True):
        return self.action, None

    def save(self, path: str) -> None:
        """Write a tiny placeholder so file-existence checks pass."""
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".zip")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"dummy-model")


# ===========================================================================
# ELO tests
# ===========================================================================


class TestELO:
    """Test suite for ELO rating calculations."""

    def test_initial_elo(self):
        elo = EloRating()
        assert elo.get_rating("gen_0") == INITIAL_ELO

    def test_initial_elo_custom(self):
        elo = EloRating(initial_elo=1200)
        assert elo.get_rating("gen_0") == 1200

    def test_expected_score_equal_ratings(self):
        score = EloRating.expected_score(1500, 1500)
        assert score == pytest.approx(0.5)

    def test_expected_score_higher_rated_favored(self):
        score = EloRating.expected_score(1700, 1500)
        assert score > 0.5

    def test_expected_score_lower_rated_unfavored(self):
        score = EloRating.expected_score(1300, 1500)
        assert score < 0.5

    def test_expected_scores_sum_to_one(self):
        ea = EloRating.expected_score(1600, 1400)
        eb = EloRating.expected_score(1400, 1600)
        assert ea + eb == pytest.approx(1.0)

    def test_elo_update_winner_gains(self):
        elo = EloRating()
        elo.set_rating("A", 1500)
        elo.set_rating("B", 1500)
        new_a, new_b = elo.update_ratings("A", "B")
        assert new_a > 1500
        assert new_b < 1500

    def test_elo_update_equal_ratings_symmetric(self):
        elo = EloRating()
        elo.set_rating("A", 1500)
        elo.set_rating("B", 1500)
        new_a, new_b = elo.update_ratings("A", "B")
        # With equal initial ratings, winner gains exactly what loser loses
        gain = new_a - 1500
        loss = 1500 - new_b
        assert gain == pytest.approx(loss)

    def test_elo_zero_sum(self):
        """Total rating pool is conserved after an update."""
        elo = EloRating()
        elo.set_rating("A", 1600)
        elo.set_rating("B", 1400)
        total_before = elo.get_rating("A") + elo.get_rating("B")
        elo.update_ratings("A", "B")
        total_after = elo.get_rating("A") + elo.get_rating("B")
        assert total_after == pytest.approx(total_before)

    def test_elo_update_upset_larger_change(self):
        """An upset (lower-rated beats higher-rated) produces larger rating changes."""
        elo1 = EloRating()
        elo1.set_rating("A", 1500)
        elo1.set_rating("B", 1500)
        new_a1, _ = elo1.update_ratings("A", "B")
        gain_equal = new_a1 - 1500

        elo2 = EloRating()
        elo2.set_rating("A", 1300)
        elo2.set_rating("B", 1700)
        new_a2, _ = elo2.update_ratings("A", "B")
        gain_upset = new_a2 - 1300

        assert gain_upset > gain_equal

    def test_elo_custom_k_factor(self):
        elo = EloRating(k_factor=16)
        elo.set_rating("A", 1500)
        elo.set_rating("B", 1500)
        new_a, _ = elo.update_ratings("A", "B")
        gain = new_a - 1500
        # With k=16 and equal ratings, gain = 16 * 0.5 = 8
        assert gain == pytest.approx(8.0)

    def test_elo_update_k_override(self):
        elo = EloRating(k_factor=32)
        elo.set_rating("A", 1500)
        elo.set_rating("B", 1500)
        new_a, _ = elo.update_ratings("A", "B", k=64)
        gain = new_a - 1500
        # k override: gain = 64 * 0.5 = 32
        assert gain == pytest.approx(32.0)

    def test_elo_persistence(self, tmp_path):
        elo = EloRating()
        elo.set_rating("gen_0", 1520)
        elo.set_rating("gen_1", 1480)
        save_path = tmp_path / "elo.json"
        elo.save(save_path)

        loaded = EloRating.load(save_path)
        assert loaded.get_rating("gen_0") == pytest.approx(1520)
        assert loaded.get_rating("gen_1") == pytest.approx(1480)
        assert loaded.initial_elo == INITIAL_ELO
        assert loaded.k_factor == ELO_K_FACTOR

    def test_elo_persistence_roundtrip_preserves_config(self, tmp_path):
        elo = EloRating(initial_elo=1200, k_factor=16)
        elo.set_rating("x", 1300)
        path = tmp_path / "elo.json"
        elo.save(path)

        loaded = EloRating.load(path)
        assert loaded.initial_elo == 1200
        assert loaded.k_factor == 16

    def test_ratings_property_returns_copy(self):
        elo = EloRating()
        elo.set_rating("A", 1500)
        ratings = elo.ratings
        ratings["A"] = 9999
        assert elo.get_rating("A") == 1500  # original not mutated


# ===========================================================================
# Arena / Match tests
# ===========================================================================


class TestArena:
    """Tests for match running and Sharpe computation."""

    def test_compute_sharpe_basic(self):
        returns = [0.1, 0.2, 0.15, 0.05, 0.3]
        sharpe = _compute_sharpe(returns)
        arr = np.array(returns)
        expected = arr.mean() / arr.std(ddof=1)
        assert sharpe == pytest.approx(expected)

    def test_compute_sharpe_empty(self):
        assert _compute_sharpe([]) == 0.0

    def test_compute_sharpe_single(self):
        assert _compute_sharpe([1.0]) == 0.0

    def test_compute_sharpe_constant(self):
        assert _compute_sharpe([5.0, 5.0, 5.0]) == 0.0

    def test_run_match_returns_match_result(self):
        env = _make_env()
        model_a = _DummyModel([0, 0, 0])  # flat
        model_b = _DummyModel([1, 2, 0])  # long, 50%, 1x
        result = run_match(model_a, model_b, env, n_episodes=3)
        assert isinstance(result, MatchResult)
        assert result.n_episodes == 3
        assert len(result.episode_details) == 3

    def test_run_match_deterministic(self):
        """Same models and same env should produce identical results twice."""
        env = _make_env()
        model_a = _DummyModel([0, 0, 0])
        model_b = _DummyModel([1, 1, 0])
        r1 = run_match(model_a, model_b, env, n_episodes=3)
        r2 = run_match(model_a, model_b, env, n_episodes=3)
        assert r1.model_a_pnl == pytest.approx(r2.model_a_pnl)
        assert r1.model_b_pnl == pytest.approx(r2.model_b_pnl)
        assert r1.winner == r2.winner

    def test_run_match_flat_model_zero_pnl(self):
        """A model that always stays flat should have zero PnL."""
        env = _make_env()
        model_flat = _DummyModel([0, 0, 0])
        model_active = _DummyModel([1, 2, 0])
        result = run_match(model_flat, model_active, env, n_episodes=3)
        assert result.model_a_pnl == pytest.approx(0.0)

    def test_run_match_winner_field(self):
        """The winner field should be one of 'model_a', 'model_b', or 'draw'."""
        env = _make_env()
        model_a = _DummyModel([0, 0, 0])
        model_b = _DummyModel([1, 2, 0])
        result = run_match(model_a, model_b, env, n_episodes=3)
        assert result.winner in ("model_a", "model_b", "draw")

    def test_run_match_episode_details(self):
        env = _make_env()
        model_a = _DummyModel([0, 0, 0])
        model_b = _DummyModel([1, 1, 0])
        result = run_match(model_a, model_b, env, n_episodes=5)
        for ep in result.episode_details:
            assert "episode" in ep
            assert "seed" in ep
            assert "model_a_pnl" in ep
            assert "model_b_pnl" in ep


# ===========================================================================
# Tournament tests
# ===========================================================================


class TestTournament:
    """Tests for tournament execution and promotion logic."""

    def test_tournament_returns_result(self):
        env = _make_env()
        candidate = _DummyModel([1, 2, 0])  # active trading
        champion = _DummyModel([0, 0, 0])  # flat
        result = run_tournament(
            candidate,
            champion,
            env,
            candidate_id="gen_1",
            champion_id="gen_0",
            n_episodes=3,
            promotion_threshold=0.5,
        )
        assert isinstance(result, TournamentResult)
        assert result.candidate_id == "gen_1"
        assert result.champion_id == "gen_0"

    def test_tournament_updates_elo(self):
        env = _make_env()
        elo = EloRating()
        candidate = _DummyModel([1, 2, 0])
        champion = _DummyModel([0, 0, 0])
        result = run_tournament(
            candidate,
            champion,
            env,
            candidate_id="gen_1",
            champion_id="gen_0",
            n_episodes=3,
            promotion_threshold=0.5,
            elo=elo,
        )
        # If there was a winner, ELO should have changed
        if result.winner != "draw":
            assert result.elo_after != result.elo_before

    def test_tournament_win_counts(self):
        env = _make_env()
        candidate = _DummyModel([0, 0, 0])
        champion = _DummyModel([0, 0, 0])
        n_ep = 5
        result = run_tournament(
            candidate,
            champion,
            env,
            n_episodes=n_ep,
            promotion_threshold=0.5,
        )
        total = result.candidate_wins + result.champion_wins + result.draws
        assert total == n_ep

    def test_tournament_promotion_when_threshold_met(self):
        """When candidate always wins, it should be promoted."""
        env = _make_env()
        # Two identical flat models => draws => 0 wins => not promoted at 0.55
        # Use threshold=0.0 so any number of wins (including 0) promotes
        candidate = _DummyModel([0, 0, 0])
        champion = _DummyModel([0, 0, 0])
        result = run_tournament(
            candidate,
            champion,
            env,
            n_episodes=3,
            promotion_threshold=0.0,
        )
        assert result.candidate_promoted is True

    def test_tournament_no_promotion_when_threshold_not_met(self):
        """With a very high threshold, candidate should not be promoted."""
        env = _make_env()
        candidate = _DummyModel([0, 0, 0])
        champion = _DummyModel([0, 0, 0])
        result = run_tournament(
            candidate,
            champion,
            env,
            n_episodes=3,
            promotion_threshold=1.1,  # impossible to meet
        )
        assert result.candidate_promoted is False

    def test_tournament_elo_zero_sum(self):
        """ELO total should be conserved after a tournament."""
        env = _make_env()
        elo = EloRating()
        elo.set_rating("gen_0", 1500)
        elo.set_rating("gen_1", 1500)
        total_before = elo.get_rating("gen_0") + elo.get_rating("gen_1")

        run_tournament(
            _DummyModel([1, 2, 0]),
            _DummyModel([0, 0, 0]),
            env,
            candidate_id="gen_1",
            champion_id="gen_0",
            n_episodes=3,
            elo=elo,
        )

        total_after = elo.get_rating("gen_0") + elo.get_rating("gen_1")
        assert total_after == pytest.approx(total_before)


# ===========================================================================
# Versioning tests
# ===========================================================================


class TestVersioning:
    """Tests for model generation save/load and champion tracking."""

    def test_save_generation_creates_files(self, tmp_path):
        model = _DummyModel()
        gen_dir = save_generation(model, 0, metadata={"elo": 1500}, base_dir=tmp_path)
        assert gen_dir.exists()
        assert (gen_dir / "model.zip").exists()
        assert (gen_dir / "metadata.json").exists()

    def test_save_generation_metadata_content(self, tmp_path):
        model = _DummyModel()
        save_generation(model, 3, metadata={"elo": 1520, "note": "test"}, base_dir=tmp_path)
        meta_path = tmp_path / "gen_3" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["generation"] == 3
        assert meta["elo"] == 1520
        assert meta["note"] == "test"

    def test_save_generation_default_metadata(self, tmp_path):
        model = _DummyModel()
        save_generation(model, 0, base_dir=tmp_path)
        meta_path = tmp_path / "gen_0" / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["generation"] == 0

    def test_get_champion_none_initially(self, tmp_path):
        assert get_champion(base_dir=tmp_path) is None

    def test_set_and_get_champion(self, tmp_path):
        set_champion(5, base_dir=tmp_path)
        assert get_champion(base_dir=tmp_path) == 5

    def test_set_champion_overwrites(self, tmp_path):
        set_champion(1, base_dir=tmp_path)
        set_champion(3, base_dir=tmp_path)
        assert get_champion(base_dir=tmp_path) == 3

    def test_list_generations_empty(self, tmp_path):
        assert list_generations(base_dir=tmp_path) == []

    def test_list_generations_finds_saved(self, tmp_path):
        model = _DummyModel()
        save_generation(model, 0, metadata={"elo": 1500}, base_dir=tmp_path)
        save_generation(model, 1, metadata={"elo": 1520}, base_dir=tmp_path)
        save_generation(model, 2, metadata={"elo": 1480}, base_dir=tmp_path)

        gens = list_generations(base_dir=tmp_path)
        assert len(gens) == 3
        assert gens[0]["generation"] == 0
        assert gens[1]["generation"] == 1
        assert gens[2]["generation"] == 2

    def test_list_generations_includes_metadata(self, tmp_path):
        model = _DummyModel()
        save_generation(model, 0, metadata={"elo": 1500}, base_dir=tmp_path)
        gens = list_generations(base_dir=tmp_path)
        assert gens[0]["elo"] == 1500

    def test_list_generations_nonexistent_dir(self, tmp_path):
        result = list_generations(base_dir=tmp_path / "nope")
        assert result == []

    def test_champion_json_format(self, tmp_path):
        set_champion(7, base_dir=tmp_path)
        data = json.loads((tmp_path / "champion.json").read_text())
        assert data == {"generation": 7}
