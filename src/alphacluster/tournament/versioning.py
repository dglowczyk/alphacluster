"""Model generation versioning and champion tracking.

Provides save/load for numbered model generations and maintains a
``champion.json`` file pointing to the current best generation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from alphacluster.config import MODELS_DIR

logger = logging.getLogger(__name__)


def _base_dir(base_dir: str | Path | None) -> Path:
    """Resolve the base models directory."""
    if base_dir is None:
        return MODELS_DIR
    return Path(base_dir)


def save_generation(
    model: Any,
    generation: int,
    metadata: dict[str, Any] | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Save a model as a numbered generation.

    Creates ``<base_dir>/gen_<N>/model.pt`` and ``<base_dir>/gen_<N>/metadata.json``.

    Parameters
    ----------
    model:
        An SB3 model with a ``policy.state_dict()`` method.
    generation:
        Generation number (non-negative integer).
    metadata:
        Optional metadata dict to persist alongside the model.
    base_dir:
        Root directory for model storage. Defaults to ``MODELS_DIR``.

    Returns
    -------
    Path
        The generation directory (``<base_dir>/gen_<N>``).
    """
    import torch

    bd = _base_dir(base_dir)
    gen_dir = bd / f"gen_{generation}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.policy.state_dict(), str(gen_dir / "model.pt"))

    meta = metadata or {}
    meta["generation"] = generation
    meta_path = gen_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    logger.info("Saved generation %d to %s", generation, gen_dir)
    return gen_dir


def load_generation(
    generation: int,
    env: Any | None = None,
    base_dir: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a model and its metadata by generation number.

    Parameters
    ----------
    generation:
        Generation number to load.
    env:
        Optional environment to bind to the loaded model.
    base_dir:
        Root directory. Defaults to ``MODELS_DIR``.

    Returns
    -------
    tuple[model, metadata]
        The loaded SB3 model and its metadata dict.
    """
    from alphacluster.agent.trainer import load_agent

    bd = _base_dir(base_dir)
    gen_dir = bd / f"gen_{generation}"

    model_path = gen_dir / "model.pt"
    model = load_agent(model_path, env=env)

    meta_path = gen_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    logger.info("Loaded generation %d from %s", generation, gen_dir)
    return model, metadata


def get_champion(base_dir: str | Path | None = None) -> int | None:
    """Return the current champion generation number, or None if unset.

    Parameters
    ----------
    base_dir:
        Root directory. Defaults to ``MODELS_DIR``.

    Returns
    -------
    int | None
        Champion generation number, or None if no champion has been set.
    """
    bd = _base_dir(base_dir)
    champ_path = bd / "champion.json"
    if not champ_path.exists():
        return None
    data = json.loads(champ_path.read_text())
    return data.get("generation")


def set_champion(
    generation: int,
    base_dir: str | Path | None = None,
) -> Path:
    """Set the current champion to *generation*.

    Writes ``<base_dir>/champion.json``.

    Parameters
    ----------
    generation:
        The generation number to crown as champion.
    base_dir:
        Root directory. Defaults to ``MODELS_DIR``.

    Returns
    -------
    Path
        Path to the ``champion.json`` file.
    """
    bd = _base_dir(base_dir)
    bd.mkdir(parents=True, exist_ok=True)
    champ_path = bd / "champion.json"
    data = {"generation": generation}
    champ_path.write_text(json.dumps(data, indent=2))
    logger.info("Champion set to generation %d", generation)
    return champ_path


def list_generations(base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """List all saved generations with their metadata.

    Parameters
    ----------
    base_dir:
        Root directory. Defaults to ``MODELS_DIR``.

    Returns
    -------
    list[dict]
        Sorted list of metadata dicts (each includes ``"generation"``).
    """
    bd = _base_dir(base_dir)
    results: list[dict[str, Any]] = []

    if not bd.exists():
        return results

    for gen_dir in sorted(bd.iterdir()):
        if not gen_dir.is_dir() or not gen_dir.name.startswith("gen_"):
            continue
        try:
            gen_num = int(gen_dir.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue

        meta_path = gen_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {}
        meta["generation"] = gen_num

        # Check if the model file exists
        meta["model_exists"] = (gen_dir / "model.pt").exists()

        results.append(meta)

    return results
