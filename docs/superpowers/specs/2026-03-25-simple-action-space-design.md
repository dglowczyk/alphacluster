# Simplified Action Space — Design Spec

## Motivation

Optuna v1 (40 trials) and v2 (20 trials) both failed to find stable trading
configurations. Every viable screening trial collapsed during 500k validation.
Win rate stuck at ~47% across all parameter combinations.

Hypothesis: the 36-action discrete space (3 directions × 4 sizes × 3 leverages)
is too complex for the agent to explore effectively. The agent must simultaneously
learn market direction AND position sizing AND leverage selection — three coupled
decisions that compound the exploration difficulty.

This spec adds a configurable **simple action mode** (3 actions: long/flat/short)
with fixed position size and leverage. The goal is diagnostic: determine whether
the agent can learn directional trading when freed from size/leverage decisions.

## Design Decisions

- **Configurable, not permanent** — `TrainingConfig.simple_actions` flag toggles
  between 3-action and 36-action modes. Enables A/B comparison.
- **Fixed size=10%, leverage=10x** — sufficient exposure for meaningful reward
  signal without excessive liquidation risk.
- **Same direction = no-op** — repeating "long" while already long does nothing
  and costs zero fees. Eliminates modify_position edge cases.
- **Identical reward function** — isolates the effect of action space reduction.
- **Tournament incompatible across modes** — models trained with different action
  space modes cannot compete in tournaments (mismatched policy heads). This is
  acceptable for a diagnostic test.
- **`step()` accepts int or array** — `Discrete(3)` returns scalar int,
  `MultiDiscrete` returns array. Type annotation updated accordingly.

## Changes

### 1. `src/alphacluster/agent/config.py` — TrainingConfig

Three new fields with defaults that preserve existing behavior:

```python
simple_actions: bool = False       # True → Discrete(3), False → MultiDiscrete([3,4,3])
fixed_size_pct: float = 0.10       # position size when simple_actions=True
fixed_leverage: int = 10           # leverage when simple_actions=True
```

### 2. `src/alphacluster/env/trading_env.py` — TradingEnv

**Constructor** — new parameters:

```python
def __init__(
    self,
    df, funding_df=None,
    window_size=WINDOW_SIZE,
    episode_length=EPISODE_LENGTH,
    simple_actions: bool = False,
    fixed_size_pct: float = 0.10,
    fixed_leverage: int = 10,
):
    self._simple_actions = simple_actions
    self._fixed_size_pct = fixed_size_pct
    self._fixed_leverage = fixed_leverage

    if simple_actions:
        self.action_space = spaces.Discrete(3)  # 0=flat, 1=long, 2=short
    else:
        self.action_space = spaces.MultiDiscrete(
            [N_DIRECTIONS, N_POSITION_SIZES, N_LEVERAGE_LEVELS]
        )
```

**step()** — update type annotation to accept scalar (Discrete) or array
(MultiDiscrete):

```python
def step(
    self, action: np.ndarray | int | np.integer,
) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
```

Action decoding at top of method:

```python
if self._simple_actions:
    direction = int(action)
    if direction == 0:
        size_pct, leverage = 0.0, 1
    else:
        size_pct = self._fixed_size_pct
        leverage = self._fixed_leverage
else:
    direction_idx, size_idx, leverage_idx = (
        int(action[0]), int(action[1]), int(action[2])
    )
    direction = direction_idx
    if direction == 0:
        size_pct = 0.0
        leverage = 1
    else:
        size_pct = POSITION_SIZE_OPTIONS[size_idx]
        leverage = LEVERAGE_OPTIONS[leverage_idx]
```

**Same direction = no-op** — in the `desired_side == current_side` branch:

```python
else:
    # Same direction as current position
    if self._simple_actions:
        pass  # no-op: fixed size/leverage, nothing to modify
    else:
        # existing modify_position logic (check size/leverage delta)
        ...
```

### 3. `src/alphacluster/agent/trainer.py` — train()

Pass config flags to env factory:

```python
env = TradingEnv(
    df=df, funding_df=funding_df,
    window_size=config.window_size,
    episode_length=config.episode_length,
    simple_actions=config.simple_actions,
    fixed_size_pct=config.fixed_size_pct,
    fixed_leverage=config.fixed_leverage,
)
```

Both training envs (SubprocVecEnv) and eval env get the same flags.

`create_agent()` requires no changes — SB3 reads `env.action_space` and creates
the appropriate policy head automatically (single Categorical for Discrete(3)).

### 4. `scripts/train.py` and `scripts/evaluate.py` — Propagate flags

Both scripts create `TradingEnv` directly (bypassing `trainer.py`). All env
instantiations must pass `simple_actions`, `fixed_size_pct`, `fixed_leverage`
from the config.

### 5. `src/alphacluster/cli.py` — Propagate flags

CLI creates envs for evaluation and tournament. Same propagation needed.

### 6. `tests/test_env.py` — New Tests

```python
class TestSimpleActions:
    def test_action_space_is_discrete_3(self):
        """simple_actions=True → Discrete(3)"""

    def test_step_long_opens_with_fixed_params(self):
        """action=1 opens long with fixed_size_pct and fixed_leverage"""

    def test_step_flat_closes_position(self):
        """action=0 closes open position"""

    def test_same_direction_is_noop(self):
        """Repeating action=1 while long → no fees, no modification"""

    def test_default_actions_unchanged(self):
        """simple_actions=False → MultiDiscrete([3,4,3])"""
```

Existing tests unaffected — default is `simple_actions=False`.

### 7. `notebooks/colab_train_simple.ipynb` — Diagnostic Notebook

Copy of `colab_train.ipynb` with:

```python
config = TrainingConfig(
    total_timesteps=500_000,
    simple_actions=True,
    fixed_size_pct=0.10,
    fixed_leverage=10,
)
```

500k steps (not 2M) — faster convergence expected with 3 actions.
Separate checkpoint dir (`simple_actions_v1/`).

## Files Changed

| File | Change |
|------|--------|
| `src/alphacluster/agent/config.py` | Add 3 fields to TrainingConfig |
| `src/alphacluster/env/trading_env.py` | Parameterize action space and step() |
| `src/alphacluster/agent/trainer.py` | Pass flags to env factory |
| `scripts/train.py` | Propagate simple_actions flags to env |
| `scripts/evaluate.py` | Propagate simple_actions flags to env |
| `src/alphacluster/cli.py` | Propagate simple_actions flags to env |
| `tests/test_env.py` | Add TestSimpleActions (5 tests) |
| `notebooks/colab_train_simple.ipynb` | New diagnostic notebook |

## Verification

1. `make test` — all existing + new tests pass
2. `make lint` — clean
3. Default behavior unchanged: `TrainingConfig()` produces 36-action env
4. `TrainingConfig(simple_actions=True)` produces 3-action env
5. Notebook runs on Colab with 500k steps, produces learning curves

## Success Criteria

After training with `simple_actions=True`:

- **Pass:** Win rate > 50% or clear upward trend in learning curves → action
  space was the bottleneck, proceed to Step 2 (additional features)
- **Fail:** Win rate ~47%, flat collapse persists → problem is deeper than action
  space, need to reconsider architecture/features before adding complexity
