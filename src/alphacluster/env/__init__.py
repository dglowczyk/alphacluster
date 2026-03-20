"""Trading environment package — registers ``CryptoPerp-v0`` with Gymnasium."""

import gymnasium

gymnasium.register(
    id="CryptoPerp-v0",
    entry_point="alphacluster.env.trading_env:TradingEnv",
)
