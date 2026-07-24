"""Per-model price table for LLM cost estimation (USD per 1M tokens).

Prices are configuration, not truth — keep them here so a reviewer can correct them
without touching call sites. Unknown models fall back to 0 cost and are flagged.
"""
from __future__ import annotations

# (input_per_mtok, output_per_mtok) in USD. Approximate list prices.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-8[1m]": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "echo": (0.0, 0.0),
    "ollama": (0.0, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> tuple[float, bool]:
    """Return (usd, known). `known=False` means the model wasn't in the price table."""
    key = model
    if key not in PRICES:
        # try a prefix match (e.g. dated snapshots)
        for k in PRICES:
            if model.startswith(k):
                key = k
                break
    if key not in PRICES:
        return 0.0, False
    pin, pout = PRICES[key]
    usd = input_tokens / 1_000_000 * pin + output_tokens / 1_000_000 * pout
    return round(usd, 6), True
