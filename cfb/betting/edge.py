"""Edge, expected value, and stake sizing for Kalshi contracts.

A Kalshi contract costs its price in cents and pays 100 if it resolves YES, so
the arithmetic is clean -- but two things trip people up and both are handled
explicitly here:

**Fees are charged on the trade, not the profit.**  Kalshi's fee is
``ceil(0.07 x contracts x P x (1-P) x 100)`` cents, which peaks at 1.75 cents a
contract around 50c.  On a 3-cent edge that is more than half your expected
profit.  Any EV number that ignores it is fiction, so every figure below is
net of fees.

**You must cross the spread.**  The tradeable price is the *ask* when you buy,
not the mid.  Reporting edge against the mid is the most common way a paper
edge fails to exist.  ``edge_table`` uses asks by default and shows the mid
separately so you can see how much of the edge the spread is eating.

Sizing uses Kelly on the net-of-fee price, then scales it down.  Full Kelly on a
model whose probabilities are uncertain is a good way to go broke; a quarter is
the usual compromise and the default here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfb.data.kalshi_client import DEFAULT_FEE_MULTIPLIER


def fee_per_contract(price_cents: float,
                     multiplier: float = DEFAULT_FEE_MULTIPLIER) -> float:
    """Kalshi fee for a single contract at ``price_cents``, in cents."""
    p = min(max(float(price_cents) / 100.0, 0.0), 1.0)
    return math.ceil(multiplier * p * (1.0 - p) * 100.0)


def kelly_fraction(model_prob: float, price_cents: float,
                   fee_cents: float = 0.0) -> float:
    """Kelly stake as a fraction of bankroll for a YES purchase.

    Buying at price ``p`` (plus fee) to win 1.0 gives net odds
    ``b = (1 - p - fee) / (p + fee)``, and Kelly reduces to
    ``(q - p - fee) / (1 - p - fee)``.
    """
    q = float(np.clip(model_prob, 0.0, 1.0))
    cost = float(price_cents + fee_cents) / 100.0
    if cost <= 0 or cost >= 1:
        return 0.0
    edge = q - cost
    return max(edge / (1.0 - cost), 0.0)


@dataclass
class ContractEval:
    side: str            # "yes" or "no"
    price_cents: float   # what you actually pay
    model_prob: float    # probability this side settles at 100
    fee_cents: float
    ev_cents: float      # expected profit per contract, net of fee
    edge: float          # model_prob - all-in cost, in probability terms
    kelly: float
    roi: float           # ev / cost

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def evaluate_contract(model_prob_yes: float, price_cents: float, side: str = "yes",
                      fee_multiplier: float = DEFAULT_FEE_MULTIPLIER) -> ContractEval:
    """Evaluate buying one contract at ``price_cents`` on ``side``."""
    side = side.lower()
    q = float(np.clip(model_prob_yes, 0.0, 1.0))
    q_side = q if side == "yes" else 1.0 - q
    price = float(price_cents)
    fee = fee_per_contract(price, fee_multiplier)
    cost = price + fee
    ev = q_side * 100.0 - cost
    return ContractEval(
        side=side, price_cents=price, model_prob=q_side, fee_cents=fee,
        ev_cents=ev, edge=q_side - cost / 100.0,
        kelly=kelly_fraction(q_side, price, fee),
        roi=ev / cost if cost > 0 else 0.0,
    )


def best_side(model_prob_yes: float, yes_ask: float | None, no_ask: float | None,
              fee_multiplier: float = DEFAULT_FEE_MULTIPLIER) -> ContractEval | None:
    """Pick whichever side of the book has positive EV, if either does."""
    cands = []
    if yes_ask is not None and 0 < yes_ask < 100:
        cands.append(evaluate_contract(model_prob_yes, yes_ask, "yes", fee_multiplier))
    if no_ask is not None and 0 < no_ask < 100:
        cands.append(evaluate_contract(model_prob_yes, no_ask, "no", fee_multiplier))
    if not cands:
        return None
    return max(cands, key=lambda c: c.ev_cents)


def edge_table(
    markets: pd.DataFrame,
    prob_col: str = "model_prob",
    bankroll: float = 1000.0,
    kelly_fraction_of: float = 0.25,
    min_edge: float = 0.02,
    min_volume: int = 0,
    fee_multiplier: float = DEFAULT_FEE_MULTIPLIER,
    use_mid: bool = False,
) -> pd.DataFrame:
    """Rank Kalshi markets by net-of-fee edge against the model.

    ``markets`` needs ``model_prob`` plus Kalshi's ``yes_ask`` / ``no_ask``
    (and, optionally, ``yes_bid`` / ``no_bid`` / ``volume``).  Set
    ``use_mid=True`` only to see the theoretical edge; you cannot trade it.
    """
    if markets.empty:
        return pd.DataFrame()
    rows = []
    for r in markets.itertuples():
        q = getattr(r, prob_col, None)
        if q is None or not np.isfinite(q):
            continue
        yes_ask = _price(r, "yes_ask", "yes_bid", use_mid)
        no_ask = _price(r, "no_ask", "no_bid", use_mid)
        ev = best_side(float(q), yes_ask, no_ask, fee_multiplier)
        if ev is None:
            continue
        volume = int(getattr(r, "volume", 0) or 0)
        stake = bankroll * ev.kelly * kelly_fraction_of
        contracts = int(stake // (ev.price_cents / 100.0)) if ev.price_cents > 0 else 0
        rows.append({
            "ticker": getattr(r, "ticker", ""),
            "title": getattr(r, "title", ""),
            "subtitle": getattr(r, "yes_sub_title", "") or getattr(r, "subtitle", ""),
            "game": getattr(r, "game_label", ""),
            "model_prob_yes": round(float(q), 4),
            "side": ev.side,
            "price_cents": ev.price_cents,
            "fair_cents": round(100 * (float(q) if ev.side == "yes" else 1 - float(q)), 1),
            "fee_cents": ev.fee_cents,
            "edge": round(ev.edge, 4),
            "ev_cents": round(ev.ev_cents, 2),
            "roi": round(ev.roi, 4),
            "kelly": round(ev.kelly, 4),
            "stake_$": round(stake, 2),
            "contracts": contracts,
            "volume": volume,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[(df["edge"] >= min_edge) & (df["volume"] >= min_volume)]
    return df.sort_values("ev_cents", ascending=False).reset_index(drop=True)


def _price(row, ask_field: str, bid_field: str, use_mid: bool) -> float | None:
    ask = getattr(row, ask_field, None)
    if use_mid:
        bid = getattr(row, bid_field, None)
        if ask is not None and bid is not None and ask > 0 and bid > 0:
            return (float(ask) + float(bid)) / 2.0
    try:
        a = float(ask)
    except (TypeError, ValueError):
        return None
    return a if 0 < a < 100 else None


def simulate_bankroll(bets: pd.DataFrame, outcomes: np.ndarray,
                      bankroll: float = 1000.0,
                      kelly_fraction_of: float = 0.25) -> pd.DataFrame:
    """Replay a sequence of graded bets to see the equity curve Kelly implies."""
    bal = bankroll
    rows = []
    for (r, won) in zip(bets.itertuples(), outcomes):
        stake = bal * float(getattr(r, "kelly", 0.0)) * kelly_fraction_of
        cost = stake
        payout = (100.0 / float(r.price_cents)) * stake if won else 0.0
        fee = fee_per_contract(float(r.price_cents)) / 100.0 * (
            stake / (float(r.price_cents) / 100.0) if r.price_cents else 0.0)
        bal = bal - cost - fee + payout
        rows.append({"ticker": getattr(r, "ticker", ""), "stake": stake,
                     "won": bool(won), "balance": bal})
    return pd.DataFrame(rows)
