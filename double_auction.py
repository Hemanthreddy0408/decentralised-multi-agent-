"""
double_auction.py - Hourly call-market Double Auction (PHASE 6).

MECHANISM (uniform-price k-double auction)
  1. Sellers submit SELL bids  (quantity, ask price)   - lowest price they accept.
  2. Buyers  submit BUY  bids  (quantity, bid price)   - highest price they pay.
  3. Sort sells by ask ASCENDING and buys by bid DESCENDING (ties: lower agent id first).
  4. Walk both lists from the top: while best_bid >= best_ask, trade
     min(remaining buy qty, remaining sell qty); partial fills are supported.
  5. The LAST (marginal) matched pair (bid_m, ask_m) defines ONE uniform clearing price
         price = ask_m + k * (bid_m - ask_m)          (k = 0.5 -> midpoint)
     and every trade of this hour is executed at that price.
     Since bid_m <= every matched bid and ask_m >= every matched ask, every participant
     gets a price at least as good as their limit (individual rationality).
  6. Anything unmatched is left for the main grid (handled by the caller).
Example: seller 3 kWh @5.5, buyer 2 kWh @6.5 -> trade 2 kWh at 6.0 INR/kWh; 1 kWh left to grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_EPS = 1e-9


@dataclass(frozen=True)
class Bid:
    agent_id: int
    side: str          # "buy" or "sell"
    quantity: float    # kWh
    price: float       # INR/kWh limit price


@dataclass(frozen=True)
class Trade:
    buyer_id: int
    seller_id: int
    quantity: float
    price: float


@dataclass
class AuctionResult:
    trades: list = field(default_factory=list)
    clearing_price: Optional[float] = None
    matched_quantity: float = 0.0
    unmatched_buy: dict = field(default_factory=dict)    # agent_id -> kWh still needed
    unmatched_sell: dict = field(default_factory=dict)   # agent_id -> kWh still for sale

    def bought_by(self, agent_id: int) -> float:
        return sum(t.quantity for t in self.trades if t.buyer_id == agent_id)

    def sold_by(self, agent_id: int) -> float:
        return sum(t.quantity for t in self.trades if t.seller_id == agent_id)


def _validate(bids) -> None:
    sides = {}
    for b in bids:
        if b.side not in ("buy", "sell"):
            raise ValueError(f"invalid bid side: {b.side}")
        if b.quantity < 0 or b.price < 0:
            raise ValueError("bid quantity and price must be non-negative")
        if sides.setdefault(b.agent_id, b.side) != b.side:
            raise ValueError(f"agent {b.agent_id} cannot buy and sell in the same auction")


def run_double_auction(bids, k: float = 0.5) -> AuctionResult:
    """Clear one auction round. See module docstring for the exact rule."""
    if not 0.0 <= k <= 1.0:
        raise ValueError("k must be in [0, 1]")
    bids = [b for b in bids if b.quantity > _EPS]
    _validate(bids)
    buys = sorted((b for b in bids if b.side == "buy"), key=lambda b: (-b.price, b.agent_id))
    sells = sorted((b for b in bids if b.side == "sell"), key=lambda b: (b.price, b.agent_id))
    rem_buy = [b.quantity for b in buys]
    rem_sell = [s.quantity for s in sells]

    matches = []
    last_bid = last_ask = None
    i = j = 0
    while i < len(buys) and j < len(sells) and buys[i].price >= sells[j].price - 1e-12:
        qty = min(rem_buy[i], rem_sell[j])
        matches.append((buys[i].agent_id, sells[j].agent_id, qty))
        last_bid, last_ask = buys[i].price, sells[j].price
        rem_buy[i] -= qty
        rem_sell[j] -= qty
        if rem_buy[i] <= _EPS:
            i += 1
        if rem_sell[j] <= _EPS:
            j += 1

    result = AuctionResult()
    if matches:
        price = last_ask + k * (last_bid - last_ask)
        result.clearing_price = price
        result.trades = [Trade(b, s, q, price) for b, s, q in matches]
        result.matched_quantity = sum(q for _, _, q in matches)
    result.unmatched_buy = {b.agent_id: r for b, r in zip(buys, rem_buy) if r > _EPS}
    result.unmatched_sell = {s.agent_id: r for s, r in zip(sells, rem_sell) if r > _EPS}
    return result
