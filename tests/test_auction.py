import pytest
from double_auction import Bid, run_double_auction


def test_valid_match_and_midpoint_price():
    res = run_double_auction([Bid(1, "sell", 3.0, 5.5), Bid(2, "buy", 2.0, 6.5)])
    assert res.matched_quantity == pytest.approx(2.0)
    assert res.clearing_price == pytest.approx(6.0)
    assert res.unmatched_sell == {1: pytest.approx(1.0)}


def test_no_match_when_bid_below_ask():
    res = run_double_auction([Bid(1, "sell", 3.0, 7.0), Bid(2, "buy", 2.0, 6.5)])
    assert res.trades == [] and res.clearing_price is None
    assert res.unmatched_buy == {2: 2.0} and res.unmatched_sell == {1: 3.0}


def test_partial_quantity_matching():
    res = run_double_auction([Bid(1, "sell", 1.0, 5.0), Bid(2, "buy", 4.0, 7.0)])
    assert res.matched_quantity == pytest.approx(1.0)
    assert res.unmatched_buy[2] == pytest.approx(3.0)


def test_multiple_buyers_and_sellers_priority_and_conservation():
    bids = [Bid(1, "sell", 2.0, 5.0), Bid(2, "sell", 2.0, 6.0), Bid(3, "sell", 2.0, 9.0),
            Bid(4, "buy", 3.0, 8.0), Bid(5, "buy", 2.0, 7.0), Bid(6, "buy", 5.0, 4.0)]
    res = run_double_auction(bids)
    assert res.matched_quantity == pytest.approx(4.0)          # only sellers 1,2 clear
    assert res.sold_by(3) == 0 and res.bought_by(6) == 0        # 9.0 ask / 4.0 bid excluded
    assert sum(res.bought_by(i) for i in (4, 5, 6)) == pytest.approx(sum(res.sold_by(i) for i in (1, 2, 3)))
    assert res.bought_by(4) == pytest.approx(3.0)               # highest bid served first


def test_individual_rationality():
    bids = [Bid(1, "sell", 2.0, 5.0), Bid(2, "sell", 2.0, 6.0), Bid(3, "buy", 2.0, 8.0), Bid(4, "buy", 2.0, 7.0)]
    res = run_double_auction(bids)
    p = res.clearing_price
    limits = {b.agent_id: b.price for b in bids}
    for t in res.trades:
        assert p <= limits[t.buyer_id] + 1e-9 and p >= limits[t.seller_id] - 1e-9


def test_agent_cannot_buy_and_sell_together():
    with pytest.raises(ValueError):
        run_double_auction([Bid(1, "sell", 1, 5), Bid(1, "buy", 1, 6)])


def test_empty_auction():
    res = run_double_auction([])
    assert res.trades == [] and res.matched_quantity == 0
