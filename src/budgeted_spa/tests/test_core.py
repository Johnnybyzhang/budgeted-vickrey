from __future__ import annotations

import numpy as np
from budgeted_spa.simulate import second_price_outcome


def test_second_price_basic():
    rng = np.random.default_rng(0)
    bids = np.array([3.0, 1.0])
    w, p = second_price_outcome(bids, rng)
    assert w == 0 and p == 1.0


def test_tie_break_price_equals_bid():
    rng = np.random.default_rng(1)
    bids = np.array([2.0, 2.0])
    w, p = second_price_outcome(bids, rng)
    assert p == 2.0

