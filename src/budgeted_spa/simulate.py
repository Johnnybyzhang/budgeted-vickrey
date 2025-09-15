from __future__ import annotations

"""Monte-Carlo simulator for budgeted repeated second-price auctions.

Fast-mode priorities:
- Vectorized core for N=2
- Aggregate outputs only (optional episodes skipped)
- Optional BLAS threads pinned via env vars
"""

import os
# Pin BLAS threads unless explicitly set
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from dataclasses import dataclass
from typing import List, Callable, Dict
import argparse
import numpy as np
import pandas as pd

from .params import Params, add_common_args, apply_tiny
from .io_utils import to_table, load_pickle


def draw_values(prm: Params, rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
    """Draw private values according to `Params.value_mode`.

    - continuous: Uniform[value_low, value_high]
    - discrete: sample with equal probability from either `value_points` (if provided)
      or the grid arange(value_low, value_high + eps, value_step).
    """
    if prm.value_mode == "continuous":
        return rng.uniform(prm.value_low, prm.value_high, size=size)
    # discrete
    if prm.value_points:
        support = np.asarray(prm.value_points, dtype=float)
    else:
        support = np.arange(prm.value_low, prm.value_high + 1e-9, prm.value_step, dtype=float)
    # sample i.i.d. from support
    flat = rng.choice(support, size=int(np.prod(size)), replace=True)
    return flat.reshape(size)


def second_price_outcome(bids: np.ndarray, rng: np.random.Generator) -> tuple[int, float]:
    """Return (winner_index, price=second-highest bid). Break ties uniformly.

    bids: shape (N,)
    """
    N = bids.shape[0]
    order = np.argsort(bids)
    top = order[-1]
    second = order[-2] if N >= 2 else order[-1]
    # Handle full tie on highest bid
    max_bid = bids[top]
    tied = np.flatnonzero(np.isclose(bids, max_bid))
    if tied.size > 1:
        win = int(rng.choice(tied))
        price = max_bid  # second-highest equals the same under tie
        return win, float(price)
    price = bids[second] if N >= 2 else 0.0
    return int(top), float(price)


def policy_truth(values: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    return np.minimum(values, budgets)


def load_threshold_policy(policy_pkl: str) -> Dict[tuple[int, float], float]:
    dp = load_pickle(policy_pkl)
    return dp.taus  # type: ignore[attr-defined]


def policy_threshold(values: np.ndarray, budgets: np.ndarray, t: int, taus: Dict[tuple[int, float], float], budget_step: float, v_high: float) -> np.ndarray:
    out = np.empty_like(values)
    for i, (v, B) in enumerate(zip(values, budgets)):
        Bg = float(np.round(B / budget_step) * budget_step)
        tau = taus.get((t, Bg), min(B, v_high))
        out[i] = min(v, tau, B)
    return out


def simulate(prm: Params, policy: str = "truth", policy_pkl: str | None = None, fast: bool = False,
             out_episodes: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(prm.seed)
    T = prm.T if not fast else min(5, prm.T)
    n_mc = prm.n_mc if not fast else min(1_000, prm.n_mc)
    N = prm.n_players
    budgets = np.full((n_mc, N), prm.B0, dtype=float)

    taus = None
    if policy == "threshold":
        assert policy_pkl is not None, "threshold policy requires --policy-pkl path"
        taus = load_threshold_policy(policy_pkl)

    agg_rows: List[dict] = []
    ep_rows: List[dict] = []
    cap = (np.inf if prm.budget_cap is None else prm.budget_cap)
    meta = prm.asdict()
    # stringify list-like for CSV friendliness
    if isinstance(meta.get("value_points"), list):
        meta["value_points"] = ",".join(str(x) for x in meta["value_points"])  # type: ignore[index]
    for t in range(1, T + 1):
        vals = draw_values(prm, rng, size=(n_mc, N))
        bids = np.zeros_like(vals)
        if policy in ("truth", "truthful_capped"):
            bids = np.minimum(vals, budgets)
        elif policy == "threshold":
            # per-episode loop only for threshold
            for i in range(n_mc):
                bids[i, :] = policy_threshold(vals[i, :], budgets[i, :], t, taus, prm.budget_step, prm.value_high)  # type: ignore[arg-type]
        else:
            raise ValueError(f"unknown policy {policy}")

        if N == 2:
            b0 = bids[:, 0]
            b1 = bids[:, 1]
            ties = np.isclose(b0, b1)
            # winners: 0 if b0>b1, 1 otherwise; break ties randomly
            rand_bits = rng.integers(0, 2, size=n_mc)
            w = np.where(ties, rand_bits, np.where(b0 > b1, 0, 1))
            price = np.where(w == 0, b1, b0)
            # Metrics (pre-update)
            hv = np.argmax(vals, axis=1)
            eff = (hv == w).astype(float)
            util0 = np.where(w == 0, vals[:, 0] - price, 0.0)
            util1 = np.where(w == 1, vals[:, 1] - price, 0.0)
            truth_cap = np.minimum(vals, budgets)
            truth_dev = np.abs(bids - truth_cap).mean(axis=1)
            # Aggregates
            row = dict(t=t, price_mean=float(price.mean()), efficiency=float(eff.mean()), mean_util=float((util0 + util1).mean()), mean_truth_dev=float(truth_dev.mean()))
            row.update(meta)
            agg_rows.append(row)
            # Update budgets
            budgets[:, 0] = np.minimum(budgets[:, 0] - np.where(w == 0, price, 0.0) + prm.refill, cap)
            budgets[:, 1] = np.minimum(budgets[:, 1] - np.where(w == 1, price, 0.0) + prm.refill, cap)
            if out_episodes:
                for i in range(n_mc):
                    ep_rows.append(dict(t=t, winner=int(w[i]), price=float(price[i]), efficiency=int(eff[i]), mean_util=float((util0[i] + util1[i])), mean_truth_dev=float(truth_dev[i])))
                    # Per-player budgets for trajectory plots (after update)
                    ep_rows.append(dict(t=t, player=0, budget=float(budgets[i, 0])))
                    ep_rows.append(dict(t=t, player=1, budget=float(budgets[i, 1])))
        else:
            # Fallback generic path (loop on episodes)
            winners = np.zeros(n_mc, dtype=int)
            price = np.zeros(n_mc)
            for i in range(n_mc):
                w_i, p_i = second_price_outcome(bids[i], rng)
                winners[i] = w_i
                price[i] = p_i
            hv = np.argmax(vals, axis=1)
            eff = (hv == winners).astype(float)
            util = np.zeros_like(vals)
            for i in range(n_mc):
                util[i, winners[i]] = vals[i, winners[i]] - price[i]
            truth_cap = np.minimum(vals, budgets)
            truth_dev = np.abs(bids - truth_cap).mean(axis=1)
            row = dict(t=t, price_mean=float(price.mean()), efficiency=float(eff.mean()), mean_util=float(util.sum(axis=1).mean()), mean_truth_dev=float(truth_dev.mean()))
            row.update(meta)
            agg_rows.append(row)
            # Update budgets
            for i in range(n_mc):
                budgets[i, winners[i]] = min(budgets[i, winners[i]] - price[i] + prm.refill, cap)
                for j in range(N):
                    if j != winners[i]:
                        budgets[i, j] = min(budgets[i, j] + prm.refill, cap)
            if out_episodes:
                for i in range(n_mc):
                    ep_rows.append(dict(t=t, winner=int(winners[i]), price=float(price[i]), efficiency=int(eff[i]), mean_util=float(util[i].sum()), mean_truth_dev=float(truth_dev[i])))
                    for j in range(N):
                        ep_rows.append(dict(t=t, player=int(j), budget=float(budgets[i, j])))

    return pd.DataFrame(ep_rows) if out_episodes else pd.DataFrame(agg_rows)


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--policy", choices=["truth", "truthful_capped", "threshold"], default="truthful_capped")
    p.add_argument("--policy-pkl", type=str, default=None)
    p.add_argument("--out-episodes", type=str, default="none")
    p.add_argument("--out-aggregates", type=str, default="results/fast.csv")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--skip-dp", action="store_true")  # accepted for compatibility; no-op here
    p.add_argument("--skip-nash", action="store_true")  # accepted for compatibility; no-op here
    args = p.parse_args(argv)

    prm = Params()
    for k, v in vars(args).items():
        if k in {"policy", "policy_pkl", "out", "fast"}:
            continue
        if v is not None:
            setattr(prm, k.replace("-", "_"), v)
    # Normalize potential comma-separated string for value_points
    if isinstance(prm.value_points, str):  # type: ignore[attr-defined]
        s: str = prm.value_points  # type: ignore[assignment]
        pts = [float(x) for x in s.split(",")] if s else []
        prm.value_points = pts  # type: ignore[assignment]
    if args.tiny:
        prm = apply_tiny(prm)
    prm.validate()
    out_eps = (args.out_episodes.lower() != "none")
    df = simulate(prm, policy=args.policy, policy_pkl=args.policy_pkl, fast=(args.fast or args.tiny), out_episodes=out_eps)
    if out_eps:
        to_table(df, args.out_episodes)
    else:
        to_table(df, args.out_aggregates)


if __name__ == "__main__":
    main()
