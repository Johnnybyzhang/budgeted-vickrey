from __future__ import annotations

"""Finite-horizon DP for a threshold policy against a truth-telling opponent.

State: (B, t). Policy class: b = min(v, tau(B), B).
Opponent: b_op = v_op (assumes ample budget); values ~ Uniform[value_low, value_high].
This yields a myopic best-response-style dynamic program over tau choices.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List
import argparse
import numpy as np
import pandas as pd

from .params import Params, add_common_args, apply_tiny
from .io_utils import save_pickle, to_table


@dataclass
class DPResult:
    params: Params
    budgets: np.ndarray  # grid
    taus: Dict[Tuple[int, float], float]  # (t, B) -> tau
    values: Dict[Tuple[int, float], float]  # (t, B) -> V_t(B)


def expected_round(B: float, tau: float, prm: Params, rng: np.random.Generator | None = None) -> Tuple[float, float]:
    """Compute expected utility and expected payment for one round.

    - Player bid: b1 = min(v1, tau, B)
    - Opponent bid: b2 = v2 (truth, ample budget)
    - Win iff b1 > b2 (ties have 0.5 prob but measure-zero under continuous grid)
    - Utility if win: v1 - b2
    - Payment if win: b2
    Returns (E[utility], E[payment]).
    Approximates integrals with a coarse value grid.
    """
    v1_grid = np.arange(prm.value_low, prm.value_high + 1e-9, prm.value_step)
    v2_grid = v1_grid
    w = 1.0 / len(v1_grid)
    util = 0.0
    pay = 0.0
    for v1 in v1_grid:
        b1 = min(v1, tau, B)
        # win when v2 < b1
        winners = v2_grid[v2_grid < b1]
        if winners.size:
            # For each v2 < b1, utility is v1 - v2
            util += w * np.mean(v1 - winners)
            pay += w * np.mean(winners)
    # Average also over v2: the inner mean already covers v2 distribution
    return float(util), float(pay)


def backward_induction(prm: Params) -> DPResult:
    # Budget grid
    Bmax = prm.budget_cap if prm.budget_cap is not None else prm.B0 + prm.T * prm.refill
    budgets = np.round(np.arange(0.0, Bmax + 1e-9, prm.budget_step), 6)
    # Candidate taus per budget (cap by B and value_high)
    def tau_candidates(B: float) -> np.ndarray:
        hi = min(B, prm.value_high)
        step = max(prm.bid_step, prm.value_step)
        if hi <= 0:
            return np.array([0.0])
        return np.round(np.arange(0.0, hi + 1e-9, step), 6)

    V: Dict[Tuple[int, float], float] = {}
    TAU: Dict[Tuple[int, float], float] = {}

    # Terminal value at t = T+1 is 0
    for t in range(prm.T, 0, -1):
        for B in budgets:
            best_v = -1e18
            best_tau = 0.0
            for tau in tau_candidates(B):
                u, pay = expected_round(B, tau, prm)
                # Expected next budget: if win, B' = B - pay + refill; if lose, B' = B + refill
                # Approximate using win probability from the same loop: P(win) ≈ E[1{v2<b1}]
                # Estimate P(win) by reusing expected payment relationship: E[b2 | win] * P(win) = E[payment]
                # With uniform v2, E[b2 | win] ≈ 0.5 * E[b1] (rough) -> approximate p_win = pay / max(1e-9, 0.5*E[b1])
                # To avoid complexity, compute P(win) via grid directly here
                v1_grid = np.arange(prm.value_low, prm.value_high + 1e-9, prm.value_step)
                p_win = 0.0
                for v1 in v1_grid:
                    b1 = min(v1, tau, B)
                    p_win += (b1 - prm.value_low) / (prm.value_high - prm.value_low)
                p_win /= len(v1_grid)
                # Expected next budget via two outcomes, then clamp to cap
                B_win = B - pay + prm.refill
                B_lose = B + prm.refill
                if prm.budget_cap is not None:
                    B_win = min(B_win, prm.budget_cap)
                    B_lose = min(B_lose, prm.budget_cap)
                # Snap to grid
                B_win_g = float(np.round(np.clip(B_win, 0, budgets[-1]) / prm.budget_step) * prm.budget_step)
                B_lose_g = float(np.round(np.clip(B_lose, 0, budgets[-1]) / prm.budget_step) * prm.budget_step)
                cont = p_win * V.get((t + 1, B_win_g), 0.0) + (1 - p_win) * V.get((t + 1, B_lose_g), 0.0)
                total = u + cont
                if total > best_v:
                    best_v = total
                    best_tau = float(tau)
            V[(t, float(B))] = best_v
            TAU[(t, float(B))] = best_tau
    return DPResult(params=prm, budgets=budgets, taus=TAU, values=V)


def export_policy(dp: DPResult, out_pkl: str, out_csv: str | None = None) -> None:
    save_pickle(dp, out_pkl)
    if out_csv:
        rows: List[dict] = []
        for (t, B), tau in dp.taus.items():
            rows.append(dict(t=t, B=B, tau=tau, V=dp.values[(t, B)]))
        to_table(pd.DataFrame(rows), out_csv)


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--policy", type=str, choices=["threshold"], default="threshold")
    p.add_argument("--out", type=str, default="results/dp_policy.pkl")
    p.add_argument("--csv", type=str, default="results/dp_policy.csv")
    p.add_argument("--skip-dp", action="store_true", help="skip DP and emit a simple heuristic policy")
    args = p.parse_args(argv)

    prm = Params()
    for k, v in vars(args).items():
        if k in {"policy", "out", "csv"}:
            continue
        if v is not None:
            setattr(prm, k.replace("-", "_"), v)
    if args.tiny:
        prm = apply_tiny(prm)
    prm.validate()

    if args.skip_dp:
        # Emit a trivial constant-threshold policy (quick heuristic)
        budgets = np.round(np.arange(0.0, (prm.B0 + prm.T * prm.refill) + 1e-9, prm.budget_step), 6)
        taus: Dict[Tuple[int, float], float] = {}
        values: Dict[Tuple[int, float], float] = {}
        const_tau = min(90.0, prm.value_high)
        for t in range(1, prm.T + 1):
            for B in budgets:
                taus[(t, float(B))] = min(const_tau, float(B))
                values[(t, float(B))] = 0.0
        dp = DPResult(params=prm, budgets=budgets, taus=taus, values=values)
        export_policy(dp, args.out, args.csv)
    elif args.policy == "threshold":
        dp = backward_induction(prm)
        export_policy(dp, args.out, args.csv)


if __name__ == "__main__":
    main()
