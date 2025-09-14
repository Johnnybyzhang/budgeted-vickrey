from __future__ import annotations

"""One-shot second-price auction with per-round budget caps.

Discretizes bids and computes Nash equilibria using NashPy for 2 players.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Dict
import argparse
import numpy as np
import pandas as pd
import nashpy as nash
from functools import lru_cache

from .params import Params, add_common_args, apply_tiny
from .io_utils import to_table

# Discrete grids for memoization (fast mode)
VALUES = tuple(range(60, 101, 10))
BUDGETS = tuple(range(60, 121, 10))
BID_STEPS = (1.0, 5.0, 10.0)


def _closest_idx(x: float, grid: Tuple[int, ...]) -> int:
    arr = np.asarray(grid, dtype=float)
    return int(np.argmin(np.abs(arr - x)))


def action_grid(v: float, B: float, step: float) -> np.ndarray:
    m = min(v, B)
    n = int(np.floor(m / step))
    return np.round(np.linspace(0.0, n * step, n + 1), 6)


def payoff_matrices(v1: float, v2: float, B1: float, B2: float, step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A1 = action_grid(v1, B1, step)
    A2 = action_grid(v2, B2, step)
    n1, n2 = len(A1), len(A2)
    U1 = np.zeros((n1, n2))
    U2 = np.zeros((n1, n2))
    price = np.zeros((n1, n2))
    alloc = np.zeros((n1, n2))  # 1 if player1 wins, 0 if player2 wins, 0.5 if tie
    for i, b1 in enumerate(A1):
        for j, b2 in enumerate(A2):
            if b1 > b2:
                U1[i, j] = v1 - b2
                U2[i, j] = 0.0
                price[i, j] = b2
                alloc[i, j] = 1.0
            elif b2 > b1:
                U1[i, j] = 0.0
                U2[i, j] = v2 - b1
                price[i, j] = b1
                alloc[i, j] = 0.0
            else:  # tie -> 50-50 expected
                U1[i, j] = 0.5 * (v1 - b2)
                U2[i, j] = 0.5 * (v2 - b1)
                price[i, j] = b1  # = b2
                alloc[i, j] = 0.5
    return U1, U2, price, alloc


def nash_equilibria(U1: np.ndarray, U2: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    game = nash.Game(U1, U2)
    eqs: List[Tuple[np.ndarray, np.ndarray]] = []
    # Combine two enumeration methods
    for method in (game.support_enumeration, game.vertex_enumeration):
        for sigma_r, sigma_c in method():
            eqs.append((np.asarray(sigma_r), np.asarray(sigma_c)))
    # Deduplicate by rounding
    dedup = []
    seen = set()
    for s1, s2 in eqs:
        key = (tuple(np.round(s1, 6)), tuple(np.round(s2, 6)))
        if key not in seen:
            seen.add(key)
            dedup.append((s1, s2))
    return dedup


def is_pure(s: np.ndarray, tol: float = 1e-6) -> bool:
    return np.sum(s > 1 - tol) == 1 and np.isclose(np.sum(s), 1.0, atol=1e-6)


@lru_cache(maxsize=None)
def _solve_cached(v1_i: int, v2_i: int, B1_i: int, B2_i: int, step_i: int) -> Dict[str, object]:
    v1, v2 = float(VALUES[v1_i]), float(VALUES[v2_i])
    B1, B2 = float(BUDGETS[B1_i]), float(BUDGETS[B2_i])
    step = float(BID_STEPS[step_i])
    U1, U2, price, alloc = payoff_matrices(v1, v2, B1, B2, step)
    A1 = action_grid(v1, B1, step)
    A2 = action_grid(v2, B2, step)
    eqs = nash_equilibria(U1, U2)
    pure_any = any(is_pure(s1) and is_pure(s2) for s1, s2 in eqs)
    # truth-capped best response check
    tt1 = np.isclose(A1, min(v1, B1)).nonzero()[0]
    tt2 = np.isclose(A2, min(v2, B2)).nonzero()[0]
    tt_br = False
    if tt1.size and tt2.size:
        i, j = int(tt1[0]), int(tt2[0])
        tt_br = (U1[i, j] >= U1[:, j].max() - 1e-9) and (U2[i, j] >= U2[i, :].max() - 1e-9)
    return dict(eq_count=len(eqs), pure_any=pure_any, mean_price=float(price.mean()), mean_alloc=float(alloc.mean()), truth_capped_br=bool(tt_br))


def run_grid(prm: Params, out: str, tiny: bool = False, skip_nash: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(prm.seed)
    if tiny:
        values = np.array([prm.value_low, (prm.value_low + prm.value_high) / 2, prm.value_high])
        budgets = np.array([60.0, prm.B0], dtype=float)
    else:
        values = np.arange(prm.value_low, prm.value_high + 1e-9, prm.value_step)
        budgets = np.array([60.0, 80.0, 100.0, prm.B0], dtype=float)
    rows = []
    for v1 in values:
        for v2 in values:
            for B1 in budgets:
                for B2 in budgets:
                    if skip_nash:
                        rows.append(dict(v1=v1, v2=v2, B1=B1, B2=B2, eq_count=0, pure_any=False, truth_capped_br=False, mean_price=min(v1, v2), mean_alloc_to_1=0.5))
                        continue
                    vi1 = _closest_idx(v1, VALUES)
                    vi2 = _closest_idx(v2, VALUES)
                    bi1 = _closest_idx(B1, BUDGETS)
                    bi2 = _closest_idx(B2, BUDGETS)
                    bsi = _closest_idx(prm.bid_step, BID_STEPS)
                    res = _solve_cached(vi1, vi2, bi1, bi2, bsi)
                    rows.append(dict(v1=v1, v2=v2, B1=B1, B2=B2, eq_count=int(res['eq_count']), pure_any=bool(res['pure_any']), truth_capped_br=bool(res['truth_capped_br']), mean_price=float(res['mean_price']), mean_alloc_to_1=float(res['mean_alloc'])))
    df = pd.DataFrame(rows)
    to_table(df, out)
    return df


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--grid", action="store_true", help="run value/budget grid and save CSV")
    p.add_argument("--out", type=str, default="results/stage_grid.csv")
    p.add_argument("--skip-nash", action="store_true", help="skip NashPy solves (fast mode)")
    return p


def main(argv: List[str] | None = None) -> None:
    p = build_argparser()
    args = p.parse_args(argv)
    prm = Params()
    for k, v in vars(args).items():
        if k in {"grid", "out"}:
            continue
        if v is not None:
            setattr(prm, k.replace("-", "_"), v)
    if args.tiny:
        prm = apply_tiny(prm)
    prm.validate()
    if args.grid:
        run_grid(prm, args.out, tiny=args.tiny, skip_nash=args.skip_nash)
    else:
        # Example single instance
        U1, U2, *_ = payoff_matrices(prm.value_high, prm.value_high, prm.B0, prm.B0, prm.bid_step)
        eqs = nash_equilibria(U1, U2)
        print(f"Found {len(eqs)} equilibria for (v1=v2={prm.value_high}, B1=B2={prm.B0}).")


if __name__ == "__main__":
    main()
