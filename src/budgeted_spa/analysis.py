from __future__ import annotations

from typing import List
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from .io_utils import read_table, to_table
from .plots import plot_price_path
from .otree_ingest import load_otree_csv, summarize_otree


def declining_price_slope(df: pd.DataFrame) -> tuple[float, float]:
    if "price" in df.columns:
        g = df.groupby("t")["price"].mean().reset_index()
    elif "price_mean" in df.columns:
        g = df[["t", "price_mean"]].dropna().rename(columns={"price_mean": "price"})
    else:
        return 0.0, float("nan")
    x = g["t"].to_numpy()
    y = g["price"].to_numpy()
    x_ = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(x_, y, rcond=None)
    slope = beta[1]
    # naive stderr
    resid = y - x_.dot(beta)
    s2 = (resid**2).sum() / max(1, len(x) - 2)
    varb = s2 * np.linalg.inv(x_.T @ x_)[1, 1]
    se = np.sqrt(varb)
    tstat = slope / se if se > 0 else np.nan
    return float(slope), float(tstat)


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=str, required=True)
    p.add_argument("--otree", type=str, default=None)
    p.add_argument("--out", type=str, default="results/summary.csv")
    args = p.parse_args(argv)

    df = read_table(args.inp)
    slope, tstat = declining_price_slope(df)
    fig_path = plot_price_path(df, outdir="figures")
    out_df = pd.DataFrame([
        dict(metric="declining_price_slope", value=slope),
        dict(metric="declining_price_tstat", value=tstat),
    ])
    to_table(out_df, args.out)
    print(f"Saved summary to {args.out} and figure {fig_path}")

    if args.otree and Path(args.otree).exists():
        ot = load_otree_csv(args.otree)
        ots = summarize_otree(ot)
        if not ots.empty:
            to_table(ots, "results/otree_summary.csv")


if __name__ == "__main__":
    main()
