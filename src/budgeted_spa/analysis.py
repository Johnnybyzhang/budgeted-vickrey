from __future__ import annotations

from typing import List
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from .io_utils import read_table, to_table
from .plots import plot_price_path, plot_stage_heatmap, plot_revenue_path
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
    p.add_argument("--episodes", type=str, default=None, help="episode-level CSV to plot budget trajectories")
    p.add_argument("--stage-grid", type=str, default=None, help="stage_game grid CSV to plot heatmaps")
    p.add_argument("--out", type=str, default="results/summary.csv")
    args = p.parse_args(argv)

    df = read_table(args.inp)
    slope, tstat = declining_price_slope(df)
    fig_path = plot_price_path(df, outdir="figures")
    rev_path = plot_revenue_path(df, outdir="figures")
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

    # Stage heatmaps and equilibrium alias for report
    if args.stage_grid and Path(args.stage_grid).exists():
        sg = read_table(args.stage_grid)
        hm = plot_stage_heatmap(sg, measure="eq_count", outdir="figures")
        if "truth_capped_br" in sg.columns:
            plot_stage_heatmap(sg, measure="truth_capped_br", outdir="figures")
        # Copy/alias to equilibrium_heatmap.png expected by report.tex
        try:
            import shutil
            dest = Path("figures/equilibrium_heatmap.png")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(hm, dest)
        except Exception:
            pass
    else:
        # Create a placeholder equilibrium_heatmap.png to satisfy LaTeX
        from matplotlib import pyplot as plt
        p = Path("figures")
        p.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "No stage grid provided", ha="center", va="center")
        fig.savefig(p / "equilibrium_heatmap.png")
        plt.close(fig)

    # Optional budget trajectories from episodes CSV
    if args.episodes and Path(args.episodes).exists():
        ep = read_table(args.episodes)
        plot_budget_trajectories(ep, outdir="figures")


if __name__ == "__main__":
    main()
