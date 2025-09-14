from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def _ensure_figures_dir(outdir: str | Path) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_price_path(df: pd.DataFrame, outdir: str | Path = "figures", dpi: int = 200) -> Path:
    p = _ensure_figures_dir(outdir)
    g = df.groupby("t")["price"]
    mean = g.mean()
    se = g.std(ddof=1) / np.sqrt(g.count())
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    ax.plot(mean.index, mean.values, label="mean price")
    ax.fill_between(mean.index, lo.values, hi.values, color="C0", alpha=0.2, label="95% CI")
    ax.set_xlabel("Round")
    ax.set_ylabel("Price")
    ax.set_title("Second-Price: Price vs Round")
    ax.legend()
    path = p / "price_vs_round.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_budget_trajectories(sim_detail: pd.DataFrame, outdir: str | Path = "figures", dpi: int = 200) -> Path:
    # Optional: if detailed budgets by episode were saved
    p = _ensure_figures_dir(outdir)
    if not {"t", "budget"}.issubset(sim_detail.columns):
        return p / "budget_traj_skipped.png"
    g = sim_detail.groupby("t")["budget"]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    ax.plot(g.median().index, g.median().values, label="median budget")
    ax.set_xlabel("Round")
    ax.set_ylabel("Budget")
    ax.set_title("Budget Trajectories (median)")
    ax.legend()
    path = p / "budget_trajectories.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path

