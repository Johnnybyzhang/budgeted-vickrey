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
    # Accept either per-episode 'price' or aggregate 'price_mean'
    if "price" in df.columns:
        g = df.groupby("t")["price"]
    elif "price_mean" in df.columns:
        tmp = df[["t", "price_mean"]].rename(columns={"price_mean": "price"})
        g = tmp.groupby("t")["price"]
    else:
        # nothing to plot
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No price column", ha="center", va="center")
        path = p / "price_vs_round.png"
        fig.savefig(path)
        plt.close(fig)
        return path
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
    # Also save under the name expected by report.tex
    fig.savefig(p / "price_path.png")
    plt.close(fig)
    return path


def plot_budget_trajectories(sim_detail: pd.DataFrame, outdir: str | Path = "figures", dpi: int = 200) -> Path:
    # Optional: if detailed budgets by episode were saved
    p = _ensure_figures_dir(outdir)
    if not {"t", "budget"}.issubset(sim_detail.columns):
        # Create a placeholder with the expected filename
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No budget detail available", ha="center", va="center")
        path = p / "budget_trajectories.png"
        fig.savefig(path)
        plt.close(fig)
        return path
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


def plot_revenue_path(df: pd.DataFrame, outdir: str | Path = "figures", dpi: int = 200) -> Path:
    """Revenue equals the price in a second-price auction; plot by round.

    Saves to figures/revenue_by_round.png to match report.tex.
    """
    p = _ensure_figures_dir(outdir)
    # Accept columns: revenue or price_mean/price
    if "revenue" in df.columns:
        g = df.groupby("t")["revenue"]
    elif "price" in df.columns:
        g = df.groupby("t")["price"]
    elif "price_mean" in df.columns:
        tmp = df[["t", "price_mean"]].rename(columns={"price_mean": "revenue"})
        g = tmp.groupby("t")["revenue"]
    else:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No revenue column", ha="center", va="center")
        path = p / "revenue_by_round.png"
        fig.savefig(path)
        plt.close(fig)
        return path
    mean = g.mean()
    se = g.std(ddof=1) / np.sqrt(g.count())
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    ax.plot(mean.index, mean.values, label="mean revenue")
    ax.fill_between(mean.index, lo.values, hi.values, color="C1", alpha=0.2, label="95% CI")
    ax.set_xlabel("Round")
    ax.set_ylabel("Revenue")
    ax.set_title("Second-Price: Revenue vs Round")
    ax.legend()
    path = p / "revenue_by_round.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_stage_heatmap(df: pd.DataFrame, measure: str = "eq_count", outdir: str | Path = "figures", dpi: int = 200) -> Path:
    """Heatmap over diagonal cases v1=v2 and B1=B2.

    measure: one of {'eq_count', 'truth_capped_br'}
    """
    p = _ensure_figures_dir(outdir)
    diag = df[(np.isclose(df.v1, df.v2)) & (np.isclose(df.B1, df.B2))].copy()
    if diag.empty or measure not in diag.columns:
        # generate a dummy figure to keep LaTeX happy
        fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi)
        ax.text(0.5, 0.5, "No stage grid data", ha="center", va="center")
        path = p / f"stage_heatmap_{measure}.png"
        fig.savefig(path)
        plt.close(fig)
        return path
    pivot = diag.pivot_table(index="v1", columns="B1", values=measure, aggfunc="mean")
    v_vals = pivot.index.values
    b_vals = pivot.columns.values
    Z = pivot.values
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=dpi)
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(b_vals)))
    ax.set_xticklabels([str(int(x)) for x in b_vals])
    ax.set_yticks(np.arange(len(v_vals)))
    ax.set_yticklabels([str(int(x)) for x in v_vals])
    ax.set_xlabel("Budget B")
    ax.set_ylabel("Value v")
    ax.set_title(f"Stage Game Heatmap: {measure}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    path = p / f"stage_heatmap_{measure}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
