from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_otree_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    return df


def summarize_otree(df: pd.DataFrame) -> pd.DataFrame:
    # Columns pattern may vary; try common pattern
    cols = [c for c in df.columns if ".group.second_price" in c]
    if not cols:
        return pd.DataFrame()
    # Derive round number
    rows = []
    for c in cols:
        # Extract round r from pattern AuctionEbay.{r}.group.second_price
        try:
            r = int(c.split(".")[1])
        except Exception:
            continue
        price = df[c].astype(float)
        rows.append(dict(t=r, price_mean=price.mean(), price_std=price.std(ddof=1)))
    return pd.DataFrame(rows).sort_values("t")

