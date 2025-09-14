from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Optional
import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_pickle(obj: Any, path: str | Path) -> None:
    p = ensure_parent(path)
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def to_table(df: pd.DataFrame, out: str | Path) -> Path:
    p = ensure_parent(out)
    if str(p).endswith(".parquet"):
        try:
            import pyarrow  # type: ignore # noqa: F401

            df.to_parquet(p, index=False)
        except Exception:
            # Fallback to CSV if pyarrow is unavailable
            p = p.with_suffix(".csv")
            df.to_csv(p, index=False)
    else:
        df.to_csv(p, index=False)
    return p


def read_table(inp: str | Path) -> pd.DataFrame:
    p = Path(inp)
    if p.suffix == ".parquet":
        try:
            return pd.read_parquet(p)
        except Exception:
            # Attempt CSV fallback by replacing extension
            alt = p.with_suffix(".csv")
            if alt.exists():
                return pd.read_csv(alt)
            raise
    if p.suffix == ".csv":
        return pd.read_csv(p)
    # Try both
    if p.with_suffix(".parquet").exists():
        return pd.read_parquet(p.with_suffix(".parquet"))
    return pd.read_csv(p.with_suffix(".csv"))

