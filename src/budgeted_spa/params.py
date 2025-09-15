from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, List
import argparse


@dataclass
class Params:
    """Global parameters for the auction environment and experiments."""

    n_players: int = 2
    T: int = 10
    value_low: float = 60.0
    value_high: float = 100.0
    B0: float = 120.0
    refill: float = 30.0  # must satisfy 0 < refill < value_low
    bid_step: float = 1.0
    value_step: float = 5.0
    budget_step: float = 1.0
    budget_cap: Optional[float] = None
    n_mc: int = 10_000
    seed: int = 42
    # Value distribution mode for simulation: "continuous" uses Uniform[value_low, value_high),
    # "discrete" samples from a finite grid.
    value_mode: str = "continuous"  # one of {"continuous", "discrete"}
    # Optional explicit support for discrete values; if empty, uses arange(value_low, value_high, value_step)
    value_points: List[float] = field(default_factory=list)

    def validate(self) -> None:
        assert self.n_players >= 2
        assert self.value_high > self.value_low >= 0
        assert 0 < self.refill < self.value_low
        assert self.bid_step > 0 and self.value_step > 0 and self.budget_step > 0
        if self.budget_cap is not None:
            assert self.budget_cap >= self.B0
        assert self.value_mode in {"continuous", "discrete"}
        if self.value_mode == "discrete":
            if self.value_points:
                # ensure points lie within [low, high]
                lo, hi = self.value_low, self.value_high
                for x in self.value_points:
                    assert lo - 1e-9 <= x <= hi + 1e-9, "value_points must lie within [value_low, value_high]"

    def asdict(self) -> dict:
        return asdict(self)


def add_common_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--n-players", type=int)
    g.add_argument("--T", type=int)
    g.add_argument("--value-low", type=float)
    g.add_argument("--value-high", type=float)
    g.add_argument("--B0", type=float)
    g.add_argument("--refill", type=float)
    g.add_argument("--budget-cap", type=float)
    g.add_argument("--bid-step", type=float)
    g.add_argument("--budget-step", type=float)
    g.add_argument("--value-step", type=float)
    g.add_argument("--value-mode", type=str, choices=["continuous", "discrete"], help="draw values from continuous uniform or a discrete grid")
    g.add_argument("--value-points", type=str, help="comma-separated list for discrete value support; defaults to arange(value_low,value_high,value_step)")
    g.add_argument("--seed", type=int)
    g.add_argument("--n-mc", type=int)
    g.add_argument("--tiny", action="store_true", help="use very small, fast defaults")


def apply_tiny(prm: Params) -> Params:
    """Override parameters for a very fast run.

    Intended to finish in a few seconds on a laptop.
    """
    prm.T = min(prm.T, 3)
    prm.n_mc = min(prm.n_mc, 1000)
    prm.bid_step = max(prm.bid_step, 5.0)
    prm.value_step = max(prm.value_step, 20.0)
    prm.budget_step = max(prm.budget_step, 20.0)
    prm.B0 = min(prm.B0, 120.0)
    prm.refill = min(prm.refill, 20.0)
    return prm


def parse_params_from_cli(argv: Optional[list[str]] = None) -> Params:
    p = argparse.ArgumentParser(prog="budgeted-spa")
    add_common_args(p)
    args = p.parse_args(argv)
    prm = Params()
    for k, v in vars(args).items():
        if v is None:
            continue
        if k == "value_points":
            # parse comma-separated floats
            pts = [float(x) for x in v.split(",")] if isinstance(v, str) else []
            setattr(prm, "value_points", pts)
        else:
            setattr(prm, k.replace("-", "_"), v)
    prm.validate()
    return prm


def main() -> None:
    prm = parse_params_from_cli()
    print(prm)


if __name__ == "__main__":
    main()
