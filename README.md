Budgeted Repeated Second-Price Auction (with uv)

Reproducible repo for PS-1: stage-game Nash equilibria with budgets, a finite-horizon dynamic program (threshold policy), Monte-Carlo simulations, analytics/plots, and a LaTeX report scaffold.

Quickstart (Python ≥ 3.10, macOS/Linux)

- Install uv (if not present): curl -LsSf https://astral.sh/uv/install.sh | sh
- Sync env: cd budgeted-vickrey && uv sync --dev
- Tiny smoke test (aggregate-only, single-threaded BLAS):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python -m budgeted_spa.simulate \
    --tiny --skip-nash --skip-dp \
    --policy truthful_capped \
    --n-mc 500 --T 5 --n-players 2 \
    --bid-step 5 --value-step 10 --budget-step 10 \
    --out-episodes none --out-aggregates results/fast.csv

Reproduce the default pipeline

- Stage-game (tiny, skip Nash for speed): uv run python -m budgeted_spa.stage_game --grid --tiny --skip-nash --out results/stage_grid_tiny.csv
- Dynamic policy (skip DP; emit heuristic): uv run python -m budgeted_spa.dynamics --tiny --skip-dp --out results/dp_policy.pkl --csv results/dp_policy.csv
- Simulation (tiny, threshold): uv run python -m budgeted_spa.simulate --tiny --policy threshold --policy-pkl results/dp_policy.pkl --out-episodes none --out-aggregates results/sim_tiny.csv
- Analytics and plots: uv run python -m budgeted_spa.analysis --in results/sim_default.csv --otree data/all_apps_wide-2025-09-11.csv

Notes

- No network calls during runs beyond initial `uv sync`.
- Parquet output is supported if `pyarrow` is installed; otherwise CSV is used.
- Use `--skip-nash` and `--skip-dp` to avoid heavy solvers in fast runs (NashPy still available via `stage_game` when enabled).
- Aggregates-only mode avoids per-episode I/O (much faster).

Quick profiling

uv run python -m cProfile -o sim.prof -m budgeted_spa.simulate --tiny --out-episodes none --out-aggregates /dev/null
uv run python - <<'PY'
import pstats; s=pstats.Stats('sim.prof'); s.sort_stats('cumtime').print_stats(30)
PY
- Use `--fast` on CLIs for a ≤30s sanity run (smaller T and MC).
