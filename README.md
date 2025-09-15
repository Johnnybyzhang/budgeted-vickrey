## Budgeted Repeated Second-Price Auction

### Abstract
We study a repeated second-price auction with per-round budget constraints and refills. We provide a one-shot (stage game) baseline using NashPy on discretized bids, a fast dynamic heuristic policy (threshold) for finite horizons, and a Monte-Carlo simulator to quantify revenue, efficiency, and price dynamics under budget pressure. A small empirical cross-check loader is included for oTree exports.

## Task Summary
- Implement stage-game equilibria for the budgeted second-price auction (2-player, discretized bids) using NashPy.
- Provide a fast repeated-auction simulator with truthful-capped and threshold policies.
- Generate analytics/plots and a LaTeX scaffold. Split documentation across economist, computational scientist, and behavioral scientist tracks.

## Reproduction Steps (Python ≥ 3.10, macOS/Linux)
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Sync env: `cd budgeted-vickrey && uv sync --dev`
- Fast simulation (aggregate-only):
  ```
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python -m budgeted_spa.simulate \
    --fast --skip-nash --skip-dp \
    --policy truthful_capped \
    --n-mc 500 --T 5 --n-players 2 \
    --bid-step 5 --value-step 10 --budget-step 10 \
    --out-episodes none --out-aggregates results/fast.csv
  ```
- Plot summary: `uv run python -m budgeted_spa.analysis --in results/fast.csv`

## Optional Components
- Stage-game grid (NashPy, small): `uv run python -m budgeted_spa.stage_game --grid --tiny --skip-nash --out results/stage_grid_tiny.csv`
- Heuristic policy artifact: `uv run python -m budgeted_spa.dynamics --skip-dp --out results/dp_policy.pkl --csv results/dp_policy.csv`

## Outputs
- Figures in `figures/` (e.g., price_vs_round.png)
- Aggregates in `results/` (fast.csv, summary.csv)

## Documentation Split
- economist/: background, citations (with pages/sections), and refs
- computational_scientist/: Colab notebook, how-to-run
- behavioral_scientist/: oTree deployment, screenshots, and LLM prompt artifacts
