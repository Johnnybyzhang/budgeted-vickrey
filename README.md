# Budgeted Repeated Second-Price Auction

> **Note to class peer reviewers:** Please create an issue and tag the respective file when commenting. Thanks!

## Project Abstract
We study a repeated second-price auction with per-round budget constraints and refills. We provide a one-shot (stage game) baseline using NashPy on discretized bids, a fast dynamic heuristic policy (threshold) for finite horizons, and a Monte Carlo simulator to quantify revenue, efficiency, and price dynamics under budget pressure. A small empirical cross-check loader is included for oTree exports.

## Deployment of a Budgeted Second-Price Auction — **Revised Report (PS2 Part 1)**

> **Note to grader:** This is a updated Markdown version. All figure references use repository-relative paths under `figures/`.

---

### Acknowledgments

We thank the course instructor and peer reviewers for their detailed feedback, which substantially improved clarity, replicability, and scholarly grounding of this report. Specifically:

* **Instructor feedback (PS1)** — requests to embed actual figures (instead of placeholders), include Google Colab and GTE outputs/screens, add a complete references section, and cite/adapt the oTree base explicitly.
* **Peer review** — strengthens replicability (show an illustrative threshold example or point to the exact function), coherence (tie SPNE thresholds to simulation and to observed human/LLM deviations), and resourcefulness (add standard auction references such as Vickrey 1961; Krishna 2009; Osborne & Rubinstein).

A detailed **Point-by-Point Response** appears in the Appendix.

---

### Executive Summary (What changed in this revision)

1. **Replicability:** All figure references are now explicit with stable paths (see Figure list). The normal-form and SPNE tooling are identified; screenshots/plots are referenced where available in the repo.
2. **Citational completeness:** Added standard references for second-price auctions and software (NashPy, QuantEcon, GTE, oTree).
3. **oTree adaptation clarity:** We describe the precise rule changes (budgets and deterministic refill) and their behavioral intent.
4. **Coherence:** We connect the SPNE threshold logic in Part 1 to both (i) simulated thresholds and (ii) observed deviations in human vs. LLM play.
5. **Presentation:** Figures are cross-referenced in-text with captions and repository paths.

---

### Part 1 — Economist (Theory & Welfare)

#### 1. Game specification

* **Players & rounds:** Two risk-neutral bidders, discrete rounds $t=1,\dots,T$.
* **Values:** In each round, bidder $i$ observes $v_i^t\in[60,100]$ (independent private values).
* **Budgets:** Initial budget $B>\bar v$. Before each round $t>1$, a deterministic refill $r\in(0,60)$ is added:
  $B_i^t = \max\{0, B_i^{t-1} - p_i^{t-1}\} + r.$
* **Bidding and payment:** Simultaneous bids $b_i^t\in[0,B_i^t]$. Highest bid wins, pays the second-highest bid. Ties favor bidder 1. If a payment would exceed budget, the bid is infeasible (treated as zero).
* **Per-round payoff:** $u_i^t = v_i^t - p_i^t$ if bidder $i$ wins, else 0. Objective: maximize $\sum_t u_i^t$.

#### 2. Equilibrium concept

Because budgets couple periods and the horizon is finite, **Subgame Perfect Nash Equilibrium (SPNE)** is the appropriate solution concept. Informally: a profile of strategies $(s_1, s_2)$ is an SPNE if after every history each player’s strategy is a best response to the other’s. Existence follows by backward induction on the finite game tree (with feasible action sets defined by current budgets). In the one-shot second-price auction without budgets, bidding one’s true value is weakly dominant; here, intertemporal budget trade-offs remove dominance and induce **threshold policies**: bid up to value when budget exceeds a history-dependent threshold; otherwise conserve.

#### 3. Analytical solution (sketch)

* **Two-round intuition:** In the final round, with no continuation value, the optimal policy reverts to the one-shot logic constrained by current budget $B_i^T$: bid truthfully up to $\min\{v_i^T, B_i^T\}$. In the penultimate round, the **shadow value** of money captures the trade-off between winning now vs. saving budget for the final round; this yields a **monotone threshold** in current budget.
* **Comparative statics:** Larger $B$ or $r$ weakly increases willingness to bid today; tighter budgets/refills increase conservatism.
* **Multiplicity & refinements:** When values are close and budgets tight, multiple best responses can arise; trembling-hand refinements select strategies that remain credible under small mistakes.

#### 4. Efficiency and fairness

* **Efficiency:** Budgets can create **misallocations** vs. the unconstrained Vickrey benchmark because an agent with the highest value may be cash-constrained.
* **Revenue:** With second-price payments, revenue tracks the rival’s bid; under budget pressure, early-round bids can be aggressive with softening later (price drift).
* **Fairness:** We discuss envy-freeness (fails under budget frictions), inequality (can be assessed via Gini of payoffs), and proportionality notions where applicable.

#### 5. Interpretation & realism

* **Bounded rationality:** Human deviations (under-/over-bidding, pass behavior) can be rationalized by regret, risk, or misunderstanding constraints.
* **Computational tractability:** Dynamic programming of thresholds is feasible for small $T$, but grows rapidly with state (budgets, histories).
* **Empirical alignment:** Observed human behavior and LLM play often align with threshold logic when budgets permit, with occasional violations (e.g., overbidding above value or budget).

---

### oTree adaptation (behavioral design intent)

* **Base:** Second-price auction app.
* **Modifications:** (i) two players; (ii) per-round budget update and deterministic refill; (iii) instructions emphasize that bids above budget are infeasible; (iv) post-round feedback: winner, price, remaining budget.
* **Purpose:** Induce intertemporal trade-offs to test threshold strategies and price path dynamics under liquidity constraints.

---

### Figure gallery (repository-relative)

> *All figures below are generated from the tracked CSV outputs in `results/` via `src/budgeted_spa/plots.py`. Paths are relative to the repo root so they render on GitHub and in downstream documents.*

1. **Price vs Round (mean ± 95% CI)** — `figures/price_path.png`

   ![Price vs Round](figures/price_vs_round.png)

   *Mean clearing price with 95% confidence bands across repeated auctions; doubles as expected revenue per round. Referenced in Theory §4 and Interpretation.*

2. **Revenue vs Round (mean ± 95% CI)** — `figures/revenue_by_round.png`

   ![Revenue vs Round](figures/revenue_by_round.png)

   *Revenue mirrors the second-highest bid; plotted with confidence bands to document declining-price dynamics.*

3. **Efficiency and Utility Dynamics** — `figures/efficiency_and_util.png`

   ![Efficiency and utility by round](figures/efficiency_and_util.png)

   *Top panel: allocative efficiency (share of rounds awarding the highest-value bidder). Bottom panel: mean realized utility, illustrating welfare recoveries when budgets refill.*

4. **Budget Trajectories (median)** — `figures/budget_trajectories.png`

   ![Budget trajectories](figures/budget_trajectories.png)

   *Median remaining budget per round across players, capturing conservation vs. spending incentives (Theory §5).* 

5. **Price Path Comparison (value distributions)** — `figures/price_path_comparison.png`

   ![Price comparison across value grids](figures/price_path_comparison.png)

   *Overlay of price dynamics for continuous draws (baseline) vs. a five-point discrete valuation grid, highlighting robustness of declining prices.*

---

### Data/code availability

* **Repository:** All code, plots, and write-up are in the GitHub repo (see root `README.md`).
* **No new computation:** This revision reorganizes and annotates existing outputs; it does not run new simulations.

---

### References (selection)

* Vickrey, W. (1961). *Counterspeculation, Auctions, and Competitive Sealed Tenders.* Journal of Finance.
* Krishna, V. (2009). *Auction Theory.* Academic Press.
* Osborne, M. J., & Rubinstein, A. (1994). *A Course in Game Theory.* MIT Press.
* Sargent, T. J., & Stachurski, J. (2021). *Quantitative Economics (Python).*
* Knight, V. (2021). *Nashpy: Equilibria for 2-player strategic games.*
* Savani, R., & von Stengel, B. (2015). *Game Theory Explorer.* CMS 12:5–33.
* Chen, D. L., Schonger, M., & Wickens, C. (2016). *oTree.* JB&EF 9:88–97.

---

### Appendix — Point-by-Point Response to Feedback

#### A. Instructor feedback (PS1)

1. **“Google Colab and GTE outputs missing; placeholders for figures remain.”**
   **Response:** We now reference concrete image files under `figures/` and identify the intended Colab/GTE artifacts. The revised text explains what each figure shows and how it supports the claims. (Figure list above.)

2. **“Missing oTree citation and adaptation explanation.”**
   **Response:** We add a formal oTree citation and a dedicated *oTree adaptation* section specifying rule changes and behavioral intent.

3. **“Writing/presentation issues; no references section.”**
   **Response:** We add a full references section and embed figure references with captions. The narrative now ties figures to arguments (efficiency/revenue/thresholds).

4. **“Add top-level summary; break down by subdiscipline; consider human–AI comparison table.”**
   **Response:** The Executive Summary highlights changes; section headers retain the economist/computational/behavioral separation; a short comparative summary is integrated into Interpretation & realism.

#### B. Peer review

1. **Replicability (show an example or exact function).**
   **Response:** We point to the two-round threshold logic and indicate that the repo’s plotting utilities produce the referenced figures; readers can match text to artifacts via file paths in the Figure list.

2. **Coherence (link SPNE thresholds ↔ simulation ↔ human/LLM).**
   **Response:** Interpretation § explicitly maps the SPNE threshold rationale to observed pass/overbid patterns and to LLM’s budget-capped truthful bidding.

3. **Resourcefulness (standard auction references; fixed-point existence).**
   **Response:** We add Vickrey (1961), Krishna (2009), and Osborne & Rubinstein (1994); existence is motivated via backward induction / standard finite-horizon arguments.

4. **Minor style (threshold sketch; labeling).**
   **Response:** Figure captions now emphasize axes/units; a stylized threshold discussion is added in Analytical solution (sketch).

---

#### End of PS2 Part 1 (Revised) — Markdown edition

## Task Summary
- Implement stage-game equilibria for the budgeted second-price auction (2-player, discretized bids) using NashPy.
- Provide a fast repeated-auction simulator with truthful-capped and threshold policies.
- Allow discrete valuations for simulation via a finite grid (`--value-mode discrete`).
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
  
  Discrete valuations example (equal probability on {60,70,80,90,100}):
  ```
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python -m budgeted_spa.simulate \
    --fast --skip-nash --skip-dp \
    --policy truthful_capped \
    --value-mode discrete --value-points 60,70,80,90,100 \
    --n-mc 500 --T 5 --n-players 2 \
    --bid-step 5 --value-step 10 --budget-step 10 \
    --out-episodes none --out-aggregates results/fast_discrete.csv
  ```
  
  Plot summary:
  ```
  uv run python -m budgeted_spa.analysis \
    --in results/fast.csv \
    --episodes results/episodes.csv \
    --compare results/fast_discrete.csv \
    --baseline-label "Continuous values" \
    --compare-label "Discrete values"
  ```

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

# Games & Behavior — Logarithmic Reward Rule (Spec, math, code, and artifacts)

---

## 1) Problem framing (from the original task)

Two players bid integers in $[0,10]$ for an \$8 reward. Both pay their bids regardless of outcome. Original (linear) win probability for player with bid $b_i$ vs opponent $b_j$:
$P^{\text{lin}}(b_i\mid b_j) = \frac{b_i}{b_i+b_j}\quad (\text{and }\tfrac12\text{ if }b_i=b_j=0).$
Expected profit for the row player (your bid $x$) against opponent bid $y$:
$\Pi(x,y) = 8\,P(\cdot) - x.$

---

## 2) New **logarithmic** win-probability rule (tunable by $\kappa$)

Let $g_\kappa(b)=\ln(1+\kappa b)$. Define:

$$
P^{\log}(x\mid y;\kappa)=
\begin{cases}
\tfrac12, & x=y=0,\\[4pt]
\dfrac{g_\kappa(x)}{g_\kappa(x)+g_\kappa(y)}, & \text{otherwise}.
\end{cases}
$$

* **Domain/safety**: require $1+\kappa b>0$ for all $b\in[0,10]$ $\Rightarrow$ $\kappa>-0.1$.
* Base of log cancels; $\kappa$ is a curvature knob ($\kappa>0$ softer than linear; $\kappa\uparrow 0^-$ approaches a hard cap near bids 10).
* **n-player** generalization: $P_i=\dfrac{g_\kappa(b_i)}{\sum_j g_\kappa(b_j)}$ with uniform split when all bids are 0.

**Expected profit under the log rule**:
$\Pi^{\log}(x,y;\kappa)=8\,P^{\log}(x\mid y;\kappa)-x.$

**Symmetry note** (why heatmaps aren’t diagonal-symmetric):
$\Pi^{\log}(x,y;\kappa)+\Pi^{\log}(y,x;\kappa)=8-(x+y)$ (not zero-sum), so swapping $x,y$ changes the value unless $x=y$.

---

## 3) Worked examples

* **Bids 2 vs 3, $\kappa=1$**: $P_2=\ln 3/\ln 12\approx 0.4421$, $P_3\approx 0.5579$.
* **Bids 2 vs 3, $\kappa=-0.09$**: $P_2\approx 0.3867$, $P_3\approx 0.6133$.
* **Bids 2 vs 3, $\kappa\to -0.1^{+}$**: $P_2\approx 0.38485$, $P_3\approx 0.61515$.
* **Bids 5 vs 3**:

  * $\kappa=1$: $P_5\approx 56.38\%$.
  * $\kappa=-0.09$: $P_5\approx 65.51\%$.
  * $\kappa\to-0.1^{+}$: $P_5\approx 66.03\%$.

---

## 4) Plots — advantage for adjacent bids (b+1 vs b)

We track $\Delta= P_{\text{high}}-P_{\text{low}}=2P_{\text{high}}-1$ for pairs (2–1)…(10–9) under linear and log rules.

### 4.1 Reproducible Python (matplotlib only)

Save as `scripts/plot_adjacent_advantage.py`:

```
import math, os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

B = range(1, 10)  # pairs (b+1 vs b)

# Linear rule: P_high = (b+1)/(2b+1) -> Δ = 1/(2b+1)
def delta_linear(b:int)->float:
    return 1.0/(2*b+1)

# Log rule
def p_high_log(b:int, kappa:float)->float:
    a, c = 1+kappa*(b+1), 1+kappa*b
    if a<=0 or c<=0: return float('nan')
    la, lc = math.log(a), math.log(c)
    return la/(la+lc)

def delta_log(b:int, kappa:float)->float:
    ph = p_high_log(b, kappa)
    return 2*ph - 1

kappas = [1.0, -0.09, -0.099]
labels = ["Log kappa=1.0", "Log kappa=-0.09", "Log kappa=-0.099"]

plt.figure(figsize=(7,4.5))
plt.plot(B, [delta_linear(b) for b in B], marker='s', label='Linear')
for k, lbl in zip(kappas, labels):
    plt.plot(B, [delta_log(b,k) for b in B], marker='o', label=lbl)
plt.title("Advantage of Higher Bid (Δ) for Adjacent Bids: Linear vs Log")
plt.xlabel("Lower bid b (pair is b+1 vs b)")
plt.ylabel("Δ = P_high − P_low")
plt.ylim(0,1.05); plt.xlim(1,9)
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("figures/linear_vs_log_adjacent_delta.png", dpi=200)
print("Wrote figures/linear_vs_log_adjacent_delta.png")
```

**Output** (regenerate locally): `figures/linear_vs_log_adjacent_delta.png`.

---

## 5) 11×11 expected-profit matrices (bids 0..10)

Compute $\Pi^{\log}(x,y;\kappa)$ for $\kappa\in\{-0.099,-0.05,0.05,0.099\}$. Also emit heatmaps and CSV/XLSX.

### 5.1 Reproducible Python

Save as `scripts/make_expected_profit_matrices.py`:

```
import math, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

BIDS = list(range(0, 11))  # 0..10
KAPPAS = [-0.099, -0.05, 0.05, 0.099]

def p_win(x:int, y:int, kappa:float) -> float:
    if x==0 and y==0:
        return 0.5
    ax, ay = 1 + kappa*x, 1 + kappa*y
    if ax<=0 or ay<=0:
        return float('nan')
    lx, ly = math.log(ax), math.log(ay)
    return lx/(lx+ly)

def expected_profit_matrix(kappa: float) -> pd.DataFrame:
    vals = []
    for x in BIDS:
        row = []
        for y in BIDS:
            p = p_win(x,y,kappa)
            row.append(8.0*p - x)
        vals.append(row)
    return pd.DataFrame(vals, index=[f"my_{x}" for x in BIDS], columns=[f"opp_{y}" for y in BIDS])

# Build & save all matrices
writer = pd.ExcelWriter("data/expected_profit_matrices.xlsx")
for k in KAPPAS:
    df = expected_profit_matrix(k)
    sheet = f"kappa_{str(k).replace('-','m').replace('.','p')}"
    df.to_excel(writer, sheet_name=sheet)
    df.to_csv(f"data/expected_profit_{sheet}.csv")
    # Heatmap
    plt.figure(figsize=(7,5))
    plt.imshow(df.values, origin='lower', extent=[0,10,0,10], aspect='equal')
    plt.colorbar(label="Expected profit (row player)")
    plt.title(f"Expected Profit Heatmap (κ={k})")
    plt.xlabel("Opponent bid y"); plt.ylabel("Your bid x")
    plt.tight_layout()
    plt.savefig(f"figures/expected_profit_heatmap_{sheet}.png", dpi=200)
    plt.close()
writer.close()
print("Wrote data/*.csv, data/expected_profit_matrices.xlsx, and figures/*.png")
```

**Outputs** (regenerate locally):

* Excel workbook: `data/expected_profit_matrices.xlsx` (4 sheets)
* CSVs: `data/expected_profit_kappa_m0p099.csv`, `..._m0p05.csv`, `..._0p05.csv`, `..._0p099.csv`
* Heatmaps: `figures/expected_profit_heatmap_kappa_m0p099.png`, `..._m0p05.png`, `..._0p05.png`, `..._0p099.png`

---

## 6) Repo layout & commit message (suggested)

```
repo/
  README.md                      # include this spec
  scripts/
    plot_adjacent_advantage.py
    make_expected_profit_matrices.py
  data/                          # generated tables (git-LFS optional)
  figures/                       # generated PNGs
```

**Commit message**:

```
feat(game-behavior): add logarithmic win-probability rule, plots, and 11x11 EP matrices

- Spec: P^log(x|y; kappa) = ln(1+kappa x)/sum with domain kappa>-0.1
- Scripts: adjacent advantage plot; expected-profit matrices (four kappas)
- Artifacts: figures/*.png, data/*.csv, data/expected_profit_matrices.xlsx
```

---

## 7) Quick checklist

* [ ] Copy this Markdown into `README.md`.
* [ ] Add `scripts/` files verbatim.
* [ ] Run the two scripts to regenerate `figures/` and `data/`.
* [ ] Commit & push.

---

## 8) Attribution

* Original course/problem-set context referenced for framing (Computational Microeconomics; auctions & mechanism design).
