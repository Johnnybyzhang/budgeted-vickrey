Economist Track — Background and Citations

Scope
- Define the modified repeated second-price auction with per-round budgets and refills.
- Discuss stage-game incentives under budget caps and truthful-capped bidding.
- Explain dynamic budget accumulation and potential declining price paths.

Key References (verify page/section in your copies)
- Vickrey (1961), Econometrica, “Counterspeculation, Auctions, and Competitive Sealed Tenders,” pp. 8–37. Core second‑price mechanism and dominant-strategy truth-telling without budget constraints.
- Krishna (2009), Auction Theory (2nd ed.), Chapter 3 (Private Values), Sections 3.1–3.2. Baseline properties of second-price auctions; adapt arguments to budget caps.
- Milgrom and Weber (1982), Econometrica, “A Theory of Auctions and Competitive Bidding,” Sections 2–3. Comparative statics and equilibrium insights.
- Che and Gale (1998), Econometrica, “Standard Auctions with Financially Constrained Bidders.” Budget constraints in auctions; existence and allocation efficiency issues.

Model Notes (for report cross-reference)
- Values: v_i^t ~ Uniform[60,100].
- Per-round refill r in (0,60), added before t≥2.
- Feasible bids: b_i ≤ B_i^t.
- Second price rule; ties split 50–50 in expectation.
- Budget transition: B_i^{t+1} = min{B_max, B_i^t − payment_i^t + r}.

Files
- refs/references.bib — BibTeX stubs to cite in LaTeX.
- Add any PDFs (license permitting) to refs/papers/.

How to Use in LaTeX
- In report.tex, add: \bibliography{economist/refs/references}

