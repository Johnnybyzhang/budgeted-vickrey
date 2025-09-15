Computational Scientist Track — How to Run and Artifacts

Goal
- Provide a reproducible pipeline to generate stage-game tables (optional), dynamic heuristic policy, fast simulations, and figures.

Zero-Compute Note
- Do not execute heavy compute in this folder on device. Use the root commands (uv) or run in Colab.

Colab Notebook
- Open notebook.ipynb in Google Colab.
- Runtime → Change runtime type → Python 3.x.
- Follow the “Setup” cell to install uv and sync dependencies inside Colab.

Local Quick Run (reference)
- See repository root README for uv commands. Main outputs land in results/ and figures/.

GTE Screenshots
- Place exports (PNGs) under gte/ with descriptive captions in this README:
  - stage-equilibria-supports.png — “Supports from NashPy (grid step=5)”
  - threshold-policy-table.png — “Threshold policy by budget grid”

Files
- notebook.ipynb — Colab-exportable walkthrough (no heavy compute by default).
- gte/ — optional screenshots/exports.

