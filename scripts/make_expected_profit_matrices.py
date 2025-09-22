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
