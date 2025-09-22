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
