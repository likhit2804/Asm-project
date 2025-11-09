import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto

def run(rain_series):
    rain = np.array(rain_series)
    threshold = np.percentile(rain[rain>0], 95)
    excesses = rain[rain > threshold] - threshold
    shape, loc, scale = genpareto.fit(excesses)
    x = np.linspace(0, np.max(excesses), 200)
    y = genpareto.pdf(x, shape, loc=0, scale=scale)
    plt.figure(figsize=(8,4))
    plt.hist(excesses, bins=30, density=True, alpha=0.6, label='Empirical')
    plt.plot(x, y, 'r-', label='Fitted GPD')
    plt.title("Extreme Value Fit (POT)")
    plt.xlabel("Excess above threshold (mm)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/evt_return_levels.png")
    plt.close()
    with open("results/summary.txt","w") as f:
        f.write(f"GPD shape={shape:.3f}, scale={scale:.3f}, threshold={threshold:.3f}\n")
    return {"shape": shape, "scale": scale, "threshold": threshold}
