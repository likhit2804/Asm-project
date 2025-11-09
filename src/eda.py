import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

def run(df):
    df = df.copy()
    counts_per_day = df.groupby(df['timestamp'].dt.date).apply(lambda g: (g['rain_mm_per_hr']>0.1).sum())
    mean, var = counts_per_day.mean(), counts_per_day.var()
    vmr = var/mean if mean > 0 else np.nan
    vals = acf(df['rain_mm_per_hr'].fillna(0).values, nlags=24)
    plt.figure(figsize=(8,4))
    plt.plot(range(len(vals)), vals, marker='o')
    plt.title("Rainfall ACF (24-hour lag)")
    plt.xlabel("Lag (hours)")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    plt.savefig("results/acf.png")
    plt.close()
    return {"vmr": vmr}
