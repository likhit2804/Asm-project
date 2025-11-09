import numpy as np
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt

def acf_plot(series, lags=48, outpath=None):
    vals = acf(series, nlags=lags, fft=True, missing='drop')
    plt.figure(figsize=(8,3))
    plt.plot(range(len(vals)), vals, marker='o')
    plt.xlabel('Lag (hours)'); plt.ylabel('ACF')
    if outpath: plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    return vals
