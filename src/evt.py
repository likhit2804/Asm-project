import numpy as np
from scipy.stats import genpareto
import matplotlib.pyplot as plt

def fit_gpd(excesses, outpath=None):
    params = genpareto.fit(excesses, floc=0)
    shape, loc, scale = params[0], params[1], params[2]
    if outpath:
        x = np.linspace(0, max(excesses)*1.1, 200)
        plt.figure(figsize=(8,3))
        plt.hist(excesses, bins=30, density=True, alpha=0.6, label='Empirical')
        plt.plot(x, genpareto.pdf(x, shape, loc=0, scale=scale), 'r-', label='GPD')
        plt.xlabel('Excess above threshold (mm)'); plt.ylabel('Density'); plt.legend()
        plt.tight_layout(); plt.savefig(outpath); plt.close()
    return {'shape':float(shape),'scale':float(scale)}
