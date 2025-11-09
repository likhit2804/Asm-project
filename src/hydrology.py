import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def linear_reservoir_runoff(rain, K):
    rain = np.array(rain)
    S = np.zeros_like(rain)
    for t in range(1, len(rain)):
        S[t] = S[t-1]*np.exp(-1/K) + K*(1-np.exp(-1/K))*rain[t]
    q = S / K
    return q, S

def run(rain):
    def rmse(K):
        qsim, _ = linear_reservoir_runoff(rain, K)
        return np.sqrt(np.mean((rain - qsim)**2))
    res = minimize(rmse, x0=[3.0], bounds=[(0.5,10)])
    K_hat = res.x[0]
    qsim, _ = linear_reservoir_runoff(rain, K_hat)
    plt.figure(figsize=(8,4))
    plt.plot(rain[:200], label='Rain input', alpha=0.6)
    plt.plot(qsim[:200], label='Runoff (sim)', alpha=0.8)
    plt.legend(); plt.title("Linear Reservoir Calibration")
    plt.xlabel("Time step"); plt.ylabel("mm/hr")
    plt.tight_layout(); plt.savefig("results/hydro_fit.png")
    plt.close()
    return {"K_hat": K_hat}
