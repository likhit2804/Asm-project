import numpy as np

def linear_reservoir_runoff(rain_series, K, dt_hours=1.0, S0=0.0):
    r = np.asarray(rain_series, dtype=float)
    n = len(r)
    S = np.zeros(n+1)
    q = np.zeros(n)
    S[0] = S0
    for t in range(n):
        alpha = np.exp(-dt_hours / K)
        S[t+1] = S[t] * alpha + K * (1 - alpha) * r[t]
        q[t] = S[t+1] / K
    return q, S[1:]
