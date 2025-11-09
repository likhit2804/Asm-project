import pandas as pd

def fit_nhpp_from_data(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    counts = df.groupby('hour')['rain_mm_per_hr'].apply(lambda x: (x > 0.1).sum())
    mean_counts = counts.mean()
    lam_t = counts / counts.sum() * mean_counts * 24
    return lam_t
