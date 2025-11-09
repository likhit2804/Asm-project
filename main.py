import os
import pandas as pd
import numpy as np
import time
from src import eda, evt, hydrology, arrival_models

def ensure_dirs():
    os.makedirs("results", exist_ok=True)

def load_input(path):
    import pandas as pd
    print(f"Reading dataset from: {path}")

    # safer read
    df = pd.read_csv(path, low_memory=False)

    # --- detect and fix timestamp column ---
    ts_col = None
    for c in ['timestamp', 'date_time', 'datetime', 'time']:
        if c in df.columns:
            ts_col = c
            break
    if not ts_col:
        raise ValueError("No timestamp/date_time column found")

    # parse datetimes robustly (handles mixed formats)
    df['timestamp'] = pd.to_datetime(
        df[ts_col], errors='coerce', infer_datetime_format=True
    )
    df = df.dropna(subset=['timestamp'])

    # --- detect rainfall column ---
    rain_col = None
    for c in ['precipMM', 'rain_mm_per_hr', 'rain', 'precipitation']:
        if c in df.columns:
            rain_col = c
            break
    if not rain_col:
        raise ValueError("No rainfall/precip column found")

    # convert rainfall column to numeric
    df['rain_mm_per_hr'] = pd.to_numeric(df[rain_col], errors='coerce').fillna(0.0)

    # --- keep only relevant columns ---
    df = df[['timestamp', 'rain_mm_per_hr']]

    print(f"✅ Cleaned: {len(df)} rows from {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df

def main(input_csv='data/example_kaggle_rainfall.csv'):
    ensure_dirs()
    t0 = time.time()

    df = load_input(input_csv)
    print(f"Loaded dataset: {len(df)} rows, columns: {list(df.columns)}")

    t1 = time.time()
    eda_res = eda.run(df)
    print(f"EDA complete. VMR: {eda_res['vmr']:.3f} ({time.time()-t1:.2f}s)")

    t2 = time.time()
    lam_t = arrival_models.fit_nhpp_from_data(df)
    lam_t.to_csv('results/lambda_hourly.csv')
    print(f"NHPP fitted to observed data. ({time.time()-t2:.2f}s)")

    t3 = time.time()
    evt_res = evt.run(df['rain_mm_per_hr'].values)
    print(f"EVT complete. shape={evt_res['shape']:.3f}, scale={evt_res['scale']:.3f} ({time.time()-t3:.2f}s)")

    t4 = time.time()
    hydro_res = hydrology.run(df['rain_mm_per_hr'].values)
    print(f"Hydrology calibration complete. K={hydro_res['K_hat']:.3f} ({time.time()-t4:.2f}s)")

    print(f"All phases complete in {time.time()-t0:.1f}s. Results in /results")

if __name__ == "__main__":
    main()
