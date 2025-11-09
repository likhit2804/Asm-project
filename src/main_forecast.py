import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import eda, evt, hydrology, forecast
import time

def load_input(path):
    df = pd.read_csv(path, low_memory=False)
    ts_col = None
    for c in ['timestamp','date_time','datetime','time']:
        if c in df.columns:
            ts_col = c; break
    if ts_col is None:
        raise ValueError('No timestamp column found')
    df['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=['timestamp'])
    rain_col = None
    for c in ['precipMM','rain_mm_per_hr','precip_mm','rain']:
        if c in df.columns:
            rain_col = c; break
    if rain_col is None:
        raise ValueError('No rainfall column found')
    df['rain_mm_per_hr'] = pd.to_numeric(df[rain_col], errors='coerce').fillna(0.0)
    df = df[['timestamp','rain_mm_per_hr']].reset_index(drop=True)
    return df

def run_all(input_csv, forecast_steps=24):
    os.makedirs('results', exist_ok=True)
    t0 = time.time()
    df = load_input(input_csv)
    print('Loaded', len(df),'rows.')

    # EDA
    eda.acf_plot(df['rain_mm_per_hr'].values, lags=24, outpath='results/acf.png')

    # EVT
    u = np.percentile(df['rain_mm_per_hr'].values, 95)
    excesses = df['rain_mm_per_hr'].values[df['rain_mm_per_hr'].values > u] - u
    if len(excesses)>0:
        gpd = evt.fit_gpd(excesses, outpath='results/evt_fit.png')
    else:
        gpd = {'shape':None,'scale':None}

    # Hydrology calibration (simple synthetic obs_q from K=4 then fit by grid)
    K_true = 4.0
    q_true, _ = hydrology.linear_reservoir_runoff(df['rain_mm_per_hr'].values, K_true)
    obs_q = q_true + np.random.default_rng(0).normal(0, 0.05, size=q_true.shape)
    Ks = np.linspace(0.5,10,20)
    bestK = Ks[0]; best_rmse=1e9
    for K in Ks:
        qsim,_ = hydrology.linear_reservoir_runoff(df['rain_mm_per_hr'].values, K)
        rmse = np.sqrt(np.mean((obs_q - qsim)**2))
        if rmse < best_rmse:
            best_rmse = rmse; bestK = K
    with open('results/summary.txt','w') as f:
        f.write(f'GPD: {gpd}\nCalibrated K: {bestK}, RMSE: {best_rmse}\n')

    # Forecasting rainfall (ARIMA)
    series = df['rain_mm_per_hr']
    try:
        arima_res = forecast.fit_arima(series, order=(2,0,2))
        mean_pred, lower, upper = forecast.forecast_arima(arima_res, steps=forecast_steps)
        method='ARIMA'
    except Exception as e:
        print('ARIMA failed:',e)
        mean_pred = forecast.persistence_forecast(series, steps=forecast_steps)
        lower = mean_pred*0.8; upper = mean_pred*1.2
        method='Persistence'

    # save rainfall forecast plot
    times = pd.date_range(start=df['timestamp'].iloc[-1], periods=forecast_steps+1, freq='H')[1:]
    plt.figure(figsize=(8,3))
    plt.plot(times, mean_pred, label='Predicted rain (mm/h)')
    plt.fill_between(times, lower, upper, color='gray', alpha=0.3, label='95% CI')
    plt.title(f'Rainfall Forecast ({method})'); plt.legend(); plt.tight_layout()
    plt.savefig('results/rain_forecast.png'); plt.close()

    # Convert rainfall forecast to runoff forecast using calibrated K
    q_forecast, _ = hydrology.linear_reservoir_runoff(mean_pred, bestK)
    plt.figure(figsize=(8,3))
    plt.plot(times, q_forecast, label='Predicted runoff (mm/h)')
    plt.title('Forecasted Runoff Hydrograph'); plt.legend(); plt.tight_layout()
    plt.savefig('results/runoff_forecast.png'); plt.close()

    print('Done. Results saved in results/. Time (s):', time.time()-t0)

if __name__ == '__main__':
    import sys
    inp = 'data/example_kaggle_rainfall.csv' if len(sys.argv)<2 else sys.argv[1]
    run_all(inp, forecast_steps=48)
