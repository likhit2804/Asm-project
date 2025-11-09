Urban Flood Forecast (v1)
-------------------------

This package extends the existing stochastic pipeline with a simple rainfall forecasting
module (ARIMA) and converts predicted rainfall into forecasted runoff using the calibrated
linear reservoir.

Usage:
    python -m src.main_forecast data/your_rainfile.csv

Outputs saved to results/: acf.png, evt_fit.png, rain_forecast.png, runoff_forecast.png, summary.txt

Requirements: numpy, pandas, matplotlib, scipy, statsmodels
