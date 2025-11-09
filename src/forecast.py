import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def persistence_forecast(series, steps=24):
    last = float(series[-1]) if len(series)>0 else 0.0
    return np.repeat(last, steps)

def fit_arima(series, order=(2,0,2)):
    model = ARIMA(series, order=order)
    res = model.fit()
    return res

def forecast_arima(res_fit, steps=24):
    pred = res_fit.get_forecast(steps=steps)
    mean = pred.predicted_mean
    conf = pred.conf_int(alpha=0.05)
    lower = conf.iloc[:,0].values
    upper = conf.iloc[:,1].values
    return mean.values, lower, upper
