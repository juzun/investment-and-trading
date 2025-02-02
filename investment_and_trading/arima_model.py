import datetime as dt
import json
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import sklearn.metrics as metrics
import statsmodels.api as sm
import tensorflow as tf
import yfinance as yf
from keras.api.callbacks import EarlyStopping
from keras.api.layers import LSTM, Dense, Dropout, Input
from keras.api.models import Model, Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller, kpss

from investment_and_trading.types import TickerSymbol


def get_arima_coefficients(series: pd.Series) -> Tuple[int, int, int]:
    p, d, q = 0, 0, 0

    # Perform differencing testing and adjust d-parameter
    while not adf_test(series):
        series = series.pct_change()
        series = series.dropna()
        d += 1

    if not kpss_test(series):
        d += 1

    # Get parameters p and q from Autocorrelation and Partial Autocorrelation functions
    p = int(sm.tsa.pacf(series).round(0).sum())
    q = int(sm.tsa.acf(series).round(0).sum())
    return p, d, q


def adf_test(series: pd.Series) -> bool:
    """
    Check for stationarity using Augmented Dickey-Fuller test.

    Attributes:
        series: input data to be tested on stationarity.

    Returns:
        bool: True if data are stationary, False if not.
    """
    result = adfuller(series)
    # ADF Statistic: result[0], p-value: result[1]
    if result[1] <= 0.05:
        return True
    else:
        return False


def kpss_test(series: pd.Series) -> bool:
    result = kpss(series)
    if result[1] >= 0.05:
        return True
    else:
        return False


ticker = TickerSymbol.UPS
raw_data = fetch_stock_history(
    tickers=[ticker], start=dt.date(1950, 1, 1), end=dt.date.today()
)
data = preprocess_stock_history(stock_history=raw_data)
if data.index.year.nunique() < 20:
    raise Warning("Data length is less than 20 years. Predictions might be inaccurate.")


arima_coefs = get_arima_coefficients(series=data["Adj Close"])
arima_model = ARIMA(data["Adj Close"], order=arima_coefs)
arima_result: ARIMAResults = arima_model.fit()


n_steps = 30

forecast = arima_result.forecast(steps=n_steps)
forecast_dates = pd.date_range(start=data.index[-1], periods=n_steps, freq="B")
forecast_series = pd.Series(forecast, index=forecast_dates)
forecast_series.iloc[0] = data["Adj Close"].iloc[-1]


n_repetitions = 20
simulations = arima_result.simulate(
    nsimulations=n_steps,
    anchor=data["Adj Close"].index[-1],
    repetitions=n_repetitions,
)
simulations.iloc[0] = data["Adj Close"].iloc[-1]


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data.index, y=data["Adj Close"], mode="lines", name="Adj Close History"
    )
)
fig.add_trace(
    go.Scatter(
        x=data.index, y=arima_result.fittedvalues, mode="lines", name="Fitted Values"
    )
)
fig.add_trace(
    go.Scatter(
        x=forecast_series.index, y=forecast_series, mode="lines", name="Prediction"
    )
)
for i, sim in enumerate(simulations):
    fig.add_trace(
        go.Scatter(
            x=forecast_series.index,
            y=simulations[sim],
            mode="lines",
            name=f"Simulation {i+1}",
        )
    )
fig.update_layout(
    title=f"ARIMA{arima_coefs} model",
    xaxis_title="Date",
    yaxis_title="Value",
    template="plotly_dark",
)
fig.show()
