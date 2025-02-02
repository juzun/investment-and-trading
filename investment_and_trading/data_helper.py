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


class DataHelper:

    def fetch_stock_history(
        tickers: List[TickerSymbol],
        start_date: dt.date = dt.date(2010, 1, 1),
        end_date=dt.date(2025, 1, 1),
    ) -> pd.DataFrame:
        tickers = [ticker.name for ticker in tickers]
        stock_history = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            actions=True,
            multi_level_index=False,
        )
        return stock_history

    def preprocess_stock_history(stock_history: pd.DataFrame) -> pd.DataFrame:
        stock_history = stock_history[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "Dividends",
                "Stock Splits",
            ]
        ]
        stock_history = stock_history.dropna()

        # Calculate returns
        stock_history["Returns"] = stock_history["Adj Close"].pct_change()

        # Moving Averages
        stock_history["MA_10"] = stock_history["Adj Close"].rolling(window=10).mean()
        stock_history["MA_50"] = stock_history["Adj Close"].rolling(window=50).mean()

        # Volatility - Rolling standard deviation of returns (10-day window)
        stock_history["Volatility_10"] = (
            stock_history["Returns"].rolling(window=10).std()
        )

        # RSI Calculation (Relative Strength Index, 14-day window)
        delta = stock_history["Adj Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        stock_history["RSI_14"] = 100 - (100 / (1 + rs))

        # Lagged Returns (1-day, 5-day, 10-day)
        stock_history["Lag_1"] = stock_history["Returns"].shift(1)
        stock_history["Lag_5"] = stock_history["Returns"].shift(5)
        stock_history["Lag_10"] = stock_history["Returns"].shift(10)

        # Drop missing values due to rolling window calculations
        stock_history = stock_history.dropna()

        # Set data frequency to business days and interpolate newly added nan values.
        stock_history = stock_history.asfreq("B").interpolate()

        # One-hot encoding for day of the week
        stock_history["DayOfWeek"] = stock_history.index.dayofweek
        stock_history = pd.get_dummies(
            stock_history, columns=["DayOfWeek"], drop_first=True
        )

        return stock_history
