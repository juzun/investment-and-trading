import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller, kpss

from investment_and_trading.data_helper import DataHelper
from investment_and_trading.lstm_model import LSTMModel, Sequential
from investment_and_trading.types import TickerSymbol


class StockModel:

    def __init__(self, ticker: TickerSymbol) -> None:
        self.ticker = ticker
        self.data_helper = DataHelper

    def prepare_data(self, start_date: dt.date, end_date: dt.date) -> None:
        raw_data = self.data_helper.fetch_stock_history(
            tickers=[self.ticker], start_date=start_date, end_date=end_date
        )
        self.data: pd.DataFrame = self.data_helper.preprocess_stock_history(
            stock_history=raw_data
        )

    def generate_lstm_model(
        self, features: Optional[List[str]], target_feature: Optional[str]
    ) -> None:
        if features is None:
            features = self.data.columns
        if target_feature is None:
            target_feature = "Adj Close"
        self.lstm_model = LSTMModel(features=features, target_feature=target_feature)

    def __save_trained_lstm_model(self) -> None:
        self.lstm_model.save_trained_lstm_model(ticker_name=self.ticker.name)

    def load_lstm_model(self) -> None:
        self.lstm_model = LSTMModel.load_trained_model()

    def generate_lstm_prediction_plot(
        self, data: pd.DataFrame, prediction_days_ahead: int = 7
    ) -> go.Figure:

        predictions = self.lstm_model.make_predictions(days_ahead=prediction_days_ahead)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Adj Close"],
                mode="lines",
                name="Adj Close History",
            )
        )
        for days, preds in predictions.items():
            future_dates = pd.date_range(
                start=data.index[-1], periods=days + 1, freq="B"
            )
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=np.insert(preds, 0, data["Adj Close"].iloc[-1]),
                    mode="lines",
                    name=f"Forecast {days}d",
                )
            )

        fig.update_layout(
            title=f"{self.ticker.name} price prediction",
            xaxis_title="Date",
            yaxis_title="Adj Close Price",
            template="plotly_dark",
        )
        return fig
