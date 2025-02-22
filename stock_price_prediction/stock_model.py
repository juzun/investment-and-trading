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

from stock_price_prediction.data_helper import DataHelper
from stock_price_prediction.lstm_model import LSTMModel
from stock_price_prediction.types import TickerSymbol


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
        self, features: Optional[List[str]] = None, target_feature: Optional[str] = None, seq_length: int = 30
    ) -> None:
        if features is None:
            features = list(self.data.columns)
        if target_feature is None:
            target_feature = "Adj Close"
        self.lstm_model = LSTMModel(data=self.data, features=features, target_feature=target_feature, seq_length=seq_length)

    def save_lstm_model(self) -> None:
        self.lstm_model._save_trained_lstm_model(ticker_name=self.ticker.name)

    def load_lstm_model(self) -> None:
        self.lstm_model = LSTMModel._load_trained_model(data=self.data, ticker_name=self.ticker.name)

    def generate_lstm_prediction_plot(
        self, prediction_days_ahead: int = 7
    ) -> go.Figure:

        predictions = self.lstm_model.make_predictions(days_ahead=prediction_days_ahead)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data["Adj Close"],
                mode="lines",
                name="Adj Close History",
            )
        )

        future_dates = pd.date_range(
            start=self.data.index[-1], periods=prediction_days_ahead + 1, freq="B"
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                mode="lines",
                name=f"Predict {prediction_days_ahead}d",
            )
        )

        fig.update_layout(
            title=f"{self.ticker.name} Adjusted Close Price Prediction",
            xaxis_title="Date",
            yaxis_title="Adj Close Price",
            template="plotly_dark",
        )
        return fig
