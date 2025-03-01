import datetime as dt
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock_price_prediction.arima_model import ARIMAModel
from stock_price_prediction.data_helper import DataHelper
from stock_price_prediction.lstm_model import LSTMModel
from stock_price_prediction.types import TickerSymbol


class StockModel:

    def __init__(self, ticker: TickerSymbol) -> None:
        self.ticker = ticker

    def prepare_data(self, start_date: dt.date, end_date: dt.date) -> None:
        raw_data = DataHelper.fetch_stock_history(
            tickers=[self.ticker], start_date=start_date, end_date=end_date
        )
        self.data: pd.DataFrame = DataHelper.preprocess_stock_history(
            stock_history=raw_data
        )

    def initialize_lstm_model(
        self,
        features: Optional[List[str]] = None,
        target_feature: Optional[str] = None,
        seq_length: int = 30,
    ) -> None:
        if features is None:
            features = list(self.data.columns)
        if target_feature is None:
            target_feature = "Adj Close"
        self.lstm_model = LSTMModel(
            data=self.data,
            features=features,
            target_feature=target_feature,
            seq_length=seq_length,
        )

    def initialize_arima_model(self, target_feature: Optional[str] = None) -> None:
        if target_feature is None:
            target_feature = "Adj Close"
        self.arima_model = ARIMAModel(data=self.data, target_feature=target_feature)

    def save_lstm_model(self) -> None:
        self.lstm_model._save_trained_lstm_model(ticker_name=self.ticker.name)

    def load_lstm_model(self) -> None:
        self.lstm_model = LSTMModel._load_trained_model(
            data=self.data, ticker_name=self.ticker.name
        )

    def generate_lstm_prediction_plot(
        self, prediction_days_ahead: int = 7
    ) -> go.Figure:
        predictions = self.lstm_model.make_predictions(days_ahead=prediction_days_ahead)
        future_dates = pd.date_range(
            start=self.data.index[-1], periods=prediction_days_ahead + 1, freq="B"
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data["Adj Close"],
                mode="lines",
                name="Adj Close History",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                mode="lines",
                name=f"{prediction_days_ahead} days prediction",
            )
        )
        fig.update_layout(
            title=f"{self.ticker.name} Adjusted Close Price Prediction",
            xaxis_title="Date",
            yaxis_title="Adj Close Price",
            template="plotly_dark",
        )
        return fig

    def generate_arima_simulations_plot(
        self, prediction_days_ahead: int = 7, number_of_simulations: int = 10
    ) -> go.Figure:
        simulations = self.arima_model.make_simulations(
            days_ahead=prediction_days_ahead, n_simulations=number_of_simulations
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data["Adj Close"],
                mode="lines",
                name="Adj Close History",
            )
        )
        for i, sim in enumerate(simulations):
            fig.add_trace(
                go.Scatter(
                    x=simulations.index,
                    y=simulations[sim],
                    mode="lines",
                    name=f"Simulation {i+1}",
                )
            )
        fig.update_layout(
            title=f"{self.ticker.name} Adjusted Close Price simulations",
            xaxis_title="Date",
            yaxis_title="Adj Close Price",
            template="plotly_dark",
        )
        return fig

    def generate_plot(
        self, prediction_days_ahead: int = 7, number_of_simulations: int = 10
    ) -> go.Figure:
        predictions = self.lstm_model.make_predictions(days_ahead=prediction_days_ahead)
        future_dates = pd.date_range(
            start=self.data.index[-1], periods=prediction_days_ahead + 1, freq="B"
        )
        simulations = self.arima_model.make_simulations(
            days_ahead=prediction_days_ahead, n_simulations=number_of_simulations
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.5, 0.5],
            subplot_titles=["LSTM Prediction", "ARIMA Simulations"],
        )

        # LSTM plot
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data["Adj Close"],
                mode="lines",
                name="Adj Close History",
                line={"color": "#636EFA"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                mode="lines",
                name=f"{prediction_days_ahead} days prediction",
                line={"color": "#ff7f0e"},
            ),
            row=1,
            col=1,
        )

        # ARIMA plot
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data["Adj Close"],
                mode="lines",
                name="Adj Close History",
                line={"color": "#636EFA"},
            ),
            row=2,
            col=1,
        )
        colors = [
            "#e41a1c",
            "#6a3d9a",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#ffcc00",
            "#a65628",
            "#f781bf",
            "#999999",
            "#17becf",
        ]
        for i, sim in enumerate(simulations):
            fig.add_trace(
                go.Scatter(
                    x=simulations.index,
                    y=simulations[sim],
                    mode="lines",
                    name=f"Simulation {i+1}",
                    line={"color": colors[i]},
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            xaxis2=dict(title="Date"),
            yaxis=dict(title="Adj Close Price"),
            yaxis2=dict(title="Adj Close Price"),
            height=700,
        )
        return fig
