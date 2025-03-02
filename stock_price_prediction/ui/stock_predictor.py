import datetime as dt

import streamlit as st

from stock_price_prediction.stock_model import StockModel
from stock_price_prediction.types import TickerSymbol


class StockPredictor:
    """
    A class to handle the preparation, training, and prediction of stock prices
    using LSTM and ARIMA models for a given stock ticker symbol.

    Attributes:
        ticker (TickerSymbol): The selected stock ticker.
        stock_model (StockModel): The StockModel instance for performing stock price prediction.
        prediction_plot (Plotly figure or None): The generated prediction plot (if available).
    """

    def __init__(self, ticker: TickerSymbol) -> None:
        """
        Initializes the StockPredictor with a specific ticker symbol.

        Args:
            ticker (TickerSymbol): The ticker symbol of the stock for prediction.
        """
        self.ticker = ticker
        self.stock_model = StockModel(ticker=ticker)
        self.prediction_plot = None

    def prepare_and_predict(self, prediction_days: int) -> bool:
        """
        Prepares the data, trains the models (LSTM and ARIMA), and generates the stock price prediction plot.

        This method fetches historical stock data, initializes the models, and generates a prediction
        for the specified number of days ahead.

        Args:
            prediction_days (int): The number of days ahead for which predictions should be generated.

        Returns:
            bool: Returns True if the prediction was successful, False otherwise.
        """
        try:
            start_date = dt.date(1900, 1, 1)
            end_date = dt.date.today()
            self.stock_model.prepare_data(start_date=start_date, end_date=end_date)
            self.stock_model.load_lstm_model()
            self.stock_model.initialize_arima_model()
            self.stock_model.arima_model.fit_arima_model()

            self.prediction_plot = self.stock_model.generate_plot(
                prediction_days_ahead=prediction_days
            )
            return True  # Success
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            return False  # Failure
