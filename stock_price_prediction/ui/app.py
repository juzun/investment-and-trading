from typing import Optional

import streamlit as st

from stock_price_prediction.types import TickerSymbol
from stock_price_prediction.ui.stock_predictor import StockPredictor


class StockPredictionApp:
    """
    A class representing the Stock Prediction Streamlit application.
    It handles the user interface, including selecting a ticker, setting prediction days,
    and displaying the stock price predictions.

    Attributes:
        predictor (Optional[StockPredictor]): An instance of StockPredictor used to prepare data and generate predictions.
        selected_ticker_name (str): The name of the selected ticker symbol.
        selected_ticker (TickerSymbol): The selected ticker symbol as an enum from TickerSymbol.
        prediction_days (int): The number of days ahead for stock price predictions.
        compute_button (bool): Boolean indicating whether the compute button was pressed.
    """

    def __init__(self) -> None:
        """
        Initializes the Streamlit page configuration, sets up the sidebar with options for the user,
        and initializes the session state.
        """
        st.set_page_config(page_title="Stock Price Predictor", layout="wide")
        self.predictor: Optional[StockPredictor] = None
        self.initialize_sidebar()
        self.initialize_session_state()

    def initialize_sidebar(self) -> None:
        """
        Initializes the sidebar where the user can select the stock ticker, set the number of prediction days,
        and press a button to compute the prediction.
        """
        with st.sidebar:
            st.header("⚙️ Settings")

            # Dropdown to select the stock ticker
            self.selected_ticker_name = st.selectbox(
                "Select a Ticker:", [ticker.name for ticker in TickerSymbol], index=0
            )
            self.selected_ticker = TickerSymbol[self.selected_ticker_name]

            # Slider to select prediction days ahead
            self.prediction_days = st.slider(
                "Prediction Days Ahead", min_value=1, max_value=30, value=28
            )

            # Button to trigger prediction computation
            self.compute_button = st.button("📊 Show Predictions")

    def initialize_session_state(self) -> None:
        """
        Initializes the session state to track whether the stock model is prepared and results are available.
        """
        if "stock_model_prepared" not in st.session_state:
            st.session_state["stock_model_prepared"] = False

    def run(self) -> None:
        """
        Runs the main logic of the application: displaying the title, handling user interactions,
        preparing and predicting stock prices, and displaying the results.
        """
        st.title("📈 Stock Price Prediction Dashboard")

        # When the 'Show Predictions' button is clicked, prepare data and generate predictions
        if self.compute_button:
            self.predictor = StockPredictor(ticker=self.selected_ticker)
            success = self.predictor.prepare_and_predict(
                prediction_days=self.prediction_days
            )

            # If prediction was successful, save the results in session state
            if success:
                st.session_state["pred_sim_plot"] = self.predictor.prediction_plot
                st.session_state["selected_ticker"] = self.selected_ticker
                st.session_state["stock_model_prepared"] = True
                st.success("✅ Predictions successfully calculated!")

        # Display the results after prediction computation
        self.display_results()

    def display_results(self) -> None:
        """
        Displays the results of the stock price prediction, including the generated plot,
        if the model has been prepared and predictions are available.
        """
        if st.session_state["stock_model_prepared"]:
            st.subheader(
                f"{st.session_state['selected_ticker'].name} Price Prediction and Simulations"
            )
            st.plotly_chart(st.session_state["pred_sim_plot"], use_container_width=True)
        else:
            st.info(
                "🔄 Please click the 'Show Predictions' button to generate the results."
            )


if __name__ == "__main__":
    app = StockPredictionApp()
    app.run()
