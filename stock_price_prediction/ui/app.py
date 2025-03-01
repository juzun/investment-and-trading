import datetime as dt

import streamlit as st

from stock_price_prediction.stock_model import StockModel
from stock_price_prediction.types import TickerSymbol

# Streamlit UI Setup
st.set_page_config(page_title="Stock Price Predictor", layout="wide")


st.title("📈 Stock Price Prediction Dashboard")

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Settings")

    # Select ticker
    selected_ticker_name = st.selectbox(
        "Select a Ticker:", [ticker.name for ticker in TickerSymbol], index=0
    )
    selected_ticker = TickerSymbol[selected_ticker_name]

    # Number of days to predict
    prediction_days = st.slider(
        "Prediction Days Ahead", min_value=1, max_value=30, value=28
    )

    # Button to trigger computations
    compute_button = st.button("📊 Show Predictions")

if "stock_model_prepared" not in st.session_state:
    st.session_state["stock_model_prepared"] = False

if compute_button:
    try:
        start_date = dt.date(1900, 1, 1)
        end_date = dt.date.today()
        stock_model = StockModel(ticker=selected_ticker)
        stock_model.prepare_data(start_date=start_date, end_date=end_date)
        stock_model.load_lstm_model()
        stock_model.initialize_arima_model()
        stock_model.arima_model.fit_arima_model()

        st.session_state["pred_sim_plot"] = stock_model.generate_plot(
            prediction_days_ahead=prediction_days
        )
        st.session_state["selected_ticker"] = selected_ticker
        st.session_state["stock_model_prepared"] = True
        st.success("✅ Predictions successfully calculated!.")

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.session_state["stock_model_prepared"] = False

if st.session_state["stock_model_prepared"]:
    st.subheader(
        f"{st.session_state['selected_ticker'].name} Price Prediction and Simulations"
    )
    st.plotly_chart(st.session_state["pred_sim_plot"], use_container_width=True)
else:
    st.info("🔄 Please click the 'Show Predictions' button to generate the results.")
