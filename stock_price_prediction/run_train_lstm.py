import datetime as dt

import plotly.graph_objects as go
import structlog
import typer

from stock_price_prediction.stock_model import StockModel
from stock_price_prediction.types import TickerSymbol

app = typer.Typer()


@app.command()
def predict_stock(
    ticker: str = typer.Option(  # noqa B008
        help="Ticker symbol of stock to predict (KO, UPS, ADBE...).",
    ),
    prediction_days_ahead: int = typer.Option(  # noqa B008
        default=28,
        help="Number of days ahead to predict. Defaults to 28.",
    ),
) -> None:
    """
    Fetch stock data, train an LSTM model, and store it with the logs.

    Args:
        ticker (str): The stock ticker symbol (e.g., "KO" for Coca-Cola).
    """

    log = structlog.get_logger()

    try:
        valid_ticker = TickerSymbol[ticker]
    except KeyError:
        log.info(
            f"Error: '{ticker}' is not a valid ticker. Choose from: {', '.join(TickerSymbol.__members__.keys())}"
        )
        raise typer.Exit(code=1)

    log.info(f"Initializing stock model for {ticker}, {valid_ticker.value}...")

    # Initialize stock model
    stock_model = StockModel(ticker=valid_ticker)

    # Prepare data
    log.info("Fetching and preprocessing stock data...")
    stock_model.prepare_data(start_date=dt.date(1900, 1, 1), end_date=dt.date.today())

    # Initialize and train LSTM model
    log.info("Initializing and training LSTM model...")
    stock_model.initialize_lstm_model()
    stock_model.lstm_model.train_model()

    # Save trained LSTM model
    log.info("Savinging the trained LSTM model...")
    custom_filename = f"{ticker}_{dt.datetime.now().strftime('%m%d%H%M%S')}"
    stock_model.save_lstm_model(custom_filename=custom_filename)
    log.info(
        f"LSTM model saved in {f'{custom_filename}_model.keras'} with its logs in {f'{custom_filename}_log.json.'}"
    )

    # Initialize and fit ARIMA model
    log.info("Initializing and fitting ARIMA model...")
    stock_model.initialize_arima_model()
    stock_model.arima_model.fit_arima_model()

    # Generate prediction plot and save it as HTML
    log.info("Generating prediction and simulations plot...")
    fig: go.Figure = stock_model.generate_plot(
        prediction_days_ahead=prediction_days_ahead
    )
    html_output_file = f"{ticker}_pred_sims_plot.html"
    fig.write_html(html_output_file)
    log.info(f"Prediction and simulations plot stored in {html_output_file}.")


if __name__ == "__main__":
    app()
