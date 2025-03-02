import datetime as dt

import plotly.graph_objects as go
import structlog
import typer

from stock_price_prediction.stock_model import StockModel
from stock_price_prediction.types import TickerSymbol

app = typer.Typer()


@app.command()
def predict_stock(
    ticker: str =typer.Option(  # noqa B008
        help="Ticker symbol of stock to predict (KO, UPS, ADBE...).",
    )
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

    log.info(f"Initializing stock model for {ticker}, {valid_ticker}...")

    # Initialize stock model
    stock_model = StockModel(ticker=valid_ticker)

    # Prepare data
    log.info("Fetching and preprocessing stock data...")
    stock_model.prepare_data(start_date=dt.date(1900, 1, 1), end_date=dt.date.today())

    # Initialize and train LSTM model
    log.info("Initializing and training LSTM model...")
    stock_model.initialize_lstm_model()
    stock_model.lstm_model.train_model()

    # Save trained model
    log.info("Savinging the trained LSTM model...")
    custom_filename = f"{ticker}_{dt.datetime.now().strftime('%m%d%H%M%S')}"
    stock_model.save_lstm_model(custom_filename=custom_filename)
    log.info(
        f"Model saved as {f'{custom_filename}_model.keras'} with its logs as {f'{custom_filename}_log.json'}"
    )

    # Generate prediction plot and save it as HTML
    log.info("Generating LSTM prediction plot...")
    fig: go.Figure = stock_model.generate_lstm_prediction_plot(prediction_days_ahead=28)
    html_output_file = f"{ticker}_lstm_prediction_plot.html"
    fig.write_html(html_output_file)
    log.info(html_output_file)


if __name__ == "__main__":
    app()

stock_model = StockModel(ticker=TickerSymbol.KO)
stock_model.prepare_data(start_date=dt.date(1900, 1, 1), end_date=dt.date.today())
stock_model.initialize_lstm_model()
stock_model.lstm_model.train_model()
stock_model.generate_lstm_prediction_plot(prediction_days_ahead=28)
