# Stock Price Prediction

This project provides a stock price prediction and simulation tool using two models: LSTM (Long Short-Term Memory) for deep learning-based price predictions and ARIMA (Autoregressive Integrated Moving Average) for time-series simulations. It is designed to facilitate stock price analysis using both models, offering a user-friendly dashboard.

## 📖 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Running the Application](#running-the-application)
  - [Streamlit UI](#streamlit-ui)
  - [CLI (Typer) Interface](#cli-typer-interface)
- [Models](#models)
  - [ARIMA Model](#arima-model)
  - [LSTM Model](#lstm-model)
- [Data](#data)

## Introduction

This application allows users to see prediction and future simulations of stock price movements for various tickers using models LSTM and ARIMA.

The predictions and simulations can be either viewed in an interactive UI built using **Streamlit** or directly generated in CLI using **Typer**.


## Installation
To get started with this project, you need to install the necessary dependencies. Poetry was used for dependency management in this project. Follow these steps to set up the environment:

#### Clone the Repository
```bash
git clone https://github.com/juzun/stock-price-prediction.git
```

#### Install Dependencies
First, install Poetry (if not installed already). Then run:
```bash
poetry install
```
This will install all the required dependencies listed in `pyproject.toml` file.

### Pre-Trained Models
Ensure that the pre-trained models for each ticker are available in the `/data/models` folder. The models are already trained and placed in this directory, ready for use in the application's UI.


## Running the Application
There are two ways to run the application:
- using the **Streamlit UI** for a graphical dashboard,
- or using the **CLI (Typer)** to train and generate prediction and simulations.

### Streamlit UI
To launch the **Streamlit UI** (the web-based dashboard), activate the virtual environment and run the following command:
```bash
streamlit run stock_price_prediction/ui/app.py
```
This will open a web page in your browser where you can select the stock ticker, set the prediction days, and view the results.

### CLI (Typer) Interface
You can also run the prediction process directly from the command line using **Typer**. To do that, run the following command:
```bash
typer stock_price_prediction/run_train_lstm.py run --ticker UPS --prediction-days-ahead 28
```
Replace UPS with the desired ticker symbol, and 28 with the number of prediction days.


## Models
### ARIMA Model
The ARIMA model is a time-series model that predicts future values based on the trend, seasonality, and noise in the data. This model is fitted on historical adjusted close price data and is used for simulating future stock price movements.

The order of ARIMA differs for each ticker. Several methods are combined to find an optimal order `(p, d, q)`. Numbers `p` and `q` are deduced from autocorrelation and partial autocorrelation functions of the price history. Number `d` is the number of differencing that had to be done so that the series became stationary. To test the stationarity, Augmented Dickey-Fuller unit root test was used and after that corrected with Kwiatkowski-Phillips-Schmidt-Shin test.

ARIMA enables us to generate simulations of further price movements which can be used e.g. for risk management. It also allows us to create simple prediction. Although this feature is implemented in this project, it is not implemented in the final results, since LSTM predictions were more precise. ARIMA forecast is more of a mean value of future simulations.

Since fitting ARIMA model is very fast in our case, no models are stored and everytime user calls for simulations either in UI or CLI, new ARIMA model is fitted. The model is estimated during the fitting using **maximum likelihood estimation** (**MLE**) function, specifically **log-likelihood**. Under an assumption of normaly distributed residuals, the maximization of the log-likelihood is equivalent to minimizing the **sum of squared errors** (**SSE**) between the actual observed values and the fitted (predicted) values.

### LSTM Model
The LSTM (Long Short-Term Memory) model is a type of Recurrent Neural Network (RNN) designed for sequential data prediction. It is trained using several features of the stock and predicts future adjusted close prices. The sequence length for trained was chosen to be 30, roughly corresponding to one month - based on tests and comparisons, this number gave the best results.

The models for each ticker are pre-trained and stored in the `/data/models` directory, allowing it to be accessed and used by UI directly without requiring retraining.

For one of the main functions evaluated during the neural network training - the loss function, **Mean squared error** (**MSE**) was used. This function penalizes large errors more than for example **MAE** and it also has continuous gradients which makes it a better choice. MSE is then also used for the training early stop. That is done by applying MSE on validation data for each epoch and if the loss value stops decreasing and starts sort of oscilating, the training is stopped before default number of epochs is reached. MSE then serves as a good metrics for the performance of the model.


## Data

The data processing for stock price predictions involves fetching historical stock data, cleaning it, and generating additional features to enhance the model's performance. The following steps are involved:

### 1. Fetching Stock History
The historical stock price data are downloaded from Yahoo Finance using the `yfinance` library. The data includes columns like **Open**, **High**, **Low**, **Close**, **Adj Close**, **Volume**, **Dividends**, and **Stock Splits**.

### 2. Preprocessing Stock Data
The data preprocessing consists of several steps to clean and enhance the stock price data:

- **Returns Calculation**: It calculates the daily returns based on the adjusted closing price (`Adj Close`).
  
- **Moving Averages**: Computes the 10-day and 50-day moving averages to capture short-term and medium-term trends.
  
- **Volatility**: The rolling standard deviation of the daily returns is computed for a 10-day window to represent the volatility of the stock.
  
- **RSI (Relative Strength Index)**: The 14-day RSI is calculated to identify whether a stock is overbought or oversold.
  
- **Lagged Returns**: The 1-day, 5-day, and 10-day lagged returns are added as features to capture short-term trends in the stock price.

- **Interpolation and One-Hot Encoding**: Missing data is interpolated to ensure continuous time series. Additionally, the day of the week is extracted and one-hot encoded to capture weekly seasonality in the data.

### 3. Breakout Point Detection with Binseg
In most cases we can separate the stock history into two parts - slow and steady growth throughout its history, and then an almost sudden rise of the gradient. Training and fitting the models on the whole history doesn't produce very reliable results - in most cases the models tended to return back to some median values affected by long history of lower, almost constant prices. Therefore only the second part of stock's history with higher gradient and volatility is used to fit the models.

To distinguish this part, one needs to find the so called **Breakout Point** - a point, where the gradient (on some window) significantly changes (in case of stocks it almost always increases). There are several methods for that and the one used here is **Binseg** algorithm (Binary Segmentation) from `ruptures` library. It is a breakout point detection algorithm that divides the series into segments that are statistically consistent.

The final processed stock data includes all the generated features, and the data is filtered to include only the data after the detected breakout point.
