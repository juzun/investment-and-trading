# Stock Price Prediction

This project provides a stock price prediction and simulation tool using two models: LSTM (Long Short-Term Memory) for deep learning-based price predictions and ARIMA (AutoRegressive Integrated Moving Average) for time-series simulations. It is designed to facilitate stock price analysis using both models, offering a user-friendly dashboard built with **Streamlit**.

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
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This application allows users to predict and simulate stock price movements for various tickers using two powerful models: **ARIMA** and **LSTM**.

The app's UI is built using **Streamlit** for a simple and interactive user interface, with pre-trained LSTM models stored in the `/data/models` directory. The predictions and simulations are visualized in real-time on the dashboard.


## Installation
To get started with this project, you need to install the necessary dependencies. Poetry was used for dependency management in this project. Follow these steps to set up the environment:

### Clone the Repository
```bash
git clone https://github.com/juzun/stock-price-prediction.git
cd stock-price-prediction-dashboard
```

### Install Dependencies
First, install Poetry (if not installed already). Then run:
```bash
poetry install
```
This will install all the required dependencies listed in `pyproject.toml` file.

### Pre-Trained Models
Ensure that the pre-trained models for each ticker are available in the `/data/models` folder. The models are already trained and placed in this directory, ready for use in the application's UI.


## Running the Application
There are two ways to run the stock prediction application:
- using the **Streamlit UI** for a graphical interface,
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
The ARIMA model is a time-series model that predicts future values based on the trend, seasonality, and noise in the data. This model is trained on historical adjusted close price data and is used for simulating future stock price movements.

### LSTM Model
The LSTM (Long Short-Term Memory) model is a type of Recurrent Neural Network (RNN) designed for sequential data prediction. It is trained using several features of the stock and predicts future adjusted close prices.

- The models for each ticker are pre-trained and stored in the `/data/models` directory, allowing it to be accessed and used by UI directly without requiring retraining.


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
