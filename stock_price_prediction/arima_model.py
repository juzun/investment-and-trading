import warnings
from typing import Optional, Tuple

import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


class ARIMAModel:
    """A class for training and forecasting using an ARIMA model.

    Attributes:
        data (pd.DataFrame): The input time series data.
        target_feature (str): The column in the data to be modeled.
        arima_result (ARIMAResults): The trained ARIMA model result after fitting.
    """

    def __init__(self, data: pd.DataFrame, target_feature: str) -> None:
        """
        Initializes the ARIMAModel instance.

        Args:
            data (pd.DataFrame): Time series data.
            target_feature (str): The column name in the dataframe to be modeled.
        """
        self.data = data
        self.target_feature = target_feature
        self.arima_result: Optional[ARIMAResults] = None

    def fit_arima_model(self) -> None:
        """Fits an ARIMA model to the target feature in the dataset."""
        arima_coefs = self.get_arima_coefficients(series=self.data[self.target_feature])
        arima_model = ARIMA(self.data[self.target_feature], order=arima_coefs)
        self.arima_result = arima_model.fit()

    def get_arima_coefficients(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        Determines optimal ARIMA model order (p, d, q).

        Args:
            series (pd.Series): The time series data.

        Returns:
            Tuple[int, int, int]: The ARIMA order (p, d, q).
        """
        p, d, q = 0, 0, 0

        # Perform differencing testing and adjust d-parameter
        while not self.adf_test(series):
            series = series.pct_change()
            series = series.dropna()
            d += 1

        if not self.kpss_test(series):
            d += 1

        # Get parameters p and q from Autocorrelation and Partial Autocorrelation functions
        p = int(sm.tsa.pacf(series).round(0).sum())
        q = int(sm.tsa.acf(series).round(0).sum())
        return p, d, q

    def adf_test(self, series: pd.Series) -> bool:
        """
        Check for stationarity using Augmented Dickey-Fuller test.

        Attributes:
            series: input data to be tested on stationarity.

        Returns:
            bool: True if data are stationary, False if not.
        """
        result = stattools.adfuller(series)

        # If p-value <= 0.05, data is stationary
        return result[1] <= 0.05

    def kpss_test(self, series: pd.Series) -> bool:
        """
        Performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.

        Args:
            series (pd.Series): The time series to test.

        Returns:
            bool: True if the series is stationary, False otherwise.
        """
        result = stattools.kpss(series)

        # If p-value >= 0.05, data is stationary
        return result[1] >= 0.05

    def make_simulations(
        self,
        days_ahead: int,
        n_simulations: int = 10,
    ) -> pd.DataFrame:
        """
        Generates multiple simulated future paths for the time series.

        Args:
            days_ahead (int): Number of days into the future to simulate.
            n_simulations (int, optional): Number of simulation repetitions. Defaults to 10.

        Returns:
            pd.DataFrame: Simulated future values for the target feature.
        """
        if self.arima_result is None:
            warnings.warn("The ARIMA model wasn't yet created.")
            return pd.DataFrame()

        simulations: pd.DataFrame = self.arima_result.simulate(
            nsimulations=days_ahead,
            anchor=self.data[self.target_feature].index[-1],
            repetitions=n_simulations,
        )
        # Ensure continuity
        simulations.iloc[0] = self.data[self.target_feature].iloc[-1]

        return simulations

    def make_forecast(
        self,
        days_ahead: int,
    ) -> pd.Series:
        """
        Predicts future values using the fitted ARIMA model.

        Args:
            days_ahead (int): Number of future steps to forecast.

        Returns:
            pd.Series: Forecasted values indexed by date.
        """
        if self.arima_result is None:
            warnings.warn("The ARIMA model wasn't yet created.")
            return pd.DataFrame()

        forecast = self.arima_result.forecast(steps=days_ahead)
        forecast_dates = pd.date_range(
            start=self.data.index[-1], periods=days_ahead, freq="B"
        )
        forecast_series = pd.Series(forecast, index=forecast_dates)
        # Ensure continuity
        forecast_series.iloc[0] = self.data[self.target_feature].iloc[-1]

        return forecast_series
