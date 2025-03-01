from typing import Tuple

import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


class ARIMAModel:

    def __init__(self, data: pd.DataFrame, target_feature: str) -> None:
        self.data = data
        self.target_feature = target_feature

    def fit_arima_model(self) -> None:
        arima_coefs = self.get_arima_coefficients(series=self.data[self.target_feature])
        arima_model = ARIMA(self.data[self.target_feature], order=arima_coefs)
        self.arima_result: ARIMAResults = arima_model.fit()

    def get_arima_coefficients(self, series: pd.Series) -> Tuple[int, int, int]:
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
        # ADF Statistic: result[0], p-value: result[1]
        if result[1] <= 0.05:
            return True
        else:
            return False

    def kpss_test(self, series: pd.Series) -> bool:
        result = stattools.kpss(series)
        if result[1] >= 0.05:
            return True
        else:
            return False

    def make_simulations(
        self,
        days_ahead: int,
        n_simulations: int = 10,
    ) -> pd.DataFrame:
        simulations: pd.DataFrame = self.arima_result.simulate(
            nsimulations=days_ahead,
            anchor=self.data[self.target_feature].index[-1],
            repetitions=n_simulations,
        )
        simulations.iloc[0] = self.data[self.target_feature].iloc[-1]

        return simulations

    def make_forecast(
        self,
        days_ahead: int,
    ) -> pd.Series:
        forecast = self.arima_result.forecast(steps=days_ahead)
        forecast_dates = pd.date_range(
            start=self.data.index[-1], periods=days_ahead, freq="B"
        )
        forecast_series = pd.Series(forecast, index=forecast_dates)
        forecast_series.iloc[0] = self.data[self.target_feature].iloc[-1]

        return forecast_series
