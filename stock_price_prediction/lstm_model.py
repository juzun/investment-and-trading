import datetime as dt
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from keras.api.callbacks import EarlyStopping
from keras.api.layers import LSTM, Dense, Dropout, Input
from keras.api.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as metrics

from stock_price_prediction.types import TickerSymbol


class LSTMModel:

    def __init__(self, data: pd.DataFrame, features: List[str], target_feature: str) -> None:
        self.data = data
        self.features = features
        self.target_feature = target_feature
        self.feature_index_map = {feat: idx for idx, feat in enumerate(features)}
        self.seq_length = 30
        self.trained = False

    def train_model(self, ) -> None:
        self.split_and_scale()

        X_train, y_train = self.create_sequences(
            data=self.train_data_scaled,
        )
        self.X_test, self.y_test = self.create_sequences(
            data=self.test_data_scaled,
        )

        self.model = self.create_model()
        self.model.compile(optimizer="adam", loss="mse")
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        self.history = self.model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_data=(self.X_test, self.y_test),
            callbacks=[early_stopping],
        )

        self.trained = True

    def split_and_scale(self) -> None:
        train_size = int(len(self.data) * 0.8)
        train_data = self.data.iloc[:train_size]
        test_data = self.data.iloc[train_size:]

        # Initialize scaler and fit only on training data
        self.scaler = MinMaxScaler()
        self.train_data_scaled = pd.DataFrame(
            self.scaler.fit_transform(train_data[self.features]),
            columns=self.features,
            index=train_data.index,
        )

        # Transform test data using the same scaler (without fitting)
        self.test_data_scaled = pd.DataFrame(
            self.scaler.transform(test_data[self.features]),
            columns=self.features,
            index=test_data.index,
        )

    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data.iloc[i : i + self.seq_length].values)
            y.append(
                data.iloc[
                    i + self.seq_length, self.feature_index_map[self.target_feature]
                ]
            )
        return np.array(X), np.array(y)

    def create_model(self) -> Sequential:
        return Sequential(
            [
                Input(shape=(self.seq_length, len(self.features))),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation="relu"),
                Dense(1),
            ]
        )

    def make_predictions(
        self,
        days_ahead: int,
    ) -> np.ndarray:
        # Start with the most recent known sequence
        if isinstance(self.test_data_scaled, pd.DataFrame):
            input_seq = self.test_data_scaled.to_numpy()[-self.seq_length :].copy()
        else:
            input_seq = self.test_data_scaled[-self.seq_length :].copy()
        
        predictions = []        

        for _ in range(days_ahead):
            predicted_value = self.model.predict(
                input_seq.reshape(1, self.seq_length, -1), verbose=0
            )[0, 0]
            predictions.append(predicted_value)

            # Update sequence by appending prediction and shifting window
            new_entry = np.roll(input_seq, shift=-1, axis=0)
            new_entry[-1] = input_seq[-1]
            new_entry[-1, self.feature_index_map[self.target_feature]] = (
                predicted_value  # Replace "Adj Close"
            )
            input_seq = new_entry

        predictions_all_fetures = np.zeros((len(predictions), len(self.features)))
        predictions_all_fetures[:, self.feature_index_map[self.target_feature]] = (
            predictions
        )
        predictions_descaled = self.scaler.inverse_transform(predictions_all_fetures)[
            :, self.feature_index_map[self.target_feature]
        ]

        return predictions_descaled

    def _save_trained_lstm_model(self, ticker_name: str) -> None:
        if self.trained:
            self.model.save(Path(__file__).parent.parent / "data" / "models" / f"lstm_{ticker_name}.keras")
            
            y_pred = self.model.predict(self.X_test)
            log = {
                "ticker": ticker_name,
                "Features": self.features,
                "Target feature": self.target_feature,
                "MSE": metrics.mean_squared_error(y_true=self.y_test, y_pred=y_pred),
                "MAE": metrics.mean_absolute_error(y_true=self.y_test, y_pred=y_pred),
                "5% of mean Adj Close": self.data["Adj Close"].mean() * 0.05,
                "5% of mean Adj Close last 10 years": self.data["Adj Close"][
                    self.data.index.date >= dt.date(self.data.index[-1].year - 10, 1, 1)
                ].mean()
                * 0.05,
            }
            with open(Path(__file__).parent.parent / "data" / "logs" / f"log_{ticker_name}.json", "w") as file:
                json.dump(log, file)
                file.write("\n")

    @classmethod
    def _load_trained_model(cls, data: pd.DataFrame, ticker_name: str) -> "LSTMModel":
        loaded_model = load_model(
            Path(__file__).parent.parent
            / "data"
            / "models"
            / f"lstm_{ticker_name}.keras"
        )
        with open(
            Path(__file__).parent.parent / "data" / "logs" / f"log_{ticker_name}.json",
            "r",
        ) as file:
            logs = json.load(file)

        lstm_model = LSTMModel(
            data=data, features=logs["Features"], target_feature=logs["Target feature"]
        )
        lstm_model.model = loaded_model
        lstm_model.trained = True
        lstm_model.split_and_scale()

        return lstm_model
