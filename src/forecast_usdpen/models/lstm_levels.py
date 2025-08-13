import json
import os
import pickle
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from .base import BaseForecastModel
from ..preprocessing.transforms import create_sequences


class LSTMAdapter(BaseForecastModel):
    """
    Adapter for a Keras-based LSTM forecasting model.
    """

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__(model_params)
        self.scaler = MinMaxScaler()
        self.history_y: Optional[pd.Series] = None

        # Default parameters if not provided
        self.model_params.setdefault("sequence_length", 12)
        self.model_params.setdefault("lstm_units_1", 50)
        self.model_params.setdefault("lstm_units_2", 50)
        self.model_params.setdefault("dropout_1", 0.2)
        self.model_params.setdefault("dropout_2", 0.2)
        self.model_params.setdefault("learning_rate", 0.001)
        self.model_params.setdefault("epochs", 100)
        self.model_params.setdefault("batch_size", 32)

    def _build_model(self):
        """Builds the Keras LSTM model."""
        model = Sequential(
            [
                LSTM(
                    self.model_params["lstm_units_1"],
                    return_sequences=True,
                    input_shape=(self.model_params["sequence_length"], 1),
                ),
                Dropout(self.model_params["dropout_1"]),
                LSTM(self.model_params["lstm_units_2"]),
                Dropout(self.model_params["dropout_2"]),
                Dense(1),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.model_params["learning_rate"]), loss="mse"
        )
        return model

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "LSTMAdapter":
        """
        Fit the LSTM model.
        """
        self.history_y = y.copy()

        # Scale data
        y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1))

        # Create sequences
        X_train, y_train = create_sequences(
            y_scaled, self.model_params["sequence_length"]
        )

        # Build and train model
        self.model = self._build_model()
        self.model.fit(
            X_train,
            y_train,
            epochs=self.model_params["epochs"],
            batch_size=self.model_params["batch_size"],
            verbose=0,
        )
        return self

    def predict(self, h: int, X_future: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate forecasts for a future horizon.
        """
        if self.history_y is None or self.model is None:
            raise RuntimeError("The model must be fitted before prediction.")

        last_sequence = self.history_y.values[
            -self.model_params["sequence_length"] :
        ].reshape(-1, 1)
        scaled_sequence = self.scaler.transform(last_sequence)

        predictions = []
        current_sequence = scaled_sequence.copy()

        for _ in range(h):
            X_test = current_sequence.reshape(
                1, self.model_params["sequence_length"], 1
            )
            pred_scaled = self.model.predict(X_test, verbose=0)
            predictions.append(pred_scaled[0, 0])

            # Append prediction and drop the first element to maintain sequence length
            current_sequence = np.append(current_sequence[1:], pred_scaled, axis=0)

        # Inverse transform predictions
        predictions_unscaled = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )

        # Create a pandas Series with a future datetime index
        last_date = self.history_y.index[-1]
        freq = pd.infer_freq(self.history_y.index)
        future_index = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq),
            periods=h,
            freq=freq,
        )

        return pd.Series(
            predictions_unscaled.flatten(), index=future_index, name="forecast"
        )

    def save(self, path: str) -> None:
        """
        Save the model, scaler, and parameters to a directory.
        """
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.h5"))
        with open(os.path.join(path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(path, "params.json"), "w") as f:
            # Convert numpy types to native python types for json serialization
            params_to_save = {
                k: (int(v) if isinstance(v, np.integer) else v)
                for k, v in self.model_params.items()
            }
            json.dump(params_to_save, f)

    @classmethod
    def load(cls, path: str) -> "LSTMAdapter":
        """
        Load a model from a directory.
        """
        with open(os.path.join(path, "params.json"), "r") as f:
            params = json.load(f)

        adapter = cls(model_params=params)

        adapter.model = tf.keras.models.load_model(os.path.join(path, "model.h5"))
        with open(os.path.join(path, "scaler.pkl"), "rb") as f:
            adapter.scaler = pickle.load(f)

        return adapter
