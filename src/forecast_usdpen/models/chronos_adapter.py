import os
import pickle
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

from .base import BaseForecastModel


class ChronosAdapter(BaseForecastModel):
    """
    Adapter for the Chronos forecasting model.
    """

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__(model_params)
        self.history_y: Optional[pd.Series] = None
        self.model_params.setdefault("checkpoint", "amazon/chronos-t5-base")
        self.model_params.setdefault("num_samples", 20)
        self.model_params.setdefault("temperature", 1.0)
        self.model_params.setdefault("top_k", 50)
        self.model_params.setdefault("top_p", 1.0)

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "ChronosAdapter":
        """
        "Fits" the Chronos model by loading the pre-trained pipeline and storing history.
        """
        self.history_y = y.copy()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Forcing bfloat16 on CPU can cause errors if not supported
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = ChronosPipeline.from_pretrained(
            self.model_params["checkpoint"],
            device_map=device,
            torch_dtype=dtype,
        )
        return self

    def predict(self, h: int, X_future: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate forecasts using the Chronos model.
        """
        if self.history_y is None or self.model is None:
            raise RuntimeError("The model must be fitted before prediction.")

        context = torch.tensor(self.history_y.values, dtype=self.model.model.dtype)

        # The model's predict function expects a context and prediction length
        forecast = self.model.predict(
            context=context,
            prediction_length=h,
            num_samples=self.model_params.get("num_samples", 20),
            temperature=self.model_params.get("temperature", 1.0),
            top_k=self.model_params.get("top_k", 50),
            top_p=self.model_params.get("top_p", 1.0),
        )

        # We take the median of the samples as the forecast
        forecast_median = np.median(forecast[0].cpu().numpy(), axis=0)

        # Create a pandas Series with a future datetime index
        last_date = self.history_y.index[-1]
        freq = pd.infer_freq(self.history_y.index)
        if freq is None:
            freq = "M"  # Defaulting to Monthly if it cannot be inferred

        future_index = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq),
            periods=h,
            freq=freq,
        )

        return pd.Series(forecast_median.flatten(), index=future_index, name="forecast")

    def save(self, path: str) -> None:
        """
        Save the ChronosAdapter instance using pickle.
        The model itself is not saved as it's loaded from HuggingFace.
        We save the adapter which contains the history and params.
        """
        # We don't save the model object, as it's large and loaded from HF
        self.model = None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ChronosAdapter":
        """
        Load a ChronosAdapter instance from a pickle file.
        The model will be re-loaded on the next `fit` call.
        """
        with open(path, "rb") as f:
            adapter = pickle.load(f)
        # Re-initialize the model on the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        adapter.model = ChronosPipeline.from_pretrained(
            adapter.model_params["checkpoint"],
            device_map=device,
            torch_dtype=dtype,
        )
        return adapter
