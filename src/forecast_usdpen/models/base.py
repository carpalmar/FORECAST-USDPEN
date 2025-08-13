from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd


class BaseForecastModel(ABC):
    """
    Abstract base class for a forecasting model adapter.
    """

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        self.model_params = model_params or {}
        self.model = None

    @abstractmethod
    def fit(
        self, y: pd.Series, X: Optional[pd.DataFrame] = None
    ) -> "BaseForecastModel":
        """
        Fit the forecasting model.

        Args:
            y (pd.Series): The target time series to train on.
            X (Optional[pd.DataFrame]): Exogenous variables, if any.

        Returns:
            self: The fitted model instance.
        """
        ...

    @abstractmethod
    def predict(self, h: int, X_future: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate forecasts for a future horizon.

        Args:
            h (int): The forecast horizon.
            X_future (Optional[pd.DataFrame]): Future exogenous variables, if any.

        Returns:
            pd.Series: A series of forecasted values.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to a specified path.

        Args:
            path (str): The directory or file path to save the model to.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseForecastModel":
        """
        Load a model from a specified path.

        Args:
            path (str): The directory or file path to load the model from.

        Returns:
            BaseForecastModel: The loaded model instance.
        """
        ...
