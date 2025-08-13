import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from forecast_usdpen.models.lstm_levels import LSTMAdapter


@pytest.fixture
def sample_series() -> pd.Series:
    """Generate a sample time series for testing."""
    dates = pd.date_range(start="2022-01-01", periods=50, freq="ME")
    data = np.linspace(3.5, 4.0, 50)
    return pd.Series(data, index=dates)


@patch("forecast_usdpen.models.lstm_levels.LSTMAdapter._build_model")
def test_lstm_adapter_save_creates_files(
    mock_build, sample_series: pd.Series, tmp_path: Path
):
    """
    Test that the save method creates the expected files, mocking the model.
    """
    params = {"sequence_length": 5}

    # Mock the Keras model object and its save method
    mock_model_instance = MagicMock()

    def mock_save_method(filepath):
        # Create a dummy file to simulate saving
        Path(filepath).touch()

    mock_model_instance.save = mock_save_method

    mock_build.return_value = mock_model_instance

    adapter = LSTMAdapter(model_params=params)
    # The fit method will call the mocked _build_model
    adapter.fit(sample_series)

    save_path = tmp_path / "lstm_model_mock"
    adapter.save(str(save_path))

    # Assert that the files were created by the save method
    assert (save_path / "model.h5").exists()
    assert (save_path / "scaler.pkl").exists()
    assert (save_path / "params.json").exists()
