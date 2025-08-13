import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from forecast_usdpen.forecasting.pipeline import run_pipeline
from forecast_usdpen.models.base import BaseForecastModel


# A mock model that conforms to the BaseForecastModel interface
class MockForecastModel(BaseForecastModel):
    def __init__(self, model_params=None):
        super().__init__(model_params)
        self.history_y = None

    def fit(self, y: pd.Series, X: pd.DataFrame = None):
        self.history_y = y
        return self

    def predict(self, h: int, X_future: pd.DataFrame = None) -> pd.Series:
        if self.history_y is None:
            raise ValueError("Fit must be called before predict")

        last_date = self.history_y.index[-1]
        freq = pd.infer_freq(self.history_y.index)
        future_index = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq),
            periods=h,
            freq=freq,
        )
        # Return a simple forecast (e.g., repeating the last value)
        forecast_values = [self.history_y.iloc[-1]] * h
        return pd.Series(forecast_values, index=future_index, name="forecast")

    def save(self, path: str) -> None:
        pass  # Not needed for this test

    @classmethod
    def load(cls, path: str) -> "MockForecastModel":
        return MockForecastModel()


@pytest.fixture
def mock_config(tmp_path: Path) -> str:
    """Create a sample config YAML and a dummy data file."""
    data_path = tmp_path / "dummy_data.csv"
    dates = pd.to_datetime(pd.date_range(start="2023-01-31", periods=20, freq="ME"))
    data = {"DATES": dates, "PEN": np.linspace(3.8, 4.0, 20)}
    pd.DataFrame(data).to_csv(data_path, index=False)

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    config_content = f"""
data:
  path: "{data_path}"
  date_col: "DATES"
  value_col: "PEN"
forecasting:
  train_split: 0.8
  horizon: 4 # Increase horizon to have more than one return
model:
  name: "mock_model"
output:
  artifacts_path: "{artifacts_dir}"
  metrics_file: "metrics.json"
  forecasts_file: "forecasts.csv"
  plot_file: "plot.png"
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return str(config_path)


def test_run_pipeline_creates_artifacts(mock_config: str):
    """
    Test that run_pipeline executes and creates all expected artifacts.
    """
    # 1. Define the mock model override
    mock_adapters = {"mock_model": MockForecastModel}

    # 2. Run the pipeline with plotting disabled
    result = run_pipeline(
        config_path=mock_config, model_adapters_override=mock_adapters, no_plot=True
    )

    # 3. Check that artifacts were created (except plot)
    artifacts = result["artifacts"]
    assert Path(artifacts["forecasts"]).exists()
    assert Path(artifacts["metrics"]).exists()
    assert artifacts["plot"] is None

    # Explicitly check that the plot file was not created on disk
    plot_path = Path(result["artifacts"]["forecasts"]).parent / "plot.png"
    assert not plot_path.exists()

    # 4. Check forecast content
    forecast_df = pd.read_csv(artifacts["forecasts"])
    assert forecast_df.shape == (4, 2)  # Horizon of 4 + index column
    assert "forecast" in forecast_df.columns
    # The mock model repeats the last training value
    last_train_value = 3.957894736842105
    assert np.allclose(forecast_df["forecast"], last_train_value)
