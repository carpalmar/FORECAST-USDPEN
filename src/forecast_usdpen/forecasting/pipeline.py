import yaml
import numpy as np
import json
import os
import tensorflow as tf
from typing import Dict, Any, Optional

from ..io.data_loaders import load_data
from ..models import evaluator
from ..viz import plots


def run_pipeline(
    config_path: str,
    model_adapters_override: Optional[Dict[str, Any]] = None,
    no_plot: bool = False,
) -> Dict[str, Any]:
    """
    Runs the full forecasting pipeline from a configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
        model_adapters_override (Optional[Dict[str, Any]]): Allows injecting mock models for testing.
        no_plot (bool): If True, suppresses plot generation.

    Returns:
        Dict[str, Any]: A dictionary containing metrics and artifact paths.
    """
    # 1. Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    fcst_cfg = config["forecasting"]
    model_cfg = config["model"]
    output_cfg = config["output"]

    # Set seed for reproducibility
    if "seed" in config:
        np.random.seed(config["seed"])
        tf.random.set_seed(config["seed"])

    # 2. Load data
    print("Loading data...")
    df = load_data(
        path=data_cfg["path"],
        date_col=data_cfg["date_col"],
        value_col=data_cfg["value_col"],
    )

    # 3. Split data
    split_point = int(len(df) * fcst_cfg["train_split"])
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]

    # 4. Instantiate model
    model_name = model_cfg["name"]

    # Lazy load adapters to avoid import errors if dependencies are not installed
    from ..models.lstm_levels import LSTMAdapter
    from ..models.chronos_adapter import ChronosAdapter

    MODEL_ADAPTERS = {
        "lstm": LSTMAdapter,
        "chronos": ChronosAdapter,
    }

    if model_adapters_override:
        MODEL_ADAPTERS.update(model_adapters_override)

    if model_name not in MODEL_ADAPTERS:
        raise ValueError(
            f"Model '{model_name}' not recognized. Available models: {list(MODEL_ADAPTERS.keys())}"
        )

    adapter_class = MODEL_ADAPTERS[model_name]
    model = adapter_class(model_params=model_cfg.get("hyperparameters", {}))

    # 5. Fit model
    print(f"Fitting {model_name} model...")
    model.fit(y=train_df[data_cfg["value_col"]])

    # 6. Generate predictions
    print("Generating forecasts...")
    horizon = fcst_cfg.get("horizon", len(test_df))
    forecasts = model.predict(h=horizon)

    # 7. Evaluate
    print("Evaluating metrics...")
    # Ensure true values align with forecast horizon
    y_true = test_df[data_cfg["value_col"]].iloc[: len(forecasts)]

    error_metrics = evaluator.get_error_metrics(y_true, forecasts)
    trading_metrics = evaluator.get_trading_performance_metrics(y_true, forecasts)

    all_metrics = {**error_metrics, **trading_metrics}

    print("Metrics:", json.dumps(all_metrics, indent=4))

    # 8. Save artifacts
    artifacts_path = output_cfg["artifacts_path"]
    os.makedirs(artifacts_path, exist_ok=True)

    # Save forecasts
    forecast_path = os.path.join(artifacts_path, output_cfg["forecasts_file"])
    forecasts.to_csv(forecast_path)
    print(f"Forecasts saved to {forecast_path}")

    # Save metrics
    metrics_path = os.path.join(artifacts_path, output_cfg["metrics_file"])
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    plot_path = None
    if not no_plot:
        # Save plot
        plot_path = os.path.join(artifacts_path, output_cfg["plot_file"])
        plots.plot_forecast(
            y_true=y_true,
            y_pred=forecasts,
            title=f"{model_name.upper()} Forecast vs. Actuals",
            output_path=plot_path,
        )
        print(f"Plot saved to {plot_path}")

    return {
        "metrics": all_metrics,
        "artifacts": {
            "forecasts": forecast_path,
            "metrics": metrics_path,
            "plot": plot_path,
        },
    }
