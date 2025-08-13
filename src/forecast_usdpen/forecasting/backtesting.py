import pandas as pd
import yaml
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from ..io.data_loaders import load_data
from ..models import evaluator


def walk_forward_validation(
    config_path: str, model_adapters_override: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Performs walk-forward validation for a forecasting model.

    Args:
        config_path (str): Path to the YAML configuration file.
        model_adapters_override (Optional[Dict[str, Any]]): Allows injecting mock models for testing.

    Returns:
        pd.DataFrame: A DataFrame containing performance metrics for each fold.
    """
    # 1. Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    fcst_cfg = config["forecasting"]
    bt_cfg = config["backtesting"]
    model_cfg = config["model"]

    # 2. Load data
    df = load_data(
        path=data_cfg["path"],
        date_col=data_cfg["date_col"],
        value_col=data_cfg["value_col"],
    )

    # 3. Setup walk-forward parameters
    initial_train_size = int(len(df) * fcst_cfg["train_split"])
    horizon = fcst_cfg.get("horizon", 1)
    stride = bt_cfg.get("stride", 1)
    retrain = bt_cfg.get("retrain", False)

    results: List[Dict[str, Any]] = []

    # Instantiate the model adapter once if not retraining
    model_name = model_cfg["name"]

    # Lazy load adapters
    from ..models.lstm_levels import LSTMAdapter
    from ..models.chronos_adapter import ChronosAdapter

    MODEL_ADAPTERS = {
        "lstm": LSTMAdapter,
        "chronos": ChronosAdapter,
    }

    if model_adapters_override:
        MODEL_ADAPTERS.update(model_adapters_override)

    adapter_class = MODEL_ADAPTERS[model_name]
    model = adapter_class(model_params=model_cfg.get("hyperparameters", {}))

    # 4. Walk-forward loop
    progress_bar = tqdm(
        range(initial_train_size, len(df) - horizon + 1, stride),
        desc="Walk-forward validation",
    )
    for i in progress_bar:
        train_end = i
        test_end = i + horizon

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        if not len(test_df) == horizon:
            continue

        # Fit model
        if retrain or i == initial_train_size:
            model.fit(y=train_df[data_cfg["value_col"]])
        else:
            # If not retraining, we need to update the history for prediction context
            if hasattr(model, "history_y"):
                model.history_y = train_df[data_cfg["value_col"]]

        # Predict
        forecasts = model.predict(h=horizon)

        # Evaluate
        y_true = test_df[data_cfg["value_col"]]
        error_metrics = evaluator.get_error_metrics(y_true, forecasts)

        fold_result = {"fold_start_date": str(test_df.index[0].date()), **error_metrics}
        results.append(fold_result)

    results_df = pd.DataFrame(results)
    print("Walk-forward validation results:")
    print(results_df)

    print("\nAggregated metrics:")
    print(results_df.mean(numeric_only=True))

    return results_df
