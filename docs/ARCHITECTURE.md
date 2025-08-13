# Architecture Overview

This document outlines the architecture of the refactored `forecast-usdpen` package. The goal of this structure is to separate concerns, improve reusability, and create a reproducible forecasting pipeline.

## Directory Structure

-   `configs/`: Contains YAML configuration files that define pipeline runs, including data sources, model parameters, and output paths.
-   `docs/`: Project documentation.
-   `notebooks/`: Jupyter notebooks for demonstration and analysis. The refactored notebooks show how to use the package.
-   `src/forecast_usdpen/`: The main Python package source code.
-   `tests/`: The test suite for the package.
-   `artifacts/`: Default directory for output files (forecasts, metrics, plots). This is not tracked by git.

## Core Modules (`src/forecast_usdpen/`)

### 1. `io`

-   **Purpose**: Handles data input and output.
-   **Key Files**:
    -   `data_loaders.py`: Functions to load time series data from sources like CSV files into pandas DataFrames.

### 2. `preprocessing`

-   **Purpose**: Contains functions for data transformation and feature engineering.
-   **Key Files**:
    -   `transforms.py`: Functions for creating model-specific data structures, like creating sequences for LSTMs.

### 3. `models`

-   **Purpose**: Defines the model interface and implements adapters for different forecasting models. This is the core of the model abstraction layer.
-   **Key Files**:
    -   `base.py`: Defines the `BaseForecastModel` abstract base class with a common interface (`fit`, `predict`, `save`, `load`).
    -   `lstm_levels.py`: The `LSTMAdapter` class that wraps a Keras/TensorFlow LSTM model.
    -   `chronos_adapter.py`: The `ChronosAdapter` class for the Chronos foundation model.
    -   `evaluator.py`: Functions to calculate performance metrics (e.g., RMSE, MAE, Sharpe Ratio).

### 4. `viz`

-   **Purpose**: Modules for creating visualizations.
-   **Key Files**:
    -   `plots.py`: Functions to plot forecasts vs. actuals, cumulative returns, etc.

### 5. `forecasting`

-   **Purpose**: Contains the high-level logic for orchestrating the forecasting and backtesting pipelines.
-   **Key Files**:
    -   `pipeline.py`: Implements `run_pipeline`, which executes a single end-to-end forecast based on a config file.
    -   `backtesting.py`: Implements `walk_forward_validation` for robust model evaluation.

### 6. `cli`

-   **Purpose**: Provides the command-line interface.
-   **Key Files**:
    -   `cli.py`: Defines the `forecast-usdpen` command and its subcommands (`run`, `backtest`).

## Data Flow for `run` command

1.  The user executes `forecast-usdpen run --config <path_to_config>`.
2.  `cli.main()` parses the arguments and calls `forecasting.pipeline.run_pipeline()`.
3.  `run_pipeline()` reads the YAML config.
4.  It calls `io.data_loaders.load_data()` to load the initial dataset.
5.  The data is split into train and test sets.
6.  The appropriate model adapter (e.g., `LSTMAdapter`) is instantiated based on the config.
7.  The adapter's `.fit()` method is called with the training data. Inside, it may call `preprocessing.transforms` if needed.
8.  The adapter's `.predict()` method is called to generate forecasts.
9.  The forecasts are evaluated using functions from `models.evaluator`.
10. The results (forecasts CSV, metrics JSON) and visualizations (`viz.plots`) are saved to the `artifacts/` directory.
