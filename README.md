# FORECAST-USDPEN: USD/PEN Exchange Rate Forecasting

This repository contains a refactored Python package for forecasting the USD/PEN exchange rate. The original implementation, based on a collection of Jupyter notebooks, has been modularized to improve reusability, maintainability, and reproducibility.

The package provides a command-line interface (CLI) to run forecasting pipelines and backtests, driven by simple YAML configuration files.

## Features

- **Modular Architecture**: Code is organized into modules for data loading, preprocessing, modeling, evaluation, and visualization.
- **Reproducible Pipelines**: Experiments are defined in YAML configuration files, making them easy to track and reproduce.
- **Model Adapters**: A flexible adapter pattern allows for easy integration of different forecasting models (e.g., LSTM, Chronos).
- **Command-Line Interface**: A simple CLI to run forecasts and backtests without needing to open a notebook.
- **Testing**: Includes a `pytest` suite for testing core functionalities.
- **Code Quality**: Uses `ruff` for linting/formatting and `pre-commit` hooks to maintain code standards.

## Project Structure

```
FORECAST-USDPEN/
├── configs/              # Configuration files for pipelines
│   └── default.yaml
├── docs/                 # Documentation
│   └── ARCHITECTURE.md
├── notebooks/            # Original and refactored demo notebooks
│   ├── 0_refactored_*.ipynb
│   └── *.ipynb
├── src/
│   └── forecast_usdpen/  # Main package source code
│       ├── io/
│       ├── models/
│       ├── preprocessing/
│       ├── forecasting/
│       ├── viz/
│       └── cli.py
├── tests/                # Pytest test suite
├── .gitignore
├── pyproject.toml        # Project metadata and dependencies
├── requirements.txt      # For pip installation
└── README.md             # This file
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/carpalmar/FORECAST-USDPEN.git
    cd FORECAST-USDPEN
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the package:**
    The package can be installed in editable mode, which is useful for development. There are two installation options:

    *   **Standard installation (for LSTM and basic models):**
        ```bash
        pip install -e .
        ```
    *   **Full installation (includes heavy dependencies like PyTorch/TensorFlow for foundation models):**
        *Note: This requires a significant amount of disk space.*
        ```bash
        pip install -e ".[full]"
        ```

## Usage

The primary way to use this package is through its command-line interface, `forecast-usdpen`.

### Running a Forecast

This command runs a single forecasting pipeline based on the specified configuration file. It will train a model on a training set, generate predictions on a test set, and save the artifacts (forecasts, metrics, plots) to disk.

1.  **Configure your run:**
    Edit a configuration file in the `configs/` directory (e.g., `configs/default.yaml`) to specify the data path, model, and hyperparameters.

2.  **Execute the pipeline:**
    ```bash
    forecast-usdpen run --config configs/default.yaml
    ```

    After the run, the generated artifacts will be available in the `artifacts/` directory (or the path specified in your config).

### Running a Backtest

This command performs a walk-forward validation to evaluate model performance over multiple time windows.

1.  **Configure your backtest:**
    Set the backtesting parameters (stride, retrain flag) in your YAML config file.

2.  **Execute the backtest:**
    ```bash
    forecast-usdpen backtest --config configs/default.yaml
    ```
    The command will output the performance metrics for each fold and the aggregated average metrics.

### Using in a Notebook

The `notebooks/` directory contains refactored versions of the original notebooks that demonstrate how to use the package programmatically. These serve as examples of how to import and use the pipeline functions in your own analyses.

---
*This project was refactored by an AI software engineer.*
