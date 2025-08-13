import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_error_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calculates standard regression error metrics.

    Args:
        y_true (pd.Series): True values.
        y_pred (pd.Series): Predicted values.

    Returns:
        dict: A dictionary with MSE, RMSE, MAE, and R2 score.
    """
    y_true_vals = y_true.values.flatten()
    y_pred_vals = y_pred.values.flatten()

    return {
        "MSE": mean_squared_error(y_true_vals, y_pred_vals),
        "RMSE": np.sqrt(mean_squared_error(y_true_vals, y_pred_vals)),
        "MAE": mean_absolute_error(y_true_vals, y_pred_vals),
        "R2": r2_score(y_true_vals, y_pred_vals),
    }


def _calculate_returns(series: pd.Series) -> pd.Series:
    """Calculates percentage change returns."""
    return series.pct_change().dropna()


def get_trading_performance_metrics(
    y_true: pd.Series, y_pred: pd.Series, benchmark_returns: pd.Series = None
) -> dict:
    """
    Calculates performance metrics for a simple trading strategy.
    Strategy: Go long if predicted return > 0, else go short.

    Args:
        y_true (pd.Series): Series of true prices.
        y_pred (pd.Series): Series of predicted prices.
        benchmark_returns (pd.Series, optional): Benchmark returns for Information Ratio.
                                                  Defaults to buy-and-hold.

    Returns:
        dict: A dictionary of trading performance metrics.
    """
    true_returns = _calculate_returns(y_true)
    pred_returns = _calculate_returns(y_pred)

    # Align indices
    common_index = true_returns.index.intersection(pred_returns.index)
    true_returns = true_returns.loc[common_index]
    pred_returns = pred_returns.loc[common_index]

    if benchmark_returns is None:
        benchmark_returns = true_returns
    else:
        benchmark_returns = benchmark_returns.loc[common_index]

    # Simple trading strategy based on predicted direction
    positions = np.where(pred_returns > 0, 1, -1)
    strategy_returns = true_returns * pd.Series(
        positions, index=true_returns.index
    ).shift(1)
    strategy_returns = strategy_returns.dropna()

    # Performance metrics
    total_return = (1 + strategy_returns).prod() - 1
    annualized_return = (1 + total_return) ** (12 / len(strategy_returns)) - 1
    sharpe_ratio = (
        np.sqrt(12) * strategy_returns.mean() / strategy_returns.std()
        if strategy_returns.std() != 0
        else 0
    )

    # Max Drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    peak = cumulative_returns.cummax()
    max_drawdown = ((cumulative_returns - peak) / peak).min()

    # Hit Ratio
    hit_ratio = (
        np.mean(
            np.sign(strategy_returns)
            == np.sign(true_returns.loc[strategy_returns.index])
        )
        * 100
        if len(strategy_returns) > 0
        else 0
    )

    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Hit Ratio": hit_ratio,
    }
