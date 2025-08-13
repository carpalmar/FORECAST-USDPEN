import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "Forecast vs. Actuals",
    output_path: Optional[str] = None,
):
    """
    Plots the true values against the predicted values.

    Args:
        y_true (pd.Series): Series of true values with datetime index.
        y_pred (pd.Series): Series of predicted values with datetime index.
        title (str): The title of the plot.
        output_path (str, optional): Path to save the plot. If None, shows the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true, label="Valores reales", color="black")
    plt.plot(y_pred.index, y_pred, label="Predicción", linestyle="--")

    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_cumulative_returns(
    returns_dict: Dict[str, pd.Series],
    title: str = "Cumulative Returns Comparison",
    output_path: Optional[str] = None,
):
    """
    Plots the cumulative returns of multiple strategies.

    Args:
        returns_dict (Dict[str, pd.Series]): A dictionary where keys are strategy names
                                             and values are Series of returns.
        title (str): The title of the plot.
        output_path (str, optional): Path to save the plot. If None, shows the plot.
    """
    plt.figure(figsize=(12, 6))

    for name, returns in returns_dict.items():
        cumulative_returns = (1 + returns).cumprod() - 1
        plt.plot(cumulative_returns.index, cumulative_returns, label=name)

    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Retorno Acumulado")
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
