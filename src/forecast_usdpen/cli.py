import argparse
import os

from .forecasting.pipeline import run_pipeline
from .forecasting.backtesting import walk_forward_validation


def main():
    """
    Main function for the forecast-usdpen CLI.
    """
    parser = argparse.ArgumentParser(
        description="A CLI for forecasting the USD/PEN exchange rate."
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # --- Run command ---
    parser_run = subparsers.add_parser("run", help="Run a single forecasting pipeline.")
    parser_run.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file."
    )

    # --- Backtest command ---
    parser_backtest = subparsers.add_parser(
        "backtest", help="Run walk-forward validation."
    )
    parser_backtest.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file."
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at '{args.config}'")
        return

    if args.command == "run":
        print(f"--- Running pipeline with config: {args.config} ---")
        run_pipeline(config_path=args.config)
        print("--- Pipeline run finished ---")
    elif args.command == "backtest":
        print(f"--- Running backtest with config: {args.config} ---")
        walk_forward_validation(config_path=args.config)
        print("--- Backtest finished ---")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
