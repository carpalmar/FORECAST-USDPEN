import pandas as pd
import pytest
from pathlib import Path
from forecast_usdpen.io.data_loaders import load_data


@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    """Create a sample CSV file for testing."""
    csv_content = """DATES,PEN,OTHER
2023-01-31,3.80,10
2023-02-28,3.82,11
2023-03-31,3.81,12
2023-04-30,,13
"""
    csv_path = tmp_path / "sample_data.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


def test_load_data_success(sample_csv: str):
    """Test successful data loading and processing."""
    df = load_data(path=sample_csv, date_col="DATES", value_col="PEN")

    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.shape == (3, 1)  # Should drop the NaN row
    assert df.columns == ["PEN"]
    assert df.index.name == "date"
    assert df.iloc[0]["PEN"] == 3.80


def test_load_data_rename_cols(sample_csv: str):
    """Test column renaming feature."""
    df = load_data(
        path=sample_csv, date_col="DATES", value_col="PEN", rename_cols=["ds", "y"]
    )
    assert df.index.name == "ds"
    assert df.columns == ["y"]


def test_load_data_file_not_found():
    """Test FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_data(path="non_existent_file.csv", date_col="D", value_col="V")


def test_load_data_key_error(tmp_path: Path):
    """Test KeyError for incorrect column names."""
    csv_content = "DATE,VALUE\n2023-01-01,1"
    csv_path = tmp_path / "bad_cols.csv"
    csv_path.write_text(csv_content)
    with pytest.raises(KeyError):
        load_data(
            path=str(csv_path),
            date_col="DATES",  # Wrong name
            value_col="PEN",
        )
