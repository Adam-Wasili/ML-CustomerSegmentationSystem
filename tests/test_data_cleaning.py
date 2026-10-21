"""Tests for the cleaning service. No web server needed."""
from datetime import date, timedelta

import pandas as pd
import pytest

from services.data_cleaning import REQUIRED_COLUMNS, DataProcessor, DatasetError

DEFAULT_COLUMNS = ("Customer_ID", "Transaction_Date", "Transaction_Amount")


def prepare(rows, columns=DEFAULT_COLUMNS):
    """Build a DataFrame the way the upload route does (all values as text)."""
    processor = DataProcessor()
    df = pd.DataFrame(rows, columns=list(columns)).astype(str)
    return processor, processor.normalize_columns(df)


def clean(rows):
    """Run the full cleaning on a small list of rows."""
    processor, df = prepare(rows)
    return processor.clean_data(df)


def dates_by_customer(clean_df):
    """Map Customer_ID -> cleaned date, to make date assertions short."""
    return dict(zip(clean_df["Customer_ID"], clean_df["Transaction_Date"]))


# ---------------------------------------------------------------- columns
def test_column_names_normalised():
    processor, df = prepare([["C1", "2025-01-05", "100"]],
                            columns=("customer id", "Transaction Date", "TRANSACTION_AMOUNT"))
    assert list(df.columns) == REQUIRED_COLUMNS
    assert processor.validate_structure(df) == []


def test_unrelated_column_not_mapped():
    # 'Quantity' and 'Notes' are not names for the required columns
    processor, df = prepare([["C1", "2", "x"]], columns=("Customer_ID", "Quantity", "Notes"))
    errors = processor.validate_structure(df)
    assert len(errors) == 2
    assert any("Transaction_Date" in e for e in errors)
    assert any("Transaction_Amount" in e for e in errors)
    assert list(df.columns) == ["Customer_ID"]     # the other columns are ignored


def test_common_alternative_names_are_recognised():
    processor, df = prepare([["C1", "2025-01-05", "100"]],
                            columns=("Client ID", "Date", "Amount"))
    assert processor.validate_structure(df) == []
    assert processor.summary["columns_renamed"] == {
        "Client ID": "Customer_ID", "Date": "Transaction_Date", "Amount": "Transaction_Amount"}


def test_revenue_column_is_read_as_the_amount():
    processor, df = prepare([["C1", "2025-01-05", "100"]],
                            columns=("Customer_ID", "Transaction_Date", "Revenue"))
    assert processor.validate_structure(df) == []
    assert "Transaction_Amount" in df.columns


def test_two_columns_for_the_same_field_is_an_error():
    processor, df = prepare([["C1", "2025-01-05", "100", "100"]],
                            columns=("Customer_ID", "Date", "Amount", "Revenue"))
    errors = processor.validate_structure(df)
    assert any("Two columns" in e for e in errors)


def test_missing_required_columns_reported():
    processor, df = prepare([["C1", "100"]], columns=("Customer_ID", "Transaction_Amount"))
    assert any("Transaction_Date" in e for e in processor.validate_structure(df))


# ------------------------------------------------------- missing values
def test_missing_values_removed():
    _, summary = clean([
        ["C1", "2025-01-05", "100"],
        ["", "2025-01-06", "100"],
        ["C2", "", "100"],
        ["C3", "2025-01-07", ""],
    ])
    assert summary["removed"]["missing_customer_id"] == 1
    assert summary["removed"]["missing_date"] == 1
    assert summary["removed"]["missing_amount"] == 1
    assert summary["rows_retained"] == 1


# ------------------------------------------------------------------ dates
def test_invalid_dates_removed():
    _, summary = clean([
        ["C1", "2025-01-05", "100"],
        ["C2", "not a date", "100"],
        ["C3", "2025-02-30", "100"],      # 30 February does not exist
    ])
    assert summary["removed"]["invalid_date"] == 2
    assert summary["rows_retained"] == 1


def test_future_dates_removed():
    future = (date.today() + timedelta(days=30)).isoformat()
    _, summary = clean([["C1", "2025-01-05", "100"], ["C2", future, "100"]])
    assert summary["removed"]["future_date"] == 1


def test_year_first_dates_accepted():
    _, summary = clean([["C1", "2025-01-05", "100"], ["C2", "2025-1-6", "100"]])
    assert summary["rows_retained"] == 2
    assert summary["dates_non_iso"] == 0


def test_day_first_inferred_from_file():
    # 25/03/2025 can only be day-first, so 05/03/2025 is read the same way
    clean_df, summary = clean([["C1", "25/03/2025", "100"], ["C2", "05/03/2025", "100"]])
    dates = dates_by_customer(clean_df)
    assert dates["C1"] == pd.Timestamp("2025-03-25")
    assert dates["C2"] == pd.Timestamp("2025-03-05")
    assert summary["date_convention"] == "day/month/year"
    assert summary["dates_non_iso"] == 2


def test_month_first_inferred_from_file():
    # 03/25/2025 can only be month-first, so 05/03/2025 means 3 May
    clean_df, summary = clean([["C1", "03/25/2025", "100"], ["C2", "05/03/2025", "100"]])
    dates = dates_by_customer(clean_df)
    assert dates["C1"] == pd.Timestamp("2025-03-25")
    assert dates["C2"] == pd.Timestamp("2025-05-03")
    assert summary["date_convention"] == "month/day/year"


def test_mixed_year_first_and_month_first_are_both_kept():
    _, summary = clean([["C1", "2025-03-05", "100"], ["C2", "03/25/2025", "100"]])
    assert summary["rows_retained"] == 2
    assert summary["removed"]["ambiguous_date"] == 0


def test_conflicting_orders_make_unclear_dates_ambiguous():
    # The file proves BOTH orders, so 05/03/2025 cannot be trusted and is removed
    clean_df, summary = clean([
        ["C1", "25/03/2025", "100"],     # only day-first is possible
        ["C2", "03/25/2025", "100"],     # only month-first is possible
        ["C3", "05/03/2025", "100"],     # could be either
    ])
    assert summary["removed"]["ambiguous_date"] == 1
    assert summary["rows_retained"] == 2
    assert "C3" not in set(clean_df["Customer_ID"])


def test_all_unclear_dates_removed_when_nothing_proves_the_order():
    _, summary = clean([
        ["C1", "2025-01-05", "100"],
        ["C2", "05/03/2025", "100"],
        ["C3", "06/04/2025", "100"],
    ])
    assert summary["removed"]["ambiguous_date"] == 2
    assert summary["rows_retained"] == 1


def test_same_day_and_month_is_not_ambiguous():
    clean_df, summary = clean([["C1", "05/05/2025", "100"]])
    assert clean_df["Transaction_Date"].iloc[0] == pd.Timestamp("2025-05-05")
    assert summary["removed"]["ambiguous_date"] == 0


def test_two_digit_year_is_invalid():
    _, summary = clean([["C1", "2025-01-05", "100"], ["C2", "05/03/25", "100"]])
    assert summary["removed"]["invalid_date"] == 1


def test_day_and_month_both_over_12_is_invalid():
    _, summary = clean([["C1", "2025-01-05", "100"], ["C2", "25/25/2025", "100"]])
    assert summary["removed"]["invalid_date"] == 1


# ---------------------------------------------------------------- amounts
def test_invalid_amounts_removed():
    _, summary = clean([["C1", "2025-01-05", "100"],
                        ["C2", "2025-01-06", "abc"],
                        ["C3", "2025-01-07", "12x"]])
    assert summary["removed"]["invalid_amount"] == 2


def test_non_positive_amounts_removed():
    _, summary = clean([["C1", "2025-01-05", "100"],
                        ["C2", "2025-01-06", "-100"],
                        ["C3", "2025-01-07", "0"]])
    assert summary["removed"]["non_positive_amount"] == 2


def test_thousands_separator_normalised():
    clean_df, summary = clean([["C1", "2025-01-05", "25,000"],
                               ["C2", "2025-01-06", "MK 30,000"]])
    assert sorted(clean_df["Transaction_Amount"].tolist()) == [25000.0, 30000.0]
    assert summary["amounts_normalised"] == 2


def test_european_decimal_format_is_rejected():
    # '25.000,50' is ambiguous, so it is removed instead of guessed
    _, summary = clean([["C1", "2025-01-05", "100"], ["C2", "2025-01-06", "25.000,50"]])
    assert summary["removed"]["invalid_amount"] == 1


# ------------------------------------------------- duplicates and grouping
def test_duplicates_removed():
    _, summary = clean([["C1", "2025-01-05", "100"],
                        ["C1", "2025-01-05", "100"],
                        ["C2", "2025-01-06", "200"]])
    assert summary["removed"]["duplicates"] == 1
    assert summary["rows_retained"] == 2


def test_customer_grouping_by_customer_id():
    clean_df, summary = clean([
        ["C003", "2025-02-10", "20000"], ["C003", "2025-02-15", "15000"],
        ["C003", "2025-03-01", "25000"],
        ["C008", "2025-02-12", "30000"], ["C008", "2025-03-05", "15000"],
    ])
    grouped = clean_df.groupby("Customer_ID")["Transaction_Amount"].agg(["count", "sum"])
    assert summary["customers"] == 2
    assert grouped.loc["C003", "count"] == 3
    assert grouped.loc["C003", "sum"] == 60000


# -------------------------------------------------------------- file level
def test_header_only_csv_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("Customer_ID,Transaction_Date,Transaction_Amount\n")
    with pytest.raises(DatasetError):
        DataProcessor.read_file(str(path), "csv")


def test_all_rows_invalid_raises_error():
    processor, df = prepare([["", "", ""]])
    with pytest.raises(DatasetError):
        processor.clean_data(df)