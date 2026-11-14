"""Date and amount formats that real files use, and the examples shown when a file is rejected."""
import pandas as pd
import pytest

from services.data_cleaning import DataProcessor, DatasetError


def prepare(rows):
    processor = DataProcessor()
    table = pd.DataFrame(rows, columns=["Customer_ID", "Transaction_Date",
                                        "Transaction_Amount"]).astype(str)
    return processor, processor.normalize_columns(table)


def clean(rows):
    processor, table = prepare(rows)
    return processor.clean_data(table)


def dates_by_customer(clean_df):
    return dict(zip(clean_df["Customer_ID"], clean_df["Transaction_Date"]))


# ------------------------------------------------------------------ dates
def test_month_name_dates_are_read():
    clean_df, summary = clean([["C1", "15-Jan-2025", "100"],
                               ["C2", "15 January 2025", "100"],
                               ["C3", "Jan 15, 2025", "100"],
                               ["C4", "January 15 2025", "100"]])
    assert summary["rows_retained"] == 4
    assert set(clean_df["Transaction_Date"]) == {pd.Timestamp("2025-01-15")}


def test_month_name_without_a_year_is_not_guessed():
    _, summary = clean([["C1", "2025-01-05", "100"], ["C2", "Jan 15", "100"]])
    assert summary["removed"]["invalid_date"] == 1


def test_two_digit_years_are_read_with_a_warning():
    clean_df, summary = clean([["C1", "25/03/24", "100"], ["C2", "05/03/24", "100"]])
    dates = dates_by_customer(clean_df)
    assert dates["C1"] == pd.Timestamp("2024-03-25")
    assert dates["C2"] == pd.Timestamp("2024-03-05")        # day-first is proven by the first row
    assert any("two-digit year" in warning for warning in summary["warnings"])


def test_two_digit_years_above_68_are_read_as_19xx():
    clean_df, _ = clean([["C1", "2025-01-05", "100"], ["C2", "05/05/99", "100"]])
    assert dates_by_customer(clean_df)["C2"] == pd.Timestamp("1999-05-05")


def test_compact_dates_are_read():
    clean_df, summary = clean([["C1", "20250105", "100"], ["C2", "2025-01-06", "100"]])
    assert summary["rows_retained"] == 2
    assert dates_by_customer(clean_df)["C1"] == pd.Timestamp("2025-01-05")


# ---------------------------------------------------------------- amounts
def test_currency_signs_and_codes_before_amounts_are_accepted():
    clean_df, summary = clean([["C1", "2025-01-05", "$1,250.50"],
                               ["C2", "2025-01-06", "£300"],
                               ["C3", "2025-01-07", "USD 40"]])
    assert sorted(clean_df["Transaction_Amount"]) == [40.0, 300.0, 1250.5]
    assert summary["amounts_normalised"] == 3


# ------------------------------------------------- examples in the message
def test_message_shows_examples_of_unreadable_dates():
    processor, table = prepare([["C1", "31st of Marchh", "100"], ["C2", "??", "100"]])
    with pytest.raises(DatasetError) as problem:
        processor.clean_data(table)
    message = str(problem.value)
    assert "Examples of dates that could not be read" in message
    assert "31st of Marchh" in message


def test_message_shows_examples_of_unreadable_amounts():
    processor, table = prepare([["C1", "2025-01-05", "abc"], ["C2", "2025-01-06", "12x"]])
    with pytest.raises(DatasetError) as problem:
        processor.clean_data(table)
    message = str(problem.value)
    assert "Examples of amounts that could not be read" in message
    assert "abc" in message


# ------------------------------------------------------------- encoding
def test_windows_encoded_csv_is_read(tmp_path):
    path = tmp_path / "ansi.csv"
    text = "Customer_ID,Transaction_Date,Transaction_Amount,Customer_Name\nC1,2025-01-05,100,Jos\u00e9\n"
    path.write_bytes(text.encode("cp1252"))
    raw = DataProcessor.read_file(str(path), "csv")
    assert raw["Customer_Name"].iloc[0] == "Jos\u00e9"