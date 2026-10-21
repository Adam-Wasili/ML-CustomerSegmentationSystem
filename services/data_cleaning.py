"""Data validation and cleaning (Module 2).

DataProcessor mirrors the class in the design diagram. The Python method names
are snake_case versions of the diagram names (cleanData -> clean_data, etc.).

Golden rule: every uploaded value is untrusted text until it has been checked.
A row is either kept exactly as the user gave it (apart from harmless format
normalisation that is reported) or removed and counted. Values are never guessed.
"""
import re

import numpy as np
import pandas as pd

MAX_ROWS = 200_000          # safety limit on dataset size
MAX_TEXT_LENGTH = 200       # optional text fields are cut to this length
# Text values that mean "nothing here" (compared in lower case)
MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "#n/a", "-"}

REQUIRED_COLUMNS = ["Customer_ID", "Transaction_Date", "Transaction_Amount"]
OPTIONAL_COLUMNS = ["Customer_Name", "Product", "Category", "Location"]
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# (key, label) pairs, in the order the cleaning rules are applied.
# The results page loops over this list, so adding a rule here adds a row there.
REMOVAL_LABELS = [
    ("missing_customer_id", "Missing Customer_ID"),
    ("missing_date", "Missing Transaction_Date"),
    ("missing_amount", "Missing Transaction_Amount"),
    ("invalid_date", "Invalid date (cannot be read)"),
    ("ambiguous_date", "Ambiguous date (day/month order unclear)"),
    ("future_date", "Date in the future"),
    ("invalid_amount", "Invalid amount (not a number)"),
    ("non_positive_amount", "Amount is zero or negative"),
    ("duplicates", "Duplicate transaction"),
]

# Dates written with the year LAST (e.g. 25/03/2026) or FIRST (2026-03-25).
# Groups: 1st part, 2nd part, 3rd part. Anything after the date (a time) is ignored.
_DATE_PATTERN = r"^(\d{1,4})[-/.](\d{1,2})[-/.](\d{1,4})(?:[ T].*)?$"

DAY_FIRST = "day/month/year"
MONTH_FIRST = "month/day/year"


def _column_key(name):
    """Make column names comparable: 'Customer ID', 'customer_id' and
    'CUSTOMER-ID' all become 'customerid'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Lookup: comparable key -> standard column name
# Alternative names the system accepts for each standard column.
# Capitals, spaces, underscores and hyphens are ignored when matching, so
# "Sales Amount", "sales_amount" and "SALES-AMOUNT" are the same name.
# To accept another name, add it to the right list (the name is only mapped
# if it appears here; the system never guesses from the data).
COLUMN_ALIASES = {
    "Customer_ID": ["Customer Code", "Customer No", "Customer Number", "Cust ID", "Client ID"],
    "Transaction_Date": ["Date", "Purchase Date", "Sale Date", "Order Date",
                         "Invoice Date", "Trans Date"],
    "Transaction_Amount": ["Amount", "Revenue", "Sales", "Sales Amount",
                           "Total Amount", "Payment Amount"],
    "Customer_Name": ["Client Name"],
    "Product": ["Product Name", "Item", "Item Name"],
    "Category": ["Product Category"],
    "Location": ["Branch", "City"],
}

# Lookup: comparable key -> standard column name (built from the standard names
# and the aliases above)
_COLUMN_KEYS = {}
for _standard_name, _aliases in COLUMN_ALIASES.items():
    _COLUMN_KEYS[_column_key(_standard_name)] = _standard_name
    for _alias in _aliases:
        _COLUMN_KEYS[_column_key(_alias)] = _standard_name


class DatasetError(Exception):
    """A problem with the uploaded dataset. The message is safe to show to users."""


class DataProcessor:
    def __init__(self):
        self.column_conflicts = []   # e.g. two columns that both look like Customer_ID
        # Everything that happened during cleaning is collected here and later
        # saved with the dataset, so the results page can report it.
        self.summary = {
            "rows_raw": 0,
            "columns_renamed": {},
            "columns_ignored": [],
            "removed": {key: 0 for key, _ in REMOVAL_LABELS},
            "rows_removed_total": 0,
            "rows_retained": 0,
            "amounts_normalised": 0,
            "dates_non_iso": 0,        # kept dates that did not start with the year
            "date_convention": None,   # DAY_FIRST, MONTH_FIRST or None (not proven)
            "customers": 0,
            "total_revenue": 0.0,
            "date_min": None,
            "date_max": None,
            "warnings": [],
        }

    # ------------------------------------------------------------------ reading
    @staticmethod
    def read_file(path, file_type):
        """Read a CSV/XLSX file with EVERY value as text.

        Reading as text means nothing is converted silently (for example an ID
        like '007' stays '007', and no amount is rounded by the reader).
        """
        try:
            if file_type == "csv":
                # utf-8-sig removes the invisible marker Excel adds to CSV files
                df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
            elif file_type == "xlsx":
                # Only the first sheet is read
                df = pd.read_excel(path, dtype=str, engine="openpyxl", keep_default_na=False)
            else:
                raise DatasetError("Unsupported file type.")
        except DatasetError:
            raise
        except pd.errors.EmptyDataError as exc:
            raise DatasetError("The file is empty.") from exc
        except UnicodeDecodeError as exc:
            raise DatasetError("The CSV file must be saved with UTF-8 encoding.") from exc
        except Exception as exc:
            # Do not expose internal error details to the user
            raise DatasetError(
                "The file could not be read. Check that it is a valid CSV or Excel file."
            ) from exc

        if df.empty:
            raise DatasetError("The file contains no data rows.")
        if len(df) > MAX_ROWS:
            raise DatasetError(f"The file has more than {MAX_ROWS:,} rows, which is the limit.")

        df.columns = [str(column).strip() for column in df.columns]
        return df.fillna("").astype(str)

    # ------------------------------------------------------------- validation
    def normalize_columns(self, df):
        """Keep recognised columns, renaming spelling variants to the standard names.

        Only spelling variants are matched ('customer id' -> 'Customer_ID').
        Unrelated columns such as 'Date' or 'Amount' are NOT mapped: they are
        ignored and reported, so the system never guesses what a column means.
        """
        chosen = {}  # standard name -> original column name in the file
        for column in df.columns:
            target = _COLUMN_KEYS.get(_column_key(column))
            if target is None:
                continue
            if target in chosen:
                self.column_conflicts.append(
                    f"Two columns ('{chosen[target]}' and '{column}') both match "
                    f"'{target}'. Keep only one of them."
                )
                continue
            chosen[target] = column

        self.summary["columns_renamed"] = {
            original: target for target, original in chosen.items() if original != target
        }
        self.summary["columns_ignored"] = [c for c in df.columns if c not in chosen.values()]

        ordered = [chosen[target] for target in ALL_COLUMNS if target in chosen]
        out = df[ordered].rename(columns=self.summary["columns_renamed"])
        # Remove spaces around every value
        return out.fillna("").astype(str).apply(lambda column: column.str.strip())

    def validate_structure(self, df):
        """Return a list of error messages (an empty list means the structure is valid)."""
        errors = list(self.column_conflicts)
        for column in REQUIRED_COLUMNS:
            if column not in df.columns:
                accepted = ", ".join([column] + COLUMN_ALIASES[column])
                errors.append(
                    f"Required column '{column}' was not found. "
                    f"Accepted names: {accepted}. "
                    "Rename the column in your file or add it."
                )
        return errors

    def inspect_quality(self, df):
        """Count problems WITHOUT changing anything (used by the preview page)."""
        missing_id = self._missing(df["Customer_ID"])
        missing_date = self._missing(df["Transaction_Date"])
        missing_amount = self._missing(df["Transaction_Amount"])
        dates, ambiguous, _, _ = self._parse_dates(df["Transaction_Date"])
        amounts, _ = self._parse_amounts(df["Transaction_Amount"])
        today = pd.Timestamp.today().normalize()
        return {
            "rows": len(df),
            "missing_customer_id": int(missing_id.sum()),
            "missing_date": int(missing_date.sum()),
            "missing_amount": int(missing_amount.sum()),
            "invalid_date": int((dates.isna() & ~missing_date & ~ambiguous).sum()),
            "ambiguous_date": int(ambiguous.sum()),
            "future_date": int((dates > today).sum()),
            "invalid_amount": int((amounts.isna() & ~missing_amount).sum()),
            "non_positive_amount": int((amounts <= 0).sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "unique_customers": int(df.loc[~missing_id, "Customer_ID"].nunique()),
        }

    # --------------------------------------------------------------- cleaning
    def clean_data(self, df):
        """Run all cleaning rules in order. Returns (clean_df, summary)."""
        self.summary["rows_raw"] = len(df)
        df = df.copy()
        for column in OPTIONAL_COLUMNS:       # optional columns may be absent from the file
            if column not in df.columns:
                df[column] = ""
        df = df[ALL_COLUMNS]

        df = self.handle_missing_values(df).copy()
        df = self._tidy_text_columns(df)
        df = self.validate_dates(df)
        df = self.validate_amounts(df)
        df = self.remove_duplicates(df)

        if df.empty:
            raise DatasetError(
                "No valid rows remain after cleaning. Please check the data in your file."
            )
        self._finalise(df)
        return df.reset_index(drop=True), self.summary

    def handle_missing_values(self, df):
        """Remove rows missing a field that RFM needs (counted per field).

        Customer_ID, date and amount are all required to calculate Recency,
        Frequency and Monetary, so a row without one of them cannot be used.
        """
        checks = [
            ("Customer_ID", "missing_customer_id"),
            ("Transaction_Date", "missing_date"),
            ("Transaction_Amount", "missing_amount"),
        ]
        for column, key in checks:
            is_missing = self._missing(df[column])
            self.summary["removed"][key] = int(is_missing.sum())
            df = df[~is_missing]
        return df

    def validate_dates(self, df):
        """Convert dates. Remove invalid, ambiguous and future dates (each counted)."""
        parsed, ambiguous, non_iso, convention = self._parse_dates(df["Transaction_Date"])

        # Ambiguous = could be read two ways and the file does not say which
        invalid = parsed.isna() & ~ambiguous
        self.summary["removed"]["ambiguous_date"] = int(ambiguous.sum())
        self.summary["removed"]["invalid_date"] = int(invalid.sum())

        df = df.copy()
        df["Transaction_Date"] = parsed
        df = df[~(invalid | ambiguous)]

        # A future date is almost always a typing error and would ruin Recency
        future = df["Transaction_Date"] > pd.Timestamp.today().normalize()
        self.summary["removed"]["future_date"] = int(future.sum())
        df = df[~future]

        self.summary["dates_non_iso"] = int(non_iso.loc[df.index].sum())
        self.summary["date_convention"] = convention
        return df

    def validate_amounts(self, df):
        """Convert amounts to numbers. Remove non-numeric and zero/negative amounts."""
        numeric, normalised = self._parse_amounts(df["Transaction_Amount"])
        invalid = numeric.isna()
        self.summary["removed"]["invalid_amount"] = int(invalid.sum())

        df = df.copy()
        df["Transaction_Amount"] = numeric
        df = df[~invalid]

        # Refunds (negative amounts) are not supported: they would distort Monetary
        non_positive = df["Transaction_Amount"] <= 0
        self.summary["removed"]["non_positive_amount"] = int(non_positive.sum())
        df = df[~non_positive]

        self.summary["amounts_normalised"] = int(normalised.loc[df.index].sum())
        return df

    def remove_duplicates(self, df):
        """Remove rows identical in every column, keeping the first copy."""
        duplicated = df.duplicated(keep="first")
        self.summary["removed"]["duplicates"] = int(duplicated.sum())
        return df[~duplicated]

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _missing(series):
        """True where a value is empty or one of the 'nothing here' tokens."""
        return series.str.strip().str.lower().isin(MISSING_TOKENS)

    def _parse_dates(self, series):
        """Read dates safely. Returns (dates, ambiguous, non_iso, convention).

        Rules:
          * Year first (2026-03-25, 2026/3/5)     -> always unambiguous.
          * Year last (25/03/2026, 03/25/2026):
              - first part > 12  -> can only be day/month/year  (proves day-first)
              - second part > 12 -> can only be month/day/year  (proves month-first)
              - both parts equal (05/05/2026) -> same either way
              - both parts <= 12 and different (05/03/2026) -> AMBIGUOUS.
                It is read using the order the file itself proves; if the file
                proves both orders, or neither, it stays ambiguous and the row
                is removed later.
          * Anything else (text, two-digit years, 25/25/2026) -> invalid (NaT).

        dates      : Timestamps (NaT where unreadable or ambiguous)
        ambiguous  : True where the order could not be decided
        non_iso    : True where the date was read from a year-last format
        convention : DAY_FIRST, MONTH_FIRST or None
        """
        text = series.str.strip()
        present = ~self._missing(text)

        # Split into three parts; rows that do not look like a date get NaN
        parts = text.where(present).str.extract(_DATE_PATTERN)
        part1, part2, part3 = parts[0], parts[1], parts[2]
        n1 = pd.to_numeric(part1, errors="coerce")
        n2 = pd.to_numeric(part2, errors="coerce")
        n3 = pd.to_numeric(part3, errors="coerce")

        year_first = part1.str.len() == 4
        year_last = (part3.str.len() == 4) & ~year_first

        # --- decide the day/month order for year-last dates ---
        proves_day_first = year_last & (n1 > 12) & (n2 <= 12)
        proves_month_first = year_last & (n2 > 12) & (n1 <= 12)
        same_either_way = year_last & (n1 == n2) & (n1 <= 12)
        ambiguous = year_last & (n1 <= 12) & (n2 <= 12) & (n1 != n2)

        # The file's order is "proven" only if it shows ONE kind of proof
        has_day_first = bool(proves_day_first.any())
        has_month_first = bool(proves_month_first.any())
        if has_day_first and not has_month_first:
            convention = DAY_FIRST
        elif has_month_first and not has_day_first:
            convention = MONTH_FIRST
        else:
            convention = None   # mixed orders, or nothing to prove the order

        day_first_rows = proves_day_first | same_either_way | (ambiguous & (convention == DAY_FIRST))
        month_first_rows = proves_month_first | (ambiguous & (convention == MONTH_FIRST))
        unresolved = ambiguous & (convention is None)

        # --- put the parts into year / month / day columns ---
        year = pd.Series(np.nan, index=series.index)
        month = pd.Series(np.nan, index=series.index)
        day = pd.Series(np.nan, index=series.index)

        year[year_first], month[year_first], day[year_first] = n1[year_first], n2[year_first], n3[year_first]
        year[year_last] = n3[year_last]
        month[day_first_rows], day[day_first_rows] = n2[day_first_rows], n1[day_first_rows]
        month[month_first_rows], day[month_first_rows] = n1[month_first_rows], n2[month_first_rows]

        # Impossible dates (31/02/2026) and unresolved rows become NaT
        dates = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": day}),
                               errors="coerce").dt.normalize()

        non_iso = year_last & dates.notna()
        return dates, unresolved, non_iso, convention

    @staticmethod
    def _parse_amounts(series):
        """Return (amounts as floats or NaN, mask of values whose format was normalised).

        Accepted formatting: an optional 'MK'/'MWK' prefix and valid thousands
        separators (25,000). Anything else that is not a plain number is invalid,
        so values are never guessed (for example '25.000,50' is rejected).
        """
        raw = series.str.strip()
        no_currency = raw.str.replace(r"^(?:MWK|MK)\s*", "", case=False, regex=True)
        # Thousands separators are removed only when they are in the right places
        grouped = no_currency.str.match(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
        cleaned = no_currency.where(~grouped, no_currency.str.replace(",", "", regex=False))
        numeric = pd.to_numeric(cleaned, errors="coerce")
        numeric = numeric.where(np.isfinite(numeric)).astype(float)   # reject inf
        normalised = (cleaned != raw) & numeric.notna()
        return numeric, normalised

    def _tidy_text_columns(self, df):
        """Limit text lengths and turn empty optional values into None (NULL in the database)."""
        df["Customer_ID"] = df["Customer_ID"].str.slice(0, 100)
        for column in OPTIONAL_COLUMNS:
            values = df[column].str.slice(0, MAX_TEXT_LENGTH)
            missing = self._missing(values)
            df[column] = [None if is_missing else value
                          for value, is_missing in zip(values, missing)]
        return df

    def _finalise(self, df):
        """Fill in the totals and the warnings shown on the results page."""
        s = self.summary
        s["rows_removed_total"] = int(sum(s["removed"].values()))
        s["rows_retained"] = int(len(df))
        s["customers"] = int(df["Customer_ID"].nunique())
        s["total_revenue"] = float(df["Transaction_Amount"].sum())
        s["date_min"] = df["Transaction_Date"].min().strftime("%Y-%m-%d")
        s["date_max"] = df["Transaction_Date"].max().strftime("%Y-%m-%d")

        # IDs that differ only by letter case: the system cannot know they are one person
        ids = df["Customer_ID"].unique()
        case_clashes = len(ids) - len({i.lower() for i in ids})
        if case_clashes:
            s["warnings"].append(
                f"{case_clashes} Customer_ID value(s) differ only by letter case (for example "
                "'c001' and 'C001'). They were treated as different customers because the "
                "system cannot assume they are the same person."
            )
        # Same customer written with different names: keep the first, tell the user
        names = df.dropna(subset=["Customer_Name"]).groupby("Customer_ID")["Customer_Name"].nunique()
        if int((names > 1).sum()):
            s["warnings"].append(
                f"{int((names > 1).sum())} customer(s) have more than one name in the file. "
                "The first name found was kept."
            )
        if s["dates_non_iso"]:
            if s["date_convention"]:
                how = f"The file shows the order is {s['date_convention']}, which was used."
            else:
                how = ("The file mixes day-first and month-first dates, so only dates "
                       "that could be read in one way were kept.")
            s["warnings"].append(
                f"{s['dates_non_iso']} date(s) were not written year-first (YYYY-MM-DD). {how}"
            )
        if s["removed"]["ambiguous_date"]:
            s["warnings"].append(
                f"{s['removed']['ambiguous_date']} row(s) were removed because their date (for "
                "example 05/03/2026) could mean two different days and the file does not show "
                "which order is used. Use YYYY-MM-DD dates to avoid this."
            )
        if s["amounts_normalised"]:
            s["warnings"].append(
                f"{s['amounts_normalised']} amount(s) had formatting such as thousands "
                "separators or an MK/MWK prefix. The format was normalised; the values were not changed."
            )
        if s["removed"]["duplicates"]:
            s["warnings"].append(
                "Rows identical in every column were treated as duplicates. If your business "
                "can record two identical sales on one day, check your source file."
            )
        if s["customers"] < 20:
            s["warnings"].append(
                "Fewer than 20 customers remain. Clustering works best with many more customers."
            )