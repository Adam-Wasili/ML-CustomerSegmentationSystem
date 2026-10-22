"""RFM analysis (Module 3): feature engineering for customer behaviour.

RFM is NOT the machine-learning step. It turns raw transactions into three
numbers per customer. The ML clustering module (next step) uses them as input.

    Recency   = days since the customer's most recent transaction
    Frequency = number of transactions the customer made
    Monetary  = total amount the customer spent

This file contains no Flask or database code, so it can be tested on its own.
"""
import numpy as np
import pandas as pd

SCORE_LEVELS = 5                          # RFM scores run from 1 (lowest) to 5 (highest)
MAX_DAYS_AFTER_LAST_TRANSACTION = 3650    # about 10 years; a later date is almost surely a typing error


class RFMError(Exception):
    """A problem with the RFM calculation. The message is safe to show to users."""


class RFMAnalysis:
    """Calculates Recency, Frequency, Monetary and their 1-5 scores.

    Methods expect a DataFrame of transactions with the columns:
    customer_id, transaction_date (datetime) and amount (number).
    """

    # ------------------------------------------------------------ reference date
    def resolve_reference_date(self, transactions, reference_date=None):
        """Return the analysis date as a Timestamp.

        Default: the latest transaction date plus one day. A custom date is
        accepted only if it is not before the latest transaction (otherwise
        some Recency values would be negative).
        """
        if transactions.empty:
            raise RFMError("There are no transactions to analyse.")

        latest = transactions["transaction_date"].max().normalize()
        if reference_date is None:
            return latest + pd.Timedelta(days=1)

        try:
            chosen = pd.Timestamp(reference_date).normalize()
        except (ValueError, OverflowError) as error:
            raise RFMError("The analysis date is not a valid date.") from error

        if chosen < latest:
            raise RFMError(
                "The analysis date cannot be earlier than the latest transaction date "
                f"({latest:%Y-%m-%d})."
            )
        if chosen > latest + pd.Timedelta(days=MAX_DAYS_AFTER_LAST_TRANSACTION):
            raise RFMError("The analysis date is more than 10 years after the latest transaction.")
        return chosen

    # ------------------------------------------------------------ R, F and M
    def calculate_recency(self, transactions, reference_date):
        """Days between each customer's latest transaction and the analysis date."""
        last_purchase = transactions.groupby("customer_id")["transaction_date"].max()
        return (reference_date - last_purchase).dt.days.astype(int)

    def calculate_frequency(self, transactions):
        """Number of transactions per customer (rows with the same customer_id are one customer)."""
        return transactions.groupby("customer_id").size().astype(int)

    def calculate_monetary(self, transactions):
        """Total amount spent per customer."""
        return transactions.groupby("customer_id")["amount"].sum().astype(float)

    def calculate_rfm(self, transactions, reference_date):
        """Combine R, F and M into one table with one row per customer."""
        missing = {"customer_id", "transaction_date", "amount"} - set(transactions.columns)
        if missing:
            raise RFMError("The transaction data is incomplete.")
        if transactions.empty:
            raise RFMError("There are no transactions to analyse.")

        rfm = pd.DataFrame({
            "recency": self.calculate_recency(transactions, reference_date),
            "frequency": self.calculate_frequency(transactions),
            "monetary": self.calculate_monetary(transactions),
        })
        rfm.index.name = "customer_id"
        return rfm.reset_index()

    # --------------------------------------------------------------- scoring
    def score_rfm(self, rfm):
        """Add r_score, f_score and m_score (1 = lowest, 5 = best).

        Recency is reversed: the most recent customers get 5.
        """
        if rfm.empty:
            raise RFMError("There are no customers to score.")
        scored = rfm.copy()
        scored["r_score"] = self._score(rfm["recency"], higher_is_better=False)
        scored["f_score"] = self._score(rfm["frequency"], higher_is_better=True)
        scored["m_score"] = self._score(rfm["monetary"], higher_is_better=True)
        return scored

    @staticmethod
    def _score(values, higher_is_better):
        """Rank customers and split them into five equal groups.

        Tied values share an average rank, so customers with the SAME value
        always receive the SAME score (important for Frequency, where many
        customers have exactly one transaction). The rounding step prevents
        tiny floating point errors from pushing a customer into the wrong group.
        """
        count = len(values)
        ranks = (values if higher_is_better else -values).rank(method="average")
        scores = np.ceil(np.round(ranks * SCORE_LEVELS / count, 9))
        return scores.clip(1, SCORE_LEVELS).astype(int)

    # ------------------------------------------------------------- summary
    def summarise(self, rfm):
        """Totals, averages and score counts for the RFM page."""
        if rfm.empty:
            raise RFMError("There are no RFM results to summarise.")

        def describe(series):
            return {
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "min": float(series.min()),
                "max": float(series.max()),
            }

        # How many customers received each score (1-5) for R, F and M
        distribution = {}
        for letter, column in (("r", "r_score"), ("f", "f_score"), ("m", "m_score")):
            counts = rfm[column].value_counts()
            distribution[letter] = {
                level: int(counts.get(level, 0)) for level in range(1, SCORE_LEVELS + 1)
            }

        return {
            "customers": int(len(rfm)),
            "transactions": int(rfm["frequency"].sum()),
            "total_monetary": float(rfm["monetary"].sum()),
            "recency": describe(rfm["recency"]),
            "frequency": describe(rfm["frequency"]),
            "monetary": describe(rfm["monetary"]),
            "score_distribution": distribution,
        }