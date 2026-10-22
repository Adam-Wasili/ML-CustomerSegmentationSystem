"""Tests for the RFM calculation service. No web server needed."""
import pandas as pd
import pytest

from services.rfm_analysis import RFMAnalysis, RFMError

# The example from the project description: C003 made three purchases
SAMPLE = [
    ["C003", "2026-02-10", 20000], ["C003", "2026-02-15", 15000], ["C003", "2026-03-01", 25000],
    ["C008", "2026-02-12", 30000], ["C008", "2026-03-05", 15000],
]


def make_transactions(rows):
    """Build the transaction table the way the route does."""
    table = pd.DataFrame(rows, columns=["customer_id", "transaction_date", "amount"])
    table["transaction_date"] = pd.to_datetime(table["transaction_date"])
    return table


def rfm_for(rows, reference_date=None):
    """Return the RFM table (indexed by customer) and the analysis date."""
    analysis = RFMAnalysis()
    transactions = make_transactions(rows)
    analysis_date = analysis.resolve_reference_date(transactions, reference_date)
    rfm = analysis.calculate_rfm(transactions, analysis_date)
    return rfm.set_index("customer_id"), analysis_date


def hundred_customers():
    """100 customers with all-different values, easy to check by hand."""
    return pd.DataFrame({
        "customer_id": range(100),
        "recency": range(1, 101),                              # customer 0 is the most recent
        "frequency": range(1, 101),
        "monetary": [1000 * i for i in range(1, 101)],
    })


# ------------------------------------------------- R, F and M calculation
def test_frequency_and_monetary_per_customer():
    rfm, _ = rfm_for(SAMPLE)
    assert rfm.loc["C003", "frequency"] == 3
    assert rfm.loc["C003", "monetary"] == 60000
    assert rfm.loc["C008", "frequency"] == 2
    assert rfm.loc["C008", "monetary"] == 45000


def test_each_customer_appears_once():
    rfm, _ = rfm_for(SAMPLE)
    assert len(rfm) == 2


def test_default_reference_date_is_last_transaction_plus_one_day():
    _, analysis_date = rfm_for(SAMPLE)
    assert analysis_date == pd.Timestamp("2026-03-06")


def test_recency_from_default_reference_date():
    rfm, _ = rfm_for(SAMPLE)
    assert rfm.loc["C003", "recency"] == 5      # last purchase 1 March
    assert rfm.loc["C008", "recency"] == 1      # last purchase 5 March


def test_custom_reference_date_changes_recency():
    rfm, analysis_date = rfm_for(SAMPLE, "2026-03-10")
    assert analysis_date == pd.Timestamp("2026-03-10")
    assert rfm.loc["C003", "recency"] == 9
    assert rfm.loc["C008", "recency"] == 5


def test_reference_date_equal_to_last_transaction_is_allowed():
    rfm, _ = rfm_for(SAMPLE, "2026-03-05")
    assert rfm.loc["C008", "recency"] == 0


def test_reference_date_before_last_transaction_is_rejected():
    with pytest.raises(RFMError):
        rfm_for(SAMPLE, "2026-03-01")


def test_reference_date_far_in_the_future_is_rejected():
    with pytest.raises(RFMError):
        rfm_for(SAMPLE, "2099-01-01")


def test_no_transactions_is_rejected():
    with pytest.raises(RFMError):
        RFMAnalysis().resolve_reference_date(make_transactions([]))


# ---------------------------------------------------------------- scoring
def test_scores_split_100_unique_customers_into_five_equal_groups():
    scored = RFMAnalysis().score_rfm(hundred_customers())
    for column in ("r_score", "f_score", "m_score"):
        assert set(scored[column]) == {1, 2, 3, 4, 5}
        assert (scored[column].value_counts() == 20).all()


def test_recency_score_is_reversed():
    scored = RFMAnalysis().score_rfm(hundred_customers())
    assert scored.loc[0, "r_score"] == 5        # most recent customer
    assert scored.loc[99, "r_score"] == 1       # least recent customer


def test_frequency_and_monetary_scores_grow_with_the_value():
    scored = RFMAnalysis().score_rfm(hundred_customers())
    assert scored.loc[99, "f_score"] == 5 and scored.loc[0, "f_score"] == 1
    assert scored.loc[99, "m_score"] == 5 and scored.loc[0, "m_score"] == 1


def test_identical_values_get_identical_scores():
    # 50 customers all made exactly one purchase: they must share one Frequency score
    table = pd.DataFrame({
        "customer_id": range(50), "recency": range(1, 51),
        "frequency": [1] * 50, "monetary": [1000 * i for i in range(1, 51)],
    })
    scored = RFMAnalysis().score_rfm(table)
    assert scored["f_score"].nunique() == 1


def test_scoring_works_with_fewer_than_five_customers():
    table = pd.DataFrame({"customer_id": [1, 2, 3], "recency": [1, 2, 3],
                          "frequency": [1, 2, 3], "monetary": [100, 200, 300]})
    scored = RFMAnalysis().score_rfm(table)
    assert scored[["r_score", "f_score", "m_score"]].min().min() >= 1
    assert scored[["r_score", "f_score", "m_score"]].max().max() <= 5
    assert scored["r_score"].tolist() == [5, 4, 2]


def test_scoring_works_with_one_customer():
    table = pd.DataFrame({"customer_id": [1], "recency": [3], "frequency": [2], "monetary": [500]})
    scored = RFMAnalysis().score_rfm(table)
    assert scored.loc[0, "r_score"] == 5


# ---------------------------------------------------------------- summary
def test_summary_numbers():
    scored = RFMAnalysis().score_rfm(hundred_customers())
    summary = RFMAnalysis().summarise(scored)
    assert summary["customers"] == 100
    assert summary["transactions"] == sum(range(1, 101))
    assert summary["monetary"]["max"] == 100000.0
    assert summary["total_monetary"] == sum(1000 * i for i in range(1, 101))
    assert sum(summary["score_distribution"]["r"].values()) == 100