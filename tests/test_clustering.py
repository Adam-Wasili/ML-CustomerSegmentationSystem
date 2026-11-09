"""Tests for the K-Means service and the naming of segments. No web server needed."""
import numpy as np
import pandas as pd
import pytest

from services.behavioral_analytics import BehavioralAnalytics
from services.clustering import MIN_CUSTOMERS, MLClustering, MLClusteringError
from services.rfm_analysis import RFMAnalysis


def three_groups():
    """90 customers in three clearly different behaviour groups (30 each)."""
    rng = np.random.default_rng(0)
    rows = []
    for recency, frequency, monetary in [(5, 20, 100000), (100, 5, 20000), (300, 1, 3000)]:
        for _ in range(30):
            rows.append({
                "recency": max(0, int(rng.normal(recency, recency * 0.1 + 1))),
                "frequency": max(1, int(rng.normal(frequency, frequency * 0.1 + 0.3))),
                "monetary": max(100.0, float(rng.normal(monetary, monetary * 0.1))),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------- preparing features
def test_features_are_scaled():
    features = MLClustering().prepare_features(three_groups())
    assert np.allclose(features.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(features.std(axis=0), 1)


def test_frequency_gets_a_log_transform():
    # With a log transform the gap between 1 and 10 is not tiny compared with 10 and 100
    table = pd.DataFrame({"recency": range(10), "frequency": [1, 10, 100] + [5] * 7,
                          "monetary": [100.0] * 10})
    features = MLClustering().prepare_features(table)
    first_gap = features[1, 1] - features[0, 1]
    second_gap = features[2, 1] - features[1, 1]
    assert first_gap / second_gap > 0.5          # without the log it would be about 0.1


def test_too_few_customers_is_rejected():
    table = pd.DataFrame({"recency": [1] * 5, "frequency": [1] * 5, "monetary": [10.0] * 5})
    with pytest.raises(MLClusteringError):
        MLClustering().prepare_features(table)
    assert MIN_CUSTOMERS == 10


# --------------------------------------------------------- choosing k
def test_evaluation_covers_every_k():
    clustering = MLClustering()
    evaluation = clustering.evaluate_k(clustering.prepare_features(three_groups()))
    assert [row["k"] for row in evaluation] == list(range(2, 11))
    assert all(row["inertia"] > 0 for row in evaluation)


def test_recommendation_finds_the_three_groups():
    clustering = MLClustering()
    evaluation = clustering.evaluate_k(clustering.prepare_features(three_groups()))
    assert clustering.recommend_k(evaluation) == 3


# --------------------------------------------------------- K-Means itself
def test_every_customer_gets_a_cluster():
    clustering = MLClustering()
    features = clustering.prepare_features(three_groups())
    outcome = clustering.perform_clustering(features, 3)
    assert len(outcome["labels"]) == 90
    assert set(outcome["labels"]) == {0, 1, 2}
    assert outcome["silhouette"] > 0.5            # clearly separated groups


def test_same_data_gives_the_same_clusters():
    table = three_groups()
    first = MLClustering()
    second = MLClustering()
    labels_one = first.perform_clustering(first.prepare_features(table), 3)["labels"]
    labels_two = second.perform_clustering(second.prepare_features(table), 3)["labels"]
    assert (labels_one == labels_two).all()


def test_invalid_number_of_clusters_is_rejected():
    clustering = MLClustering()
    features = clustering.prepare_features(three_groups())
    with pytest.raises(MLClusteringError):
        clustering.perform_clustering(features, 1)
    with pytest.raises(MLClusteringError):
        clustering.perform_clustering(features, 11)


def test_cluster_statistics_use_the_original_values():
    table = RFMAnalysis().score_rfm(three_groups())
    labels = np.repeat([0, 1, 2], 30)             # the three groups, in the order they were built
    profile = MLClustering.profile_clusters(table, labels)
    assert profile["customers"].tolist() == [30, 30, 30]
    assert profile["percent"].sum() == pytest.approx(100)
    assert profile.loc[0, "avg_recency"] == pytest.approx(table["recency"][:30].mean())
    assert profile.loc[2, "avg_monetary"] == pytest.approx(table["monetary"][60:].mean())


# --------------------------------------------------- naming the segments
def profile_of(*scores):
    """A cluster table built from (r, f, m) average scores."""
    rows = []
    for number, (r, f, m) in enumerate(scores):
        rows.append({"cluster": number, "customers": 10, "percent": 20.0,
                     "avg_recency": 30.0, "avg_frequency": 5.0, "avg_monetary": 1000.0,
                     "avg_r_score": r, "avg_f_score": f, "avg_m_score": m})
    return pd.DataFrame(rows)


def test_names_come_from_the_measured_scores():
    named = BehavioralAnalytics().identify_characteristics(
        profile_of((5, 5, 5), (1, 4, 4), (1, 1, 1), (3, 1.5, 1.5), (4, 3.5, 3.5)))
    assert named["segment_name"].tolist() == [
        "High-Value / Loyal Customers", "At-Risk Customers", "Inactive Customers",
        "Occasional Customers", "Regular Customers"]


def test_the_name_does_not_depend_on_the_cluster_number():
    low_first = BehavioralAnalytics().identify_characteristics(profile_of((1, 1, 1), (5, 5, 5)))
    high_first = BehavioralAnalytics().identify_characteristics(profile_of((5, 5, 5), (1, 1, 1)))
    assert low_first["segment_name"].tolist() == ["Inactive Customers", "High-Value / Loyal Customers"]
    assert high_first["segment_name"].tolist() == ["High-Value / Loyal Customers", "Inactive Customers"]


def test_duplicate_names_are_made_unique():
    named = BehavioralAnalytics().identify_characteristics(profile_of((5, 5, 5), (5, 5, 4.8)))
    assert len(set(named["segment_name"])) == 2
    assert named["segment_name"][0].endswith("(Cluster 0)")


def test_descriptions_describe_but_do_not_predict():
    named = BehavioralAnalytics().identify_characteristics(
        profile_of((5, 5, 5), (1, 4, 4), (1, 1, 1), (3, 1.5, 1.5), (4, 3.5, 3.5)))
    for description in named["description"]:
        assert "predict" not in description.lower()
    assert "low Recency" in named["description"][0]       # recent customers have LOW Recency (days)