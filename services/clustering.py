"""ML clustering (Module 4): K-Means on the RFM features.

RFM (the previous module) only DESCRIBES each customer with three numbers.
This module is the machine-learning step: K-Means, an unsupervised algorithm,
groups customers whose Recency, Frequency and Monetary values are alike.

This file has no Flask or database code, so it can be tested on its own.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_CUSTOMERS = 10               # fewer customers cannot be grouped in a meaningful way
MIN_K = 2                        # smallest number of clusters tried
MAX_K = 10                       # largest number of clusters tried
RANDOM_STATE = 42                # fixed, so the same data always gives the same clusters
N_INIT = 10                      # K-Means is started 10 times and the best result is kept
SILHOUETTE_SAMPLE_SIZE = 5000    # silhouette on huge datasets is slow, so a sample is used


class MLClusteringError(Exception):
    """A problem with the clustering. The message is safe to show to users."""


class MLClustering:
    """Prepares the RFM features, chooses the number of clusters and runs K-Means."""

    algorithm = "K-Means"

    def __init__(self):
        self.scaler = None

    # ------------------------------------------------------------ preparing
    def prepare_features(self, rfm):
        """Turn Recency, Frequency and Monetary into scaled features.

        1. Frequency and Monetary are skewed (a few customers have very large values),
           so log(1 + value) is applied. Recency is not skewed in the same way.
        2. All three features are standardized (mean 0, spread 1).
           Scaling is necessary because K-Means works with distances: without it,
           Monetary (thousands) would decide the groups almost alone, while
           Recency (days) and Frequency (small counts) would hardly matter.
        """
        if len(rfm) < MIN_CUSTOMERS:
            raise MLClusteringError(
                f"At least {MIN_CUSTOMERS} customers are needed for clustering. "
                f"This dataset has {len(rfm)}."
            )
        features = pd.DataFrame({
            "recency": rfm["recency"].astype(float).to_numpy(),
            "frequency": np.log1p(rfm["frequency"].astype(float).to_numpy()),
            "monetary": np.log1p(rfm["monetary"].astype(float).to_numpy()),
        })
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(features)

    @staticmethod
    def max_clusters(features):
        """The largest k that makes sense for this data."""
        distinct = len(np.unique(features, axis=0))     # customers with exactly equal values count once
        limit = min(MAX_K, len(features) - 1, distinct)
        if limit < MIN_K:
            raise MLClusteringError("The customers behave too similarly to be split into groups.")
        return limit

    # ------------------------------------------------ choosing the number of clusters
    def evaluate_k(self, features):
        """Try every k and record the elbow (inertia) and silhouette values.

        inertia    : how tightly the customers sit around their cluster centres (lower is tighter).
                     The 'elbow' is where adding more clusters stops helping much.
        silhouette : from -1 to 1, higher means clusters are more clearly separated.
        """
        results = []
        for k in range(MIN_K, self.max_clusters(features) + 1):
            model = self.train_model(features, k)
            results.append({"k": k, "inertia": float(model.inertia_),
                            "silhouette": self._silhouette(features, model.labels_)})
        return results

    @staticmethod
    def recommend_k(evaluation):
        """The k with the best silhouette score (the smaller k wins a tie)."""
        scored = [row for row in evaluation if row["silhouette"] is not None]
        if not scored:
            return MIN_K
        return max(scored, key=lambda row: (row["silhouette"], -row["k"]))["k"]

    @staticmethod
    def _silhouette(features, labels):
        """Silhouette score, or None when it cannot be calculated."""
        if len(set(labels)) < 2:
            return None
        sample = min(len(features), SILHOUETTE_SAMPLE_SIZE)
        return float(silhouette_score(features, labels, sample_size=sample,
                                      random_state=RANDOM_STATE))

    # ---------------------------------------------------------- clustering
    def train_model(self, features, k):
        """Fit K-Means with k clusters."""
        return KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(features)

    def perform_clustering(self, features, k):
        """Cluster the customers into k groups. Returns labels, inertia and silhouette."""
        limit = self.max_clusters(features)
        if not MIN_K <= k <= limit:
            raise MLClusteringError(f"Choose a number of clusters between {MIN_K} and {limit}.")
        model = self.train_model(features, k)
        return {
            "k": k,
            "labels": model.labels_,                    # one cluster number per customer
            "inertia": float(model.inertia_),
            "silhouette": self._silhouette(features, model.labels_),
        }

    @staticmethod
    def profile_clusters(rfm, labels):
        """Statistics of each cluster, calculated from the ORIGINAL (unscaled) RFM values."""
        data = rfm[["recency", "frequency", "monetary", "r_score", "f_score", "m_score"]].copy()
        data["cluster"] = np.asarray(labels)
        profile = data.groupby("cluster").agg(
            customers=("recency", "size"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            avg_r_score=("r_score", "mean"),
            avg_f_score=("f_score", "mean"),
            avg_m_score=("m_score", "mean"),
        ).reset_index()
        profile["percent"] = profile["customers"] / len(data) * 100
        return profile