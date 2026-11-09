"""Behavioural analytics (Module 5, started in Step 4).

For now this file turns the measured statistics of each cluster into a business
name and a plain description. Step 5 adds the analytics page that uses it.

The names are NOT tied to cluster numbers. They are chosen only after looking at the
cluster's average RFM scores (1-5, where 5 is best: recent, frequent, high spending).
The descriptions describe past behaviour only; they do not predict anything.
"""
import pandas as pd

HIGH_SCORE = 3.5     # an average score at or above this is called "high"
LOW_SCORE = 2.5      # an average score at or below this is called "low"

# What each segment means for the business (kept free of predictions)
INTERPRETATIONS = {
    "High-Value / Loyal Customers":
        "This indicates a highly active and valuable group that is worth rewarding and keeping.",
    "Regular Customers":
        "This indicates a steady group that buys fairly regularly; staying in touch helps keep this activity.",
    "Occasional Customers":
        "This indicates customers who buy rarely and spend little; encouraging repeat purchases could be considered.",
    "At-Risk Customers":
        "This indicates customers who used to buy often or spend well but have not bought recently; "
        "contacting them to find out why could be considered.",
    "Inactive Customers":
        "This indicates customers who have not bought recently and bought little; they can be treated as lapsed.",
}


class BehavioralAnalytics:
    """Finds the characteristics of each customer segment."""

    def identify_characteristics(self, profile):
        """Add a segment_name and a description to every cluster in the profile table."""
        result = profile.copy()
        base_names = [self._choose_name(row.avg_r_score, row.avg_f_score, row.avg_m_score)
                      for row in result.itertuples(index=False)]

        # Two clusters can deserve the same name; add the cluster number so names stay unique
        counts = pd.Series(base_names).value_counts()
        result["segment_name"] = [
            f"{name} (Cluster {cluster})" if counts[name] > 1 else name
            for name, cluster in zip(base_names, result["cluster"])
        ]
        result["description"] = [
            self._describe(row, name)
            for row, name in zip(result.itertuples(index=False), base_names)
        ]
        return result

    @staticmethod
    def _choose_name(r_score, f_score, m_score):
        """Pick a name from the average scores of a cluster (5 = recent, frequent, high spending)."""
        spending = (f_score + m_score) / 2          # how much the group buys and spends
        overall = (r_score + f_score + m_score) / 3

        if r_score <= LOW_SCORE:                    # has not bought recently
            return "At-Risk Customers" if spending >= 3.0 else "Inactive Customers"
        if overall >= 4.2:                          # recent, frequent and high spending
            return "High-Value / Loyal Customers"
        if spending >= 2.5:
            return "Regular Customers"
        return "Occasional Customers"

    @staticmethod
    def _level(score, reverse=False):
        """Turn an average score into the word low, moderate or high."""
        if score >= HIGH_SCORE:
            word = "high"
        elif score <= LOW_SCORE:
            word = "low"
        else:
            word = "moderate"
        if reverse:                                  # a HIGH recency score means LOW Recency (days)
            word = {"high": "low", "low": "high", "moderate": "moderate"}[word]
        return word

    def _describe(self, row, base_name):
        """A short description built from the measured values of the cluster."""
        recency_word = self._level(row.avg_r_score, reverse=True)
        frequency_word = self._level(row.avg_f_score)
        monetary_word = self._level(row.avg_m_score)
        return (
            f"Cluster {row.cluster} contains {row.customers} customers ({row.percent:.1f}% of all customers). "
            f"Their last purchase was on average {row.avg_recency:.0f} days before the analysis date "
            f"({recency_word} Recency); they made {row.avg_frequency:.1f} transactions on average "
            f"({frequency_word} Frequency) and spent {row.avg_monetary:,.0f} on average "
            f"({monetary_word} Monetary value). {INTERPRETATIONS[base_name]}"
        )