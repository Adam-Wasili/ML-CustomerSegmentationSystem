"""Behavioural analytics (Module 5).

This file does two jobs:
  1. identify_characteristics: turns the measured statistics of each cluster into a business
     name and a plain description (the names are chosen from the cluster's average RFM scores,
     never from the cluster number).
  2. analyze_behavior / generate_observations: adds each segment's share of the revenue and
     writes the key observations shown on the analytics page and in the report.

In the class diagram, generateSegmentProfile corresponds to the description built in _describe.
The descriptions describe past behaviour only; they do not predict anything.
"""
import pandas as pd

HIGH_SCORE = 3.5     # an average score at or above this is called "high"
LOW_SCORE = 2.5      # an average score at or below this is called "low"
VERY_LOW_SCORE = 1.5     # an average score at or below this is called "very low"

# What each segment means for the business (kept free of predictions)
INTERPRETATIONS = {
    "High-value/Loyal customers":
        "This indicates a highly active and valuable group that is worth rewarding and keeping.",
    "Regular customers":
        "This indicates a steady group that buys fairly regularly; staying in touch helps keep this activity.",
    "Occasional customers":
        "This indicates customers who buy rarely and spend little; encouraging repeat purchases could be considered.",
    "Moderate Active Customers":
        "This indicates customers who used to buy often or spend well but have not bought recently; "
        "contacting them to find out why could be considered.",
    "Inactive Customers":
        "This indicates a group with a long gap since their last purchase and below-average buying "
        "frequency. Check their monetary value before deciding how much effort to spend on winning them back.",
    "At-Risk Customers":
        "This indicates the group with the longest gap since the last purchase. They are the least active "
        "customers; review their value before choosing between a win-back offer and a low-cost reminder.",
}


class BehavioralAnalytics:
    """Finds the characteristics and the business meaning of each customer segment."""

    # ---------------------------------------------- names and descriptions
    def identify_characteristics(self, profile):
        """Add a segment_name and a description to every cluster in the profile table."""
        result = profile.copy()
        base_names = [self._choose_name(row.avg_r_score, row.avg_f_score, row.avg_m_score)
                      for row in result.itertuples(index=False)]

        # The inactive cluster with the longest gap (lowest R+F) is also "Moderate Active Customers".
        # The choice comes from the scores, not from the cluster number.
        inactive = [index for index, name in enumerate(base_names) if name == "Inactive Customers"]
        if len(inactive) > 1:
            worst = min(inactive, key=lambda index: (
                result.iloc[index]["avg_r_score"] + result.iloc[index]["avg_f_score"],
                result.iloc[index]["avg_r_score"]))
            base_names[worst] = "Moderate Active Customers"

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

        if r_score <= VERY_LOW_SCORE and spending < 2.0:                    # has not bought recently
            return "At-Risk Customers"
        if r_score <= LOW_SCORE:
            return "Moderate Active Customers" if spending >= 3.0 else "Inactive Customers"
        if overall >= 4.2:                          # recent, frequent and high spending
            return "High-value/Loyal customers"
        if spending >= 2.5:
            return "Regular customers"
        return "Occasional customers"

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

    # -------------------------------------------------- analysing the segments
    @staticmethod
    def analyze_behavior(segments):
        """Add each segment's total spending and its share of all spending.

        segments is a list of dictionaries (name, customers, percent, avg_recency,
        avg_frequency, avg_monetary ...). The average times the number of customers gives
        the exact total, so no extra data is needed.
        """
        totals = [segment["customers"] * segment["avg_monetary"] for segment in segments]
        overall = sum(totals)
        analysed = []
        for segment, total in zip(segments, totals):
            row = dict(segment)
            row["total_monetary"] = total
            row["revenue_share"] = (total / overall * 100) if overall else 0.0
            analysed.append(row)
        return analysed

    @staticmethod
    def generate_observations(analysed):
        """Plain sentences about the most important differences between the segments."""
        if not analysed:
            return []
        largest = max(analysed, key=lambda s: s["customers"])
        top_revenue = max(analysed, key=lambda s: s["revenue_share"])
        most_recent = min(analysed, key=lambda s: s["avg_recency"])
        most_frequent = max(analysed, key=lambda s: s["avg_frequency"])
        return [
            f"The largest segment is {largest['name']} with {largest['customers']} customers "
            f"({largest['percent']:.1f}% of all customers).",
            f"{top_revenue['name']} brings in the most revenue: {top_revenue['revenue_share']:.1f}% of the "
            f"total, from {top_revenue['percent']:.1f}% of the customers.",
            f"{most_recent['name']} has the most recent buyers (on average {most_recent['avg_recency']:.0f} "
            f"days since the last purchase).",
            f"{most_frequent['name']} buys most often (on average {most_frequent['avg_frequency']:.1f} "
            f"transactions per customer).",
        ]