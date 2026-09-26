"""
Chapter 8 -- Which Variables Matter
===================================

Compass: a feature that looks most informative by raw mutual
information is not the same as the most VALUABLE feature, once you
account for how much complexity (entropy) it costs to encode it.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import rank_features, quadrant_report
from robustkit.information.visualization import plot_feature_space


def make_churn_data(seed=51, n=2000):
    """
    A customer-churn dataset with four candidate features:

    - region: 4 categories, genuinely predictive of churn.
    - zip_code: 48 categories, each nested within exactly one region --
      carries almost the SAME underlying signal as region, just spread
      thinly across twelve times as many categories.
    - tenure_months: continuous, genuinely predictive, moderate
      cardinality once binned.
    - signup_channel: 3 categories, unrelated to churn -- pure noise.
    """
    rng = np.random.default_rng(seed)

    region = rng.choice(["North", "South", "East", "West"], n)
    region_churn_bump = {"North": 0.08, "South": 0.40, "East": 0.22, "West": 0.14}

    zip_map = {r: [f"{r[0]}{i:02d}" for i in range(12)] for r in region_churn_bump}
    zip_code = np.array([rng.choice(zip_map[r]) for r in region])

    tenure = rng.uniform(1, 60, n)
    signup_channel = rng.choice(["Web", "Referral", "Phone"], n)

    churn_prob = np.array([region_churn_bump[r] for r in region]) + (60 - tenure) / 60 * 0.35
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churned = (rng.uniform(0, 1, n) < churn_prob).astype(int)

    return pd.DataFrame({
        "zip_code": zip_code, "region": region, "tenure_months": tenure,
        "signup_channel": signup_channel, "churned": churned,
    })


def main():
    df = make_churn_data()
    print(f"Rows: {len(df)}, churn rate: {df['churned'].mean():.1%}")
    print(f"zip_code: {df['zip_code'].nunique()} distinct values, region: {df['region'].nunique()} distinct values")

    # --- Rank every feature by mutual information and efficiency ---
    ranking = rank_features(df, target="churned")
    print("\n=== rank_features (sorted by information_efficiency) ===")
    print(ranking.to_string(index=False))

    print(
        "\nNotice zip_code: it has the HIGHEST raw mutual information of"
        "\nany feature here -- higher even than region. Read only that"
        "\nnumber, and zip_code looks like the star of the dataset. But"
        "\nits information_efficiency is barely half of region's, because"
        "\nit costs 5.6 bits of entropy (48 categories) to deliver only a"
        "\nlittle more predictive power than region's 2.0 bits (4"
        "\ncategories) already provide on its own."
    )

    # --- Classify into quadrants ---
    quad = quadrant_report(ranking=ranking)
    print("\n=== quadrant_report ===")
    print(quad[["feature", "mutual_information", "information_efficiency", "quadrant"]].to_string(index=False))

    print(
        "\nregion lands as a Star (high MI, high efficiency) -- the"
        "\nclearest case of a feature worth keeping as-is. zip_code lands"
        "\nas Predictive: genuinely informative, but expensive, in the"
        "\nsense that most of what it tells you is already available more"
        "\ncheaply from region. tenure_months is Efficient: a modest but"
        "\ncompact contributor. signup_channel is Weak on both axes --"
        "\ncorrectly identified as noise."
    )

    # --- Chart: information space ---
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_feature_space(ranking=quad, ax=ax)
    ax.set_title("Information space: efficiency vs. mutual information")
    fig.tight_layout()
    fig.savefig("information_space.png", dpi=150)
    print("\nChart saved to information_space.png")


if __name__ == "__main__":
    main()
