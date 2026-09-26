"""
Chapter 10 -- Finding Outliers Correctly
===================================

Compass: a naive mean/standard-deviation outlier detector can be
corrupted by the very outliers it's trying to find -- a phenomenon
called MASKING -- while a median/MAD-based detector stays stable.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import mad_outlier_report


def make_salary_data_with_outliers(seed=71, n=200):
    """
    A clean salary-vs-age dataset, with four genuine, injected
    underpayment cases: three moderate ($6,000-$7,000 below trend) and
    one severe ($25,000 below trend). The severe one is deliberately
    the largest -- exactly the kind of case that should be the easiest
    to catch.
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(25, 60, n)
    salary = 30000 + 400 * age + rng.normal(0, 800, n)
    df = pd.DataFrame({"JobFamily": ["ENG"] * n, "age": age, "salary": salary,
                        "emp_id": [f"E{i}" for i in range(n)]})

    true_outliers = []
    for i, drop in enumerate([6000, 6500, 7000]):
        df.loc[i, "salary"] -= drop
        true_outliers.append(df.loc[i, "emp_id"])
    df.loc[3, "salary"] -= 25000  # the severe case
    true_outliers.append(df.loc[3, "emp_id"])

    return df, true_outliers


def main():
    df, true_outliers = make_salary_data_with_outliers()
    print(f"n={len(df)}, true injected outliers: {true_outliers}")

    x = df["age"].to_numpy(dtype=float)
    y = df["salary"].to_numpy(dtype=float)

    # --- The naive approach: fit a trend, flag by mean +/- k*std of the residuals ---
    coeffs = np.polyfit(x, y, 1)
    expected_naive = np.polyval(coeffs, x)
    residual_naive = y - expected_naive
    mean_r, std_r = residual_naive.mean(), residual_naive.std()
    k = 3.0
    flagged_naive = set(df["emp_id"][np.abs(residual_naive - mean_r) > k * std_r])

    print(f"\n=== Naive: flag if |residual - mean| > {k} x std ===")
    print(f"mean(residual)={mean_r:,.0f}, std(residual)={std_r:,.0f}")
    print(f"Flagged: {sorted(flagged_naive)}")
    missed = set(true_outliers) - flagged_naive
    print(f"MISSED: {sorted(missed)}" if missed else "(caught everything)")

    # --- The robust approach: mad_outlier_report (median/MAD-based) ---
    report = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                                 min_size=20, k=k, id_cols=["emp_id"])
    flagged_robust = set(report[report["flagged"]]["emp_id"])

    print(f"\n=== Robust: mad_outlier_report, same k={k} ===")
    print(f"Flagged: {sorted(flagged_robust)}")
    missed_robust = set(true_outliers) - flagged_robust
    print(f"MISSED: {sorted(missed_robust)}" if missed_robust else "(caught everything)")

    print(
        "\nThe naive method misses E0 -- a genuine $6,000 underpayment,"
        "\nlarger than typical noise in this data -- because E3's $25,000"
        "\ncase inflated the standard deviation enough to push E0 back"
        "\nunder the 3-sigma line. One extreme outlier masked another,"
        "\nsmaller one. The robust method, using the median and MAD"
        "\ninstead, isn't dragged around by E3 the same way, and catches"
        "\nall four."
    )

    # --- Chart: residuals with both thresholds shown ---
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["red" if e in true_outliers else "steelblue" for e in df["emp_id"]]
    ax.scatter(x, residual_naive, c=colors, s=25, alpha=0.7, zorder=3)

    ax.axhline(mean_r + k * std_r, color="orange", linestyle="--", label=f"Naive threshold (mean +/- {k}\u00d7std)")
    ax.axhline(mean_r - k * std_r, color="orange", linestyle="--")

    median_r = float(np.median(residual_naive))
    mad_r = float(1.4826 * np.median(np.abs(residual_naive - median_r)))
    ax.axhline(median_r + k * mad_r, color="green", linestyle="-", label=f"Robust threshold (median +/- {k}\u00d7MAD)")
    ax.axhline(median_r - k * mad_r, color="green", linestyle="-")

    for e in true_outliers:
        row = df[df["emp_id"] == e].iloc[0]
        ax.annotate(e, (row["age"], row["salary"] - np.polyval(coeffs, row["age"])),
                    fontsize=9, xytext=(6, 0), textcoords="offset points")

    ax.set_xlabel("age")
    ax.set_ylabel("residual from trend ($)")
    ax.set_title("Naive threshold moves to accommodate the extreme case; robust threshold doesn't")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("masking_effect.png", dpi=150)
    print("\nChart saved to masking_effect.png")


if __name__ == "__main__":
    main()
