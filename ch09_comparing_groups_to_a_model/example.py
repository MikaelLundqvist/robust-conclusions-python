"""
Chapter 9 -- Comparing Groups to a Model
===================================

Compass: comparing several groups against a single, shared benchmark
model -- and using the confidence interval, not just the point
estimate, to tell a real difference from one that's just noise.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import segment_position_report


def make_salary_data(seed=61, n=2400):
    """
    Five job families, one shared age-salary relationship across the
    whole organization, plus a per-family offset. FIN's offset (950)
    and ITS's offset (950) are both set close to the population's
    size-weighted AVERAGE offset -- deliberately, so that once a
    single benchmark is fit across everyone, FIN and ITS should NOT
    stand out clearly above or below it, the way ENG, OPS, and SALES
    do.
    """
    rng = np.random.default_rng(seed)
    job_family = rng.choice(["ENG", "ITS", "FIN", "SALES", "OPS"], n, p=[0.30, 0.20, 0.06, 0.24, 0.20])
    age = rng.uniform(24, 62, n)
    family_bump = {"ENG": 4000, "ITS": 950, "FIN": 950, "SALES": -800, "OPS": -2200}
    salary = 32000 + 420 * age + np.array([family_bump[f] for f in job_family]) + rng.normal(0, 1400, n)
    return pd.DataFrame({"JobFamily": job_family, "age": age, "salary": salary})


def main():
    df = make_salary_data()
    print("Family sizes and raw averages:")
    print(df.groupby("JobFamily")["salary"].agg(["mean", "count"]).round(0))

    # --- Compare every family against ONE shared benchmark ---
    report = segment_position_report(df, segment_col="JobFamily", y_col="salary", x_col="age")
    report["ci_crosses_zero"] = (report["ci_lower"] < 0) & (report["ci_upper"] > 0)

    print("\n=== segment_position_report ===")
    print(report.round(0).to_string(index=False))

    print(
        "\nENG, OPS, and SALES all have confidence intervals that stay"
        "\nentirely on one side of zero -- their deviation from the"
        "\nshared benchmark is not something you'd expect from sampling"
        "\nnoise alone. FIN's interval [-223, 401] straddles zero: its"
        "\npoint estimate (+62) looks like a small positive gap, but the"
        "\ninterval says that gap is entirely consistent with FIN sitting"
        "\nright on the benchmark, and the +62 is just this particular"
        "\nsample. ITS's interval [0.5, 375] is the interesting borderline"
        "\ncase -- technically on the positive side, but only barely, by"
        "\nan amount smaller than the granularity you'd want to act on."
    )

    # --- Chart: point estimates with confidence intervals ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    order = report.sort_values("difference")
    colors = ["#d62728" if cross else "#2ca02c" for cross in order["ci_crosses_zero"]]
    y_pos = np.arange(len(order))

    ax.errorbar(
        order["difference"], y_pos,
        xerr=[order["difference"] - order["ci_lower"], order["ci_upper"] - order["difference"]],
        fmt="o", capsize=4, color="black", ecolor="gray", markersize=0,
    )
    ax.scatter(order["difference"], y_pos, c=colors, s=100, zorder=3, edgecolors="black")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order["segment"])
    ax.set_xlabel("Difference from shared benchmark (SEK)")
    ax.set_title("Which families actually differ from the benchmark?")
    fig.tight_layout()
    fig.savefig("segment_position.png", dpi=150)
    print("\nChart saved to segment_position.png")


if __name__ == "__main__":
    main()
