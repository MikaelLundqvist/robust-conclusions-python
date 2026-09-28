"""
Chapter 13 -- Custom Benchmarks and Reference Curves
===================================

Compass: a benchmark model can use as many columns as you give it --
and the choice between reference="model", "curve", and
"benchmark_curve" isn't just about how smooth the chart looks, it can
determine whether a genuine underpayment is caught at all.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from robustkit import mad_outlier_report


def make_data(seed=101, n=1000):
    """
    Salary depends on age, job family, AND level -- three factors, not
    one. A benchmark that only knows about age will systematically
    misjudge anyone whose family/level combination differs from the
    population average, in either direction.
    """
    rng = np.random.default_rng(seed)
    job_family = rng.choice(["ENG", "ITS", "FIN"], n, p=[0.4, 0.35, 0.25])
    level = rng.choice(["L1", "L2", "L3"], n, p=[0.5, 0.35, 0.15])
    age = rng.uniform(25, 60, n)
    family_bump = {"ENG": 3000, "ITS": 1000, "FIN": 0}
    level_bump = {"L1": 0, "L2": 6000, "L3": 14000}
    salary = (32000 + 380 * age + np.array([family_bump[f] for f in job_family])
              + np.array([level_bump[l] for l in level]) + rng.normal(0, 900, n))
    df = pd.DataFrame({"JobFamily": job_family, "level": level, "age": age, "salary": salary,
                        "emp_id": [f"E{i}" for i in range(n)]})
    return df


class CustomBenchmark:
    """Any object with predict(dataframe) -> array works as benchmark_fit
    -- robustkit never needs to know it uses three columns, not one."""
    def __init__(self, df):
        self.X_columns = pd.get_dummies(df[["age", "JobFamily", "level"]], columns=["JobFamily", "level"], drop_first=True).columns
        X = pd.get_dummies(df[["age", "JobFamily", "level"]], columns=["JobFamily", "level"], drop_first=True)
        self.model = HuberRegressor()
        self.model.fit(X, df["salary"])

    def predict(self, sub_df):
        Xs = pd.get_dummies(sub_df[["age", "JobFamily", "level"]], columns=["JobFamily", "level"], drop_first=True)
        Xs = Xs.reindex(columns=self.X_columns, fill_value=0)
        return self.model.predict(Xs)


def main():
    df = make_data()

    # A genuine underpayment: an FIN, L3 (senior-level), age 29 employee
    # docked $9,000 -- but L3 is a rare, high-paying level (14 of the
    # bumps), so on a simple age-only trend this barely registers.
    fn_idx = df[(df["level"] == "L3") & (df["age"] < 35) & (df["JobFamily"] == "FIN")].index[0]
    df.loc[fn_idx, "salary"] -= 9000
    fn_id = df.loc[fn_idx, "emp_id"]
    print(f"Injected case: {fn_id}, {df.loc[fn_idx, ['JobFamily','level','age']].to_dict()}, docked $9,000")

    # --- Simple benchmark: age only ---
    report_simple = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                                        min_size=20, k=3.0, id_cols=["emp_id"])
    residual_simple = report_simple[report_simple["emp_id"] == fn_id]["residual"].values[0]
    flagged_simple = report_simple[report_simple["emp_id"] == fn_id]["flagged"].values[0]

    # --- Custom benchmark: age + JobFamily + level ---
    custom_fit = CustomBenchmark(df)
    report_custom = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                                        benchmark_fit=custom_fit, min_size=20, k=3.0, id_cols=["emp_id"])
    residual_custom = report_custom[report_custom["emp_id"] == fn_id]["residual"].values[0]
    flagged_custom = report_custom[report_custom["emp_id"] == fn_id]["flagged"].values[0]

    print(f"\nSimple (age-only) benchmark: residual=${residual_simple:,.0f}, flagged={flagged_simple}")
    print(f"Custom (age+family+level) benchmark: residual=${residual_custom:,.0f}, flagged={flagged_custom}")

    # --- Now compare all three `reference` modes against the SAME custom model ---
    print(f"\n=== reference mode comparison, all using the custom benchmark ===")
    for ref in ["model", "curve", "benchmark_curve"]:
        report = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                                     benchmark_fit=custom_fit, min_size=20, k=3.0, id_cols=["emp_id"], reference=ref)
        row = report[report["emp_id"] == fn_id]
        print(f"reference={ref!r}: residual=${row['residual'].values[0]:,.0f}, flagged={row['flagged'].values[0]}, "
              f"total flagged in dataset={int(report['flagged'].sum())}")

    print(
        "\nreference='model' catches the case: it uses the custom"
        "\nmodel's own prediction for THIS person, which already knows"
        "\nthey're L3 (a high-paying level). reference='curve' and"
        "\n'benchmark_curve' both MISS it -- because both fit a curve"
        "\nagainst x_col (age) ALONE, within the FIN segment. Level isn't"
        "\npart of x_col, so fitting a 1-dimensional curve over age"
        "\nsmooths right over the L1/L2/L3 differences, re-absorbing the"
        "\nlevel effect the same way a simple age-only benchmark would."
        "\nA custom, multivariate model only pays off with reference='model'"
        "\n-- pairing it with a curve-based reference throws away exactly"
        "\nthe extra structure that made the custom model worth building."
    )

    # --- Chart: the same person's residual under each approach ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Simple\n(age only)", "Custom +\nreference='model'", "Custom +\nreference='curve'", "Custom +\nreference='benchmark_curve'"]
    residuals = [residual_simple]
    for ref in ["model", "curve", "benchmark_curve"]:
        report = mad_outlier_report(df, y_col="salary", segment_cols=["JobFamily"], x_col="age",
                                     benchmark_fit=custom_fit, min_size=20, k=3.0, id_cols=["emp_id"], reference=ref)
        residuals.append(report[report["emp_id"] == fn_id]["residual"].values[0])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ["#d62728" if r < -3000 else "#7f7f7f" for r in residuals]
    ax.bar(labels, residuals, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(-9000, color="orange", linestyle="--", linewidth=1, label="Actual underpayment ($9,000)")
    ax.set_ylabel("Residual for this one employee ($)")
    ax.set_title("Same person, same $9,000 underpayment, four different verdicts")
    ax.legend()
    fig.tight_layout()
    fig.savefig("reference_mode_comparison.png", dpi=150)
    print("\nChart saved to reference_mode_comparison.png")


if __name__ == "__main__":
    main()
