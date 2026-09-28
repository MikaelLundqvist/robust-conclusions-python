"""
Chapter 12 -- Is the Pipeline Itself Trustworthy?
===================================

Compass: a robust statistical method can be perfectly correct while
sitting on top of a pipeline that silently dropped rows -- a join
that quietly lost a whole department, or a fit that used fewer rows
than you thought, without ever raising an error.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import check_row_integrity, compare_row_sets, segment_consistency_report


def make_pipeline_data(seed=91, n=500):
    """
    A salary dataset with three deliberately realistic data issues,
    each invisible unless specifically checked for:

    1. A department-name lookup table has "ops" in lowercase, while
       the salary data uses "OPS" -- an inner join on department code
       silently drops every OPS employee, rather than erroring.
    2. A few employees have no department code at all (e.g. a new
       hire not yet assigned) -- invisible unless you check that every
       row landed somewhere after grouping.
    3. A few FIN employees have a missing age value -- Huber fitting
       doesn't error on this, it just silently drops those rows,
       fitting on fewer observations than the segment's row count
       would suggest.
    """
    rng = np.random.default_rng(seed)
    dept_codes = rng.choice(["ENG", "SALES", "OPS", "FIN"], n).astype(object)
    age = rng.uniform(25, 60, n)
    salary = 30000 + 400 * age + rng.normal(0, 900, n)
    df = pd.DataFrame({"emp_id": [f"E{i}" for i in range(n)], "dept_code": dept_codes, "age": age, "salary": salary})

    missing_idx = rng.choice(n, size=7, replace=False)
    df.loc[missing_idx, "dept_code"] = np.nan

    fin_idx = df[df["dept_code"] == "FIN"].index
    missing_age_idx = rng.choice(fin_idx, size=5, replace=False)
    df.loc[missing_age_idx, "age"] = np.nan

    lookup = pd.DataFrame({
        "dept_code": ["ENG", "SALES", "ops", "FIN"],  # <- the bug: lowercase "ops"
        "dept_name": ["Engineering", "Sales", "Operations", "Finance"],
    })
    return df, lookup


def main():
    df, lookup = make_pipeline_data()
    print(f"Raw data: {len(df)} rows")

    # --- Check 1: does every row have a usable segment label? ---
    integrity = check_row_integrity(df, group_col="dept_code")
    print("\n=== check_row_integrity ===")
    for k, v in integrity.items():
        print(f"{k}: {v}")

    # --- The pipeline step that goes wrong: merging in department names ---
    merged = df.merge(lookup, on="dept_code", how="inner")
    print(f"\nAfter merging in department names: {len(merged)} rows (was {len(df)})")

    # --- Check 2: what actually changed across that merge? ---
    comparison = compare_row_sets(df, merged, key_col="emp_id")
    print("\n=== compare_row_sets ===")
    for k, v in comparison.items():
        if k not in ("dropped_keys", "added_keys"):
            print(f"{k}: {v}")
    dropped_df = df[df["emp_id"].isin(comparison["dropped_keys"])]
    print("\nDropped employees, by department code:")
    print(dropped_df["dept_code"].value_counts().to_string())

    # --- Check 3: within each segment, did the fit actually use every row? ---
    clean = df.dropna(subset=["dept_code"])
    consistency = segment_consistency_report(clean, segment_col="dept_code", x_col="age", y_col="salary", min_size=20)
    print("\n=== segment_consistency_report ===")
    print(consistency.to_string(index=False))

    print(
        "\nNotice FIN: fit_ok is True -- fitting a Huber trend on FIN"
        "\nsucceeded without any error -- but n_valid_xy (121) is lower"
        "\nthan n_total (126). The fit didn't fail; it just silently used"
        "\n5 fewer rows than the segment's row count would suggest,"
        "\nbecause Huber fitting (like almost any numerical fit) simply"
        "\ndrops missing values rather than raising an exception. A"
        "\nsuccessful fit is not the same guarantee as a fit that used"
        "\nall the data you thought it did."
    )

    # --- Chart: rows remaining at each pipeline stage ---
    stages = ["Raw data", "Has dept label", "After dept-name merge", "Has valid age (FIN)"]
    counts = [
        len(df),
        len(df.dropna(subset=["dept_code"])),
        len(merged),
        len(merged.dropna(subset=["age"])),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stages, counts, color=["#7f7f7f", "#7f7f7f", "#d62728", "#7f7f7f"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 5, str(count), ha="center")
    ax.set_ylabel("Rows remaining")
    ax.set_title("Where did the rows go?")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig("pipeline_stages.png", dpi=150)
    print("\nChart saved to pipeline_stages.png")


if __name__ == "__main__":
    main()
