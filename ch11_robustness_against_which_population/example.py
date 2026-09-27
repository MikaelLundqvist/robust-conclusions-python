"""
Chapter 11 -- Robustness Against Which Population?
===================================

Compass: the same person can be flagged as an outlier, or not,
depending entirely on which reference population they're compared
against -- their own segment, or the whole organization.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import dual_reference_outlier_report


def make_salary_data(seed=81):
    """
    Four job families sharing the same underlying age-salary
    relationship, with three deliberate exceptions:

    - OPS (n=300) is collectively shifted $6,500 below the benchmark,
      but with very tight internal spread -- everyone in it is paid
      similarly LOW, consistently.
    - One ENG employee has a severe, genuine $22,000 shortfall --
      unusual by any standard.
    - One FIN employee (FIN's own spread is unusually tight) has a
      modest $900 dip -- small in absolute terms, but large relative
      to how tightly FIN's salaries normally cluster.
    """
    rows = []

    def add(fam, n_i, noise, bump, seed_i):
        r = np.random.default_rng(seed_i)
        age = r.uniform(25, 60, n_i)
        salary = 30000 + 400 * age + bump + r.normal(0, noise, n_i)
        for a, s in zip(age, salary):
            rows.append({"JobFamily": fam, "age": a, "salary": s, "emp_id": f"{fam}_{len(rows)}"})

    add("ENG", 400, 900, 0, 1)
    add("SALES", 400, 900, 0, 2)
    add("OPS", 300, 350, -6500, 3)
    add("FIN", 100, 250, 0, 4)

    df = pd.DataFrame(rows)
    idx_a = df[df["JobFamily"] == "ENG"].index[0]
    df.loc[idx_a, "salary"] -= 22000

    idx_b = df[df["JobFamily"] == "FIN"].index[5]
    df.loc[idx_b, "salary"] -= 900

    return df


def main():
    df = make_salary_data()

    report = dual_reference_outlier_report(
        df, y_col="salary", x_col="age", segment_cols=["JobFamily"],
        min_size=20, k=3.0, id_cols=["emp_id"],
    )

    print("=== reference_type distribution ===")
    print(report["reference_type"].value_counts().to_string())

    merged = report.merge(df[["emp_id", "JobFamily"]], on="emp_id")
    print("\n=== reference_type by JobFamily ===")
    print(pd.crosstab(merged["JobFamily"], merged["reference_type"]))

    type_c = merged[merged["reference_type"] == "C"]
    print(f"\nType C (n={len(type_c)}) -- a sample:")
    print(type_c[["emp_id", "local_residual", "local_flagged", "global_residual", "global_flagged"]].head(5).round(0).to_string(index=False))

    print(
        "\nAlmost the entire OPS department (298 of 300 people) lands as"
        "\nType C: their local_residual sits near zero -- typical, once"
        "\nyou compare them only to their own OPS colleagues -- but their"
        "\nglobal_residual sits around -$6,000, clearly flagged against"
        "\nthe organization-wide benchmark. Nobody in OPS looks unusual"
        "\nto their manager. Every one of them looks underpaid to anyone"
        "\ncomparing against the company as a whole."
    )

    # --- Chart: local vs global residual, colored by type ---
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = {"A": "#d62728", "B": "#ff7f0e", "C": "#1f77b4", "D": "#7f7f7f"}
    for rt, sub in merged.groupby("reference_type"):
        ax.scatter(sub["local_residual"], sub["global_residual"], s=18, alpha=0.6,
                   color=colors[rt], label=f"Type {rt} (n={len(sub)})")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Local residual (vs. own segment)")
    ax.set_ylabel("Global residual (vs. organization benchmark)")
    ax.set_title("Same people, two reference populations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("dual_reference.png", dpi=150)
    print("\nChart saved to dual_reference.png")


if __name__ == "__main__":
    main()
