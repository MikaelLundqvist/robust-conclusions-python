"""
Chapter 1 -- Why Robust Analysis
================================

Compass: show OLS and a robust method reaching genuinely different
conclusions on the same real-feeling data -- see WHY robustness
matters before being taught HOW (that's Chapter 2).

Run with:

    python example.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import fit_ols_trend, fit_huber_trend, predict_trend


def make_salary_data(seed=7):
    """
    A small, realistic-feeling dataset: monthly salary by age at a
    mid-sized company. Most employees follow an ordinary, gently
    rising trend. A handful of senior executives are paid far above
    that trend -- not data errors, just genuinely different roles.
    That's the point: nothing here needs to be "cleaned" before
    analysis, and yet it still causes trouble for an ordinary fit.
    """
    rng = np.random.default_rng(seed)

    n_employees = 120
    age = rng.uniform(23, 62, n_employees)
    salary = 28000 + 550 * (age - 23) + rng.normal(0, 1500, n_employees)

    exec_age = np.array([45.0, 51.0, 58.0, 47.0, 55.0])
    exec_salary = np.array([210000.0, 195000.0, 225000.0, 180000.0, 240000.0])

    age_all = np.concatenate([age, exec_age])
    salary_all = np.concatenate([salary, exec_salary])
    return age_all, salary_all


def main():
    age, salary = make_salary_data()

    ols_fit = fit_ols_trend(age, salary, degree=2)
    huber_fit = fit_huber_trend(age, salary, degree=2)

    check_ages = np.array([30, 40, 50])
    ols_pred = predict_trend(ols_fit, check_ages)
    huber_pred = predict_trend(huber_fit, check_ages)

    print(f"Dataset: {len(age)} employees (5 of them senior executives)\n")
    print(f"{'Age':>5} {'OLS predicts':>15} {'Huber predicts':>16} {'Difference':>18}")
    for a, o, h in zip(check_ages, ols_pred, huber_pred):
        diff = o - h
        pct = diff / h * 100
        print(f"{a:>5} {o:>14,.0f} {h:>15,.0f} {diff:>+13,.0f} ({pct:+.1f}%)")

    print(
        "\nFive people out of 125 -- the five most senior executives --"
        "\nare enough to visibly pull the OLS trend upward at every age,"
        "\neven for someone in their early 30s who has never worked"
        "\nanywhere near an executive. Huber barely notices them."
        "\n\nBoth methods saw the exact same data. They do not agree on"
        "\nwhat a typical 30-, 40-, or 50-year-old earns. That"
        "\ndisagreement -- not a formula -- is the reason this book"
        "\nexists."
    )

    grid = np.linspace(age.min(), age.max(), 200)
    ols_curve = predict_trend(ols_fit, grid)
    huber_curve = predict_trend(huber_fit, grid)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(age, salary, s=20, alpha=0.5, color="gray", label="Employees")
    ax.plot(grid, ols_curve, color="crimson", linewidth=2, label="OLS")
    ax.plot(grid, huber_curve, color="steelblue", linewidth=2, label="Huber")
    ax.set_xlabel("Age")
    ax.set_ylabel("Monthly salary")
    ax.set_title("Same data, two conclusions: OLS vs. Huber")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("ols_vs_huber.png", dpi=150)
    print("\nChart saved to ols_vs_huber.png")


if __name__ == "__main__":
    main()
