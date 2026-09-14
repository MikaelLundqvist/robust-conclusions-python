"""
Chapter 2 -- Huber, Tukey, OLS Side by Side
============================================

Compass: show fit_huber_trend / fit_tukey_trend / fit_ols_trend on the
identical dataset, with a clear visual/numeric comparison -- not just
three separate examples.

Run with:

    python example.py

Requires statsmodels (for fit_tukey_trend) in addition to robustkit's
core dependencies.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import fit_ols_trend, fit_huber_trend, fit_tukey_trend, predict_trend


def make_housing_data(seed=11):
    """
    House size (square meters) vs. price, for a mid-sized town.

    Most homes follow an ordinary trend. A small handful are nicer
    than average -- renovated, better location -- and sit moderately
    above the trend. A separate, smaller handful are genuinely
    extreme: large waterfront estates whose price has almost nothing
    to do with size alone. Nothing here is a data error; all three
    groups are real homes that really sold for what's listed.

    Deliberately, the three estates sit at the LARGE end of the size
    range -- which turns out to matter a great deal for what happens
    next.
    """
    rng = np.random.default_rng(seed)

    n_homes = 150
    size = rng.uniform(40, 180, n_homes)
    price = 800 + 28 * size + rng.normal(0, 350, n_homes)

    nice_size = rng.uniform(90, 160, 6)
    nice_price = 800 + 28 * nice_size + rng.uniform(2500, 4500, 6)

    estate_size = np.array([210.0, 240.0, 195.0])
    estate_price = np.array([28000.0, 34000.0, 25000.0])

    size_all = np.concatenate([size, nice_size, estate_size])
    price_all = np.concatenate([price, nice_price, estate_price])
    return size_all, price_all


def main():
    size, price = make_housing_data()

    # A straight line (degree=1), deliberately -- with a curved fit,
    # the three estates distort the curve's SHAPE as well as its
    # level, which is a real and interesting phenomenon but a
    # confusing place to start. A straight line isolates the effect
    # we want to look at first.
    ols_fit = fit_ols_trend(size, price, degree=1)
    huber_fit = fit_huber_trend(size, price, degree=1)
    tukey_fit = fit_tukey_trend(size, price, degree=1)

    check_sizes = np.array([50, 110, 170])
    ols_pred = predict_trend(ols_fit, check_sizes)
    huber_pred = predict_trend(huber_fit, check_sizes)
    tukey_pred = predict_trend(tukey_fit, check_sizes)

    # A validation anchor, not a fourth method: what does OLS say if
    # we just delete the three estates outright? If a robust method's
    # numbers land close to THIS, that's independent confirmation it's
    # doing roughly what "ignore the estates" would do by hand.
    size_no_estates, price_no_estates = size[:-3], price[:-3]
    reference_fit = fit_ols_trend(size_no_estates, price_no_estates, degree=1)
    reference_pred = predict_trend(reference_fit, check_sizes)

    print(f"Dataset: {len(size)} homes (6 nicer-than-average, 3 extreme waterfront estates)\n")
    print(f"{'Size (sqm)':>10} {'OLS':>10} {'Huber':>10} {'Tukey':>10} {'w/o estates':>12}")
    for s, o, h, t, r in zip(check_sizes, ols_pred, huber_pred, tukey_pred, reference_pred):
        print(f"{s:>10} {o:>10,.0f} {h:>10,.0f} {t:>10,.0f} {r:>12,.0f}")

    print(
        "\nNotice the SIGN of OLS's disagreement flips. At size 50 --"
        "\nfar from the three estates -- OLS UNDER-predicts, sitting"
        "\nbelow Huber, Tukey, and the 'estates removed' reference"
        "\nalike. At size 170 -- close to the estates -- OLS swings the"
        "\nother way and OVER-predicts, by a wide margin."
        "\n\nThis is a lever-arm effect. The three estates sit only at"
        "\none end of the size range. Least squares has to draw ONE"
        "\nstraight line that minimizes squared error across every"
        "\npoint at once, and the cheapest way to accommodate three"
        "\nenormously expensive homes clustered at the high end is to"
        "\ntilt the WHOLE line -- pulling it up near the estates, which"
        "\nnecessarily drags it down everywhere else to compensate."
        "\n\nHuber and Tukey both resist that tilt, and both land close"
        "\nto what you get by just deleting the three estates and"
        "\nrefitting with OLS -- without either of them ever being told"
        "\nthose three points exist."
    )

    grid = np.linspace(size.min(), size.max(), 200)
    ols_curve = predict_trend(ols_fit, grid)
    huber_curve = predict_trend(huber_fit, grid)
    tukey_curve = predict_trend(tukey_fit, grid)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(size, price, s=20, alpha=0.5, color="gray", label="Homes")
    ax.plot(grid, ols_curve, color="crimson", linewidth=2, label="OLS")
    ax.plot(grid, huber_curve, color="steelblue", linewidth=2, label="Huber")
    ax.plot(grid, tukey_curve, color="seagreen", linewidth=2, linestyle="--", label="Tukey")
    ax.set_xlabel("Size (sqm)")
    ax.set_ylabel("Price (thousands)")
    ax.set_title("Same data, three conclusions: OLS vs. Huber vs. Tukey")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("ols_huber_tukey.png", dpi=150)
    print("\nChart saved to ols_huber_tukey.png")


if __name__ == "__main__":
    main()
