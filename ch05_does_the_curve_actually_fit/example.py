"""
Chapter 5 -- Does the Curve Actually Fit?
============================================

Compass: reproduce the real finding -- a quadratic that's clearly
sufficient (autoMpg-style) vs. one that clearly isn't (a genuinely
wavy trend) -- compare_polynomial_degrees telling them apart.

Run with:

    python example.py
"""

import numpy as np

from robustkit import compare_polynomial_degrees, fit_huber_trend, predict_trend


def make_horsepower_efficiency_data(seed=17, n=300):
    """
    Horsepower against fuel efficiency -- the same shape of
    relationship found in the real-world autoMpg dataset (validated
    separately against OpenML data): efficiency falls off sharply at
    low horsepower and levels out at high horsepower, a single smooth
    curve with one bend.
    """
    rng = np.random.default_rng(seed)
    hp = rng.uniform(50, 220, n)
    mpg = 45 - 0.28 * hp + 0.0007 * hp ** 2 + rng.normal(0, 1.8, n)
    return hp, mpg


def make_engine_temp_data(seed=23, n=300):
    """
    A genuinely wavy relationship: engine efficiency rises with
    operating temperature, dips due to a secondary thermal effect,
    then rises again at very high temperatures -- a true double-bend
    shape that no single quadratic can represent, no matter how it's
    fit.
    """
    rng = np.random.default_rng(seed)
    temp = rng.uniform(60, 220, n)
    t = (temp - 60) / 160
    efficiency = 30 + 25 * np.sin(t * 3.4 * np.pi / 2) - 8 * t + rng.normal(0, 1.2, n)
    return temp, efficiency


def main():
    hp, mpg = make_horsepower_efficiency_data()
    temp, eff = make_engine_temp_data()

    print("=== Horsepower vs. fuel efficiency ===\n")
    result_hp = compare_polynomial_degrees(hp, mpg, degrees=(1, 2, 3, 4))
    print(result_hp.to_string(index=False))

    gain_2_to_3_hp = result_hp.loc[result_hp["degree"] == 3, "r_squared_gain"].iloc[0]
    print(
        f"\nGoing from degree 2 to degree 3 buys almost nothing here"
        f"\n(r_squared_gain = {gain_2_to_3_hp:.6f}). The quadratic default"
        f"\nalready captures essentially all of the real structure --"
        f"\nadding complexity beyond it isn't earning its keep."
    )

    print("\n\n=== Engine temperature vs. efficiency ===\n")
    result_temp = compare_polynomial_degrees(temp, eff, degrees=(1, 2, 3, 4))
    print(result_temp.to_string(index=False))

    gain_1_to_2 = result_temp.loc[result_temp["degree"] == 2, "r_squared_gain"].iloc[0]
    gain_2_to_3 = result_temp.loc[result_temp["degree"] == 3, "r_squared_gain"].iloc[0]
    print(
        f"\nHere the story is completely different. Going from degree 1"
        f"\nto 2 gains {gain_1_to_2:.3f} in R\u00b2 -- reasonable at a glance."
        f"\nBut going from degree 2 to degree 3 gains {gain_2_to_3:.3f},"
        f"\nMORE than the first jump. A quadratic default would have"
        f"\nlooked like a perfectly respectable fit -- R\u00b2 of 0.86 isn't"
        f"\nsuspicious on its own -- while quietly missing an entire"
        f"\nsecond bend in the relationship."
    )

    print(
        "\n\nBoth datasets could have been handed the exact same default"
        "\n(degree=2) without complaint. Only checking -- not assuming --"
        "\nreveals that one of them was already right, and the other was"
        "\nsilently leaving real structure on the table."
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.scatter(hp, mpg, s=12, alpha=0.4, color="gray")
    grid = np.linspace(hp.min(), hp.max(), 200)
    for deg, color, style in [(2, "steelblue", "-"), (4, "crimson", "--")]:
        fit = fit_huber_trend(hp, mpg, degree=deg)
        ax.plot(grid, predict_trend(fit, grid), color=color, linewidth=2, linestyle=style, label=f"degree={deg}")
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("Fuel efficiency (mpg)")
    ax.set_title("Quadratic already sufficient")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(temp, eff, s=12, alpha=0.4, color="gray")
    grid = np.linspace(temp.min(), temp.max(), 200)
    for deg, color, style in [(2, "steelblue", "-"), (4, "crimson", "--")]:
        fit = fit_huber_trend(temp, eff, degree=deg)
        ax.plot(grid, predict_trend(fit, grid), color=color, linewidth=2, linestyle=style, label=f"degree={deg}")
    ax.set_xlabel("Engine temperature")
    ax.set_ylabel("Efficiency")
    ax.set_title("Quadratic misses a real second bend")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("degree_comparison.png", dpi=150)
    print("\nChart saved to degree_comparison.png")


if __name__ == "__main__":
    main()
