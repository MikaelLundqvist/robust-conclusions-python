"""
Chapter 4 -- How Certain Is the Curve?
=========================================

Compass: show bootstrap_band and bca_bootstrap_ci narrowing as sample
size grows -- the reader should see the confidence band shrink with
more data, not just read that it does.

Run with:

    python example.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import bootstrap_band, bca_bootstrap_ci, fit_huber_trend, predict_trend


def make_ad_spend_data(seed=21, n=5000):
    """
    A marketing dataset: daily ad spend against daily revenue, for a
    single, fixed underlying relationship. Every sample size checked
    below is drawn from this SAME population -- what changes is only
    how much of it we happened to observe.
    """
    rng = np.random.default_rng(seed)
    spend = rng.uniform(10, 500, n)
    revenue = 40 + 3.2 * spend + rng.normal(0, 180, n)
    return spend, revenue


def main():
    spend_full, revenue_full = make_ad_spend_data()
    sample_sizes = [30, 100, 300, 1000, 5000]
    check_spend = 250.0

    # --- Part 1: bootstrap_band -- confidence band around the whole curve ---
    print("=== bootstrap_band: CI width around the trend at spend=250 ===\n")
    print(f"{'n':>6} {'lower':>10} {'upper':>10} {'width':>10}")

    band_widths = []
    for n in sample_sizes:
        spend_n, revenue_n = spend_full[:n], revenue_full[:n]
        band = bootstrap_band(spend_n, revenue_n, degree=1, n_boot=500, ci=95)
        grid = band["grid"]
        gi = np.argmin(np.abs(grid - check_spend))
        width = band["upper"][gi] - band["lower"][gi]
        band_widths.append(width)
        print(f"{n:>6} {band['lower'][gi]:>10.1f} {band['upper'][gi]:>10.1f} {width:>10.1f}")

    print(
        f"\nGoing from n=30 to n=5000 -- roughly 167x more data -- the"
        f"\nband width at spend=250 shrank from {band_widths[0]:.0f} to"
        f"\n{band_widths[-1]:.0f}, about {band_widths[0]/band_widths[-1]:.1f}x"
        f"\nnarrower. All five estimates used the exact same underlying"
        f"\nrelationship between spend and revenue -- nothing about the"
        f"\nTRUTH changed. What changed is how confident we can be that"
        f"\nour curve reflects it."
    )

    # --- Part 2: bca_bootstrap_ci -- confidence interval for a specific statistic ---
    def median_revenue(x, y):
        return float(np.median(y))

    print("\n\n=== bca_bootstrap_ci: CI for median revenue ===\n")
    print(f"{'n':>6} {'estimate':>10} {'lower':>10} {'upper':>10} {'width':>10}")

    ci_widths = []
    for n in sample_sizes:
        spend_n, revenue_n = spend_full[:n], revenue_full[:n]
        result = bca_bootstrap_ci(spend_n, revenue_n, statistic_fn=median_revenue, n_boot=500, ci=95, seed=0)
        width = result["upper"] - result["lower"]
        ci_widths.append(width)
        print(f"{n:>6} {result['estimate']:>10.1f} {result['lower']:>10.1f} {result['upper']:>10.1f} {width:>10.1f}")

    print(
        f"\nSame pattern, different question. bootstrap_band asked 'how"
        f"\nuncertain is the CURVE at this x value?' bca_bootstrap_ci"
        f"\nasked 'how uncertain is this SPECIFIC NUMBER (the median"
        f"\nrevenue)?' Both narrow the same way as n grows, because both"
        f"\nare answering versions of the same underlying question: how"
        f"\nmuch would this estimate wobble if we happened to have"
        f"\ncollected a different, equally valid sample of the same size?"
    )

    # --- Visual: overlay bands for a few sample sizes ---
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {30: "salmon", 300: "steelblue", 5000: "seagreen"}
    for n, color in colors.items():
        spend_n, revenue_n = spend_full[:n], revenue_full[:n]
        band = bootstrap_band(spend_n, revenue_n, degree=1, n_boot=500, ci=95)
        ax.fill_between(band["grid"], band["lower"], band["upper"], alpha=0.3, color=color, label=f"n={n}")
        ax.plot(band["grid"], (band["lower"] + band["upper"]) / 2, color=color, linewidth=1)

    ax.set_xlabel("Daily ad spend")
    ax.set_ylabel("Daily revenue")
    ax.set_title("Same relationship, shrinking uncertainty as n grows")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("shrinking_band.png", dpi=150)
    print("\nChart saved to shrinking_band.png")


if __name__ == "__main__":
    main()
