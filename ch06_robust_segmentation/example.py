"""
Chapter 6 -- Robust Segmentation
===================================

Compass: a global analysis reaching one conclusion, then the same
data segmented reaching a materially different one.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import fit_huber_trend, predict_trend, apply_by_segment


def make_store_satisfaction_data(seed=31):
    """
    Five stores, each with its own average wait time and its own
    baseline customer satisfaction. The store with the LONGEST average
    wait is also, genuinely, the most-loved store overall -- it's
    busy precisely because people like it. Within every single store,
    though, the ordinary, expected relationship holds: a customer who
    personally waits longer than others at THAT store rates their
    visit lower.
    """
    rng = np.random.default_rng(seed)
    stores = {
        "Store A": (60, 3, 150),
        "Store B": (65, 5, 150),
        "Store C": (70, 8, 150),
        "Store D": (75, 12, 150),
        "Store E": (85, 18, 150),
    }

    rows = []
    for store, (baseline, avg_wait, n) in stores.items():
        wait = np.clip(rng.normal(avg_wait, avg_wait * 0.3, n), 0.5, None)
        satisfaction = baseline - 1.2 * (wait - avg_wait) + rng.normal(0, 4, n)
        satisfaction = np.clip(satisfaction, 0, 100)
        for w, s in zip(wait, satisfaction):
            rows.append({"store": store, "wait_minutes": w, "satisfaction": s})
    return pd.DataFrame(rows)


def wait_time_effect(x, y):
    """Huber-fit satisfaction against wait time; report the predicted
    change in satisfaction from the shortest to the longest wait seen
    in this data -- a single, interpretable number."""
    fit = fit_huber_trend(x, y, degree=1)
    lo, hi = float(x.min()), float(x.max())
    pred = predict_trend(fit, np.array([lo, hi]))
    return {
        "effect_full_range": float(pred[1] - pred[0]),
        "pred_at_min_wait": float(pred[0]),
        "pred_at_max_wait": float(pred[1]),
    }


def main():
    df = make_store_satisfaction_data()

    print("Store-level averages:")
    print(df.groupby("store")[["wait_minutes", "satisfaction"]].mean().round(1))

    # --- GLOBAL: pool everyone, ignore which store they visited ---
    x_all = df["wait_minutes"].to_numpy()
    y_all = df["satisfaction"].to_numpy()
    global_result = wait_time_effect(x_all, y_all)

    print("\n=== GLOBAL analysis (store ignored) ===")
    print(f"Effect of going from shortest to longest wait: {global_result['effect_full_range']:+.1f} points")
    print(
        "\nRead naively, this says waiting longer makes customers HAPPIER."
        "\nThat's not a typo in the analysis -- it's what the pooled numbers"
        "\nactually show."
    )

    # --- SEGMENTED: fit independently within each store ---
    segmented = apply_by_segment(
        df, segment_col="store", x_col="wait_minutes", y_col="satisfaction",
        analysis_fn=wait_time_effect, min_points=20,
    )

    print("\n=== SEGMENTED analysis (per store) ===")
    print(segmented[["segment", "n", "effect_full_range"]].to_string(index=False))

    print(
        "\nEvery single store shows the opposite sign from the global"
        "\nnumber. Within each store, longer waits are associated with"
        "\nLOWER satisfaction -- the ordinary, unsurprising relationship"
        "\nyou'd expect. The global analysis wasn't measuring the effect"
        "\nof waiting at all; it was measuring which store you happened"
        "\nto be standing in."
    )

    # --- Chart: global trend vs. per-store trends ---
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"Store A": "#1f77b4", "Store B": "#ff7f0e", "Store C": "#2ca02c",
              "Store D": "#d62728", "Store E": "#9467bd"}

    for store, color in colors.items():
        sub = df[df["store"] == store]
        ax.scatter(sub["wait_minutes"], sub["satisfaction"], s=10, alpha=0.35, color=color, label=store)
        fit = fit_huber_trend(sub["wait_minutes"].to_numpy(), sub["satisfaction"].to_numpy(), degree=1)
        grid = np.linspace(sub["wait_minutes"].min(), sub["wait_minutes"].max(), 50)
        ax.plot(grid, predict_trend(fit, grid), color=color, linewidth=2)

    grid_all = np.linspace(x_all.min(), x_all.max(), 100)
    global_fit = fit_huber_trend(x_all, y_all, degree=1)
    ax.plot(grid_all, predict_trend(global_fit, grid_all), color="black", linewidth=3,
             linestyle="--", label="Global trend (store ignored)")

    ax.set_xlabel("Wait time (minutes)")
    ax.set_ylabel("Satisfaction score")
    ax.set_title("Global trend rises; every single store's trend falls")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("segmentation_paradox.png", dpi=150)
    print("\nChart saved to segmentation_paradox.png")


if __name__ == "__main__":
    main()
