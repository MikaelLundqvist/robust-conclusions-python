"""
Chapter 15 -- From Analysis to Report
===================================

Compass: the same analysis needs to look different depending on who's
going to see it -- an analyst debugging a model wants every data
point visible; a manager getting a PDF of six warehouse-shift
combinations wants a clean trend line and nothing else. Producing
both from the same underlying fit, without re-deriving anything, is
what this chapter is about.

Run with:

    python example.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import (
    plot_huber_iqr, export_huber_iqr_pdf, export_huber_iqr_images,
    export_outlier_pdf, mad_outlier_report,
)


def make_delivery_data(seed=301, n=1500):
    """
    Delivery times for a courier network: three warehouses, two
    shifts, with a real relationship between route length (number of
    stops) and delivery time, plus a real night-shift penalty. Six
    routes are given a genuine, severe delay -- a vehicle breakdown --
    unrelated to route length or shift.
    """
    rng = np.random.default_rng(seed)
    warehouse = rng.choice(["North", "South", "East"], n, p=[0.4, 0.35, 0.25])
    shift = rng.choice(["Day", "Night"], n, p=[0.7, 0.3])
    stops = rng.uniform(5, 60, n)
    base = 20 + 2.2 * stops + rng.normal(0, 12, n)
    night_penalty = np.where(shift == "Night", 15, 0)
    delivery_minutes = base + night_penalty
    df = pd.DataFrame({
        "warehouse": warehouse, "shift": shift, "stops": stops,
        "delivery_minutes": delivery_minutes, "route_id": [f"R{i}" for i in range(n)],
    })
    breakdown_idx = rng.choice(n, size=6, replace=False)
    df.loc[breakdown_idx, "delivery_minutes"] += 90
    return df, set(df.loc[breakdown_idx, "route_id"])


def main():
    df, breakdowns = make_delivery_data()
    print(f"n={len(df)}, {len(breakdowns)} genuine breakdowns injected")

    # --- plot_huber_iqr directly: the function behind every chart in this book ---
    # Every earlier chapter built its own matplotlib chart by hand with
    # fit_huber_trend/predict_trend. plot_huber_iqr is the single
    # function that does this directly, with one parameter controlling
    # who the chart is for.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_huber_iqr(df["stops"].to_numpy(), df["delivery_minutes"].to_numpy(), ax=axes[0], show_points=False)
    axes[0].set_title("Publisher (show_points=False)")
    plot_huber_iqr(df["stops"].to_numpy(), df["delivery_minutes"].to_numpy(), ax=axes[1], show_points=True)
    axes[1].set_title("Analyst (show_points=True)")
    fig.tight_layout()
    fig.savefig("publisher_vs_analyst.png", dpi=150)
    print("Saved publisher_vs_analyst.png")

    # --- export_huber_iqr_pdf: one PDF, one page per segment ---
    path1 = export_huber_iqr_pdf(
        df, x_col="stops", y_col="delivery_minutes", segment_cols=["warehouse", "shift"],
        path="delivery_trends.pdf", min_size=20, min_points_to_plot=10,
    )
    import pypdf
    print(f"export_huber_iqr_pdf: {len(pypdf.PdfReader(path1).pages)} pages in {path1}")

    # --- export_huber_iqr_images: the same charts, as individual files ---
    # Useful when the destination is a slide deck or a wiki page that
    # wants one image at a time, not a combined PDF.
    out_dir = "delivery_images"
    image_paths = export_huber_iqr_images(
        df, x_col="stops", y_col="delivery_minutes", segment_cols=["warehouse", "shift"],
        output_dir=out_dir, min_size=20, min_points_to_plot=10,
    )
    print(f"export_huber_iqr_images: {len(image_paths)} images in {out_dir}/")

    # --- export_outlier_pdf: find the breakdowns ---
    # direction="positive" because here a LONG delivery is the problem
    # -- the opposite of earlier chapters' salary examples, where
    # direction="negative" (underpayment) was the default worth
    # flagging. Getting this backwards silently flags nothing: with
    # direction="negative" on this data, zero of the six genuine
    # breakdowns get caught, because none of them are unusually FAST.
    report = mad_outlier_report(
        df, y_col="delivery_minutes", segment_cols=["warehouse", "shift"], x_col="stops",
        min_size=20, k=3.0, id_cols=["route_id"], direction="positive",
    )
    flagged = set(report[report["flagged"]]["route_id"])
    print(f"\nFlagged {len(flagged)} routes, caught {len(flagged & breakdowns)}/{len(breakdowns)} genuine breakdowns")

    path3 = export_outlier_pdf(
        df, y_col="delivery_minutes", segment_cols=["warehouse", "shift"], x_col="stops",
        path="delivery_outliers.pdf", min_size=20, min_points_to_plot=10, k=3.0,
        id_cols=["route_id"], annotate_flagged_with="route_id", direction="positive",
    )
    print(f"export_outlier_pdf: {len(pypdf.PdfReader(path3).pages)} pages in {path3}")


if __name__ == "__main__":
    main()
