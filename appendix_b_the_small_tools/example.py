"""
Appendix B -- The Small Tools
=============================

Compass: eleven small functions that the chapters did not need on the
way, but that you will reach for when you work: a first look at a
table, two spread measures, the slope of a trend, the shape of a
segment's residuals, a four-way split, a bootstrap interval for any
statistic, a one-call feature map and the two standard trend charts.

The table is a synthetic set of used-car listings and its structure
is built in:

  * price follows 24,000 - 2,100 x age + 55 x age^2 dollars (falling
    and flattening with age), times a brand multiplier (Volt x1.12,
    Nova x1.00, Terra x0.92, Zeta x1.05); the true slope is therefore
    -2,100 + 110 x age dollars per year,
  * the noise grows with age: sigma = 0.03 + 0.008 x age (older cars
    differ more from each other),
  * the South region has the SAME median price as the others but
    twice the noise,
  * region and listing_id have no effect on the level of price,
  * 3% of age values and 1% of region values are missing.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import (
    profile, print_profile, iqr, dispersion_ratio, dispersion_by_bin, fit_huber_trend, trend_derivative,
    fit_huber_benchmark, residual_summary, segment_position_report, classify_quadrants,
    bca_bootstrap_ci_by_index, feature_map, plot_analyst_view, plot_publisher_view,
    benchmark_predict, bootstrap_band,
)

MULT = {"Volt": 1.12, "Nova": 1.00, "Terra": 0.92, "Zeta": 1.05}


def make_listings(seed=1701, n=5000):
    rng = np.random.default_rng(seed)
    region = rng.choice(["North", "Central", "South"], n, p=[0.35, 0.40, 0.25])
    brand = rng.choice(list(MULT), n, p=[0.30, 0.25, 0.25, 0.20])
    age = np.clip(rng.normal(7, 4, n), 0, 20).round(1)
    sigma = (0.03 + 0.008 * age) * np.where(region == "South", 2.0, 1.0)
    base = 24000 - 2100 * age + 55 * age ** 2
    price = (base * np.array([MULT[f] for f in brand]) * np.exp(rng.normal(0, sigma))).round(-1)
    df = pd.DataFrame({"price": price, "age": age, "brand": brand, "region": region,
                       "listing_id": rng.integers(0, 200, n).astype(str)})
    df.loc[rng.random(n) < 0.03, "age"] = np.nan
    df.loc[rng.random(n) < 0.01, "region"] = np.nan
    return df


def main():
    raw = make_listings()
    print(f"{len(raw):,} used-car listings; target price. Built in: price follows 24,000 - 2,100 x + 55 x^2 "
          f"(slope -2,100 + 110 x), noise grows with age, South has twice the noise and the same median.\n")

    # ------------------------------------------------------------------
    # 1. A first look.
    # ------------------------------------------------------------------
    print("=== 1. print_profile and profile ===")
    print_profile(raw, target="price")
    p = profile(raw, target="price")
    print(f"profile() returns a dict with keys: {', '.join(p)}")
    df = raw.dropna(subset=["age", "region"]).reset_index(drop=True)
    print(f"\nAfter dropping rows with a missing age or region: {len(df):,} rows.")

    # ------------------------------------------------------------------
    # 2. Two spread measures.
    # ------------------------------------------------------------------
    print("\n=== 2. iqr and dispersion_ratio ===")
    print(f"whole stock: iqr {iqr(df.price):,.0f}, dispersion_ratio {dispersion_ratio(df.price):.3f}")
    bins = pd.qcut(df.age, 8)
    rows = []
    for b, g in df.groupby(bins, observed=True):
        rows.append({"age": str(b), "n": len(g), "median": g.price.median(),
                     "iqr": iqr(g.price), "dispersion_ratio": dispersion_ratio(g.price)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    d = dispersion_by_bin(df.age, df.price, n_bins=4)
    g0 = df[df.age <= df.age.quantile(1 / 4)]
    print(f"dispersion_by_bin, first of 4 bins: iqr {d.iqr.iloc[0]:,.0f}, dispersion_ratio "
          f"{d.dispersion_ratio.iloc[0]:.3f}; iqr() on the same people {iqr(g0.price):,.0f}")

    # ------------------------------------------------------------------
    # 3. The slope of the trend.
    # ------------------------------------------------------------------
    print("\n=== 3. trend_derivative ===")
    fit = fit_huber_trend(df.age.to_numpy(), df.price.to_numpy(), degree=2)
    xs = np.array([1.0, 5.0, 10.0, 15.0])
    slope = trend_derivative(fit, xs)
    print(pd.DataFrame({"age": xs, "estimated_slope": slope.round(0), "true_slope": -2100 + 110 * xs}
                       ).to_string(index=False))
    print(f"cars older than 12 years: {int((df.age > 12).sum())} of {len(df):,}; older than 15: {int((df.age > 15).sum())}")
    slope_h = trend_derivative(fit, xs, h=1.0)
    print(f"with h=1.0 instead of 0.01 the largest change is {np.max(np.abs(slope_h - slope)):.4f} dollars per year")

    # ------------------------------------------------------------------
    # 4. The shape of a segment's residuals.
    # ------------------------------------------------------------------
    print("\n=== 4. residual_summary against segment_position_report ===")
    bench = fit_huber_benchmark(df.age.to_numpy(), df.price.to_numpy(), degree=2)
    rs = residual_summary(df, "region", "price", benchmark_fit=bench, x_col="age")
    print(rs.round(0).to_string(index=False))
    sp = segment_position_report(df, "region", "price", benchmark_fit=bench, x_col="age")
    print(sp[["segment", "n", "difference", "ci_lower", "ci_upper"]].round(0).to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Four-way split.
    # ------------------------------------------------------------------
    print("\n=== 5. classify_quadrants ===")
    expected = benchmark_predict(bench, df, x_col="age")
    df["abs_gap"] = np.abs(df.price - expected) / expected          # relative distance from the benchmark
    q = classify_quadrants(df, "age", "abs_gap", x_threshold="median", y_threshold=0.9,
                           labels={"high_high": "older, far off", "high_y_only": "newer, far off",
                                   "high_x_only": "older, close", "low_low": "newer, close"})
    print(q.quadrant.value_counts().sort_index().to_string())
    print(f"thresholds in attrs: age {q.attrs['x_threshold']:.1f}, abs_gap {q.attrs['y_threshold']:.1%}")
    far = q[q.quadrant.str.contains("far off")]
    share = pd.DataFrame({"share_of_listings": q.region.value_counts(normalize=True),
                          "share_of_far_off": far.region.value_counts(normalize=True)}).round(3)
    print(share.to_string())

    # ------------------------------------------------------------------
    # 6. Interval for any statistic.
    # ------------------------------------------------------------------
    print("\n=== 6. bca_bootstrap_ci_by_index ===")
    south = df[df.region == "South"].reset_index(drop=True)
    res = south.price.to_numpy() - benchmark_predict(bench, south, x_col="age")

    def median_gap(idx):
        return float(np.median(res[idx]))

    by_hand = bca_bootstrap_ci_by_index(len(south), median_gap)
    row = sp[sp.segment == "South"].iloc[0]
    print(f"South median gap by hand: {by_hand['estimate']:,.0f} [{by_hand['lower']:,.0f}, {by_hand['upper']:,.0f}]")
    print(f"segment_position_report:  {row.difference:,.0f} [{row.ci_lower:,.0f}, {row.ci_upper:,.0f}]")

    def share_below(idx):
        return float(np.mean(res[idx] < 0))

    sb = bca_bootstrap_ci_by_index(len(south), share_below)
    print(f"share of South listings below benchmark: {sb['estimate']:.3f} [{sb['lower']:.3f}, {sb['upper']:.3f}]")

    def spread_p90_p10(idx):
        a, b = np.percentile(res[idx], [10, 90])
        return float(b - a)

    other = df[df.region != "South"].reset_index(drop=True)
    res_o = other.price.to_numpy() - benchmark_predict(bench, other, x_col="age")
    so = bca_bootstrap_ci_by_index(len(other), lambda idx: float(np.subtract(*np.percentile(res_o[idx], [90, 10]))))
    ss = bca_bootstrap_ci_by_index(len(south), spread_p90_p10)
    print(f"p90-p10 of residuals: South {ss['estimate']:,.0f} [{ss['lower']:,.0f}, {ss['upper']:,.0f}]; "
          f"others {so['estimate']:,.0f} [{so['lower']:,.0f}, {so['upper']:,.0f}]")
    print(f"keys returned: {', '.join(by_hand)}")

    # ------------------------------------------------------------------
    # 7. Feature map.
    # ------------------------------------------------------------------
    print("\n=== 7. feature_map ===")
    fm_df = df[["price", "age", "brand", "region", "listing_id"]]
    plt.close("all")
    fm = feature_map(fm_df, target="price")
    print(fm[["feature", "mutual_information", "information_efficiency", "quadrant"]].round(3).to_string(index=False))
    fm_fig = plt.gcf()
    fm_fig.savefig("feature_map.png", dpi=130)

    # ------------------------------------------------------------------
    # 8. The two standard trend charts.
    # ------------------------------------------------------------------
    print("\n=== 8. plot_analyst_view and plot_publisher_view ===")
    x, y = df.age.to_numpy(), df.price.to_numpy()
    rng = np.random.default_rng(5)
    small = rng.choice(len(df), 500, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharey='row')
    out = {}
    for j, (label, idx) in enumerate([("500 cars", small), (f"{len(df):,} cars", np.arange(len(df)))]):
        band = plot_analyst_view(x[idx], y[idx], ax=axes[0, j], title=f"Analyst view, {label}")
        pub = plot_publisher_view(x[idx], y[idx], ax=axes[1, j], title=f"Publisher view, {label}")
        w = (np.asarray(band["upper"]) - np.asarray(band["lower"])) / np.abs(np.asarray(band["median"])) * 100
        out[label] = (float(np.median(w)), float(pub.dispersion_ratio.median()))
    for label, (w, dr) in out.items():
        print(f"{label:>13}: median bootstrap band width {w:.2f}% of the bootstrap median curve; median dispersion_ratio of the bins {dr:.3f}")
    print(f"publisher view returns: {', '.join(pub.columns)}")
    fig.tight_layout()
    fig.savefig("trend_views.png", dpi=130)
    print("\nSaved feature_map.png and trend_views.png")


if __name__ == "__main__":
    main()
