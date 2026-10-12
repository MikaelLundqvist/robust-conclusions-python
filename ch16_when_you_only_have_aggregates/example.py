"""
Chapter 16 -- When You Only Have Aggregates
===========================================

Compass: a published table of Q1 / median / Q3 per group is not a
dataset with the rows removed. Some questions the table answers
directly; some it answers only if you reconstruct individuals from it
(and then you inherit the assumptions of the reconstruction); and some
it cannot answer at all. The skill is telling these three apart.

Because this chapter needs to know the right answer, it starts from
synthetic INDIVIDUAL-level data (which an agency has and you don't),
"publishes" the aggregates an agency would, and then checks what each
robustkit tool recovers against the truth it can no longer see.

Run with:

    python example.py
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import (
    load_json_stat, plot_quantile_trend, quantile_trend_dispersion,
    expand_aggregated_group, expand_aggregated_table, check_reconstruction_quality,
    fit_huber_trend, predict_trend, plot_huber_iqr, dispersion_by_bin,
)

YEARS = list(range(2012, 2025))
REGIONS = ["North", "Central", "South"]
STATS = [("N", "Number of patients"), ("Q1", "First quartile"),
         ("MED", "Median"), ("Q3", "Third quartile")]


# ---------------------------------------------------------------------------
# The truth (hidden from the analyst) and what gets published from it
# ---------------------------------------------------------------------------

def make_truth(seed=401):
    """
    Individual-level waiting times (days) for non-urgent surgery:
    13 years x 3 regions. Each region is built to teach something.

      North: steady rise in the median, constant relative spread.
      South:  the median rises only modestly, but the spread WIDENS --
            the long waits get longer while the typical wait barely does.
      Central: two kinds of patient (80% routine, 20% complex), so the
            distribution has two humps. Its quartiles look unremarkable.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for region in REGIONS:
        n = {"North": 2400, "Central": 3600, "South": 3000}[region]
        for year in YEARS:
            t = year - 2012
            if region == "North":
                days = rng.lognormal(np.log(40 * np.exp(0.050 * t)), 0.55, n)
            elif region == "South":
                days = rng.lognormal(np.log(45 * np.exp(0.020 * t)), 0.45 + 0.030 * t, n)
            else:
                is_complex = rng.random(n) < 0.20
                days = np.where(
                    is_complex,
                    rng.lognormal(np.log(160 * np.exp(0.035 * t)), 0.30, n),
                    rng.lognormal(np.log(30 * np.exp(0.035 * t)), 0.35, n),
                )
            parts.append(pd.DataFrame({"region": region, "year": year, "days": days}))
    return pd.concat(parts, ignore_index=True)


def publish_json_stat(truth, path):
    """
    Aggregate to what an agency publishes -- n, Q1, median, Q3 per
    region and year, rounded to whole days -- and write it as a
    JSON-stat file. (A real SCB file carries Swedish dimension labels
    such as "år"; load_json_stat uses whatever labels the file has.)
    """
    g = truth.groupby(["region", "year"])["days"]
    agg = pd.DataFrame({
        "N": g.size(), "Q1": g.quantile(0.25).round(0),
        "MED": g.quantile(0.50).round(0), "Q3": g.quantile(0.75).round(0),
    })
    values = [float(agg.loc[(r, y), code]) for r in REGIONS for code, _ in STATS for y in YEARS]
    dataset = {
        "dimension": {
            "Region": {"label": "region", "category": {
                "index": {r: i for i, r in enumerate(REGIONS)}, "label": {r: r for r in REGIONS}}},
            "Statistik": {"label": "statistic", "category": {
                "index": {c: i for i, (c, _) in enumerate(STATS)}, "label": dict(STATS)}},
            "Tid": {"label": "year", "category": {
                "index": {str(y): i for i, y in enumerate(YEARS)}, "label": {str(y): str(y) for y in YEARS}}},
            "id": ["Region", "Statistik", "Tid"],
            "size": [len(REGIONS), len(STATS), len(YEARS)],
            "role": {"time": ["Tid"]},
        },
        "value": values,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset}, f)


def to_wide(long):
    """load_json_stat returns one row per (region, statistic, year);
    the quantile tools want one row per (region, year) with q1/median/q3 columns."""
    wide = long.pivot_table(index=["region", "year"], columns="statistic", values="value").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"Number of patients": "n", "First quartile": "q1",
                                "Median": "median", "Third quartile": "q3"})
    wide["year"] = wide["year"].astype(int)
    return wide


def huber_slope(x, y):
    """Average change per year (days/year) from a straight Huber line."""
    fit = fit_huber_trend(np.asarray(x, float), np.asarray(y, float), degree=1)
    p = predict_trend(fit, np.array([2012.0, 2024.0]))
    return (p[1] - p[0]) / 12


def make_queue_data(seed=601, n=6000):
    """
    Individual waits (days) against the number of patients ahead in the
    queue at referral: 80% routine patients, 20% complex ones who wait
    about 2.5 times as long. Used in the last section, where robustkit's
    OWN chart is the thing being examined.
    """
    rng = np.random.default_rng(seed)
    ahead = rng.uniform(5, 60, n)
    typical = 20 + 2.2 * ahead
    is_complex = rng.random(n) < 0.20
    wait = np.where(is_complex, rng.lognormal(np.log(2.5 * typical), 0.20),
                    rng.lognormal(np.log(typical), 0.25))
    return ahead, wait


def matched_unimodal(x, y, bins=15, seed=7):
    """
    A second dataset with the same x, built so that its bars match the
    first: within each of plot_huber_iqr's own bins, y is drawn from a
    single lognormal calibrated to that bin's Q1/median/Q3 -- the
    reconstruction from earlier in this chapter. It is, by construction,
    "what the bars say, and nothing else".
    """
    binned = pd.qcut(x, q=bins, duplicates="drop")
    y_matched = np.empty_like(y)
    groups = pd.Series(range(len(x))).groupby(binned, observed=True).groups
    for i, (_, idx) in enumerate(groups.items()):
        idx = np.array(list(idx))
        q1, med, q3 = np.percentile(y[idx], [25, 50, 75])
        y_matched[idx] = expand_aggregated_group(len(idx), q1, med, q3, seed=seed + i)
    return y_matched


def info_box(x, y, metric):
    """The text robustkit puts in the chart's corner box, read back off the axes."""
    fig, ax = plt.subplots()
    plot_huber_iqr(x, y, ax=ax, residual_box_metric=metric)
    text = ax.texts[0].get_text().replace("\n", "   ")
    plt.close(fig)
    return text


def main():
    truth = make_truth()
    publish_json_stat(truth, "waiting_times.json")
    print(f"Hidden truth: {len(truth):,} individual waits. Published: waiting_times.json\n")

    # ------------------------------------------------------------------
    # 1. The quantiles ARE the answer
    # ------------------------------------------------------------------
    wide = to_wide(load_json_stat("waiting_times.json"))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    print("=== 1. Read straight from the published table ===")
    for j, region in enumerate(REGIONS):
        sub = wide[wide["region"] == region]
        plot_quantile_trend(sub, "year", "q1", "median", "q3", ax=axes[0, j], title=region)
        axes[0, j].set_ylim(0, 150)
        axes[0, j].set_xlabel("")  # shared x-axis: label only the bottom row
        d = quantile_trend_dispersion(sub, "year", "q1", "median", "q3")
        axes[1, j].plot(d["year"], d["dispersion_ratio"], marker="o", color="darkorange")
        axes[1, j].set_ylim(0.4, 1.4)
        axes[1, j].grid(True, alpha=0.3)
        axes[1, j].set_xlabel("year")
        first, last = d.iloc[0], d.iloc[-1]
        print(f"{region}: median {first['median']:.0f} -> {last['median']:.0f} days "
              f"({last['median'] / first['median'] - 1:+.0%});  "
              f"dispersion (Q3-Q1)/median {first['dispersion_ratio']:.2f} -> {last['dispersion_ratio']:.2f}")
    axes[0, 0].set_ylabel("waiting time (days)")
    axes[1, 0].set_ylabel("(Q3 - Q1) / median")
    fig.tight_layout()
    fig.savefig("published_quantile_trends.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Reconstruct individuals -- what comes back, checked against truth
    # ------------------------------------------------------------------
    recon = expand_aggregated_table(wide, n_col="n", q1_col="q1", median_col="median", q3_col="q3",
                                    group_cols=["region", "year"], value_name="days")

    print("\n=== 2. Trend slope (days per year): truth vs reconstruction vs the medians alone ===")
    print(f"{'region':7s} {'truth':>7s} {'reconstructed':>14s} {'fit to medians':>15s}")
    for region in REGIONS:
        t = truth[truth.region == region]
        r = recon[recon.region == region]
        w = wide[wide.region == region]
        print(f"{region:7s} {huber_slope(t.year, t.days):7.2f} "
              f"{huber_slope(r.year, r.days):14.2f} {huber_slope(w.year, w['median']):15.2f}")

    print("\ncheck_reconstruction_quality, 2024 (percent error against the published quartiles):")
    for region in ["North", "Central"]:
        row = wide[(wide.region == region) & (wide.year == 2024)].iloc[0]
        vals = recon[(recon.region == region) & (recon.year == 2024)]["days"]
        e = check_reconstruction_quality(vals, row.q1, row["median"], row.q3)["pct_error"]
        print(f"  {region}: Q1 {e['q1']:.1f}%   median {e['median']:.1f}%   Q3 {e['q3']:.1f}%")

    # At a few thousand draws, a few percent of error is mostly sampling
    # noise, so a single call cannot tell a well-fitting group from a
    # badly-fitting one. Reconstruct the SAME published triple at a very
    # large n and the noise disappears; what is left cannot be fixed by
    # drawing more values -- it is how far the triple is from the
    # log-symmetric shape a lognormal imposes.
    print("\nSame check with sampling noise removed (every group reconstructed at n = 1,000,000):")
    print(f"{'region':7s} {'typical error':>14s} {'worst group':>12s}")
    for region in REGIONS:
        errs = []
        for _, row in wide[wide.region == region].iterrows():
            v = expand_aggregated_group(1_000_000, row.q1, row["median"], row.q3, seed=0)
            e = check_reconstruction_quality(v, row.q1, row["median"], row.q3)["pct_error"]
            errs.append(max(e["q1"], e["q3"]))
        print(f"{region:7s} {np.mean(errs):13.1f}% {np.max(errs):11.1f}%")

    # ------------------------------------------------------------------
    # 3. What the three published numbers cannot carry
    # ------------------------------------------------------------------
    print("\n=== 3. A question about the tail, 2024: waits over 120 days ===")
    print(f"{'region':7s} {'true share':>11s} {'reconstructed':>14s} {'true p95':>9s} {'recon p95':>10s}")
    for region in REGIONS:
        t = truth[(truth.region == region) & (truth.year == 2024)]["days"]
        c = recon[(recon.region == region) & (recon.year == 2024)]["days"]
        print(f"{region:7s} {(t > 120).mean():11.1%} {(c > 120).mean():14.1%} "
              f"{t.quantile(.95):9.0f} {c.quantile(.95):10.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    bins = np.logspace(np.log10(5), np.log10(900), 50)
    for ax, region in zip(axes, ["North", "Central"]):
        t = truth[(truth.region == region) & (truth.year == 2024)]["days"]
        c = recon[(recon.region == region) & (recon.year == 2024)]["days"]
        ax.hist(t, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color="steelblue", label="truth (hidden)")
        ax.hist(c, bins=bins, density=True, histtype="step", linewidth=2, color="darkorange", label="reconstructed")
        ax.axvline(120, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("waiting time (days, log scale)")
        ax.set_title(f"{region} 2024:  over 120 days  truth {(t > 120).mean():.0%},  reconstructed {(c > 120).mean():.0%}")
        ax.legend()
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig("reconstruction_tail.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. The size of the group you reconstruct changes the tail, not the quartiles
    # ------------------------------------------------------------------
    row = wide[(wide.region == "North") & (wide.year == 2024)].iloc[0]
    print(f"\n=== 4. Same published quartiles ({row.q1:.0f} / {row['median']:.0f} / {row.q3:.0f}), "
          f"reconstructed at different n ===")
    print(f"{'n':>12s} {'Q1 err':>7s} {'median err':>11s} {'Q3 err':>7s} {'longest wait':>13s}")
    for n in [2_400, 240_000, 2_400_000]:
        v = expand_aggregated_group(n, row.q1, row["median"], row.q3, seed=7)
        e = check_reconstruction_quality(v, row.q1, row["median"], row.q3)["pct_error"]
        print(f"{n:12,d} {e['q1']:6.2f}% {e['median']:10.2f}% {e['q3']:6.2f}% {v.max():10,.0f} days")

    # ------------------------------------------------------------------
    # 5. robustkit's own charts summarise data the same way
    # ------------------------------------------------------------------
    # dispersion_by_bin -- the function behind plot_huber_iqr's error bars --
    # produces exactly the table an agency would publish: n, Q1, median, Q3 per bin.
    x, y_a = make_queue_data()
    y_b = matched_unimodal(x, y_a)
    bars_a, bars_b = dispersion_by_bin(x, y_a, n_bins=15), dispersion_by_bin(x, y_b, n_bins=15)

    print("\n=== 5. robustkit's own chart: two datasets with the same bars ===")
    print("A = 20% complex cases;  B = one kind of case, bars matched to A")
    print("Per-bin bar endpoints, B against A (relative difference):")
    for col in ["q1", "median", "q3"]:
        d = np.abs(bars_b[col].to_numpy() / bars_a[col].to_numpy() - 1)
        print(f"  {col:6s} average {d.mean():.1%}   largest {d.max():.1%}")

    bins = pd.qcut(x, 15, duplicates="drop")
    med_a = pd.Series(y_a).groupby(bins, observed=True).transform("median").to_numpy()
    med_b = pd.Series(y_b).groupby(bins, observed=True).transform("median").to_numpy()
    print(f"Share of waits more than twice their bin's median:  A {np.mean(y_a > 2 * med_a):.1%}   B {np.mean(y_b > 2 * med_b):.1%}")
    print("Corner box, default:  A  " + info_box(x, y_a, "r2"))
    print("                      B  " + info_box(x, y_b, "r2"))
    print("Corner box, mdape:    A  " + info_box(x, y_a, "mdape"))
    print("                      B  " + info_box(x, y_b, "mdape"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for j, (y, title) in enumerate([(y_a, "A: two kinds of patient (20% complex)"),
                                    (y_b, "B: one kind, bars matched to A")]):
        plot_huber_iqr(x, y, ax=axes[0, j], show_points=False, ylim=(0, 330), title=title)
        plot_huber_iqr(x, y, ax=axes[1, j], show_points=True, ylim=(0, 700), title="")
    axes[0, 0].set_ylabel("publisher view (no points)")
    axes[1, 0].set_ylabel("analyst view (every point)")
    axes[1, 0].set_xlabel("patients ahead in queue")
    axes[1, 1].set_xlabel("patients ahead in queue")
    fig.tight_layout()
    fig.savefig("own_charts_blind_spot.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
