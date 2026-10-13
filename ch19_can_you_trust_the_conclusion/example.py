"""
Chapter 19 -- Can You Trust the Conclusion?
============================================

Compass: a conclusion is trustworthy to the degree that it survives
changing the fitting method, dropping the awkward points, and
changing the segment. Measure each, then ask where the weakness is.

The staff table is synthetic and its weakness is built in:

  * pay is a concave function of experience times 6% lognormal noise,
  * a "bonus" multiplies pay by 1.6 for some people: 15% of SAL staff
    and 2% of everyone else.

A least-squares fit chases that bonus, a Huber or Tukey fit largely
ignores it. So the expected disagreement between methods is, to a good
approximation, the average pay uplift the bonus creates: about 9% in
SAL (15% x 60%), about 1.2% elsewhere (2% x 60%).

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import (
    feature_robustness_report, plot_feature_robustness,
    segment_stability_report, segment_stability_drilldown_report, segment_quality_report,
)

FAMILIES = ["ENG", "SAL", "OPS", "FIN"]
SEGMENT_COLS = ["job_family", "level", "site"]
MIN_SIZE = 100
FEATURES = ["experience", "tenure", "age", "commute_km", "team_size"]


def make_staff(seed=1301, n=6000):
    rng = np.random.default_rng(seed)
    site = rng.choice(["North", "Central", "South"], n, p=[0.35, 0.40, 0.25])
    level = rng.choice(["L1", "L2", "L3"], n, p=[0.40, 0.38, 0.22])
    family = rng.choice(FAMILIES, n, p=[0.30, 0.25, 0.25, 0.20])
    mean_exp = {"L1": 3, "L2": 9, "L3": 16}
    exp = np.clip(rng.normal([mean_exp[lv] for lv in level], 3.5), 0, 35)
    clean = (32000 + 2200 * exp - 40 * exp ** 2) * np.exp(rng.normal(0, 0.06, n))
    bonus = rng.random(n) < np.where(family == "SAL", 0.15, 0.02)
    salary = np.where(bonus, clean * 1.6, clean)
    df = pd.DataFrame({"site": site, "level": level, "job_family": family,
                       "experience": exp.round(1), "salary": salary.round(-1), "bonus": bonus})
    df["tenure"] = np.clip(df.experience * 0.7 + rng.normal(0, 3, n), 0, None).round(1)
    df["age"] = (22 + df.experience + rng.normal(0, 4, n)).round(0)
    df["commute_km"] = rng.uniform(0, 60, n).round(1)                     # unrelated to pay
    team = rng.integers(3, 12, n).astype(float)
    team[rng.choice(n, 20, replace=False)] = rng.uniform(80, 100, 20)       # 20 extreme values
    df["team_size"] = team
    df["clean_salary"] = clean
    return df


def main():
    df = make_staff()
    print(f"{len(df):,} employees. Bonus (x1.6): "
          + ", ".join(f"{f} {df.loc[df.job_family == f, 'bonus'].mean():.1%}" for f in FAMILIES)
          + f"; everyone {df.bonus.mean():.1%}.")
    uplift = df.salary.sum() / df.clean_salary.sum() - 1
    print(f"Average pay uplift from the bonus: {uplift:.1%} overall, "
          + ", ".join(f"{f} {df.loc[df.job_family == f, 'salary'].sum() / df.loc[df.job_family == f, 'clean_salary'].sum() - 1:.1%}"
                      for f in FAMILIES) + ".\n")

    # ------------------------------------------------------------------
    # 1. A map of features.
    # ------------------------------------------------------------------
    rep = feature_robustness_report(df, target="salary", features=FEATURES)
    print("=== 1. feature_robustness_report, default (median) thresholds ===")
    print(rep.round(3).to_string())

    # ------------------------------------------------------------------
    # 2. The labels are relative.
    # ------------------------------------------------------------------
    subset = feature_robustness_report(df, target="salary", features=["experience", "commute_km", "team_size"])
    by_hand = rep.set_index("feature").apply(
        lambda r: "within 5% and 2%" if r.stability_pct <= 5 and r.cook_impact_pct <= 2 else "outside", axis=1)
    print("\n=== 2. the same features, another candidate set, or a fixed tolerance ===")
    lab = pd.DataFrame({"default (5 features)": rep.set_index("feature").quadrant,
                        "3 features": subset.set_index("feature").quadrant,
                        "fixed tolerance, by hand": by_hand}).reindex(FEATURES)
    print(lab.fillna("-").to_string())
    e1 = rep.set_index("feature").loc["experience"]
    e2 = subset.set_index("feature").loc["experience"]
    print(f"experience: stability {e1.stability_pct:.3f} / impact {e1.cook_impact_pct:.3f} in the first, "
          f"{e2.stability_pct:.3f} / {e2.cook_impact_pct:.3f} in the second")

    # ------------------------------------------------------------------
    # 3. Where is the weakness? Segment by segment.
    # ------------------------------------------------------------------
    ss = segment_stability_report(df, x_col="experience", y_col="salary", segment_cols=SEGMENT_COLS, min_size=MIN_SIZE)
    fine = ss[ss.segment_level == 0].copy()
    fine["family"] = fine.segment.str.split("_").str[0]
    print("\n=== 3. segment_stability_report, finest cells ===")
    print(f"{len(ss)} segments, {len(fine)} at the finest level")
    print(fine.sort_values("median_pct_diff", ascending=False)[["segment", "n", "median_pct_diff", "max_pct_diff"]]
          .head(5).round(2).to_string(index=False))
    byfam = fine.groupby("family")["median_pct_diff"].agg(["count", "min", "median", "max"]).round(2).reindex(FAMILIES)
    print("median_pct_diff of the finest cells, by job family:")
    print(byfam.to_string())

    # ------------------------------------------------------------------
    # 4. Does granularity hide it?
    # ------------------------------------------------------------------
    dd = segment_stability_drilldown_report(df, x_col="experience", y_col="salary",
                                            segment_cols=SEGMENT_COLS, min_size=MIN_SIZE)
    lv = dd.groupby("segment_level")["median_pct_diff"].agg(["count", "min", "median", "max"]).round(2)
    print("\n=== 4. segment_stability_drilldown_report, median_pct_diff by hierarchy level ===")
    print(lv.to_string())

    # ------------------------------------------------------------------
    # 5. One table per segment.
    # ------------------------------------------------------------------
    q = segment_quality_report(df, x_col="experience", y_col="salary", segment_cols=SEGMENT_COLS, min_size=MIN_SIZE)
    qf = q[q.segment_level == 0].copy()
    qf["family"] = qf.segment.str.split("_").str[0]
    print("\n=== 5. segment_quality_report, finest cells ===")
    cols = ["segment", "n", "median_pct_diff", "n_flagged", "cook_impact_pct", "mdape", "iqr_resid"]
    print(qf.sort_values("median_pct_diff", ascending=False)[cols].head(5).round(2).to_string(index=False))
    print("median by job family:")
    print(qf.groupby("family")[["median_pct_diff", "cook_impact_pct", "mdape", "iqr_resid"]].median()
          .round(2).reindex(FAMILIES).to_string())
    top8 = qf.sort_values("median_pct_diff", ascending=False).head(8)
    print(f"Top 8 cells by stability spread that are SAL: {int((top8.family == 'SAL').sum())}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    plot_feature_robustness(report=rep, ax=ax[0])
    ax[0].set_title("Feature robustness map (median thresholds)")
    colours = {"SAL": "#d95f02", "ENG": "#1b9e77", "OPS": "#7570b3", "FIN": "#e7298a"}
    for fam in FAMILIES:
        sub = fine[fine.family == fam]
        ax[1].scatter(sub["n"], sub["median_pct_diff"], color=colours[fam], label=fam, s=40)
    ax[1].set_xlabel("people in the finest cell")
    ax[1].set_ylabel("method disagreement, median %")
    ax[1].set_title("Where the methods disagree")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("conclusion_trust.png", dpi=130)
    print("\nSaved conclusion_trust.png")


if __name__ == "__main__":
    main()
