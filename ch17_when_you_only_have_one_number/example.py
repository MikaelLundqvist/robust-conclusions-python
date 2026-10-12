"""
Chapter 17 -- When You Only Have One Number
===========================================

Compass: Chapter 16 had three numbers per group (Q1, median, Q3). Often
a table publishes only one -- a mean, or a median. Which one it is
decides which questions the table can answer, and no reconstruction
can change that.

As in Chapter 16, the individual-level data are synthetic and hidden,
so every answer can be checked against the truth. Length of hospital
stay by age band, two departments: heavily right-skewed, far more old
patients than young ones, and the spread WIDENS with age.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

from robustkit import (
    fit_ols_trend, fit_huber_trend, predict_trend,
    expand_aggregated_table_flat, expand_aggregated_table_borrowed_dispersion,
    expand_aggregated_group_borrowed_dispersion, compare_reconstruction_methods,
)

AGES = np.arange(22.5, 90, 5.0)          # 14 age-band midpoints
DEPTS = ["Medical", "Surgical"]
TAIL_DAYS = 14                           # "a long stay" for the tail question


def make_truth(seed=501):
    """Hidden individual-level length of stay (days)."""
    rng = np.random.default_rng(seed)
    parts = []
    for dept in DEPTS:
        for age in AGES:
            d = (age - 20) / 10.0                          # decades above 20
            if dept == "Medical":
                n = int(round(250 * np.exp(0.43 * d)))
                med, sigma = 3.0 * np.exp(0.18 * d), 0.55 + 0.045 * d
            else:
                n = int(round(350 * np.exp(0.33 * d)))
                med, sigma = 2.5 * np.exp(0.10 * d), 0.50 + 0.030 * d
            parts.append(pd.DataFrame({"department": dept, "age": age,
                                       "los": rng.lognormal(np.log(med), sigma, n)}))
    return pd.concat(parts, ignore_index=True)


def publish(truth):
    """The two tables an agency might publish: n + mean, or n + median,
    per department and age band, rounded to 0.1 day."""
    g = truth.groupby(["department", "age"])["los"]
    base = pd.DataFrame({"n": g.size()}).reset_index()
    return base.assign(value=g.mean().round(1).values), base.assign(value=g.median().round(1).values)


def slope(fit_fn, x, y):
    """Average change in days per DECADE of age, over the observed range."""
    f = fit_fn(np.asarray(x, float), np.asarray(y, float), degree=1)
    p = predict_trend(f, np.array([AGES[0], AGES[-1]]))
    return (p[1] - p[0]) / ((AGES[-1] - AGES[0]) / 10)


def slopes(fit_fn, df, col):
    return {d: slope(fit_fn, df[df.department == d].age, df[df.department == d][col]) for d in DEPTS}


def pct(x, ref):
    v = round((x - ref) / ref * 100)
    return "0%" if v == 0 else f"{v:+d}%"


def main():
    truth = make_truth()
    means_table, medians_table = publish(truth)
    med_pat = truth[truth.department == "Medical"]
    print(f"Hidden truth: {len(truth):,} patients. Age bands hold between "
          f"{means_table.n.min()} and {means_table.n.max()} of them.\n")

    # ------------------------------------------------------------------
    # 1. The mean and the median ask different questions
    # ------------------------------------------------------------------
    T_mean = slopes(fit_ols_trend, truth, "los")        # trend in the MEAN
    T_typ = slopes(fit_huber_trend, truth, "los")       # trend of the TYPICAL stay
    print("=== 1. Two true trends (days per decade of age), from the hidden individuals ===")
    print(f"{'':40s} {'Medical':>8s} {'Surgical':>9s}")
    print(f"{'trend in the MEAN (OLS)':40s} {T_mean['Medical']:8.2f} {T_mean['Surgical']:9.2f}")
    print(f"{'trend of the TYPICAL stay (Huber)':40s} {T_typ['Medical']:8.2f} {T_typ['Surgical']:9.2f}")
    print("\nMedical, hidden truth by age band:")
    print(f"{'age':>6s} {'median':>7s} {'mean':>6s} {'mean/median':>12s} {'(Q3-Q1)/median':>15s}")
    for age in [AGES[0], AGES[5], AGES[-1]]:
        v = med_pat[med_pat.age == age].los
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        print(f"{age:6.1f} {q2:7.1f} {v.mean():6.1f} {v.mean() / q2:12.2f} {(q3 - q1) / q2:15.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    g = med_pat.groupby("age")["los"]
    axes[0].fill_between(AGES, g.quantile(.25), g.quantile(.75), color="gray", alpha=0.25,
                         label="Q1-Q3 (hidden)")
    axes[0].plot(AGES, g.median(), color="darkorange", linewidth=2, marker="o", label="median")
    axes[0].plot(AGES, g.mean(), color="steelblue", linewidth=2, marker="s", label="mean")
    axes[0].set_xlabel("age band midpoint"); axes[0].set_ylabel("length of stay (days)")
    axes[0].set_title("Medical: the mean pulls away as the spread widens"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    width = 1.6
    for k, dept in enumerate(DEPTS):
        nn = means_table[means_table.department == dept].n
        axes[1].bar(AGES + (k - 0.5) * width, nn, width=width, label=dept)
    axes[1].set_xlabel("age band midpoint"); axes[1].set_ylabel("patients in the band")
    axes[1].set_title("Far more old patients than young ones"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("mean_vs_median.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Flat reconstruction: exact for the question the table can answer
    # ------------------------------------------------------------------
    def flat(table):
        return expand_aggregated_table_flat(table, "n", "value", group_cols=["department", "age"],
                                            value_name="los")
    flat_means, flat_meds = flat(means_table), flat(medians_table)

    print("\n=== 2. Which table answers which question?  (Medical, days per decade) ===")
    cells = {
        ("means", "mean"): slopes(fit_ols_trend, flat_means, "los")["Medical"],
        ("medians", "mean"): slopes(fit_ols_trend, flat_meds, "los")["Medical"],
        ("means", "typical"): slopes(fit_huber_trend, flat_means, "los")["Medical"],
        ("medians", "typical"): slopes(fit_huber_trend, flat_meds, "los")["Medical"],
    }
    print(f"{'published table':16s} {'question: trend in the MEAN':>30s} {'question: TYPICAL stay':>30s}")
    print(f"{'  (truth)':16s} {T_mean['Medical']:>30.2f} {T_typ['Medical']:>30.2f}")
    for tbl in ["means", "medians"]:
        a, b = cells[(tbl, "mean")], cells[(tbl, "typical")]
        print(f"{tbl:16s} {a:>22.2f} ({pct(a, T_mean['Medical']):>5s})     "
              f"{b:>22.2f} ({pct(b, T_typ['Medical']):>5s})")
    print("(mean question: flat reconstruction + OLS;  typical question: flat reconstruction + Huber)")
    print(f"\nflat + OLS on the means table: {cells[('means', 'mean')]:.3f}   |   "
          f"OLS on the hidden individuals: {T_mean['Medical']:.3f}")

    # Why flat is exact: repeating a mean n times IS weighting it by n.
    p = means_table[means_table.department == "Medical"]
    w = HuberRegressor(epsilon=1.35, alpha=1e-4).fit(
        p.age.to_numpy().reshape(-1, 1), p.value.to_numpy(), sample_weight=p.n.to_numpy())
    print(f"\nflat + Huber on the means table: {slopes(fit_huber_trend, flat_means, 'los')['Medical']:.3f}   |   "
          f"Huber on the 14 means, weighted by n: {w.coef_[0] * 10:.3f}")
    print(f"unweighted OLS on the 14 means (a different question: the average band, "
          f"not the average patient): {slope(fit_ols_trend, p.age, p.value):.2f}")

    # Only the RELATIVE group sizes matter, so n can be scaled down -- useful
    # when a real table reports millions per group.
    print("\nFlat reconstruction with every n scaled down by a common factor (Medical means table):")
    print(f"{'n scaled by':>12s} {'rows':>8s} {'OLS':>8s} {'Huber':>8s}")
    for k in [1, 0.1, 0.01]:
        scaled = p.assign(n=np.maximum(1, (p.n * k).round()).astype(int))
        r = expand_aggregated_table_flat(scaled, "n", "value", group_cols=["age"], value_name="los")
        print(f"{k:12g} {len(r):8,d} {slope(fit_ols_trend, r.age, r.los):8.4f} "
              f"{slope(fit_huber_trend, r.age, r.los):8.4f}")

    # ------------------------------------------------------------------
    # 3. What neither number carries: the spread
    # ------------------------------------------------------------------
    print("\n=== 3. How much do conclusions depend on the spread you assume? ===")
    ratios = [0.2, 0.4, 0.8, 1.2, 1.6]
    sweep = []
    for r in ratios:
        b = expand_aggregated_table_borrowed_dispersion(
            medians_table, "n", "value", r, group_cols=["department", "age"], value_name="los")
        sweep.append(slopes(fit_huber_trend, b, "los")["Medical"])
    print("Typical-stay trend (Medical) from the medians table, borrowed ratio varied "
          f"(truth {T_typ['Medical']:.2f}; flat {cells[('medians', 'typical')]:.2f}):")
    print("  " + "   ".join(f"{r}: {s:.2f}" for r, s in zip(ratios, sweep)))

    top = medians_table[(medians_table.department == "Medical") & (medians_table.age == AGES[-1])].iloc[0]
    n_top, med_top = int(top.n), float(top.value)
    true_share = (med_pat[med_pat.age == AGES[-1]].los > TAIL_DAYS).mean()
    tail = {"flat": float((np.full(n_top, med_top) > TAIL_DAYS).mean())}
    for r in [0.4, 0.8, 1.2]:
        tail[f"ratio {r}"] = float((expand_aggregated_group_borrowed_dispersion(n_top, med_top, r, seed=3) > TAIL_DAYS).mean())
    print(f"\nShare of oldest-band Medical stays longer than {TAIL_DAYS} days "
          f"(truth {true_share:.0%}), from the published median {med_top}:")
    print("  " + "   ".join(f"{k}: {v:.0%}" for k, v in tail.items()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot(ratios, sweep, marker="o", color="darkorange", label="borrowed dispersion")
    axes[0].axhline(T_typ["Medical"], color="black", linestyle="--", label="truth")
    axes[0].axhline(cells[("medians", "typical")], color="steelblue", linestyle=":", label="flat")
    axes[0].set_ylim(0, 2.0); axes[0].set_xlabel("borrowed dispersion ratio (Q3-Q1)/median")
    axes[0].set_ylabel("days per decade"); axes[0].set_title("Question 1, the trend: barely moves")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    labels = list(tail.keys())
    bars = axes[1].bar(labels, [tail[k] * 100 for k in labels], color="darkorange")
    for bar, k in zip(bars, labels):
        axes[1].text(bar.get_x() + bar.get_width() / 2, tail[k] * 100 + 1, f"{tail[k]:.0%}", ha="center")
    axes[1].axhline(true_share * 100, color="black", linestyle="--", label=f"truth ({true_share:.0%})")
    axes[1].set_ylabel(f"% of stays over {TAIL_DAYS} days"); axes[1].set_ylim(0, 60)
    axes[1].set_title("Question 2, the tail: depends entirely on the assumption")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig("spread_sensitivity.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------
    # 4. A mean in the median's slot
    # ------------------------------------------------------------------
    mean_top = float(means_table[(means_table.department == "Medical") & (means_table.age == AGES[-1])].value.iloc[0])
    print(f"\n=== 4. compare_reconstruction_methods, oldest Medical band, ratio 0.8 ===")
    for label, val in [("published MEAN", mean_top), ("published MEDIAN", med_top)]:
        out = compare_reconstruction_methods(n_top, val, 0.8, seed=1)
        b = out["borrowed_dispersion"]
        print(f"fed with the {label} ({val}):  reconstructed median {np.median(b):.1f},  "
              f"reconstructed mean {np.mean(b):.1f}")


if __name__ == "__main__":
    main()
