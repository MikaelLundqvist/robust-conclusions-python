"""
Chapter 21 -- Does the Flag Survive?
====================================

Compass: an outlier flag is a statement about a person AND a
reference group. Before acting on it, change the reference group and
see whether the flag is still there.

The staff table is synthetic and its structure is built in:

  * Pay is a function of experience, times a site multiplier (North
    -8%, Central 0, South +3%), times noise. The noise differs by job
    family: OPS is tight (4%), FIN 7%, SAL 8%, ENG loose (10%).
  * 36 people are planted with a pay cut relative to their own cell:
      "deep"  : 12 people (ENG/SAL/FIN) paid 30% less than expected,
      "tight" : 12 OPS people paid 15% less than expected,
      "noisy" : 12 ENG people paid 15% less than expected.
    The other 5,964 are "none": nothing was done to them.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import (
    mad_outlier_report, mad_outlier_drilldown_report, outlier_drilldown_summary,
    dual_reference_outlier_report, dual_reference_outlier_drilldown_report,
)

SITES = ["North", "Central", "South"]
LEVELS = ["L1", "L2", "L3"]
FAMILIES = ["ENG", "SAL", "OPS", "FIN"]
SITE_MULT = {"North": 0.92, "Central": 1.00, "South": 1.03}
NOISE = {"ENG": 0.10, "SAL": 0.08, "OPS": 0.04, "FIN": 0.07}
SEGMENT_COLS = ["job_family", "level", "site"]      # most specific first
MIN_SIZE = 100
GROUPS = ["deep", "tight", "noisy"]


def make_staff(seed=901, n=6000):
    rng = np.random.default_rng(seed)
    site = rng.choice(SITES, n, p=[0.35, 0.40, 0.25])
    level = rng.choice(LEVELS, n, p=[0.40, 0.38, 0.22])
    family = rng.choice(FAMILIES, n, p=[0.30, 0.25, 0.25, 0.20])
    mean_exp = {"L1": 3, "L2": 9, "L3": 16}
    exp = np.clip(rng.normal([mean_exp[lv] for lv in level], 3.5), 0, 35)
    base = 32000 + 1500 * exp - 20 * exp ** 2
    sd = np.array([NOISE[f] for f in family])
    salary = base * np.array([SITE_MULT[s] for s in site]) * np.exp(rng.normal(0, sd))
    df = pd.DataFrame({"employee_id": [f"E{i:05d}" for i in range(n)], "site": site, "level": level,
                       "job_family": family, "experience": exp.round(1), "salary": salary})
    df["planted"] = "none"

    def plant(mask, k, mult, label):
        idx = rng.choice(df.index[mask & (df.planted == "none")], k, replace=False)
        df.loc[idx, "salary"] *= mult
        df.loc[idx, "planted"] = label

    plant(df.job_family.isin(["ENG", "SAL", "FIN"]), 12, 0.70, "deep")
    plant(df.job_family == "OPS", 12, 0.85, "tight")
    plant(df.job_family == "ENG", 12, 0.85, "noisy")
    df["salary"] = df["salary"].round(-1)
    return df


def main():
    df = make_staff()
    truth = df.set_index("employee_id")["planted"]
    print(f"{len(df):,} employees. Planted: " + ", ".join(f"{g} {int((df.planted == g).sum())}" for g in GROUPS)
          + f"; untouched {int((df.planted == 'none').sum()):,}.\n")
    kw = dict(y_col="salary", x_col="experience", segment_cols=SEGMENT_COLS, min_size=MIN_SIZE,
              id_cols=["employee_id"])

    # ------------------------------------------------------------------
    # 1. The exclusive report: one reference group per person.
    # ------------------------------------------------------------------
    ex = mad_outlier_report(df, **kw)
    flagged = ex[ex.flagged]
    kinds = truth.loc[flagged.employee_id].value_counts()
    print("=== 1. mad_outlier_report: each person compared with one segment ===")
    print(f"{len(ex):,} rows, {len(flagged)} flagged. Of them: "
          + ", ".join(f"{g} {int(kinds.get(g, 0))}" for g in GROUPS + ["none"]))

    # ------------------------------------------------------------------
    # 2. The drilldown: every level, independently.
    # ------------------------------------------------------------------
    dd = mad_outlier_drilldown_report(df, **kw)
    per_level = dd.groupby("segment_level").agg(rows=("flagged", "size"), flagged=("flagged", "sum"))
    print("\n=== 2. mad_outlier_drilldown_report: the same person at every level ===")
    print(per_level.to_string())
    print(f"{len(dd):,} rows for {df.employee_id.nunique():,} people")

    # ------------------------------------------------------------------
    # 3. How many levels does each flag survive?
    # ------------------------------------------------------------------
    summ = outlier_drilldown_summary(dd, id_col="employee_id")
    summ["planted"] = truth.loc[summ.employee_id].to_numpy()
    tab = (summ.groupby(["planted", "outlier_levels"]).size().unstack(fill_value=0)
           .reindex(GROUPS + ["none"]))
    tab.insert(0, "never", [int((truth == g).sum()) - int(tab.loc[g].sum()) for g in tab.index])
    tab.columns = ["never flagged"] + [f"{c} level" + ("s" if c > 1 else "") for c in tab.columns[1:]]
    print("\n=== 3. outlier_drilldown_summary: levels flagged, by what was planted ===")
    print(tab.to_string())
    top = summ[summ.outlier_levels == 3]
    kinds3 = truth.loc[top.employee_id].value_counts()
    print(f"Flagged at all 3 levels: {len(top)} people, of whom "
          + ", ".join(f"{g} {int(kinds3.get(g, 0))}" for g in GROUPS + ["none"]))
    fam_none = df.set_index("employee_id").loc[top.employee_id[truth.loc[top.employee_id].to_numpy() == "none"],
                                                 "job_family"].value_counts()
    print("The untouched people flagged at all 3 levels, by job family: "
          + ", ".join(f"{f} {int(v)}" for f, v in fam_none.items()))
    only0 = summ[(summ.outlier_levels == 1) & summ.levels.apply(lambda v: list(v) == [0])]
    kinds0 = truth.loc[only0.employee_id].value_counts()
    print(f"Flagged at the finest level only: {len(only0)} people, of whom "
          + ", ".join(f"{g} {int(kinds0.get(g, 0))}" for g in GROUPS + ["none"]))

    # ------------------------------------------------------------------
    # 4. Why: the threshold moves with the reference group.
    # ------------------------------------------------------------------
    dd["family"] = dd.employee_id.map(df.set_index("employee_id")["job_family"])
    dd["threshold_pct"] = dd["threshold"] / dd["expected"] * 100
    thr = dd.groupby(["family", "segment_level"])["threshold_pct"].median().unstack().round(1)
    print("\n=== 4. Median flagging threshold, % of expected pay, by job family and level ===")
    print(thr.loc[["OPS", "FIN", "SAL", "ENG"]].to_string())

    # ------------------------------------------------------------------
    # 5. Two references: colleagues and the organisation.
    # ------------------------------------------------------------------
    dual = dual_reference_outlier_drilldown_report(df, y_col="salary", x_col="experience",
                                                   segment_cols=SEGMENT_COLS, min_size=MIN_SIZE,
                                                   id_cols=["employee_id"])
    types = dual.groupby(["segment_level", "reference_type"]).size().unstack(fill_value=0)
    print("\n=== 5. dual_reference_outlier_drilldown_report: reference types by level ===")
    print(types.to_string())
    dual["site"] = dual.employee_id.map(df.set_index("employee_id")["site"])
    c_rows = dual[dual.reference_type == "C"]
    print("Type C rows by site: " + ", ".join(f"{s} {int((c_rows.site == s).sum())}" for s in SITES))
    a_all = (dual[dual.reference_type == "A"].groupby("employee_id")["segment_level"].nunique())
    a_all = a_all[a_all == 3]
    kindsA = truth.loc[a_all.index].value_counts()
    print(f"Type A at all 3 levels: {len(a_all)} people, of whom "
          + ", ".join(f"{g} {int(kindsA.get(g, 0))}" for g in GROUPS + ["none"]))
    both = set(a_all.index) & set(top.employee_id)
    print(f"Also flagged at all 3 levels by the MAD drilldown: {len(both)} of them")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = ["never flagged", "1 level", "2 levels", "3 levels"]
    colours = ["#d9d9d9", "#9ecae1", "#4292c6", "#08306b"]
    left = np.zeros(len(GROUPS))
    for c, colour in zip(cols, colours):
        vals = tab.loc[GROUPS, c].to_numpy() if c in tab.columns else np.zeros(len(GROUPS))
        ax[0].barh(GROUPS, vals, left=left, color=colour, label=c)
        left += vals
    ax[0].invert_yaxis()
    ax[0].set_xlabel("planted people (12 each)")
    ax[0].set_title("How many levels does the flag survive?")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=8, frameon=False)
    for fam, colour in [("OPS", "#d95f02"), ("ENG", "#1b9e77")]:
        ax[1].plot(thr.columns, thr.loc[fam], marker="o", label=f"{fam} (noise {NOISE[fam]:.0%})", color=colour)
    ax[1].axhline(15, color="grey", ls="--", lw=1)
    ax[1].text(0.02, 15.6, "planted cut: 15%", fontsize=8, color="grey")
    ax[1].set_xticks(list(thr.columns))
    ax[1].set_xlabel("hierarchy level (0 = finest)")
    ax[1].set_ylabel("threshold, % of expected pay")
    ax[1].set_title("The bar a person must clear")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("flag_survival.png", dpi=130)
    print("\nSaved flag_survival.png")


if __name__ == "__main__":
    main()
