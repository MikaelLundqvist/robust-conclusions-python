"""
Appendix A -- Which Features Make a Good Chart?
===============================================

Compass: the feature that predicts best is not always the one that
communicates best, and two features that are each useful may be
redundant together. Measure relevance, legibility, redundancy and
synergy separately before choosing what goes on the page.

The staff table is synthetic and its structure is built in:

  * pay depends strongly on experience (continuous) and on level,
    which is itself driven by experience (so the two overlap),
  * job family is a moderate multiplier (ENG x1.12, SAL x1.00,
    OPS x0.92, FIN x1.05),
  * commission_plan (yes/no, assigned at random) adds 30% -- but only
    for SAL staff (an interaction),
  * site has no effect on pay,
  * office_code (40 values) and badge_number (200 values) are noise.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import (
    rank_features, rank_by_communication, communication_score, pair_redundancy,
    pair_synergy, conditional_mutual_information, rank_communicative_pairs,
)

FEATURES = ["experience", "level", "job_family", "commission_plan", "site", "office_code", "badge_number"]
PAIR_FEATURES = ["experience", "level", "job_family", "site", "commission_plan"]


def make_staff(seed=1401, n=4000):
    rng = np.random.default_rng(seed)
    site = rng.choice(["North", "Central", "South"], n, p=[0.35, 0.40, 0.25])
    level = rng.choice(["L1", "L2", "L3"], n, p=[0.40, 0.38, 0.22])
    family = rng.choice(["ENG", "SAL", "OPS", "FIN"], n, p=[0.30, 0.25, 0.25, 0.20])
    mean_exp = {"L1": 3, "L2": 9, "L3": 16}
    exp = np.clip(rng.normal([mean_exp[lv] for lv in level], 3.5), 0, 35)
    plan = rng.choice(["yes", "no"], n)
    base = 32000 + 2200 * exp - 40 * exp ** 2
    mult = {"ENG": 1.12, "SAL": 1.0, "OPS": 0.92, "FIN": 1.05}
    salary = (base * np.array([mult[f] for f in family])
              * np.where((family == "SAL") & (plan == "yes"), 1.30, 1.0) * np.exp(rng.normal(0, 0.06, n)))
    return pd.DataFrame({
        "salary": salary.round(-1), "experience": exp.round(1), "level": level, "job_family": family,
        "site": site, "commission_plan": plan,
        "badge_number": rng.integers(0, 200, n).astype(str),
        "office_code": rng.choice([f"O{i:02d}" for i in range(40)], n)})


def main():
    df = make_staff()
    print(f"{len(df):,} employees; target salary. Built in: experience and level drive pay; job family "
          f"moderate; commission_plan +30% in SAL only; site, office_code, badge_number no effect.\n")

    # ------------------------------------------------------------------
    # 1. Relevance only (the Chapter 8 view).
    # ------------------------------------------------------------------
    rf = rank_features(df, target="salary")
    print("=== 1. rank_features: relevance to salary ===")
    print(rf.round(3).to_string())

    # ------------------------------------------------------------------
    # 2. Relevance plus legibility.
    # ------------------------------------------------------------------
    rc = rank_by_communication(df, target="salary")
    print("\n=== 2. rank_by_communication, default weights ===")
    print(rc.round(3).to_string())

    # ------------------------------------------------------------------
    # 3. One feature in isolation.
    # ------------------------------------------------------------------
    one = communication_score(df, "job_family", "salary")
    print("\n=== 3. communication_score for job_family alone ===")
    print(f"MI normalised against itself: {one['mutual_information']:.3f} "
          f"(raw {one['raw_mutual_information']:.3f}; in the ranking above raw was "
          f"{float(rc.loc[rc.feature == 'job_family', 'raw_mutual_information'].iloc[0]):.3f}, normalised "
          f"{float(rc.loc[rc.feature == 'job_family', 'mutual_information'].iloc[0]):.3f})")
    print(f"communication_index alone {one['communication_index']:.3f} vs in the ranking "
          f"{float(rc.loc[rc.feature == 'job_family', 'communication_index'].iloc[0]):.3f}")

    # ------------------------------------------------------------------
    # 4. The audience: weights and priors.
    # ------------------------------------------------------------------
    w = {"mutual_information": 0.25, "stability": 0.15, "group_size_score": 0.10,
         "compressibility": 0.10, "interpretability": 0.40}
    prior = {"experience": 0.2, "level": 1.0, "job_family": 1.0}
    rp = rank_by_communication(df, target="salary", weights=w, interpretability=prior)
    merged = rank_by_communication(df, target="salary", interpretability=prior, weights={"interpretability": 0.5})
    print("\n=== 4. weights and an audience prior (experience 0.2, level 1.0, job_family 1.0) ===")
    print("weights summing to 1 (interpretability 0.40):")
    print(rp[["feature", "communication_index"]].round(3).to_string(index=False))
    print(f"weights={{'interpretability': 0.5}} alone is merged with the defaults: weights sum to "
          f"{sum({**{'mutual_information': 0.35, 'stability': 0.25, 'group_size_score': 0.2, 'compressibility': 0.1, 'interpretability': 0.1}, 'interpretability': 0.5}.values()):.2f}; "
          f"top index {merged.communication_index.max():.3f}")

    # ------------------------------------------------------------------
    # 5. Redundancy between two features.
    # ------------------------------------------------------------------
    print("\n=== 5. pair_redundancy (0 = independent, 1 = redundant) ===")
    print(f"experience / level   {pair_redundancy(df, 'experience', 'level'):.3f}")
    print(f"site / job_family    {pair_redundancy(df, 'site', 'job_family'):.3f}")

    # ------------------------------------------------------------------
    # 6. Synergy and conditional information.
    # ------------------------------------------------------------------
    sub = {"SAL": df[df.job_family == "SAL"], "other": df[df.job_family != "SAL"]}
    mi_within = {k: float(rank_features(v[["salary", "commission_plan"]], target="salary")
                          .set_index("feature").loc["commission_plan", "mutual_information"]) for k, v in sub.items()}
    print("\n=== 6. synergy and conditional information ===")
    print(f"commission_plan MI with salary: whole staff "
          f"{float(rf.set_index('feature').loc['commission_plan', 'mutual_information']):.3f}, "
          f"within SAL {mi_within['SAL']:.3f}, within the rest {mi_within['other']:.3f}")
    print(f"pair_synergy(job_family, commission_plan) = {pair_synergy(df, 'job_family', 'commission_plan', 'salary'):.3f}")
    print(f"pair_synergy(commission_plan, job_family) = {pair_synergy(df, 'commission_plan', 'job_family', 'salary'):.3f}")
    print(f"pair_synergy(site, job_family)            = {pair_synergy(df, 'site', 'job_family', 'salary'):.3f}")
    print(f"pair_synergy(experience, job_family)      = {pair_synergy(df, 'experience', 'job_family', 'salary'):.3f}")
    print(f"pair_synergy(job_family, experience)      = {pair_synergy(df, 'job_family', 'experience', 'salary'):.3f}")
    print(f"conditional_mutual_information(commission_plan; salary | job_family) = "
          f"{conditional_mutual_information(df, 'commission_plan', 'salary', 'job_family'):.3f}")
    print(f"conditional_mutual_information(commission_plan; salary | site)       = "
          f"{conditional_mutual_information(df, 'commission_plan', 'salary', 'site'):.3f}")

    # ------------------------------------------------------------------
    # 7. Pairs ranked.
    # ------------------------------------------------------------------
    pairs = rank_communicative_pairs(df, target="salary", features=PAIR_FEATURES)
    print("\n=== 7. rank_communicative_pairs ===")
    print(pairs.round(3).to_string())

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    r = rc.iloc[::-1]
    ax[0].barh(r.feature, r.communication_index, color="#4292c6", label="communication_index")
    ax[0].scatter(r.mutual_information, r.feature, color="#d95f02", zorder=3, label="MI, normalised")
    ax[0].set_title("Relevance vs communication index")
    ax[0].legend(fontsize=8, loc="lower right")
    p = pairs.iloc[::-1]
    labels = [f"{a} + {b}" for a, b in zip(p.feature_1, p.feature_2)]
    ax[1].barh(labels, p.pair_score, color="#7570b3")
    ax[1].set_title("Pair score (relevance x (1 - redundancy) + synergy)")
    fig.tight_layout()
    fig.savefig("feature_communication.png", dpi=130)
    print("\nSaved feature_communication.png")


if __name__ == "__main__":
    main()
