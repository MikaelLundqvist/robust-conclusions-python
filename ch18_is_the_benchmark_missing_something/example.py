"""
Chapter 18 -- Is the Benchmark Missing Something?
==================================================

Compass: a benchmark is a claim about what pay "should" be. Before
comparing anyone with it, ask what it leaves out, and let the
residuals say where to look.

The staff table is synthetic and its structure is built in:

  * pay is quadratic in experience (concave),
  * job family is a multiplier: ENG x1.12, SAL x1.00, OPS x0.92, FIN x1.05,
  * level L3 staff in South get a further +10% (an interaction),
  * two columns are pure noise: cost_centre (8 values) and floor (4).

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import (
    fit_huber_benchmark, benchmark_predict, benchmark_goodness_of_fit,
    benchmark_screening_report, segment_position_report, plot_segment_vs_benchmark,
)

FAMILY_MULT = {"ENG": 1.12, "SAL": 1.00, "OPS": 0.92, "FIN": 1.05}


def make_staff(seed=1101, n=6000):
    rng = np.random.default_rng(seed)
    site = rng.choice(["North", "Central", "South"], n, p=[0.35, 0.40, 0.25])
    level = rng.choice(["L1", "L2", "L3"], n, p=[0.40, 0.38, 0.22])
    family = rng.choice(list(FAMILY_MULT), n, p=[0.30, 0.25, 0.25, 0.20])
    mean_exp = {"L1": 3, "L2": 9, "L3": 16}
    exp = np.clip(rng.normal([mean_exp[lv] for lv in level], 3.5), 0, 35)
    base = 32000 + 2200 * exp - 40 * exp ** 2
    inter = np.where((level == "L3") & (site == "South"), 1.10, 1.0)
    salary = base * np.array([FAMILY_MULT[f] for f in family]) * inter * np.exp(rng.normal(0, 0.06, n))
    df = pd.DataFrame({
        "site": site, "level": level, "job_family": family,
        "cost_centre": rng.choice([f"CC{i}" for i in range(8)], n),
        "floor": rng.choice(["F1", "F2", "F3", "F4"], n),
        "experience": exp.round(1), "salary": salary.round(-1)})
    df["cell"] = df.level + "_" + df.site
    df["fam_cell"] = df.job_family + "_" + df.cell
    return df


class AdjustedBenchmark:
    """A Huber trend on experience times one multiplier per category, for each column in `by`.
    The multipliers are the median ratio of actual to predicted pay, re-estimated three times.
    Any object with predict(dataframe) can serve as a benchmark; this is a minimal one."""

    def __init__(self, df, y_col, x_col, degree, by=()):
        self.x_col, self.by = x_col, list(by)
        self.fit = fit_huber_benchmark(df[x_col], df[y_col], degree=degree)
        self.mult = {c: {g: 1.0 for g in df[c].unique()} for c in self.by}
        for _ in range(3):
            for c in self.by:
                ratio = pd.Series(df[y_col].to_numpy() / self.predict(df), index=df.index).groupby(df[c]).median()
                for g in self.mult[c]:
                    self.mult[c][g] *= ratio[g]

    def predict(self, d):
        out = np.asarray(benchmark_predict(self.fit, d, x_col=self.x_col), dtype=float)
        for c in self.by:
            out = out * d[c].map(self.mult[c]).to_numpy()
        return out


def show_screen(res, cols=None, n=6):
    t = res["interactions"].copy()
    t["information_efficiency"] = t["information_efficiency"].round(3)
    t["mutual_information"] = t["mutual_information"].round(3)
    return t[["feature", "mutual_information", "information_efficiency"]].head(n).to_string(index=False)


def main():
    df = make_staff()
    print(f"{len(df):,} employees. Built in: concave pay in experience; job family multipliers; "
          f"L3 in South +10%; cost_centre and floor are noise.\n")

    # ------------------------------------------------------------------
    # 1. Screen a deliberately simple benchmark.
    # ------------------------------------------------------------------
    b0 = fit_huber_benchmark(df.experience, df.salary, degree=1)
    CANDIDATES = ["site", "level", "job_family", "cost_centre", "floor"]
    s0 = benchmark_screening_report(df, y_col="salary", benchmark_fit=b0, x_col="experience",
                                    candidate_cols=CANDIDATES)
    print("=== 1. benchmark_screening_report on a straight-line benchmark ===")
    print(f"fit: R2 {s0['fit']['r_squared']:.3f}, RMSE {s0['fit']['rmse']:,.0f}")
    print("polynomial degree on the raw (experience, salary) relationship:")
    print(s0["polynomial"][["degree", "r_squared", "r_squared_gain"]].round(4).to_string(index=False))
    print("categorical candidates against the benchmark's residual:")
    print(show_screen(s0, n=6))
    top = s0["interactions"]
    noise_rows = top[top.feature.isin(["cost_centre", "floor"])]
    print("pure-noise columns: " + ", ".join(f"{r.feature} {r.information_efficiency:.3f}"
                                              for r in noise_rows.itertuples()))

    # ------------------------------------------------------------------
    # 2. Fix what was found, then screen the residual again.
    # ------------------------------------------------------------------
    b1 = AdjustedBenchmark(df, "salary", "experience", degree=2, by=["job_family"])
    g1 = benchmark_goodness_of_fit(df, "salary", b1)
    s1 = benchmark_screening_report(df, y_col="salary", benchmark_fit=b1,
                                    candidate_cols=["site", "level", "floor", "cost_centre"])
    print("\n=== 2. degree 2 and a job-family multiplier, screened again ===")
    print("job family multipliers found: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(b1.mult["job_family"].items())))
    print(f"fit: R2 {g1['r_squared']:.3f}, RMSE {g1['rmse']:,.0f}")
    print(show_screen(s1, n=5))

    # ------------------------------------------------------------------
    # 3. Follow the lead with a position report.
    # ------------------------------------------------------------------
    pos = segment_position_report(df, segment_col="cell", y_col="salary", benchmark_fit=b1)
    pos["pct"] = pos["difference"] / pos["expected_median"] * 100
    print("\n=== 3. segment_position_report by level x site, against the second benchmark ===")
    print(pos[["segment", "n", "difference", "pct"]].round(1).to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Fix the cell and screen a third time.
    # ------------------------------------------------------------------
    b2 = AdjustedBenchmark(df, "salary", "experience", degree=2, by=["job_family", "cell"])
    g2 = benchmark_goodness_of_fit(df, "salary", b2)
    s2 = benchmark_screening_report(df, y_col="salary", benchmark_fit=b2,
                                    candidate_cols=["site", "level", "floor", "cost_centre"])
    print("\n=== 4. a level x site multiplier added, screened a third time ===")
    print(f"L3_South multiplier {b2.mult['cell']['L3_South'] / np.median(list(b2.mult['cell'].values())):.3f} "
          f"relative to the median cell (built in 1.100)")
    print(f"fit: R2 {g2['r_squared']:.3f}, RMSE {g2['rmse']:,.0f}")
    print(show_screen(s2, n=5))

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    for a, (s, title) in zip(ax[:2], [(s0, "straight-line benchmark"), (s1, "after degree 2 and job family")]):
        t = s["interactions"].head(6).iloc[::-1]
        a.barh(t.feature, t.information_efficiency, color="#4292c6")
        a.set_title(title)
        a.set_xlabel("information efficiency of the residual")
    plot_segment_vs_benchmark(df, segment_col="fam_cell", segment_values=["ENG_L3_South", "ENG_L3_Central"],
                              y_col="salary", x_col="experience", benchmark_fit=b1, ax=ax[2])
    ax[2].set_title("ENG, level L3: South vs Central")
    ax[2].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("benchmark_screening.png", dpi=130)
    print("\nSaved benchmark_screening.png")


if __name__ == "__main__":
    main()
