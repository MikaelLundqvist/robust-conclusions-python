"""
Chapter 13 -- Custom Benchmarks and Reference Curves
====================================================

Compass: a benchmark model can use as many columns as you give it.
reference="model" uses all of them for every person. The curve-based
modes (reference="curve" / "benchmark_curve") fit a curve over x_col
inside each segment, so they keep the model's information only for
the variables the segmentation holds constant. Add the model's
variables to segment_cols, one at a time, and watch the information
come back.

Run with:

    python example.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from robustkit import mad_outlier_report

warnings.filterwarnings("ignore")

N = 600            # employees per simulated company
N_INJECT = 12      # underpaid employees per company
CUT = 6000         # size of each underpayment
N_SEEDS = 30       # simulated companies

FEATURES = ["age", "JobFamily", "level", "location"]
CATS = ["JobFamily", "level", "location"]

SEGMENTATIONS = {
    "Family": ["JobFamily"],
    "Family + level": ["JobFamily", "level"],
    "Family + level + location": ["JobFamily", "level", "location"],
}
MODES = ["model", "benchmark_curve", "curve"]


def make_data(seed, n=N):
    """
    Salary depends on age, job family, level AND location. `truth` is
    the noise-free salary, so we can measure how close each reference's
    expected value gets to it. N_INJECT employees are underpaid by CUT.
    """
    rng = np.random.default_rng(seed)
    job_family = rng.choice(["ENG", "ITS", "FIN"], n, p=[0.4, 0.35, 0.25])
    level = rng.choice(["L1", "L2", "L3"], n, p=[0.5, 0.35, 0.15])
    location = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    age = rng.uniform(25, 60, n)
    family_bump = {"ENG": 3000, "ITS": 1000, "FIN": 0}
    level_bump = {"L1": 0, "L2": 6000, "L3": 14000}
    location_bump = {"A": 0, "B": 4000, "C": 8000}
    truth = (32000 + 380 * age
             + np.array([family_bump[f] for f in job_family])
             + np.array([level_bump[l] for l in level])
             + np.array([location_bump[o] for o in location]))
    df = pd.DataFrame({"JobFamily": job_family, "level": level, "location": location,
                       "age": age, "salary": truth + rng.normal(0, 900, n), "truth": truth,
                       "emp_id": np.arange(n)})
    injected = rng.choice(n, N_INJECT, replace=False)
    df["injected"] = False
    df.loc[injected, "injected"] = True
    df.loc[injected, "salary"] -= CUT
    return df


class CustomBenchmark:
    """Any object with predict(dataframe) -> array works as benchmark_fit
    -- robustkit never needs to know it uses four columns, not one."""
    def __init__(self, df):
        X = pd.get_dummies(df[FEATURES], columns=CATS, drop_first=True)
        self.X_columns = X.columns
        self.model = HuberRegressor(max_iter=500)
        self.model.fit(X, df["salary"])

    def predict(self, sub_df):
        Xs = pd.get_dummies(sub_df[FEATURES], columns=CATS, drop_first=True)
        return self.model.predict(Xs.reindex(columns=self.X_columns, fill_value=0))


def evaluate(df, segment_cols, reference, benchmark_fit):
    rep = mad_outlier_report(df, y_col="salary", segment_cols=segment_cols, x_col="age",
                             benchmark_fit=benchmark_fit, min_size=20, k=3.0,
                             id_cols=["emp_id"], reference=reference)
    m = rep.merge(df[["emp_id", "truth", "injected"]], on="emp_id")
    return {"recall": (m["flagged"] & m["injected"]).sum() / N_INJECT,
            "false_flags": int((m["flagged"] & ~m["injected"]).sum()),
            "rmse": float(np.sqrt(np.mean((m["expected"] - m["truth"]) ** 2)))}


def main():
    rows = []
    for seed in range(N_SEEDS):
        df = make_data(seed)
        fit = CustomBenchmark(df)
        for seg_name, seg_cols in SEGMENTATIONS.items():
            # Age-only baseline: robustkit's default single-column trend.
            rows.append({"segmentation": seg_name, "mode": "age-only (no model)",
                         **evaluate(df, seg_cols, "model", None)})
            for mode in MODES:
                rows.append({"segmentation": seg_name, "mode": mode,
                             **evaluate(df, seg_cols, mode, fit)})
    res = pd.DataFrame(rows)
    summary = (res.groupby(["segmentation", "mode"], sort=False)
                  [["recall", "false_flags", "rmse"]].mean())

    print(f"{N_SEEDS} simulated companies, {N} employees each, {N_INJECT} underpaid by ${CUT:,}\n")
    print("recall = share of underpaid people flagged; false_flags = other people flagged per company;")
    print("rmse = error of the expected value against the noise-free salary ($)\n")
    print(summary.round(2).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    segs = list(SEGMENTATIONS)
    modes = ["age-only (no model)"] + MODES
    colors = {"age-only (no model)": "#bbbbbb", "model": "#1f77b4",
              "benchmark_curve": "#2ca02c", "curve": "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    width = 0.2
    for ax, col, title, fmt in [(axes[0], "rmse", "Error of the expected value ($, lower is better)", "{:,.0f}"),
                                (axes[1], "recall", "Underpaid people caught (higher is better)", "{:.0%}")]:
        for i, mode in enumerate(modes):
            vals = [summary.loc[(s, mode), col] for s in segs]
            xs = np.arange(len(segs)) + (i - 1.5) * width
            ax.bar(xs, vals, width, label=mode, color=colors[mode])
        ax.set_xticks(np.arange(len(segs)))
        ax.set_xticklabels([s.replace(" + ", "\n+ ") for s in segs])
        ax.set_title(title, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Segmenting by the model's variables brings its information back into the curves")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig("reference_mode_comparison.png", dpi=150)
    print("\nChart saved to reference_mode_comparison.png")


if __name__ == "__main__":
    main()
