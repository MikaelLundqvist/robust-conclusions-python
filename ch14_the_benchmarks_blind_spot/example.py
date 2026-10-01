"""
Chapter 14 -- The Benchmark's Blind Spot
===================================

Compass: a benchmark model is a NORMATIVE statement about what
"expected" means, not a prediction-accuracy contest. A benchmark
sophisticated enough to explicitly model a segment's own quirk will
stop being able to detect that quirk as an anomaly -- even as its R^2
goes up. Chasing R^2 can make a benchmark WORSE at the one thing it's
for.

Run with:

    python example.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from robustkit import segment_position_report, benchmark_goodness_of_fit, fit_huber_benchmark


def make_data(seed=201, n=3000):
    """
    Salary depends on age, JobFamily, CareerLevel, and OT eligibility
    -- plus one hidden, deliberate quirk: ITS employees at CareerLevel
    P2 specifically are underpaid by $1,000, regardless of OT status.
    This is exactly the kind of segment-specific anomaly a benchmark
    comparison is supposed to surface.
    """
    rng = np.random.default_rng(seed)
    job_family = rng.choice(["ENG", "ITS"], n)
    career_level = rng.choice(["P1", "P2", "P3"], n, p=[0.4, 0.4, 0.2])
    ot = rng.choice(["Yes", "No"], n)
    age = rng.uniform(25, 60, n)

    family_bump = {"ENG": 1000, "ITS": 800}
    level_bump = {"P1": 0, "P2": 5000, "P3": 12000}
    ot_bump = {"Yes": 500, "No": 0}
    its_p2_penalty = np.where((job_family == "ITS") & (career_level == "P2"), -1000, 0)

    salary = (30000 + 380 * age
              + np.array([family_bump[f] for f in job_family])
              + np.array([level_bump[l] for l in career_level])
              + np.array([ot_bump[o] for o in ot])
              + its_p2_penalty
              + rng.normal(0, 700, n))

    return pd.DataFrame({"JobFamily": job_family, "CareerLevel": career_level, "OT": ot, "age": age, "salary": salary})


def fit_custom_benchmark(df, cat_columns):
    """
    A correctly-built custom benchmark_fit. Note the encoder is fit
    ONCE here, during training, on every category actually present --
    and the returned object's predict() re-uses that SAME fitted
    encoder every time, rather than calling pd.get_dummies() fresh on
    whatever data predict() happens to receive.

    This matters more than it looks. segment_position_report and
    plot_segment_vs_benchmark both call predict() on a SINGLE segment
    at a time -- data that, by construction, has only ONE value in
    each of its defining categorical columns. Call pd.get_dummies() on
    data like that directly, and it generates dummy columns only for
    the categories actually present in that call -- which, for a
    single, fully-specified segment, is often none. The classic (and
    easy to reach for) reindex(columns=X.columns, fill_value=0) trick
    then silently fills the missing dummy with 0 instead of the
    correct 1, quietly corrupting every per-segment prediction. Using
    a fitted OneHotEncoder (or an equivalent fixed-categories approach)
    avoids this entirely, because it always knows the full set of
    categories from training, regardless of what a later single call
    contains.
    """
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    cat_encoded = encoder.fit_transform(df[cat_columns])
    cat_cols_out = encoder.get_feature_names_out(cat_columns)
    X = pd.DataFrame(cat_encoded, columns=cat_cols_out, index=df.index)
    X["age"] = df["age"].to_numpy()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    model = HuberRegressor(max_iter=1000)
    model.fit(X_scaled, df["salary"])

    class Benchmark:
        def predict(self, sub_df):
            cat_enc = encoder.transform(sub_df[cat_columns])  # same fitted encoder -- always correct
            Xs = pd.DataFrame(cat_enc, columns=cat_cols_out, index=sub_df.index)
            Xs["age"] = sub_df["age"].to_numpy()
            Xs_scaled = pd.DataFrame(scaler.transform(Xs), columns=Xs.columns, index=Xs.index)
            return model.predict(Xs_scaled)

    return Benchmark()


def main():
    df = make_data()
    df["segment"] = df["JobFamily"] + "_" + df["CareerLevel"] + "_" + df["OT"]
    df["FamilyLevel"] = df["JobFamily"] + "_" + df["CareerLevel"]  # a combined category, for the "too absorbing" model

    # --- Three benchmarks of increasing sophistication ---
    too_coarse = fit_huber_benchmark(df["age"].to_numpy(dtype=float), df["salary"].to_numpy(dtype=float), degree=1)
    just_right = fit_custom_benchmark(df, ["JobFamily", "CareerLevel", "OT"])            # additive, no interaction
    too_absorbing = fit_custom_benchmark(df, ["FamilyLevel", "OT"])                       # explicit JobFamily x CareerLevel interaction

    print("=== Overall fit quality (higher R^2 looks 'better' by the usual metric) ===")
    for name, bf in [("1. Too coarse (age only)", too_coarse),
                      ("2. Just right (additive: family + level + OT)", just_right),
                      ("3. Too absorbing (family x level interaction)", too_absorbing)]:
        fit = benchmark_goodness_of_fit(df, y_col="salary", benchmark_fit=bf, x_col="age")
        print(f"{name}: R^2={fit['r_squared']:.4f}, RMSE={fit['rmse']:.0f}")

    print("\n=== But does each benchmark still SEE the ITS_P2 anomaly? ===")
    for name, bf in [("1. Too coarse (age only)", too_coarse),
                      ("2. Just right (additive)", just_right),
                      ("3. Too absorbing (explicit interaction)", too_absorbing)]:
        report = segment_position_report(df, segment_col="segment", y_col="salary", benchmark_fit=bf, x_col="age")
        its_p2 = report[report["segment"].str.startswith("ITS_P2")]
        print(f"\n{name}:")
        print(its_p2[["segment", "n", "difference", "ci_lower", "ci_upper"]].round(0).to_string(index=False))

    print(
        "\nModel 3 has the HIGHEST R^2 of the three (0.986 vs. 0.984 for"
        "\nmodel 2) -- by the usual 'better fit' standard, it looks like"
        "\nthe best benchmark. But its confidence interval for ITS_P2"
        "\nstraddles zero -- the $1,000 anomaly is gone. Not because it"
        "\nwas fixed or explained away by legitimate structure, but"
        "\nbecause the model's own FamilyLevel term learned 'ITS_P2 is"
        "\njust lower' and baked that directly into its definition of"
        "\n'expected' -- the exact thing this analysis exists to catch."
    )

    # --- Chart: ITS_P2_Yes's difference and CI under all three models ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for name, bf in [("1. Too coarse\n(age only)", too_coarse),
                      ("2. Just right\n(additive)", just_right),
                      ("3. Too absorbing\n(explicit interaction)", too_absorbing)]:
        report = segment_position_report(df, segment_col="segment", y_col="salary", benchmark_fit=bf, x_col="age")
        row = report[report["segment"] == "ITS_P2_Yes"].iloc[0]
        rows.append({"model": name, "difference": row["difference"], "ci_lower": row["ci_lower"], "ci_upper": row["ci_upper"]})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y_pos = np.arange(len(plot_df))
    crosses_zero = (plot_df["ci_lower"] < 0) & (plot_df["ci_upper"] > 0)
    colors = ["#d62728" if c else "#2ca02c" for c in crosses_zero]

    ax.errorbar(plot_df["difference"], y_pos,
                xerr=[plot_df["difference"] - plot_df["ci_lower"], plot_df["ci_upper"] - plot_df["difference"]],
                fmt="o", capsize=4, color="black", ecolor="gray", markersize=0)
    ax.scatter(plot_df["difference"], y_pos, c=colors, s=120, zorder=3, edgecolors="black")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["model"])
    ax.set_xlabel("ITS_P2_Yes difference from benchmark ($)")
    ax.set_title("Same $1,000 anomaly, three benchmarks, three different verdicts")
    fig.tight_layout()
    fig.savefig("benchmark_blind_spot.png", dpi=150)
    print("\nChart saved to benchmark_blind_spot.png")


if __name__ == "__main__":
    main()
