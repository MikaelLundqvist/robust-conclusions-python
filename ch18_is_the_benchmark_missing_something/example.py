"""
Chapter 18 -- Is the Benchmark Missing Something?
==================================================

Compass: a benchmark is a claim about what something "should" cost.
Before comparing anything with it, ask what it leaves out, and let
the residuals say where to look.

The shipment table is synthetic and its structure is built in:

  * the cost of a shipment is quadratic in distance (concave: long hauls
    cost less per kilometre),
  * the carrier is a multiplier: Arrow x1.12, Beacon x1.00, Cobalt x0.92,
    Delta x1.05,
  * heavy shipments to the South region carry a further +10% (an
    interaction),
  * two columns are pure noise: driver (8 values) and dock (4).

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

CARRIER_MULT = {"Arrow": 1.12, "Beacon": 1.00, "Cobalt": 0.92, "Delta": 1.05}


def make_shipments(seed=1101, n=6000):
    rng = np.random.default_rng(seed)
    region = rng.choice(["North", "Central", "South"], n, p=[0.35, 0.40, 0.25])
    wclass = rng.choice(["Light", "Medium", "Heavy"], n, p=[0.40, 0.38, 0.22])
    carrier = rng.choice(list(CARRIER_MULT), n, p=[0.30, 0.25, 0.25, 0.20])
    mean_dist = {"Light": 3, "Medium": 9, "Heavy": 16}
    dist = np.clip(rng.normal([mean_dist[w] for w in wclass], 3.5), 0, 35)   # in 100 km
    base = 320 + 22 * dist - 0.4 * dist ** 2
    inter = np.where((wclass == "Heavy") & (region == "South"), 1.10, 1.0)
    cost = base * np.array([CARRIER_MULT[c] for c in carrier]) * inter * np.exp(rng.normal(0, 0.06, n))
    df = pd.DataFrame({
        "region": region, "weight_class": wclass, "carrier": carrier,
        "driver": rng.choice([f"D{i}" for i in range(8)], n),
        "dock": rng.choice(["K1", "K2", "K3", "K4"], n),
        "distance": dist.round(1), "cost": cost.round(1)})
    df["cell"] = df.weight_class + "_" + df.region
    df["carrier_cell"] = df.carrier + "_" + df.cell
    return df


class AdjustedBenchmark:
    """A Huber trend on distance times one multiplier per category, for each column in `by`.
    The multipliers are the median ratio of actual to predicted cost, re-estimated three times.
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
    df = make_shipments()
    print(f"{len(df):,} shipments. Built in: concave cost in distance; carrier multipliers; "
          f"heavy shipments to South +10%; driver and dock are noise.\n")

    # ------------------------------------------------------------------
    # 1. Screen a deliberately simple benchmark.
    # ------------------------------------------------------------------
    b0 = fit_huber_benchmark(df.distance, df.cost, degree=1)
    CANDIDATES = ["region", "weight_class", "carrier", "driver", "dock"]
    s0 = benchmark_screening_report(df, y_col="cost", benchmark_fit=b0, x_col="distance",
                                    candidate_cols=CANDIDATES)
    print("=== 1. benchmark_screening_report on a straight-line benchmark ===")
    print(f"fit: R2 {s0['fit']['r_squared']:.3f}, RMSE {s0["fit"]["rmse"]:,.1f}")
    print("polynomial degree on the raw (distance, cost) relationship:")
    print(s0["polynomial"][["degree", "r_squared", "r_squared_gain"]].round(4).to_string(index=False))
    print("categorical candidates against the benchmark's residual:")
    print(show_screen(s0, n=6))
    top = s0["interactions"]
    noise_rows = top[top.feature.isin(["driver", "dock"])]
    print("pure-noise columns: " + ", ".join(f"{r.feature} {r.information_efficiency:.3f}"
                                              for r in noise_rows.itertuples()))

    # ------------------------------------------------------------------
    # 2. Fix what was found, then screen the residual again.
    # ------------------------------------------------------------------
    b1 = AdjustedBenchmark(df, "cost", "distance", degree=2, by=["carrier"])
    g1 = benchmark_goodness_of_fit(df, "cost", b1)
    s1 = benchmark_screening_report(df, y_col="cost", benchmark_fit=b1,
                                    candidate_cols=["region", "weight_class", "dock", "driver"])
    print("\n=== 2. degree 2 and a carrier multiplier, screened again ===")
    print("carrier multipliers found: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(b1.mult["carrier"].items())))
    print(f"fit: R2 {g1['r_squared']:.3f}, RMSE {g1["rmse"]:,.1f}")
    print(show_screen(s1, n=5))

    # ------------------------------------------------------------------
    # 3. Follow the lead with a position report.
    # ------------------------------------------------------------------
    pos = segment_position_report(df, segment_col="cell", y_col="cost", benchmark_fit=b1)
    pos["pct"] = pos["difference"] / pos["expected_median"] * 100
    print("\n=== 3. segment_position_report by weight class x region, against the second benchmark ===")
    print(pos[["segment", "n", "difference", "pct"]].round(1).to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Fix the cell and screen a third time.
    # ------------------------------------------------------------------
    b2 = AdjustedBenchmark(df, "cost", "distance", degree=2, by=["carrier", "cell"])
    g2 = benchmark_goodness_of_fit(df, "cost", b2)
    s2 = benchmark_screening_report(df, y_col="cost", benchmark_fit=b2,
                                    candidate_cols=["region", "weight_class", "dock", "driver"])
    print("\n=== 4. a weight class x region multiplier added, screened a third time ===")
    print(f"Heavy_South multiplier {b2.mult['cell']['Heavy_South'] / np.median(list(b2.mult['cell'].values())):.3f} "
          f"relative to the median cell (built in 1.100)")
    print(f"fit: R2 {g2['r_squared']:.3f}, RMSE {g2["rmse"]:,.1f}")
    print(show_screen(s2, n=5))

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    for a, (s, title) in zip(ax[:2], [(s0, "straight-line benchmark"), (s1, "after degree 2 and carrier")]):
        t = s["interactions"].head(6).iloc[::-1]
        a.barh(t.feature, t.information_efficiency, color="#4292c6")
        a.set_title(title)
        a.set_xlabel("information efficiency of the residual")
    plot_segment_vs_benchmark(df, segment_col="carrier_cell", segment_values=["Arrow_Heavy_South", "Arrow_Heavy_Central"],
                              y_col="cost", x_col="distance", benchmark_fit=b1, ax=ax[2])
    ax[2].set_title("Arrow, heavy: South vs Central")
    ax[2].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("benchmark_screening.png", dpi=130)
    print("\nSaved benchmark_screening.png")


if __name__ == "__main__":
    main()
