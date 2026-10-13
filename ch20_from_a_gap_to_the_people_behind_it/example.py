"""
Chapter 20 -- From a Gap to the People Behind It
================================================

Compass: a gap in a group is a finding about the group. Before it
turns into a conversation about people, find out WHERE in the
hierarchy it arises, and know what each view of the data cannot see.

The staff table is synthetic and its structure is built in, so every
step can be checked against the truth:

  * North pays everyone 8% less (a STRUCTURAL effect: it belongs to
    the whole site).
  * ENG staff at level L3 in North are paid a further 10% less (a
    LOCAL effect: one cell, on top of the site's).
  * SAL staff at level L2 in South are paid a further 8% more (a local
    premium, on top of South's +3%).

The benchmark knows only years of experience.

Run with:

    python example.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import (
    segment_benchmark_report, segment_benchmark_drilldown_report, segment_contribution_report,
    deviation_report, benchmark_report_suite, export_benchmark_excel, export_benchmark_excel_no_deps,
)

SITES = ["North", "Central", "South"]
LEVELS = ["L1", "L2", "L3"]
FAMILIES = ["ENG", "SAL", "OPS", "FIN"]
SITE_MULT = {"North": 0.92, "Central": 1.00, "South": 1.03}
LOCAL_MULT = {("ENG", "L3", "North"): 0.90, ("SAL", "L2", "South"): 1.08}
SEGMENT_COLS = ["job_family", "level", "site"]      # most specific first
MIN_SIZE = 100
SPECIAL = {"ENG_L3_North": -10, "SAL_L2_South": +8}     # built-in local effects, in percent


def make_staff(seed=801, n=6000):
    """Six thousand employees: annual salary in dollars is a function of experience, times a
    site multiplier, times (in two cells only) a local multiplier, times noise."""
    rng = np.random.default_rng(seed)
    site = rng.choice(SITES, n, p=[0.35, 0.40, 0.25])
    level = rng.choice(LEVELS, n, p=[0.40, 0.38, 0.22])
    family = rng.choice(FAMILIES, n, p=[0.30, 0.25, 0.25, 0.20])
    mean_exp = {"L1": 3, "L2": 9, "L3": 16}
    exp = np.clip(rng.normal([mean_exp[lv] for lv in level], 3.5), 0, 35)
    base = 32000 + 1500 * exp - 20 * exp ** 2
    mult = np.array([SITE_MULT[s] * LOCAL_MULT.get((f, lv, s), 1.0) for s, lv, f in zip(site, level, family)])
    salary = base * mult * rng.lognormal(0, 0.08, n)
    return pd.DataFrame({"employee_id": [f"E{i:05d}" for i in range(n)], "site": site, "level": level,
                         "job_family": family, "experience": exp.round(1), "salary": salary.round(-1)})


def add_pct(report):
    """Differences are in dollars, which scale with salary level; percent of the
    expected median is what can be compared across levels."""
    out = report.copy()
    out["pct"] = out["difference"] / out["expected_median"] * 100
    return out


def main():
    df = make_staff()
    print(f"{len(df):,} employees. Built in: North -8% for everyone; ENG/L3/North a further -10%; "
          f"SAL/L2/South a further +8%.\n")
    kw = dict(y_col="salary", x_col="experience", min_size=MIN_SIZE)

    # ------------------------------------------------------------------
    # 1. Which groups sit below the benchmark? Each employee counted once.
    # ------------------------------------------------------------------
    excl = add_pct(segment_benchmark_report(df, segment_cols=SEGMENT_COLS, **kw))
    print("=== 1. segment_benchmark_report: each employee in exactly one segment ===")
    show = ["segment", "segment_level", "n", "difference", "pct"]
    print(excl.sort_values("pct")[show].head(5).round(1).to_string(index=False))
    print("   ...")
    print(excl.sort_values("pct")[show].tail(2).round(1).to_string(index=False))
    nord = excl[excl.segment.str.endswith("North")]
    rest = excl[~excl.segment.str.endswith("North") & (excl.segment != "ALL")]
    print(f"North segments: {len(nord)}, all below the benchmark: {bool((nord.pct < 0).all())} "
          f"(highest {nord.pct.max():+.1f}%). The other {len(rest)} (excluding ALL): "
          f"any below? {bool((rest.pct < 0).any())} (lowest {rest.pct.min():+.1f}%)")

    # ------------------------------------------------------------------
    # 2. The same label can mean different populations in two reports.
    # ------------------------------------------------------------------
    drill = segment_benchmark_drilldown_report(df, segment_cols=SEGMENT_COLS, **kw)
    print(f"\n=== 2. exclusive vs drilldown (min_size={MIN_SIZE}) ===")
    print(f"exclusive: {len(excl)} rows, n adds up to {excl.n.sum():,} (staff {len(df):,}); "
          f"levels {excl.segment_level.value_counts().sort_index().to_dict()}")
    print(f"drilldown: {len(drill)} rows, n adds up to {drill.n.sum():,}; "
          f"levels {drill.segment_level.value_counts().sort_index().to_dict()}")
    n_ex = int(excl.loc[excl.segment == "L3_North", "n"].iloc[0])
    n_dd = int(drill.loc[drill.segment == "L3_North", "n"].iloc[0])
    print(f"'L3_North': {n_ex} people in the exclusive report, {n_dd} in the drilldown")
    in_l3_nord = drill[(drill.segment_level == 0) & drill.segment.str.endswith("_L3_North")]
    print(f"   = {' + '.join(str(int(v)) for v in in_l3_nord.n)} in its qualifying finest cells "
          f"({', '.join(in_l3_nord.segment)}) + {n_dd - int(in_l3_nord.n.sum())} left over")
    all_row = excl[excl.segment == "ALL"].iloc[0]
    print(f"'ALL' row: {int(all_row.n)} people, assigned to no more specific segment")
    sizes = df.groupby(SEGMENT_COLS).size()
    small = sizes[sizes < MIN_SIZE].sort_values()
    print("finest cells below min_size: " + ", ".join(f"{'_'.join(k)} {v}" for k, v in small.items()))

    # ------------------------------------------------------------------
    # 3. Where in the hierarchy does the gap arise?
    # ------------------------------------------------------------------
    con = add_pct(segment_contribution_report(df, segment_cols=SEGMENT_COLS, **kw))
    parent_n = con.set_index("segment")["n"]
    fine = con[con.segment_level == 0].copy()
    fine["share"] = fine["n"] / fine["parent_segment"].map(parent_n)
    fine["contrib_pct"] = fine["contribution"] / fine["expected_median"] * 100
    fine["implied_local"] = fine["contrib_pct"] / (1 - fine["share"])

    print("\n=== 3. segment_contribution_report, finest cells ===")
    print(f"{'cell':12s} {'n':>4s} {'share of parent':>16s} {'difference':>11s} {'contribution':>13s} "
          f"{'implied local':>14s} {'built in':>9s}")
    for seg, built in SPECIAL.items():
        r = fine[fine.segment == seg].iloc[0]
        print(f"{seg:12s} {int(r.n):4d} {r.share:15.1%} {r.pct:+10.1f}% {r.contrib_pct:+12.1f}% "
              f"{r.implied_local:+13.1f}% {built:+8d}%")

    print("\nSiblings of the two special cells (same parent), predicted vs observed contribution:")
    for seg, built in SPECIAL.items():
        r = fine[fine.segment == seg].iloc[0]
        sibs = fine[(fine.parent_segment == r.parent_segment) & (fine.segment != seg)]
        predicted = -built * r.share
        obs = ", ".join(f"{s} {c:+.1f}%" for s, c in zip(sibs.segment, sibs.contrib_pct))
        print(f"  parent {r.parent_segment}: predicted {predicted:+.1f}%, observed {obs}")

    affected = {"L3_North", "L2_South"}
    rest = fine[~fine.parent_segment.isin(affected)]
    nord_rest = rest[rest.segment.str.endswith("_North")]
    print(f"\nNorth cells outside the affected parents ({len(nord_rest)}): median difference "
          f"{nord_rest.pct.median():+.1f}% from the benchmark, median contribution "
          f"{nord_rest.contrib_pct.median():+.1f}%")
    print(f"All {len(rest)} cells outside the affected parents: largest |contribution| "
          f"{rest.contrib_pct.abs().max():.1f}%")

    # ------------------------------------------------------------------
    # 4. Who is at the bottom of the list?
    # ------------------------------------------------------------------
    cell = (df.job_family == "ENG") & (df.level == "L3") & (df.site == "North")
    cell_ids = set(df.loc[cell, "employee_id"])
    print(f"\n=== 4. deviation_report: the k employees furthest below the benchmark ===")
    print(f"ENG/L3/North is {cell.mean():.1%} of staff ({len(cell_ids)} people); North is {(df.site == 'North').mean():.1%}")
    print(f"{'k':>4s} {'from the cell':>14s} {'share':>6s} {'from North':>10s}")
    ks = [10, 25, 50, 100, 200, 400]
    share_cell, share_nord = [], []
    for k in ks:
        d = deviation_report(df, y_col="salary", x_col="experience", top_n=k, id_cols=["employee_id", "site"])
        hit = int(d.employee_id.isin(cell_ids).sum())
        share_cell.append(hit / k)
        share_nord.append((d.site == "North").mean())
        print(f"{k:4d} {hit:14d} {hit / k:6.0%} {share_nord[-1]:10.0%}")

    # ------------------------------------------------------------------
    # 5. Several groupings at once, one benchmark.
    # ------------------------------------------------------------------
    suite = benchmark_report_suite(df, ["site", "level", "job_family"], y_col="salary", x_col="experience")
    print("\n=== 5. benchmark_report_suite: the same benchmark for each column ===")
    for name, rep in suite.items():
        r = add_pct(rep)
        lo = rep["ci_lower"] / rep["expected_median"] * 100
        hi = rep["ci_upper"] / rep["expected_median"] * 100
        print(f"{name}:")
        for seg, n, p, a, b in zip(r.segment, r.n, r.pct, lo, hi):
            print(f"  {seg:5s} n={n:5d}  {p:+5.1f}%  [{a:+5.1f}, {b:+5.1f}]")
    s = add_pct(suite["site"]).set_index("segment")["pct"]
    print(f"site gaps against Central: North {s['North'] - s['Central']:+.1f} points (built in -8), "
          f"South {s['South'] - s['Central']:+.1f} points (built in +3)")

    # ------------------------------------------------------------------
    # 6. Handing it over.
    # ------------------------------------------------------------------
    p1 = export_benchmark_excel(suite, "benchmarks.xlsx")
    p2 = export_benchmark_excel_no_deps(suite, "benchmarks_no_deps.xlsx")
    print("\n=== 6. Excel export ===")
    try:
        import openpyxl  # noqa: F401
        a = pd.read_excel(p1, sheet_name="site")
        b = pd.read_excel(p2, sheet_name="site")
        books = {os.path.basename(p): {k: v.shape for k, v in pd.read_excel(p, sheet_name=None).items()} for p in (p1, p2)}
        for name, sheets in books.items():
            print(f"{name}: {sheets}")
        numeric = [c for c in a.columns if c != "ci_available" and pd.api.types.is_numeric_dtype(a[c])]
        same = all(np.allclose(a[c], b[c], equal_nan=True) for c in numeric)
        print(f"same headers: {list(a.columns) == list(b.columns)}, same numbers: {same}, "
              f"same labels: {a.segment.tolist() == b.segment.tolist()}")
        print(f"ci_available (a boolean) reads back as {a.ci_available.dtype} from the first "
              f"workbook and {b.ci_available.dtype} from the no-dependency one")
    except ImportError:
        print("openpyxl is not installed, so the workbooks were written but not read back.")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    f = fine.sort_values("contrib_pct").reset_index(drop=True)
    colors = ["tab:red" if s == "ENG_L3_North" else "tab:green" if s == "SAL_L2_South" else "lightgray" for s in f.segment]
    axes[0].bar(range(len(f)), f.contrib_pct, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    notes = {"ENG_L3_North": ("ENG_L3_North", "left"), "SAL_L2_South": ("SAL_L2_South", "right"),
             "OPS_L3_North": ("OPS_L3_North\n(sibling of ENG_L3_North)", "right")}
    for i, s in enumerate(f.segment):
        if s in notes:
            text, align = notes[s]
            axes[0].annotate(text, (i, f.contrib_pct[i]), xytext=(0, -26 if f.contrib_pct[i] < 0 else 5),
                             textcoords="offset points", ha=align, fontsize=9)
    axes[0].set_ylim(f.contrib_pct.min() - 2.2, f.contrib_pct.max() + 1.6)
    axes[0].set_xticks([])
    axes[0].set_ylabel("contribution (% of expected salary)")
    axes[0].set_title("Local effect of each finest cell, against its parent")
    axes[1].plot(ks, [v * 100 for v in share_cell], marker="o", color="tab:red", label="in ENG/L3/North")
    axes[1].plot(ks, [v * 100 for v in share_nord], marker="s", color="steelblue", label="from North")
    axes[1].axhline(cell.mean() * 100, color="tab:red", linestyle=":", label="cell's share of all staff")
    axes[1].set_xscale("log"); axes[1].set_xlabel("k (list length)"); axes[1].set_ylabel("% of the k people")
    axes[1].set_title("deviation_report: who is at the bottom?")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig("gap_decomposition.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
