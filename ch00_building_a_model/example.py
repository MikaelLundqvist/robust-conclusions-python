"""
Introduction -- Building a Model, in Four Steps
============================================

Compass: the rest of the book assumes you can fit a curve, check it
and say how sure you are. This chapter shows those four steps once,
on one small synthetic table, and points to a free textbook for the
theory.

Built in (so every number can be checked against the truth):

  * crop yield (tonnes per hectare) = 3.0 + 0.045 x nitrogen
    - 0.00011 x nitrogen^2, with nitrogen in kg per hectare (a curve
    with diminishing returns, not a line), with 6% multiplicative noise,
  * 5% of the plots have a drainage problem and yield 40% less than
    the curve says (contamination).

Run with:

    python example.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robustkit import fit_ols_trend, fit_huber_trend, predict_trend, bootstrap_band


def truth(x):
    return 3.0 + 0.045 * np.asarray(x) - 0.00011 * np.asarray(x) ** 2


def make_plots(seed=1801, n=600):
    rng = np.random.default_rng(seed)
    nitrogen = np.clip(rng.normal(90, 50, n), 0, 200).round(0)
    crop = truth(nitrogen) * np.exp(rng.normal(0, 0.06, n))
    crop = np.where(rng.random(n) < 0.05, crop * 0.6, crop)
    return pd.DataFrame({"nitrogen": nitrogen, "yield_t_ha": crop.round(2)})


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    df = make_plots()
    rng = np.random.default_rng(7)
    test_mask = rng.random(len(df)) < 0.3
    train, test = df[~test_mask], df[test_mask]
    xt, yt = train.nitrogen.to_numpy(), train.yield_t_ha.to_numpy()
    xs, ys = test.nitrogen.to_numpy(), test.yield_t_ha.to_numpy()
    print(f"{len(df)} field plots: {len(train)} to fit on, {len(test)} held out. Built in: a curve "
          f"(3.0 + 0.045 n - 0.00011 n^2 tonnes per hectare), 6% noise, 5% of plots yield 40% less.\n")

    # 1. Choose the form: held-out error by degree.
    print("=== 1. Choose a form: Huber fits of increasing degree, error on held-out plots ===")
    rows, warned = [], []
    for d in range(1, 7):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            f = fit_huber_trend(xt, yt, degree=d)
        if caught:
            warned.append(d)
        rows.append({"degree": d, "train_rmse": rmse(yt, predict_trend(f, xt)),
                     "held_out_rmse": rmse(ys, predict_trend(f, xs)),
                     "held_out_median_abs_err": float(np.median(np.abs(ys - predict_trend(f, xs))))})
    sel = pd.DataFrame(rows)
    print(sel.round(3).astype({"degree": int}).to_string(index=False))
    print("optimiser did not converge within its iteration limit at degree: "
          + (", ".join(map(str, warned)) if warned else "none"))

    # 2. Fit: OLS against Huber.
    print("\n=== 2. Fit degree 2: OLS against Huber, error against the true curve ===")
    grid = np.linspace(0, 190, 61)
    fo, fh = fit_ols_trend(xt, yt, degree=2), fit_huber_trend(xt, yt, degree=2)
    for name, f in [("OLS", fo), ("Huber", fh)]:
        err = predict_trend(f, grid) - truth(grid)
        print(f"{name:>5}: mean error {err.mean():7.3f}, largest error {np.abs(err).max():6.3f} tonnes per hectare")

    # 3. Check: residuals.
    print("\n=== 3. Check the residuals (Huber, degree 2) ===")
    res = yt - predict_trend(fh, xt)
    med = np.median(res)
    mad = 1.4826 * np.median(np.abs(res - med))
    big = int(np.sum(res < -3 * mad))
    low = res[xt < 50]; mid = res[(xt >= 50) & (xt < 120)]; high = res[xt >= 120]
    print(f"median residual {med:.3f}; robust spread (MAD) {mad:.3f}; {big} of {len(res)} below -3 MAD")
    print(f"median residual by nitrogen: under 50 kg {np.median(low):.3f}, 50-120 {np.median(mid):.3f}, "
          f"120 and over {np.median(high):.3f}")

    # 4. Say how sure: a bootstrap band.
    print("\n=== 4. Say how sure: bootstrap band for the Huber trend ===")
    band = bootstrap_band(xt, yt, degree=2, n_boot=300, n_points=31, seed=0)
    g = band["grid"]
    tr = truth(g)
    inside = float(np.mean((band["lower"] <= tr) & (tr <= band["upper"])))
    width = (band["upper"] - band["lower"]) / band["median"] * 100
    inside_n = int(np.sum((band["lower"] <= tr) & (tr <= band["upper"])))
    miss = ~((band["lower"] <= tr) & (tr <= band["upper"]))
    print(f"95% band contains the true curve at {inside_n} of {len(g)} grid points")
    print(f"it misses at nitrogen {g[miss].min():.0f} to {g[miss].max():.0f} kg, where the fitted curve is "
          f"{np.mean(tr[miss] - band['median'][miss]):.3f} t/ha too low (about "
          f"{np.mean((tr[miss] - band['median'][miss]) / tr[miss]) * 100:.1f}% of the true yield)")
    print(f"band width: {width.min():.1f}% to {width.max():.1f}% of the curve; widest at x = {g[width.argmax()]:.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(sel.degree, sel.train_rmse, "o-", label="fitted plots")
    ax[0].plot(sel.degree, sel.held_out_rmse, "s-", label="held-out plots")
    ax[0].plot(sel.degree, sel.held_out_median_abs_err, "^-", label="held-out, median abs. error")
    ax[0].set_xlabel("polynomial degree"); ax[0].set_ylabel("error (tonnes per hectare)"); ax[0].legend()
    ax[0].set_title("1. Choosing the form")
    ax[1].scatter(xt, yt, s=8, color="0.7", label="plots")
    ax[1].plot(grid, truth(grid), "k--", label="truth")
    ax[1].plot(grid, predict_trend(fo, grid), color="#d95f02", label="OLS")
    ax[1].plot(grid, predict_trend(fh, grid), color="#1b9e77", label="Huber")
    ax[1].fill_between(g, band["lower"], band["upper"], color="#1b9e77", alpha=0.2, label="95% band")
    ax[1].set_xlabel("nitrogen (kg per hectare)"); ax[1].set_title("2-4. Fit, check, band"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("model_building.png", dpi=130)
    print("\nSaved model_building.png")


if __name__ == "__main__":
    main()
