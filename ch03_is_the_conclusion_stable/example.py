"""
Chapter 3 -- Is the Conclusion Stable?
========================================

Compass: demonstrate a case where model_stability_pct is LOW
(trustworthy) and a case where it's HIGH (method choice changes the
answer), plus cooks_diagnostic/cook_impact showing a handful of points
driving a result.

Run with:

    python example.py

Requires statsmodels (for model_stability_pct, which fits Tukey
internally) in addition to robustkit's core dependencies.
"""

import numpy as np

from robustkit import (
    fit_ols_trend, fit_huber_trend, predict_trend,
    model_stability_pct, cooks_diagnostic, cook_impact,
)


def make_sensor_data(seed=5, n=120, with_glitches=True):
    """
    A calibration experiment: a sensor's reading against a known,
    true reference value. Well-behaved sensors track the reference
    almost linearly with small measurement noise.

    with_glitches=True injects three genuine sensor glitches --
    moments of electrical interference that produced a wildly wrong
    reading regardless of the true value. These are real recorded
    readings, not typos: the sensor really did report those numbers.
    """
    rng = np.random.default_rng(seed)
    true_value = rng.uniform(0, 100, n)
    reading = 2.0 + 0.98 * true_value + rng.normal(0, 1.5, n)

    glitch_idx = None
    if with_glitches:
        glitch_idx = rng.choice(n, size=3, replace=False)
        reading[glitch_idx] = rng.uniform(150, 220, 3)

    return true_value, reading, glitch_idx


def make_sparse_endpoint_data(seed=13, n=100):
    """
    A second, deliberately different scenario: a workforce's age vs.
    salary, where almost everyone is middle-aged or older, and only
    TWO people are young -- one earning an ordinary junior salary, one
    earning far more than anyone that age normally would. Both real,
    both legitimate; nothing here needs cleaning.

    The point of this dataset isn't the outlier itself -- it's that
    the outlier sits in a region with almost no other data to anchor
    it, unlike Part 3-4's glitches, which were surrounded by plenty of
    ordinary neighbors.
    """
    rng = np.random.default_rng(seed)
    age_bulk = rng.uniform(35, 65, n - 2)
    salary_bulk = 28000 + 550 * (age_bulk - 22) + rng.normal(0, 1500, len(age_bulk))

    age_young = np.array([24.0, 23.0])
    salary_young = np.array([29000.0, 95000.0])

    age = np.concatenate([age_bulk, age_young])
    salary = np.concatenate([salary_bulk, salary_young])
    return age, salary, len(age) - 1


def main():
    # --- Part 1: a LOW-stability (trustworthy) case ---
    x_clean, y_clean, _ = make_sensor_data(with_glitches=False)
    stability_clean = model_stability_pct(x_clean, y_clean, degree=1)

    print("=== Clean calibration data (no glitches) ===")
    print(f"median_pct_diff: {stability_clean['median_pct_diff']:.2f}%")
    print(f"max_pct_diff:    {stability_clean['max_pct_diff']:.2f}%")
    print(
        "\nHuber, Tukey, and OLS barely disagree here. Whichever one you"
        "\nhappened to reach for, you'd have gotten essentially the same"
        "\nanswer -- this conclusion doesn't depend on that choice."
    )

    # --- Part 2: a HIGH-stability (method choice matters) case ---
    x_glitch, y_glitch, glitch_idx = make_sensor_data(with_glitches=True)
    stability_glitch = model_stability_pct(x_glitch, y_glitch, degree=1)

    print("\n=== Same experiment, with 3 genuine sensor glitches ===")
    print(f"median_pct_diff: {stability_glitch['median_pct_diff']:.2f}%")
    print(f"max_pct_diff:    {stability_glitch['max_pct_diff']:.2f}%")
    print(
        "\nA sharp jump from the clean case. Three readings, out of 120,"
        "\nare enough to make OLS, Huber, and Tukey meaningfully disagree"
        "\nabout the true calibration curve. If you'd only run one method"
        "\nand never checked, you would have had no way to know that."
    )

    # --- Part 3: which three points, and do they actually matter? ---
    diag = cooks_diagnostic(x_glitch, y_glitch, degree=1)
    print(f"\n=== cooks_diagnostic: {len(diag['flagged_indices'])} points flagged ===")
    print(f"Flagged row indices: {sorted(diag['flagged_indices'].tolist())}")
    print(f"Actual glitch indices: {sorted(glitch_idx.tolist())}")
    print(f"Cook's distance threshold (4/n): {diag['threshold']:.5f}")

    impact = cook_impact(x_glitch, y_glitch, diag["flagged_indices"], degree=1)
    print(f"\n=== cook_impact: refit Huber with those points excluded ===")
    print(f"median_pct_change: {impact['median_pct_change']:.2f}%")
    print(f"max_pct_change:    {impact['max_pct_change']:.2f}%")
    print(
        "\nHere's the twist. cooks_diagnostic correctly found exactly the"
        "\nthree glitches -- no false alarms, nothing missed. But"
        "\ncook_impact, which asks a DIFFERENT question -- does the"
        "\nHuber-fitted curve actually move if we drop them? -- comes"
        "\nback with barely any change at all."
        "\n\nThat's not a contradiction. Cook's distance measures leverage"
        "\nover an ORDINARY LEAST SQUARES fit. Huber was never listening"
        "\nto those three points very closely in the first place, so"
        "\nremoving them doesn't cost the Huber curve much. Diagnosis and"
        "\naction are two separate questions: 'which points have"
        "\nsuspicious leverage?' and 'does my actual, robust conclusion"
        "\ndepend on them?' -- and this dataset answers them differently."
    )

    # --- Part 5: the same two questions, but with sparse data at the edge ---
    age, salary, outlier_idx = make_sparse_endpoint_data()
    diag2 = cooks_diagnostic(age, salary, degree=2)
    impact2 = cook_impact(age, salary, diag2["flagged_indices"], degree=2)

    print("\n\n=== A second dataset: one outlier, almost no neighbors ===")
    print(f"Flagged row indices: {sorted(diag2['flagged_indices'].tolist())}")
    print(f"The actual outlier: index {outlier_idx}")
    print(f"median_pct_change: {impact2['median_pct_change']:.2f}%")
    print(f"max_pct_change:    {impact2['max_pct_change']:.2f}%")

    full_fit = fit_huber_trend(age, salary, degree=2)
    mask = np.ones(len(age), dtype=bool)
    mask[diag2["flagged_indices"]] = False
    reduced_fit = fit_huber_trend(age[mask], salary[mask], degree=2)
    for check_age in (24, 30, 50):
        with_outlier = predict_trend(full_fit, np.array([float(check_age)]))[0]
        without_outlier = predict_trend(reduced_fit, np.array([float(check_age)]))[0]
        pct = abs(with_outlier - without_outlier) / abs(without_outlier) * 100
        print(f"  age={check_age}: with={with_outlier:,.0f}, without={without_outlier:,.0f}, change={pct:.1f}%")

    print(
        "\nNotice two things Part 3-4 didn't show. First, cooks_diagnostic"
        "\nflags BOTH young employees here, not just the one with the"
        "\nunusual salary -- sparse regions inflate leverage for everyone"
        "\nliving in them, regardless of whether their own value is odd."
        "\nSecond, cook_impact is no longer negligible: the Huber curve"
        "\nmoves several percent right near the sparse end, fading to"
        "\nalmost nothing by the dense middle of the age range."
        "\n\nThe lesson from Part 3-4 wasn't 'high Cook's distance never"
        "\nmatters to a robust fit' -- it was 'it depends on what's"
        "\naround the flagged point.' A high-leverage point buried in a"
        "\ndense neighborhood gets outvoted. A high-leverage point"
        "\nstanding nearly alone at the edge of your data has far fewer"
        "\nneighbors to outvote it, and the impact check is what tells"
        "\nyou the difference -- rather than assuming one or the other."
    )


if __name__ == "__main__":
    main()
