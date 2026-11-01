# Chapter 19: Can You Trust the Conclusion?

**What this chapter's code must prove:** method disagreement (Huber/Tukey/OLS) and Cook impact measure how firmly a fitted conclusion holds; the spread found matches a weakness built into synthetic data (a bonus concentrated in one job family); aggregates hide it; quadrant labels are relative to the candidate set.

## Status
Fully verified end-to-end (statsmodels 0.14.6 from user-supplied wheels).
Covers 7 functions: feature_robustness_report, plot_feature_robustness,
segment_stability_report, segment_stability_drilldown_report,
segment_quality_report, pointwise_stability_report, plot_pointwise_stability
(the last two were added to the package after the first draft of this chapter). 

## Design
6,000 synthetic employees, seed 1301. Salary = (32000 + 2200 x exp - 40 x exp^2)
x lognormal(0, 0.06); bonus x1.6 for 15% of SAL and 2% of others (realised
14.4% SAL, 2.1-2.5% others, 5.3% overall). Realised pay uplift: 3.1% overall,
SAL 8.6%, ENG 1.2%, OPS 1.4%, FIN 1.2%. Features: experience, tenure, age
(noisy copies), commute_km (unrelated), team_size (20 extreme values).

## Verified results (see out.txt)
- Feature map: experience stability 3.285 (close to the 3.1% uplift),
  all five between 1.75 and 3.29, Cook impact < 1% for all.
- Labels are relative: with features experience, commute_km, team_size,
  experience changes structural_sensitivity -> fragile with identical numbers.
  Numeric stability_threshold/impact_threshold are quantiles in [0, 1]
  (5.0 raises ValueError); a fixed tolerance has to be applied by hand.
- Segment stability: 34 segments (32 finest); SAL cells 6.08-8.81 vs 0.45-3.19
  for other families; top 8 cells by spread all SAL.
- Drilldown: level 0 range 0.45-8.81, level 1 2.68-3.99, level 2 3.11-3.44.
- Quality report: SAL worst on every measure but only stability separates
  clearly (8.15 vs 1.44-2.06).

- Pointwise (3,000 staff, 60 under 3 years, 2% bonus): method spread 0.82% (x=0.06) to 1.73% (x=34.28),
  built-in gap 2% x 60% = 1.2%; bootstrap band width 2.55% at the young end (37 observations nearby)
  vs about 0.6-0.8% where there are over 300; dominant = sampling for 8 grid points (x 0.1 to 5.0),
  method for 42. Seed 1601, n_boot=300.
- Same numbers from model_stability_pct + bootstrap_band + NumPy: spread, band width and local_n all equal the report's; 8 of 50 grid points are 'sampling'.
- Gap group (300 under 8 years, 5 over 25, seed 1): no observation nearby at 27 of 50 grid points (x 10.0-26.2);
  method spread there at most 1.37%, band width at least 3.05%; at x=30.57 spread 0.34%, band 13.72%.

## Caveats stated in the chapter
- feature_robustness_report with features=None fails on text columns.
- plot_feature_robustness draws no threshold lines and returns the report, not Axes.
- Non-SAL cells' medians (1.4-2.1) are above the 1.2-1.4% bonus uplift; the
  excess is not decomposed.
