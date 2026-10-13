# Chapter 19: Can You Trust the Conclusion?

**What this chapter's code must prove:** method disagreement (Huber/Tukey/OLS) and Cook impact measure how firmly a fitted conclusion holds; the spread found matches a weakness built into synthetic data (a bonus concentrated in one job family); aggregates hide it; quadrant labels are relative to the candidate set.

## Status
Fully verified end-to-end (statsmodels 0.14.6 from user-supplied wheels).
Covers 5 functions: feature_robustness_report, plot_feature_robustness,
segment_stability_report, segment_stability_drilldown_report,
segment_quality_report. 

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

## Caveats stated in the chapter
- feature_robustness_report with features=None fails on text columns.
- plot_feature_robustness draws no threshold lines and returns the report, not Axes.
- Non-SAL cells' medians (1.4-2.1) are above the 1.2-1.4% bonus uplift; the
  excess is not decomposed.
