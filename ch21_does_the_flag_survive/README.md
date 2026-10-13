# Chapter 21: Does the Flag Survive?

**What this chapter's code must prove:** an outlier flag depends on the reference group; counting how many hierarchy levels a flag survives separates strong candidates from artefacts of one narrow definition, but it does not remove chance flags -- all checked against pay cuts planted in synthetic data.

## Status
Fully verified end-to-end. Covers 3 functions: mad_outlier_drilldown_report,
dual_reference_outlier_drilldown_report, outlier_drilldown_summary (the
exclusive mad_outlier_report is used for comparison). None use statsmodels.

## Design
6,000 synthetic employees (site x level x job_family, experience, salary).
Salary = f(experience) x site multiplier (North 0.92, Central 1.00,
South 1.03) x lognormal noise with job-family-specific sigma (OPS 0.04,
FIN 0.07, SAL 0.08, ENG 0.10). Planted: 12 "deep" (ENG/SAL/FIN, -30%),
12 "tight" (OPS, -15%), 12 "noisy" (ENG, -15%); 5,964 untouched. min_size=100,
k=3, direction negative. Seed 901.

## Verified results (see out.txt, the exact script output)
- mad_outlier_report: 29 flagged = deep 8, tight 8, noisy 0, none 13.
- Drilldown: levels 0/1/2 have 5,576 / 6,000 / 6,000 rows and 27 / 49 / 57
  flags; 17,576 rows in total. The 424 missing level-0 rows are the people in
  finest cells below min_size (checked against the cell counts).
- outlier_drilldown_summary (never-flagged added from the truth): deep
  3/1/1/7, tight 4/5/2/1, noisy 6/2/4/0, none 5908/31/19/6 for
  never/1/2/3 levels. Flagged at 3 levels: 14 (7 deep, 1 tight, 6 none; the
  six are 3 SAL + 3 ENG). Flagged at the finest level only: 8 (5 tight, 3 none).
- Median threshold as % of expected pay, level 0/1/2: OPS 12.8/20.4/20.2,
  FIN 20.0/20.4/20.3, SAL 22.4/20.5/20.2, ENG 29.9/20.4/20.3.
- Dual drilldown: type C rows are 30 North, 2 Central, 0 South (32 in all).
  Type A at all 3 levels: 9 people (7 deep, 2 none), all among the 14.

## Caveats stated in the chapter
- The chance flags (6 untouched people at 3 levels) are a property of this
  seed and noise design; the chapter reports the count, not a rate.
- outlier_drilldown_summary accepts only the MAD drilldown output; the dual
  count of type-A levels is computed by hand in the example.
- Levels are not independent tests; counting them is a heuristic.

## Files
- example.py -- run with PYTHONPATH pointing at robustkit.
- out.txt -- its output (every code-block output line in the chapter equals a
  line of this file, trailing whitespace aside).
- flag_survival.png -- the figure.
