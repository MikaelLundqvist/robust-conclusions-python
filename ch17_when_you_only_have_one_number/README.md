# Chapter 17: When You Only Have One Number

**What this chapter's code must prove:** with one number per group, WHICH number it is (mean or median) decides which question a table can answer; flat reconstruction is exactly n-weighting; and the borrowed-dispersion method's conclusions are stable for some questions and decided by the assumption for others.

## Status
Fully verified end-to-end. Covers: expand_aggregated_table_flat (calls
expand_aggregated_group_flat), expand_aggregated_table_borrowed_dispersion,
expand_aggregated_group_borrowed_dispersion, compare_reconstruction_methods.

Design: hidden synthetic individuals (41,611 patients; length of stay by
age band, 2 departments, 14 bands, n 278-4,555 per band, right-skewed,
spread widening with age). Published as n+mean OR n+median, rounded 0.1.

Verified results (Medical, days per decade):
- True mean trend (OLS on individuals) 1.95; true typical trend (Huber on
  individuals) 1.16. They differ because mean/median grows 1.18 -> 1.45
  as (Q3-Q1)/median widens 0.80 -> 1.26.
- Flat + OLS on MEANS table: 1.948 vs truth 1.950 (exact up to rounding;
  OLS on individuals == OLS on band means weighted by n when the regressor
  is group-level).
- flat + Huber on means 2.029 == Huber on the 14 means weighted by n 2.029.
  Unweighted OLS on the 14 means: 1.66 (different question; -15%).
- Cross-question errors: means table asked the typical question +75%
  (2.03 vs 1.16); medians table asked the mean question -37% (1.24 vs
  1.95); medians table asked the typical question +10% (1.28 vs 1.16).
- Borrowed-ratio sweep (medians table, typical trend): 0.2:1.21 0.4:1.20
  0.8:1.19 1.2:1.16 1.6:1.13 (truth 1.16, flat 1.28) -> <7% movement over
  an 8x change in ratio. Tail share >14 days, oldest band (truth 37%):
  flat 0%, 0.4:16%, 0.8:33%, 1.2:39%.
- borrowed fed a MEDIAN reproduces it (10.4 -> 10.3); fed a MEAN it centres
  the median there and the reconstructed mean overshoots the published mean
  by 22% (15.0 -> 18.3).

Iteration notes (what the experiments changed):
- First comparison used one "truth" (Huber on individuals) for every
  method; none matched it. That was a mismatch of estimands, not a method
  failure -- the table's mean answers a question about the mean. The chapter
  was rebuilt around two truths and the 2x2 table/question matrix.
- User pointed out that the single number might be a median rather than a
  mean; both table types are now covered. This also exposed that the
  borrowed method is internally consistent for a median input and not for a
  mean input.
- Early notes quoted the middle age band as 52.5; the script uses 47.5
  (AGES[5]); chapter text uses the script's values.

Workaround framing (decided with the user): robustkit's fit_*_trend
functions take no weights, and adding them was judged out of scope (19
call sites across 6 files, downstream effects unassessed). The chapter
therefore presents _flat as a WORKAROUND for n-weighting, not a feature,
and documents that n can be scaled down by a common factor. Scaling check
(Medical means table, OLS / Huber slope): n x1 -> 1.9483 / 2.0289 (22,385
rows); x0.1 -> 1.9485 / 2.0292 (2,239 rows); x0.01 -> 1.9545 / 2.0379
(225 rows; smallest group 3 rows; <0.5% change). Applies to _flat only --
for the lognormal reconstructions of Chapter 16 the draw count changes the
tail, not just the weight.

Honest limits stated in the chapter: error sizes (+75%, -37%) are specific
to a dataset whose spread widens strongly with age; "+10%" for the medians
table on the typical question is reported without a tested explanation.
Fit functions in robustkit take no weights; flat reconstruction is
therefore the way to obtain n-weighting inside the package.

Figures: mean_vs_median.png (mean/median/hidden IQR by age + band sizes),
spread_sensitivity.png (trend vs ratio; tail-share bars with value
labels). Both visually inspected.
