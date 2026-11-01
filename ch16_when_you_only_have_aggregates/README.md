# Chapter 16: When You Only Have Aggregates

**What this chapter's code must prove:** a published table of n/Q1/median/Q3 answers some questions directly, supports others only through a reconstruction whose assumptions you inherit, and cannot answer a third kind at all -- and each of the three can be told apart by checking against a hidden truth.

## Status
Fully verified end-to-end. Covers 7 functions: load_json_stat,
prepare_quantile_trend (used internally by the two below),
plot_quantile_trend, quantile_trend_dispersion, expand_aggregated_group,
expand_aggregated_table, check_reconstruction_quality. The means-only
functions (flat / borrowed dispersion / compare_reconstruction_methods)
are held for Chapter 17.

Design: synthetic INDIVIDUAL data (117,000 waits, 3 regions x 13 years)
is hidden, aggregated to a rounded n/Q1/median/Q3 table, written as a
JSON-stat file, and reloaded -- so each tool is checked against a truth
it cannot see. North = lognormal, steady rise. South = lognormal, median
rises modestly but spread widens. Central = bimodal (80% routine / 20%
complex). North and South double as controls for Central.

Verified results:
- JSON-stat round trip reproduces n/Q1/median/Q3 exactly.
- Section 1 (no reconstruction): South median +29% but dispersion
  (Q3-Q1)/median 0.62 -> 1.19; North/Central dispersion flat.
- Section 2: Huber slope (days/yr) truth 2.57/1.32/1.05, reconstructed
  2.66/1.34/1.04, fit to published medians 2.67/1.39/1.09 (North/Central/South)
  -- all within ~5%; reconstruction adds no information beyond the
  medians, only convenience.
- check_reconstruction_quality at true n: errors of a few percent in
  every region, mostly sampling noise (Central Q1 4.8%/Q3 5.3%, North
  up to 4.1%) -- does NOT separate Central.
- Same check at n=1,000,000 (noise removed): typical error North 1.3%,
  South 1.1%, Central 7.3% (worst 9.8%) -- the error that survives large n
  DOES separate Central. (Reason: calibration forces Q1/Q3 symmetric about
  the median in log space; Central's published triple is not.)
- Section 3 (2024, share > 120 days / p95): North 18.4% vs 19.3%
  (183 vs 187), South 18.5% vs 20.1% (221 vs 224), Central 19.9% vs 5.2%
  (305 vs 121). Controls match; the bimodal region fails.
- Section 4: North 2024 triple 50/72/106 reconstructed at n=2,400 /
  240,000 / 2,400,000: quartile error falls (3.3% -> ~1%) while the
  longest wait grows 380 -> 894 -> 1,570 days.

Section 5 added later ("Your own charts summarise the same way"), at the
user's suggestion that the book should say robustkit's own measures have
the same blind spots. Also covers dispersion_by_bin (the function behind
plot_huber_iqr's error bars; same table an agency publishes).
Design: dataset A = waits vs queue length, 20% complex cases (~2.5x as
long); dataset B = same x, each of plot_huber_iqr's own 15 bins redrawn
from one lognormal calibrated to A's Q1/median/Q3 (so B = "what A's bars
say and nothing else"). Verified (n=6000):
- bar endpoints B vs A: Q1 avg 4.5% (max 10.8%), median avg 2.6% (max
  5.4%), Q3 avg 5.5% (max 9.6%) -- close, NOT identical (the bimodal
  triple is not log-symmetric; same effect as the Chapter 16 floor).
- share of waits > 2x their bin's median: A 15.0%, B 3.2%.
- default corner box (R2/MAE/RMSE): A 0.31/42.9/71.2, B 0.43/31.3/44.9
  -> the default box DOES register the tail (but not where or why).
- mdape corner box: A 26.2% / IQR(resid) 50.0, B 24.9% / 44.6 ->
  nearly blind, as expected since both statistics describe the middle half.
Iteration note: the planned claim was "robustkit's bars are blind to a
second population". Testing narrowed it: the bars and the mdape box are
(nearly) blind; the default info box is not -- it reports a larger RMSE,
which cannot say whether the cause is a poor trend or a heavy tail.
Limits: B is built from A's bars, so similar bars are close to
by-construction; the finding is that the chart gives no way to tell A from
B, not that the bars match to the decimal.

Revision (user request): the Swedish region names Nord / Mitt / Syd were
changed to North / Central / South (same order, same sizes, same random
draws; column widths in the script's printouts widened from 6 to 7 for
"Central"). Verified against the previously shipped version: its output
(46 lines) is identical to the new output once the names are mapped and
whitespace is ignored, so every number is unchanged. The chapter's output
blocks were then checked line by line against the script's output (30
lines); one table header that had carried an extra "(days per year)" was
made exact and the unit moved into the preceding sentence. Swedish labels
that describe REAL SCB files (e.g. the dimension label "ar") are kept on
purpose.

Honest limits stated in the chapter: the large-n check only sees three
numbers, so a non-lognormal group with log-symmetric quartiles would
pass it; the tail-growth point depends on the domain having a ceiling.
The "all three within ~5%" claim is a statement about THIS experiment.

Iteration note: an initial hypothesis that check_reconstruction_quality
could not detect the bimodal group was WRONG when tested at large n; the
chapter reports the corrected finding (noise hides it at small n, the
surviving error reveals it).

Figures: published_quantile_trends.png (2x3: quantile band + dispersion
ratio per region), reconstruction_tail.png (North vs Central 2024, truth vs
reconstruction, log-x histograms). Both visually inspected.
