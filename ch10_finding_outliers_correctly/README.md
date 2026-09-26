# Chapter 10: Finding Outliers Correctly

**What this chapter's code must prove:** a naive mean/standard-deviation outlier detector can be corrupted by the very outliers it's trying to find (masking), while a median/MAD-based detector (mad_outlier_report) stays stable.

## Status
Fully verified end-to-end, no external dependencies beyond core
robustkit.

Four injected outliers: three moderate ($6,000/$6,500/$7,000 below
trend) and one severe ($25,000 below trend). Result:
- Naive (mean +/- 3*std): flags E1, E2, E3 -- MISSES E0 (the $6,000
  case). E3's extreme value inflated the standard deviation enough to
  push the threshold past E0.
- Robust (mad_outlier_report, same k=3.0): flags all four correctly.

This is a clean, textbook demonstration of "masking" -- verified with
real numbers, not just asserted. Currency fixed to $ throughout
(lesson carried over from the Chapter 9 SEK/kronor fix).

Chart (masking_effect.png) generated and visually confirmed: E0 sits
clearly above the naive (orange dashed) threshold but clearly below
the robust (green solid) threshold -- the masking effect made visible
in a single glance.
