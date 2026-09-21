# Chapter 6: Robust Segmentation

**What this chapter's code must prove:** A global analysis reaching one conclusion, then the same data segmented reaching a materially different one.

## Status
Fully verified end-to-end, no external dependencies beyond core
robustkit (no statsmodels needed -- fit_huber_trend is sklearn-based).

Classic Simpson's paradox construction: 5 stores, each with its own
baseline satisfaction and average wait time, with the longest-wait
store also being the most-loved store overall (confound). Within
every store, longer wait -> lower satisfaction (built in identically
across all 5).

Results:
- GLOBAL (pooled, store ignored): effect = +26.3 (longer wait ->
  HIGHER satisfaction -- backwards)
- SEGMENTED (apply_by_segment, per store): effect = -6.9, -14.0,
  -15.4, -21.6, -38.1 for stores A-E respectively -- ALL five
  negative, the correct direction, none even close to the global sign

Chart (`segmentation_paradox.png`) generated and visually confirmed:
5 per-store trend lines all slope downward; the global trend (thick
black dashed) slopes upward, visibly tracking the between-store
pattern rather than the within-store one.
