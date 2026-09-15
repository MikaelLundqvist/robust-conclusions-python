# Chapter 3 -- Is the Conclusion Stable?

**What this chapter's code must prove:**
Code must demonstrate a case where model_stability_pct is LOW (trustworthy) and a case where it's HIGH (method choice changes the answer), plus cooks_diagnostic/cook_impact showing a handful of points driving a result.

## Status
Fully verified end-to-end by Mikael, including model_stability_pct
(statsmodels available in his environment). Real numbers:

- Clean data: median_pct_diff=0.09%, max_pct_diff=2.60%
- With 3 glitches: median_pct_diff=6.20%, max_pct_diff=118.22%
  (the three-way max exceeded even the OLS-vs-Huber-only proxy's 98%,
  confirming Tukey resists the glitches at least as strongly as Huber)
- cooks_diagnostic flags exactly the 3 injected glitches (indices 15,
  38, 83), no false positives/negatives
- cook_impact on the dense-neighborhood glitches: median 0.12%, max
  3.05% (barely moves the Huber curve)
- Part 5 (sparse-endpoint scenario, added after Mikael's real-world
  observation): cooks_diagnostic flags both young employees; cook_impact
  reaches 4.8% near the sparse end, fading to 0.3% in the dense middle

Chapter text updated with the real numbers throughout.
