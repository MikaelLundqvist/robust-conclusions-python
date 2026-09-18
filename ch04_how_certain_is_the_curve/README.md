# Chapter 4: How Certain Is the Curve?

**What this chapter's code must prove:** bootstrap_band and bca_bootstrap_ci -- showing the confidence band narrowing as sample size grows.

## Status
Fully verified end-to-end, no external dependencies beyond core
robustkit (no statsmodels needed -- both functions use Huber/sklearn
internally). Ran cleanly across all five sample sizes (n=30 to 5,000):

- bootstrap_band width at spend=250: 127 -> 75 -> 38 -> 22 -> 11
  (about 12x narrower going from n=30 to n=5,000)
- bca_bootstrap_ci width for median revenue: 552 -> 333 -> 164 -> 97 -> 48

Chart (`shrinking_band.png`) generated and visually confirmed: three
overlaid bootstrap bands (n=30/300/5000) show clearly decreasing width
while converging on the same underlying trend line.
