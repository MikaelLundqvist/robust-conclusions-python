# Chapter 1 -- Why Robust Analysis

**What this chapter's code must prove:**
Code must show OLS and a robust method reaching genuinely different conclusions on the same real-feeling data -- the reader should see WHY robustness matters before being taught how.

## Status
Code written and verified. Produces a +24.4% disagreement between
OLS and Huber at age 50, from just 5 legitimate outliers (senior
executives) out of 125 employees. Chart (`ols_vs_huber.png`) confirms
Huber tracks the bulk of the data while OLS visibly tilts toward the
five executive points.
