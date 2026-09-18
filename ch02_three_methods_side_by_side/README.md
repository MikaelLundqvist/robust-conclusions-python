# Chapter 2 -- Huber, Tukey, OLS Side by Side

**What this chapter's code must prove:**
Code must show fit_huber_trend / fit_tukey_trend / fit_ols_trend on the identical dataset, with a clear visual/numeric comparison -- not just three separate examples.

## Status
Fully verified -- run end-to-end by Mikael, including the Tukey
portion (statsmodels unavailable in the sandbox used to build this,
so that part could only be verified externally). One theoretical
prediction in the chapter text turned out to be WRONG when checked
against real output (Tukey was expected to track the "estates
removed" reference more closely than Huber; in this dataset it
actually tracks it less closely, 122 vs. 86 average gap) -- corrected
in the chapter rather than silently fixed, since it's a genuinely
useful lesson about the gap between a method's theoretical property
and its behavior on any one specific, finite dataset.
