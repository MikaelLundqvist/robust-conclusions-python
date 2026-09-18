# Chapter 5: Does the Curve Actually Fit?

**What this chapter's code must prove:** Code must reproduce the real finding: a quadratic that's clearly sufficient (autoMpg-style) vs. one that clearly isn't (a genuinely wavy trend) -- compare_polynomial_degrees telling them apart.

## Status
Fully verified end-to-end, no external dependencies beyond core
robustkit (no statsmodels needed).

- Horsepower -> efficiency: r_squared_gain degree 2->3 = 0.000006
  (essentially zero) -- quadratic confirmed sufficient
- Engine temp -> efficiency: r_squared_gain degree 2->3 = 0.135,
  LARGER than the degree 1->2 gain (0.092) -- quadratic confirmed
  insufficient, a real second bend the quadratic default misses
  entirely while still producing a respectable-looking R^2 of 0.86

Chart (`degree_comparison.png`) generated and visually confirmed:
degree=2 and degree=4 curves overlap almost perfectly on the
sufficient case; visibly diverge (missing the second bend) on the
insufficient case.

Domain pattern (sufficient quadratic for horsepower/efficiency vs.
insufficient for a more complex relationship) mirrors the real,
previously-validated OpenML autoMpg and Ames Housing findings from
this project's own validation suite, referenced in the chapter text.
