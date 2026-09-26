# Chapter 9: Comparing Groups to a Model

**What this chapter's code must prove:** comparing several groups against a single, shared benchmark model, and using the confidence interval (not just the point estimate) to tell a real difference from noise.

## Status
Fully verified end-to-end.

IMPORTANT DISCOVERY while building this chapter: segment_position_report
(the exact function this chapter showcases) had the SAME unfixed
.iloc + repeated benchmark_predict pattern inside its jackknife/bootstrap
loop that was found and fixed earlier today in segment_contribution_report
and segment_benchmark_drilldown_report -- this one had simply not been
touched yet. Fixed identically (precompute residual once, add
jackknife_cap passthrough, default 1000) before writing this chapter,
since it would otherwise carry the same latent O(n^2) risk on large
real data. Verified: 60,000-row segment completes in ~1.1s (previously
would have been the same order of magnitude slowdown as the other two
functions). Regression test added to test_jackknife_cap_performance.py.
All existing segment_position_report tests re-verified passing
(3 pre-existing unrelated harness failures: missing statsmodels,
missing pytest.approx, missing monkeypatch fixture -- none related to
this change).

Dataset engineered so two families (FIN, ITS) have true offsets close
to the population's size-weighted average offset, so neither should
clearly stand out once a single, population-wide benchmark is fit:
- FIN (n=139): difference=+62, CI=[-223, 401] -- clearly straddles
  zero, the central "ambiguous" demonstration
- ITS (n=452): difference=+197, CI=[1, 375] -- borderline, technically
  positive but by a hair
- ENG/SALES/OPS: all confidently on one side of zero (large |difference|,
  narrow relative CI)

Chart (segment_position.png): a forest-plot-style horizontal
error-bar chart, FIN colored red (CI crosses zero) vs. the other four
green, with a dashed zero-reference line. Visually confirmed: FIN's
error bar visibly straddles the dashed line while all others sit
clearly to one side.
