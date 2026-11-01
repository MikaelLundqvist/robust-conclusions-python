# Appendix B: The Small Tools

Eleven small functions on one synthetic table of used-car listings (5,000 rows, seed 1701): `print_profile`, `profile`, `iqr`, `dispersion_ratio`, `trend_derivative`, `residual_summary`, `classify_quadrants`, `bca_bootstrap_ci_by_index`, `feature_map`, `plot_analyst_view`, `plot_publisher_view`.

Built in: price = (24,000 - 2,100 age + 55 age^2) x brand multiplier (Volt 1.12, Nova 1.00, Terra 0.92, Zeta 1.05) x lognormal noise with sigma = 0.03 + 0.008 age, doubled in South (same median); true slope -2,100 + 110 age; 3% of ages and 1% of regions missing.

Run: `python example.py` (needs `robustkit`, statsmodels for the bootstrap band). Output in `out.txt`.

Verified results:

- Missing: 153 age, 38 region; 4,809 rows remain.
- Dispersion ratio in eight age groups: 0.169 to 0.215 across seven, 0.315 in the wide last group; whole stock 0.634 mixes in the trend; with four groups the first is 0.253 (0.172 with eight).
- Slope estimates -2,054 / -1,592 / -1,014 / -436 at 1 / 5 / 10 / 15 years against truth -1,990 / -1,550 / -1,000 / -450; 100 cars older than 15.
- South residual MAD 2,393 vs 1,415 and 1,440; South median gap 98 [-62, 236]; Central -80 [-171, -24] (0.6%, benchmark ignores brand).
- Far-from-benchmark (relative gap above 22.5%): South 63% of them with 25% of listings; older 384 vs newer 97.
- South p90-p10 6,047 [5,773, 6,479] vs 3,698 [3,576, 3,817].
- Bootstrap band 3.60% (500 cars) to 1.43% (4,809); dispersion ratio of bins 0.192 to 0.184.

Figures: `feature_map.png`, `trend_views.png`.
