# Chapter 20: From a Gap to the People Behind It

**What this chapter's code must prove:** a segment-level gap can be split into a structural part (inherited from a coarser grouping) and a local part, a list of individuals cannot tell the two apart on its own, and a one-column breakdown can miss a local effect entirely -- all checked against effects built into synthetic data.

## Status
Fully verified end-to-end. Covers 7 functions: segment_benchmark_report,
segment_benchmark_drilldown_report, segment_contribution_report,
deviation_report, benchmark_report_suite, export_benchmark_excel,
export_benchmark_excel_no_deps.

Why this chapter and not "Part 1" of the planned two-chapter salary
capstone: statsmodels is not installable in the authoring environment, so
nothing touching Tukey (model_stability_pct, feature_robustness_report,
plot_feature_robustness, segment_stability_report, the stability columns
of segment_quality_report) could be run or verified. None of this chapter's
functions use Tukey. The trust-the-benchmark chapter is therefore still
unwritten; the chapter says only that "whether the benchmark itself
deserves trust is a separate question" and promises nothing.

Design: 6,000 synthetic employees (site x level x job_family, experience,
salary). Salary = f(experience) x site multiplier x local multiplier x noise.
Built in: North -8% for everyone (structural); ENG/L3/North a further -10%
and SAL/L2/South a further +8% (local). Benchmark knows experience only
(global Huber, degree 2). min_size=100.

Verified results:
- Exclusive report: 33 rows, n sums to 6,000; levels {0:30, 1:2, 3:1}.
  Drilldown: 42 rows, n sums to 17,453; levels {0:30, 1:9, 2:3}.
  'L3_North' = 186 (exclusive) vs 439 (drilldown) = 125 + 128 + 186.
  'ALL' = 97 people; six finest cells are below 100 (FIN_L3_Central = 97 is
  the only one in Central). That ALL = FIN_L3_Central is an INFERENCE from the
  counts; the report does not list members.
- All 11 North segments below the benchmark (highest -4.5%); none of the
  other 21 (excluding ALL) below (lowest +1.1%).
- Contribution: ENG_L3_North -7.4% (share of parent 28.5%) -> implied local
  -10.4% (built in -10); SAL_L2_South +6.4% (22.8%) -> +8.3% (built in +8).
  The (1 - share) rule is an approximation (medians), derived from
  contribution = child - parent with the parent containing the child.
- Siblings: L3_North predicted +2.8%, observed OPS +4.1%; L2_South predicted
  -1.8%, observed ENG -1.7 / FIN -1.3 / OPS -0.5. Sign and order of
  magnitude right; prediction is rough.
- North cells outside the affected parents (8): median difference -6.1%,
  median contribution +0.1%; all 24 cells outside affected parents:
  largest |contribution| 1.5%.
- deviation_report precision (cell = 2.1% of staff): k=10 90%, 25 84%,
  50 62%, 100 52%, 200 38%, 400 24%; share from North 100% .. 86%.
- Suite: site Central +1.9 / North -6.3 / South +5.5 (relative to a population
  benchmark, so Central is not zero); North-Central -8.1 pts (built in -8),
  South-Central +3.6 (built in +3). level and job_family all near zero; ENG
  interval [-1.1, -0.0].
- Excel: both writers give identical headers, labels and numbers; only
  ci_available differs (bool vs int64: the no-deps writer writes anything
  that parses as a number as a number).

Iteration notes:
- Prototype first showed the contribution looked UNDERSTATED vs the built-in
  effect; the (1 - share) explanation was then derived and checked
  numerically on both special cells.
- Sibling prediction for L3_North is off by ~1.3 points (OPS +4.1 vs +2.8);
  reported as rough, not hidden.
- An initial claim that "every North segment is below the benchmark" was
  checked and then added to the script's output so it is traceable.

Revision (user request): the book works in dollars, so salary is now
described as an ANNUAL salary in dollars (the numbers are unchanged; they
were already on the same scale as earlier chapters, e.g. 32000 + 420*age),
and the Swedish site names were changed to North / Central / South (same
order, same probabilities, same random draws). Verified that the script's
output is identical to the pre-change output once the names are mapped and
whitespace is ignored (0 differing lines, confidence intervals included).
Chapter 16 still uses Nord / Mitt / Syd as region names.

Limits: single built-in structural effect and two local effects; real data
has level and family effects that this construction leaves out, so the
clean separation here is by construction. Percent differences are the
script's own derived column (difference / expected_median); the report
itself returns dollars.

Figure: gap_decomposition.png (left: finest-cell contributions with the two
built-in cells and the OPS sibling annotated; right: share of the k lowest
who are in the cell / in North vs k). Visually inspected.
