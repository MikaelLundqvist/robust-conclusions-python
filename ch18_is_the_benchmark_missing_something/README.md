# Chapter 18: Is the Benchmark Missing Something?

**What this chapter's code must prove:** a screen of a benchmark's residual finds built-in missing structure (shape, a main effect, a cell), pairs inherit from their members, and overall fit does not show what is missing -- all checked against structure built into synthetic data.

## Status
Fully verified end-to-end (statsmodels 0.14.6 installed from user-supplied
wheels). Covers 2 functions: benchmark_screening_report,
plot_segment_vs_benchmark. Also uses benchmark_goodness_of_fit,
segment_position_report, fit_huber_benchmark, benchmark_predict.


## Design
6,000 synthetic employees, seed 1101. Salary = (32000 + 2200 x exp - 40 x exp^2)
x family multiplier (ENG 1.12, SAL 1.00, OPS 0.92, FIN 1.05) x 1.10 for
L3 in South x lognormal(0, 0.06). cost_centre (8 values) and floor (4) are noise.
Benchmarks: straight-line Huber; degree 2 x job_family multiplier; plus a
level x site cell multiplier. The multiplier class (AdjustedBenchmark) is a
median-ratio construction defined in example.py, not part of robustkit.

## Verified results (see out.txt)
- Straight line: R2 0.768; degree 2 gains 0.0186, degrees 3 and 4 gain 0.0003, 0.0002.
- Screen: job_family 0.273; pairs containing it follow (0.184, 0.160, 0.137,
  0.107 -- inheritance, not findings); level 0.043; floor 0.003, cost_centre 0.000.
- Second benchmark: R2 0.906; multipliers relative to SAL = ENG 1.118, FIN 1.048,
  OPS 0.921 (built in 1.12, 1.05, 0.92). Screen: level 0.040, site x level 0.031.
- Position report: L3_South +8.1% (built in +10%); L3_Central -1.5%, L3_North -1.0%.
- Third benchmark: R2 0.920; screen top level 0.021 (not zero; not chased).

## Caveats stated in the chapter
- plot_segment_vs_benchmark builds the benchmark's grid only from columns that
  are constant within the segment: with a multi-column benchmark every column
  the benchmark reads must be constant, otherwise KeyError (hit with segment
  "cell" and a benchmark reading job_family). Worked around with a fam_cell column.
- The explanation that the missing cell pulled L3_Central/L3_North down is
  consistent with the numbers, not isolated.
- Residual level score 0.021 after the third benchmark is unexplained.
