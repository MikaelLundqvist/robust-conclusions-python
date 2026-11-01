# Chapter 18: Is the Benchmark Missing Something?

**What this chapter's code must prove:** a screen of a benchmark's residual finds built-in missing structure (shape, a main effect, a cell), pairs inherit from their members, and overall fit does not show what is missing -- all checked against structure built into synthetic data. The domain is freight (not pay) on purpose, to show the method is not specific to pay data.

## Status
Fully verified end-to-end (statsmodels 0.14.6 installed from user-supplied
wheels). Covers 2 functions: benchmark_screening_report,
plot_segment_vs_benchmark. Also uses benchmark_goodness_of_fit,
segment_position_report, fit_huber_benchmark, benchmark_predict.

## Design
6,000 synthetic shipments, seed 1101. cost = (320 + 22 x dist - 0.4 x dist^2)
x carrier multiplier (Arrow 1.12, Beacon 1.00, Cobalt 0.92, Delta 1.05) x 1.10 for
Heavy in South x lognormal(0, 0.06); distance in hundreds of km. driver (8 values)
and dock (4) are noise. Benchmarks: straight-line Huber; degree 2 x carrier
multiplier; plus a weight class x region cell multiplier. The multiplier class
(AdjustedBenchmark) is a median-ratio construction defined in example.py, not
part of robustkit. (Same random draws as the earlier pay version of this chapter,
rescaled to dollars per shipment, so the statistical results are unchanged.)

## Verified results (see out.txt)
- Straight line: R2 0.768; degree 2 gains 0.0186, degrees 3 and 4 gain 0.0003, 0.0002.
- Screen: carrier 0.272; pairs containing it follow (0.184, 0.160, 0.137,
  0.107 -- inheritance, not findings); weight_class 0.044; dock 0.004, driver 0.000.
- Second benchmark: R2 0.906; multipliers relative to Beacon = Arrow 1.118, Delta 1.048,
  Cobalt 0.921 (built in 1.12, 1.05, 0.92). Screen: weight_class 0.040, region x weight_class 0.030.
- Position report: Heavy_South +8.1% (built in +10%); Heavy_Central -1.5%, Heavy_North -1.0%.
- Third benchmark: R2 0.920; screen top weight_class 0.019 (not zero; not chased).

## Caveats stated in the chapter
- plot_segment_vs_benchmark builds the benchmark's grid only from columns that
  are constant within the segment: with a multi-column benchmark every column
  the benchmark reads must be constant, otherwise KeyError (hit with segment
  "cell" and a benchmark reading carrier). Worked around with a carrier_cell column.
- The explanation that the missing cell pulled Heavy_Central/Heavy_North down is
  consistent with the numbers, not isolated.
- Residual weight_class score 0.019 after the third benchmark is unexplained.
