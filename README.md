# Robust Conclusions in Python -- example code

This repository contains the runnable example code for each chapter
of *Robust Conclusions in Python*, a book about robust statistical
analysis using [robustkit](https://github.com/MikaelLundqvist/robustkit).

## How this repo relates to the book

One folder per chapter or appendix (`ch00_...` through `ch21_...`,
`appendix_a_...`, `appendix_b_...`). Each folder contains:
- `README.md` -- the specific point that chapter's code exists to
  demonstrate, and the verified results (where the chapter has them).
- `example.py` -- the runnable code discussed in that chapter.
- `out.txt` -- the output of `example.py` that the chapter quotes
  (the Introduction, Chapters 18 to 21 and the appendices).
- the figures the script writes.

Code comes first: each chapter's `example.py` is written and verified
before the surrounding prose in the book. If you're reading the book
and want to run the code alongside it, clone this repo and run the
matching chapter's `example.py` directly.

## Getting started

```bash
git clone https://github.com/MikaelLundqvist/robust-conclusions-python.git
cd robust-conclusions-python
pip install -r requirements.txt
python ch00_building_a_model/example.py
```

Some functions (Tukey fitting and the functions built on it) need
`statsmodels`, which the requirements install.

## Requirements

Pinned against a specific `robustkit` version (see `requirements.txt`)
so the examples keep working exactly as printed in the book, even as
`robustkit` itself continues to evolve. Chapters 18 to 21, the Introduction and
the appendices use functions added after `robustkit` 0.6.0, so they need
the release that contains them; the pin must be updated to that release
before these chapters are run from a fresh install.

## Chapters

| # | Title | Data | Main robustkit functions |
|---|---|---|---|
| Intro | Building a Model, in Four Steps | agriculture (crop yield) | `fit_huber_trend`, `predict_trend`, `fit_ols_trend`, `bootstrap_band` |
| 1 | Why Robust Analysis | pay | `fit_ols_trend`, `fit_huber_trend`, `predict_trend` |
| 2 | Huber, Tukey, OLS Side by Side | housing | `fit_ols_trend`, `fit_huber_trend`, `fit_tukey_trend`, `predict_trend` |
| 3 | Is the Conclusion Stable? | sensors | `model_stability_pct`, `cooks_diagnostic`, `cook_impact` |
| 4 | How Certain Is the Curve? | advertising | `bootstrap_band`, `bca_bootstrap_ci` |
| 5 | Does the Curve Actually Fit? | engine data | `compare_polynomial_degrees` |
| 6 | Robust Segmentation | retail stores | `fit_huber_trend`, `predict_trend`, `apply_by_segment` |
| 7 | Automatic Hierarchy | support tickets | `hierarchical_segment` |
| 8 | Which Variables Matter | customer churn | `rank_features`, `quadrant_report` |
| 9 | Comparing Groups to a Model | pay | `segment_position_report` |
| 10 | Finding Outliers Correctly | pay | `mad_outlier_report` |
| 11 | Robustness Against Which Population? | pay | `dual_reference_outlier_report` |
| 12 | Is the Pipeline Itself Trustworthy? | pay | `check_row_integrity`, `compare_row_sets`, `segment_consistency_report` |
| 13 | Custom Benchmarks and Reference Curves | pay | -- |
| 14 | The Benchmark's Blind Spot | pay | -- |
| 15 | From Analysis to Report | delivery | `plot_huber_iqr`, `export_huber_iqr_pdf`, `mad_outlier_report`, `export_outlier_pdf` |
| 16 | When You Only Have Aggregates | queues, official statistics | `load_json_stat`, `plot_quantile_trend`, `quantile_trend_dispersion`, `expand_aggregated_table` |
| 17 | When You Only Have One Number | hospital stays | -- |
| 18 | Is the Benchmark Missing Something? | freight | `fit_huber_benchmark`, `benchmark_screening_report`, `benchmark_goodness_of_fit`, `segment_position_report`, `plot_segment_vs_benchmark` |
| 19 | Can You Trust the Conclusion? | pay | `feature_robustness_report`, `plot_feature_robustness`, `segment_stability_report`, `segment_stability_drilldown_report`, `segment_quality_report` |
| 20 | From a Gap to the People Behind It | pay | `segment_benchmark_report`, `segment_contribution_report`, `deviation_report`, `benchmark_report_suite`, `export_benchmark_excel` |
| 21 | Does the Flag Survive? | pay | `mad_outlier_report`, `mad_outlier_drilldown_report`, `outlier_drilldown_summary`, `dual_reference_outlier_drilldown_report` |
| App. A | Which Features Make a Good Chart? | pay | `rank_features`, `rank_by_communication`, `communication_score`, `rank_communicative_pairs` |
| App. B | The Small Tools | used cars | `print_profile`, `profile`, `iqr`, `dispersion_ratio`, `fit_huber_trend` |

## License

MIT -- see [LICENSE](LICENSE). Example code is meant to be copied,
adapted, and reused freely.
