# Robust Conclusions in Python -- example code

This repository contains the runnable example code for each chapter
of *Robust Conclusions in Python*, a book about robust statistical
analysis using [robustkit](https://github.com/MikaelLundqvist/robustkit).

## How this repo relates to the book

One folder per chapter (`ch01_...` through `ch12_...`). Each folder
contains:
- `README.md` -- a one-line "compass": the specific point that
  chapter's code exists to demonstrate, not just a description of
  what functions it calls.
- `example.py` -- the runnable code discussed in that chapter.

Code comes first: each chapter's `example.py` is written and verified
before the surrounding prose in the book. If you're reading the book
and want to run the code alongside it, clone this repo and run the
matching chapter's `example.py` directly.

## Getting started

```bash
git clone https://github.com/MikaelLundqvist/robust-conclusions-python.git
cd robust-conclusions-python
pip install -r requirements.txt
python ch01_why_robust_analysis/example.py
```

## Requirements

Pinned against a specific `robustkit` version (see `requirements.txt`)
so the examples keep working exactly as printed in the book, even as
`robustkit` itself continues to evolve. If you want the latest
`robustkit` instead, install it separately -- the examples should
still work in spirit, but exact numbers/output may differ slightly
across versions.

## Chapters

| # | Chapter | robustkit module(s) |
|---|---|---|
| 1 | Why Robust Analysis | -- |
| 2 | Huber, Tukey, OLS Side by Side | `core.trend` |
| 3 | Is the Conclusion Stable? | `core.stability`, `core.diagnostics` |
| 4 | How Certain Is the Curve? | `core.uncertainty` |
| 5 | Does the Curve Actually Fit? | `core.goodness_of_fit` |
| 6 | Robust Segmentation | `segmentation` |
| 7 | Automatic Hierarchy | `segment_awareness` |
| 8 | Which Variables Matter | `information` |
| 9 | Comparing Groups to a Model | `benchmark` |
| 10 | Finding Outliers Correctly | `segment_awareness` (MAD outliers) |
| 11 | From Analysis to Report | `report`, `benchmark` (reporting/export) |
| 12 | When You Only Have Aggregates | `quantiles` |

## License

MIT -- see [LICENSE](LICENSE). Example code is meant to be copied,
adapted, and reused freely.
