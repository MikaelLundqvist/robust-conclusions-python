# Introduction: Building a Model, in Four Steps

One synthetic table (600 field plots, seed 1801; 432 to fit, 168 held out): crop yield (t/ha) = 3.0 + 0.045 n - 0.00011 n^2 for nitrogen n in kg/ha, 6% noise, and 5% of plots with a drainage problem (yield x 0.6). Run: `python example.py`; output in `out.txt`. Reference text: ISLP (James, Witten, Hastie, Tibshirani, Taylor; Springer 2023; free PDF at statlearning.com).

Verified results:

- Median absolute held-out error by degree: 0.308 (line), 0.227 (degree 2), no gain after. RMSE about 0.64 for every degree from 2 (the drainage plots dominate it). Optimiser did not converge at degree 6.
- OLS mean error -0.104 t/ha (largest 0.146); Huber -0.042 (largest 0.053): robust fitting limits but does not remove the one-sided contamination.
- Residuals (Huber): median -0.010, MAD 0.353, 21 of 432 below -3 MAD; medians by nitrogen -0.019 / 0.010 / -0.017.
- 95% bootstrap band contains the truth at 25 of 31 points; it misses at 13-47 kg where the fit is 0.046 t/ha (1.1%) too low: the band covers sampling, not bias. Width 1.4% to 4.9%, widest at 200 kg.

Figure: `model_building.png`.
