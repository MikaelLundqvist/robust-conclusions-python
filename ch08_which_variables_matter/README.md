# Chapter 8: Which Variables Matter

**What this chapter's code must prove:** a feature that looks most informative by raw mutual information is not necessarily the most valuable feature, once you account for how much complexity (entropy) it costs to encode.

## Status
Fully verified end-to-end. Uses only the "core" information module
(rank_features, quadrant_report) -- deliberately scoped this way; the
communication-score / rank_by_communication side of the information
module is less empirically grounded and is earmarked for a separate
Appendix rather than this chapter, per earlier discussion.

Churn dataset engineered so zip_code (48 categories, nested within
region) carries almost the same signal as region (4 categories) but
spread thinly:
- zip_code: mutual_information=0.0719 (HIGHEST of all features),
  entropy_bits=5.57, information_efficiency=0.0129
- region: mutual_information=0.0539 (lower than zip_code),
  entropy_bits=2.00, information_efficiency=0.0270 (more than DOUBLE
  zip_code's)

This is the core demonstration: raw MI ranking and efficiency ranking
DISAGREE about which feature is second-best (zip_code vs
tenure_months), which is the entire point of computing both.

quadrant_report classification confirmed: region=star,
zip_code=power (Predictive: informative but expensive),
tenure_months=efficient, signup_channel=weak (correctly identified as
noise).

Chart (information_space.png) generated via the package's own
plot_feature_space (not a custom chart) and visually confirmed:
zip_code's bubble (sized by entropy) is visibly the largest, sitting
highest on the MI axis but well left of region on the efficiency axis.
