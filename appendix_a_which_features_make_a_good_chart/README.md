# Appendix A: Which Features Make a Good Chart?

**What this chapter's code must prove:** relevance, legibility, redundancy and synergy are separate measurements; the communication index carries the user's priorities (a heavy prior can rank a useless legible feature above the best predictor); synergy is complementarity, not interaction -- all checked against structure built into synthetic data.

## Status
Fully verified end-to-end. Covers 6 functions: communication_score,
rank_by_communication, rank_communicative_pairs, conditional_mutual_information,
pair_redundancy, pair_synergy (rank_features recapped from Ch8). None use statsmodels.

## Design
4,000 synthetic employees, seed 1401. Salary = (32000 + 2200 x exp - 40 x exp^2)
x family multiplier (ENG 1.12, SAL 1.00, OPS 0.92, FIN 1.05) x 1.30 for
commission_plan = yes within SAL only x lognormal(0, 0.06). level is driven by
experience (so redundant with it); site, office_code (40), badge_number (200) are noise.

## Verified results (see out.txt)
- rank_features MI: experience 0.897, level 0.432, job_family 0.186, commission_plan 0.031, site 0.000.
- Default communication index: experience 0.816 ... site 0.377 (zero MI) > office_code 0.306 > badge_number 0.221.
- communication_score alone gives job_family 0.720 vs 0.442 in the ranking (MI normalised against itself).
- Weights summing to 1 with interpretability 0.40 and prior experience 0.2: experience falls to fifth (0.550), below site (0.576).
- weights={'interpretability': 0.5} merges with defaults, sum 1.40, index up to 1.025.
- Redundancy experience/level 0.502, site/job_family 0.001.
- commission_plan MI: whole staff 0.031, within SAL 0.284, rest 0.000.
- pair_synergy: (job_family, commission_plan) 0.042 / reverse 0.052; (site, job_family) 0.005;
  (experience, job_family) 0.424 / reverse 0.609 although no interaction is built in.
- rank_communicative_pairs: experience+site 0.914 > experience+level 0.661 (the strong member lifts any pair).

## Caveats stated in the chapter
- The communication index is not objective; priorities are the weights and priors.
- Synergy measures complementarity (additive effects included), not interaction.
- The pair score is better at rejecting redundant pairs than at ranking good ones.
