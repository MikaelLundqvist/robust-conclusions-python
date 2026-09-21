# Chapter 7: Automatic Hierarchy

**What this chapter's code must prove:** hierarchical_segment / segment_awareness handling a small segment via fallback, plus the drilldown vs. exclusive distinction.

## Status
Fully verified end-to-end, no external dependencies beyond core
robustkit (hierarchical_segment is pandas-only).

Dataset deliberately engineered to demonstrate all 4 possible fallback
depths in one run:
- Level 0 (agent): 385 tickets -- established agents, individually
  sufficient
- Level 1 (team): 25 tickets -- 5 new Invoicing agents (5 each),
  insufficient alone but pooled together clear min_size=20
- Level 2 (department): 26 tickets -- new Network agents (8 total)
  and a new Support team (18 total) neither clear min_size alone at
  team level, but pooled together at department level do
- Level 3 (ALL): 8 tickets -- 2 brand-new Upsell agents with no one
  else left unassigned in Sales to pool with

Important subtlety documented in the chapter text (verified via the
segment assignment table): fallback only pools STILL-UNASSIGNED
observations at each level, never merges newcomers into an
already-settled larger group's segment. This was worth getting right
and explaining clearly, since the naive reading ("small group's data
gets pooled with the rest of its team") is not what actually happens.

Exclusive vs. drilldown comparison verified: exclusive mode assigns
all 444 tickets to exactly one of 10 segments (444 total). Drilldown
mode (levels 0-1 only, for comparison) produces 12 qualifying groups
and 803 ticket-appearances -- confirms deliberate double-counting.

Chart (fallback_levels.png) generated and visually confirmed: clear
bar chart showing the 385/25/26/8 split across the four levels.
