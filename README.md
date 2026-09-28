# Code Crushers: Business Entity Resolution (Amazon ML Challenge 2026)

Matches business records from three noisy sources to deduplicated reference entities across the US, India and France.
**Public leaderboard: 0.987 macro F0.5.**

- `Documentation_template.md`: methodology, blocking strategy, models, features and results.
- `code/business_entity_resolution/`: the runnable pipeline; see its `README.md` for setup, the exact commands and runtimes.

The output files (`output/matching_results.tsv`, `output/candidate_pairs.tsv`) are not stored in this repository; the export stage regenerates them.
The challenge dataset is not included.
