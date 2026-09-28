# Code Crushers: Business Entity Resolution (Amazon ML Challenge 2026)

Matches business records from three noisy sources to deduplicated reference entities across the US, India and France.
**Public leaderboard: 0.987 macro F0.5.**

- `Documentation_template.md`: methodology, blocking strategy, models, features and results.
- `code/business_entity_resolution/`: the runnable pipeline; see its `README.md` for setup, the exact commands and runtimes.

- `output/matching_results.tsv`: final matches (the 0.987 leaderboard submission). `output/candidate_pairs.tsv`: the candidate set the model scored. Both are stored with Git LFS; install it (`git lfs install`) before cloning.
The challenge dataset is not included.
