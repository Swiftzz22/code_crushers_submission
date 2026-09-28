# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
The pipeline resolves S2/S3 records to deduplicated S1 entities in stages:
1. Data-mined normalization, including romanization of 9 Indic scripts through a token dictionary learned from the training pairs.
2. A fine-tuned multilingual bi-encoder (multilingual-e5-small, MIT) for GPU kNN candidate generation within each country.
3. Three stacked LightGBM models:
   - A pairwise model (Stage A), which includes *word-identity* features that tell decoy qualifiers ("Group", "North") from noise words ("Dr", "The").
   - A context model (Stage B) that adds competition and peer-consensus signals.
   - A final model (Stage C) that adds a multilingual **cross-encoder** reading the raw text of the uncertain pairs.
4. A decision layer that picks, for every S1, the match set with the highest *expected* F0.5, with "predict nothing" as an explicit option.

Training and tuning simulate the test set's distractor density (5.75 records per S1 vs 4.67 in train) by hiding 19% of training S1 entities. On that test-like holdout (fold 0 minus the hidden S1) the final pipeline reaches **macro F0.5 = 0.9902**, with pair precision 0.998 and recall 0.972. The previous version (v4) scored **0.982** on the public leaderboard.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA was run on the full data plus a 300k sample of positive pairs (details in `docs/EDA.md`).
- **Scale:** train has 2.21M S1, 10.3M S2+S3 and 7.64M links. Test has 1.73M S1 (15% France, no French training data) and 9.97M S2+S3. 26% of S2/S3 records match no S1 (distractors). 5.6% of S1 have no match. No link crosses countries, and no record links to two S1.
- **Names:** after basic normalization only 15–26% of true pairs have identical names. Observed noise:
  - Typos, digit-for-letter swaps (`HELI0S`) and injected accents.
  - Legal suffixes swapped, dropped or moved to the front or middle ("Private Pioneer Products Limited").
  - Junk prefixes and suffixes (`--`, `>>`, `| www.x.com`, `[LLC]`), duplicated tokens and injected generic words ("Services", "Center", "Dr").
  - Trade names: unrelated brand names, "X f/k/a …", hashtags, domains.
  - **17% of Indian pairs carry the name in a native script** spread over 9 scripts (Devanagari 9.6%, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Odia, Gurmukhi).
- **Addresses:**
  - Case changes, component reordering, abbreviated street types.
  - States given as a full name, a code or in native script.
  - Added PO Box / PMB / Unit parts; dropped components. 4–5% of addresses are empty.
  - House numbers are noised: zero-padded, `#`-prefixed, off by one digit, truncated (2333→233). **12–15% of true pairs share no house number.**
- **Hard negatives:** only 5% of distractors have an S1 with the identical name, and those almost always sit in a different city or state. S1 names collide heavily (35–44% of S1 share their name with another S1; the largest group is 253), so the address is often the only discriminator. France shows the same pattern (e.g. "Bordeaux Club SARL" ×205).

### 2.2 Solution Strategy
**Approach Type:** Blocking (dense bi-encoder kNN) + stacked GBDT classifier + expected-F0.5 set selection.  
**Core Innovation:**
1. Every dictionary is mined from the training pairs, so native-script names match their S1 core name exactly in 92% of cases after normalization.
2. **Word-identity features.** The data generator builds decoys by adding a branch or qualifier word to a real name, while genuine copies get honorifics or articles:

   | extra word in the record's name | share of true matches (train) |
   |---|---|
   | group, holdings, north, downtown, clinic | 0% |
   | dr, smt, sri, the | 67–86% |

   We encode each differing word by its out-of-fold match log-odds. We also predict that value from the word's multilingual embedding, so unseen or French words ("groupe", "développement") inherit the behaviour of their translations.
3. A context ("Stage B") model uses cross-candidate competition and peer consensus to reject name twins and noisy near-duplicates.
4. A cross-encoder, cross-fitted on disjoint folds, rescores the ~3.5% of pairs whose Stage-A probability is uncertain. It reads the raw text of both records, so it catches character-level noise signatures that the similarity features flatten.
5. A Poisson-binomial dynamic program chooses the F0.5-optimal match set per S1.
6. Training and tuning under **simulated test distractor density**: 19% of training S1 are hidden, so their records become realistic distractors.

Country is used **only** as a blocking partition. It is never a model feature, so France is handled by the same country-agnostic model.

**Validation:** 5 folds over S1, stratified by country × match count. Fold 0 (441k S1) is the untouched holdout. All candidates and context features are computed on the full train universe, so the competition between S1 entities is realistic. The local scorer replicates the official macro F0.5 including the singleton rule and is unit-tested against the brief's example (0.714).

---

## 3. Candidate Generation (Blocking)
- **Normalization before blocking:**
  - NFKD accent folding. Junk and URL stripping. DBA-marker splitting (`f/k/a`, `formerly`, `dba`).
  - Legal-form extraction for US, Indian and French forms, including dotted forms like `S.A.R.L.`.
  - Two dictionaries mined from training pairs:
    - Native-script name tokens → Latin, 1,312 entries (e.g. `लिमिटेड`→limited).
    - Address token equivalences, 10,381 entries (`mh`/`महाराष्ट्र`→maharashtra, `dr`→drive).
  - Rule-based Indic romanization (indic-transliteration, MIT) as a fallback.
  - **Unlabelled-partition geography map:** 6 address-component equivalences mined from confidently matched test pairs of the partition without training labels (France).
    - Department → region: Nord and Pas-de-Calais → Hauts-de-France, Gironde → Nouvelle-Aquitaine, Loire-Atlantique → Pays de la Loire.
    - St → Saint in city names.
    - Each entry has 4k–82k supporting pairs and share 1.0. Without it, a department-vs-region difference looks like a *different state*, which the labelled data teaches is a strong sign of a different business.
- **Blocking keys used:** a dense bi-encoder embedding of `"{core name} {legal form} | {normalized address}"`.
  - Model: `intfloat/multilingual-e5-small` (118M parameters, MIT), fine-tuned for one pass over 300k training pairs from folds 1–4 only.
  - Loss: InfoNCE with in-batch negatives plus one hard negative per pair. In 49% of pairs the hard negative is another S1 with the *same core name*, which teaches the encoder to separate name twins by address.
  - Exact inner-product kNN on the GPU (chunked torch matmul + top-k, one database block on the 8 GB card at a time) within each country partition, in both directions: top-k S1 per S2/S3 record, and top-k records per S1.
- **Final candidate set (the exact set the model scores = `candidate_pairs.tsv`):** the union of each record's top-3 S1 and each S1's top-10 records.
- **Candidate pairs generated:** 34,609,587 on test (≈ 20 per S1, ≈ 3.5 per S2/S3 record), about 5×10⁻⁶ of the 6.7×10¹² within-country S1 × record cross product (reduction ratio > 0.99999). Train: 38.3M.
- **How true matches were kept:**
  - The record-centric direction bounds the work per record.
  - The S1-centric direction recovers matches that lose to name twins.
  - Holdout blocking recall by candidate budget:

| candidate budget | holdout link recall | candidates / S1 |
|---|---|---|
| top-3 per record | 98.31% | 14.0 |
| **top-3 per record + top-10 per S1 (used)** | **98.82%** | 17.3 |
| top-5 per record + top-20 per S1 | 99.16% | 32.6 |
| top-20 per record + top-30 per S1 | 99.44% | 102.4 |

  Recall is balanced across country × source (India-S3 is the lowest at 98.3–99.2% depending on budget). Exact lexical keys (name, number+street, first name token+number) reached only 81% recall at 63M pairs, so the dense blocker replaced them.

---

## 4. Matching Model

**Features used** (64, all vectorised with `rapidfuzz.process.cpdist`, computed in 5M-pair chunks):
- **Name:**
  - ratio, token-set, token-sort, partial, partial-token-set and Jaro-Winkler on the core name.
  - Similarity to the DBA alternative name.
  - Ratios on the raw cleaned name and on the space-free name (catches `#sloanmountain`, `grandlandscaping`).
  - Length and token-count differences, exact core-name equality, legal-form equality and missingness, and a native-script flag.
  - **Name genericity:** the number of S1 and records in the country sharing the core name.
- **Address:**
  - The same fuzzy ratios on the normalized address. Alphabetic token overlap and coverage in both directions.
  - House-number set Jaccard and coverage. First-number equality, Levenshtein distance and partial ratio. The S1 main number found anywhere in the record, which catches truncation and reordering.
  - Empty-address flag and component counts.
- **Other:**
  - Bi-encoder cosine, and the pair's rank in each direction.
  - Source (S2 vs S3, since their noise styles differ).
  - Cheap competition features within each record's and each S1's candidate list (best cosine, gap to the best, rank).
- **Stage B context features**, computed from Stage-A out-of-fold probabilities:
  - Record side: best probability, runner-up, this pair's rank, gap to the best competitor, and how many S1 score above 0.5.
  - S1 side: sum and count of confident candidates, and this pair's rank.
  - **Peer consensus:** the probability-weighted agreement of the S1's *other* candidates on first-number equality, address similarity, name similarity and empty address, plus this pair's difference from them. A candidate that disagrees with the S1's confident siblings is suspicious.
- **Word-identity features** (10, added to Stage A): for the words only in the record's core name ("extra") and only in the S1's ("missing"):
  - their count;
  - min and max of the out-of-fold smoothed match log-odds of each word (from training candidate pairs);
  - min and max of the same log-odds predicted by a ridge regression on the word's e5 embedding. The ridge is fit on 66,871 extra-words and 56,348 missing-words seen ≥ 30 times, with correlation 0.51 / 0.32 to the exact values. This is the cross-lingual path to French.
- **Cross-encoder (Stage C input):**
  - Backbone: the fine-tuned e5-small encoder with a 1-logit head. Input: `"name | address"` of S1 and record, raw text, max 128 tokens.
  - Training: one epoch each on two disjoint halves of the training folds ({1,2} and {3,4}), restricted to pairs with Stage-A probability in [0.01, 0.99] (1.76M pairs, 40% positive). At inference the band is widened to [0.001, 0.999], adding 1.3M train and 1.6M test pairs scored by the same cross-fitted models. This covers confident-but-wrong pairs, which matter most for France.
  - Each model scores the other half; the holdout and test get the mean of both.
  - Holdout AUC on this hard band: **0.970**. Test band: 2.16M pairs. Pairs outside the band have no score (missing value) plus an in-band flag.

**Model type:** LightGBM binary classifiers, 5-fold cross-fitted by S1, 255 leaves, learning rate 0.1, early stopping. Each fold trains on a 35% S1-level subsample of the other folds (about 8.7M pairs) to fit in RAM. Test predictions are the mean of the 5 fold models.
- Stage A: pair features + word features (best iteration about 420–450).
- Stage B: Stage A features + context features + pA (about 120–130).
- Stage C: Stage B features + cross-encoder logit (about 110–130). It trains on a 50% S1 subsample.
- The most important context features are pA, the gap to the best competing S1 (`rb_gap`), the record-side rank, and the peer-consensus features.

**Test-like training:** 19% of training S1 (419,100) are hidden before training and tuning. Their 7.3M candidate pairs are removed, their records stay as distractors, and record-side context features are recomputed. This matches the test ratio of 5.75 records per S1 (train 4.67). Holdout entities that are hidden are excluded from scoring.

**Threshold selection method:** tuned on the holdout. Two rules were compared:
- **Exclusivity:** each record keeps only its best S1, and is dropped if the runner-up is within a margin (0.15 chosen).
- **Per-S1 expected-F0.5 set selection:** candidates are sorted by probability. For each prefix size k = 0…n, the expected F0.5 is computed *exactly* under independent Bernoulli outcomes (a Poisson-binomial DP compiled with numba, verified against brute-force enumeration in unit tests). The best k is chosen, where k = 0 ("no match") scores P(no true link).

After Stage B, the 8 best settings in the grid (margins {0, 0.05, 0.15} × thresholds {0.3…0.7} or missing-mass {0, 0.05, 0.15}) were all expected-F0.5, so it is used. Before Stage B a global threshold of 0.7 was marginally better (0.98255 vs 0.98234), because the Stage-A probabilities were less well calibrated.

---

## 5. Results & Error Analysis

**Holdout (fold 0, 441,370 S1, never used for training):**

| configuration | macro F0.5 | US | India | singleton acc. | pair P | pair R |
|---|---|---|---|---|---|---|
| Lexical blocking only (smoke test, 30k S1) | 0.897 | 0.926 | 0.853 | 0.944 | 0.990 | 0.793 |
| Dense blocking + Stage A + threshold 0.7 | 0.98255 | 0.9829 | 0.9820 | 0.9759 | 0.9936 | 0.9625 |
| + Stage B + expected-F0.5 (v1) | 0.98550 | 0.9860 | 0.9848 | 0.9796 | 0.9960 | 0.9657 |

**Test-like holdout** (same fold, 19% of S1 hidden → test distractor density). Each version is compared with the public leaderboard:

| version | macro F0.5 (local) | US | India | singleton acc. | public LB |
|---|---|---|---|---|---|
| v1 models, evaluated under test density | 0.98156 | 0.9822 | 0.9806 | — | 0.974 |
| v2: trained/tuned under test density | 0.98460 | 0.9853 | 0.9836 | 0.9783 | 0.9746 |
| v3: + word-identity features | 0.98767 | 0.9878 | 0.9874 | 0.9858 | 0.976 |
| v4: + cross-encoder, Stage C | 0.99002 | 0.9898 | 0.9904 | 0.9924 | 0.982 |
| v5: + France geography map, wider cross-encoder band, Stage C on 50% | 0.99021 | — | — | — | 0.984 |
| **v6: + self-trained cross-encoder for France (submitted)** | **0.99021** (US/India unchanged) | — | — | — | (pending) |

Oracle ceiling with the current candidate set (perfect decisions): 0.9966.

**France (unlabelled).** Assuming US/India score on the leaderboard as they do locally, France's implied public score is about 0.92 (v2), 0.91 (v3) and 0.94 (v4). The word features did not move France. The multilingual cross-encoder did. France is the main remaining gap to the best public scores.

v5 changes 11.2% of France rows relative to v4, against 0.5% for US/India.

**v6: self-training for the unlabelled partition.** Each cross-fitted cross-encoder gets one short fine-tuning pass (lr 1e-5) on a mix of:
- 70k confident pseudo-positives from France (pC ≥ 0.995, the record's best S1, runner-up < 0.05);
- 70k confident pseudo-negatives (pC ≤ 0.02 with bi-encoder cosine ≥ 0.8, i.e. confusable look-alikes);
- 70k labelled training pairs from the model's own half.

Only France pairs (877k) are rescored, so US/India predictions are byte-identical to v5. As a guard, AUC on a labelled holdout band *improved* from 0.979 to 0.981 for both models. v6 changes 10.7% of France rows. Its France singleton rate moves to 5.6%, which equals the train rate.

The France effects of v5 and v6 can only be measured on the leaderboard.

Sanity checks on the test predictions (v6, no labels), compared with train:
- Mean predicted matches per S1: France 3.31 (train truth 3.46).
- Predicted singleton rate: France 5.6% (train truth 5.6%).
- Records assigned: France 61.6%, India 57.9%, US 58.7%.

**Where the remaining loss is (holdout, v1 analysis):**
- **72% of the lost F0.5 comes from entities where we only *miss* matches**, 20% from entities with a wrong merge, and 8% from singletons that received a false match.
- Of 1,527,900 true links, 52,449 are missed. 17,969 (1.2%) never reach the candidate set and 34,480 are rejected by the model.

**Common false negatives (missed matches):**
- **Records with an empty address:** 18.1k of the 34.5k model rejections. With no address, a generic name can't be tied to one of many same-name S1 entities.
- House-number noise that makes the number disagree (8.8k).
- Trade or URL names with little lexical overlap (3.9k).
- Blocking misses: the same categories, mostly DBA names plus empty addresses.

**Common false positives (wrong merges, 5,884 in total):**
- A different house number on otherwise matching name and street (2.3k). This is the designed hard negative, and it overlaps with the noise on true pairs.
- Near-identical records whose true S1 is a name twin (1.9k).
- Empty-address records assigned to a same-name S1 (0.8k).

---

## 6. Conclusion
A country-agnostic pipeline reaches 0.9902 macro F0.5 on a test-like local holdout and 0.984 on the public leaderboard (v5). v6 adds self-training for France. It combines:
- data-mined normalization;
- a fine-tuned multilingual bi-encoder for blocking;
- word-identity features that expose how the data's decoys were built;
- stacked pairwise, context and cross-encoder models;
- an exact expected-F0.5 decision rule.

The largest single gains came from reading *which* words differ between two names (+0.003) and from a cross-encoder on the uncertain band (+0.0024 locally, and the largest gain on France). The remaining gap is mostly France, where no labels exist. The next levers are more French-aware training for the cross-encoder, a wider cross-encoder band, and a larger candidate budget (99.4% recall is available).

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` contains:
- `src/ber/`: the package. It has a single entry point, `python -m ber.run --stage <stage> --split <train|test|both>`, driven by `configs/default.yaml`.
- `tests/`: 32 unit tests.
- `README.md`: exact commands and per-stage runtimes (≈ 8 h end to end on an RTX 4070 Laptop 8 GB).
- `requirements.txt`: pinned.

Stage order: `prep → folds → dict → normalize → encoder_train → embed → block → prune → features → wordfeat → train → cross → predict → predict_c → export`. The v5 France refresh then runs `geo → normalize/features/wordfeat/predict (test) → cross_ext (train, test) → train_c → predict_c → export`. The export stage writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the official validator (PASS).

### B. Additional Results
**Models and licenses:** `intfloat/multilingual-e5-small` (117.65M parameters, MIT). It is fine-tuned twice: as the bi-encoder, and, with a 1-logit head, as the cross-encoder. LightGBM models and the word-embedding ridge regression are trained from scratch on the provided training data. Libraries are MIT, Apache-2.0 or BSD.

**Dictionary provenance:**
- The native-script name map and the address-equivalence map are **mined from the training pairs only**.
- The legal-form and street-type tables are small generic lists; their French entries were confirmed by frequency in the test France text.
- The France geography map (6 entries) is mined from the provided test inputs: co-occurrence of address components in confidently matched pairs. No external gazetteer is used.
- Indic romanization uses `indic-transliteration` rules (MIT).
- No external data, geocoding, registry or web lookup is used anywhere.
