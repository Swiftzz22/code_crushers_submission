# Business Entity Resolution: reproduction guide

This pipeline turns the three sources into `output/matching_results.tsv` and `output/candidate_pairs.tsv`. The methodology is described in `Documentation_template.md` in the zip root.

## Layout
```
src/ber/          the pipeline package (entry point: python -m ber.run)
  config.py       YAML config loader
  prep.py         TSV -> parquet with int32 row ids, row-count / id-integrity checks
  folds.py        5 S1 folds (fold 0 = holdout) + dev subset
  metrics.py      exact official macro F0.5 (singleton rule included)
  dictionary.py   token equivalences mined from train positive pairs + Indic romanization
  normalize.py    cleaning, legal-form extraction, token canonicalization
  encoder.py      bi-encoder fine-tuning (multilingual-e5-small) and embedding
  blocking.py     GPU kNN per country (record->S1 and S1->record), lexical keys, recall report
  features.py     ~64 vectorised pair features (rapidfuzz cpdist) + context features
  wordfeat.py     word-identity features (which words differ, OOF log-odds + cross-lingual embedding ridge)
  cross.py        cross-encoder on the uncertain band (2-way cross-fit); extend() widens the band
  geo.py          address-component map (dept -> region) mined from confident unlabelled pairs
  adapt.py        self-training of the cross-encoders for the unlabelled partition (France)
  model.py        LightGBM 5-fold cross-fitting
  decide.py       exclusivity + per-S1 expected-F0.5 decision (numba DP)
  pipeline.py     prune / train (A,B) / train_c / predict / predict_c / export (+ official validator);
                  simulate_test_density() hides 19% of train S1 so training matches test distractor density
configs/default.yaml   all paths and hyper-parameters
tests/            unit tests (metric incl. the brief's 0.714 example, decision DP vs brute force)
scripts/          EDA and diagnostics (not needed to reproduce the output)
ber.sh            runner: activates the venv, sets PYTHONPATH, runs python
```

## Environment
- Linux (tested on WSL2 Ubuntu 24.04). Python 3.12. An NVIDIA GPU with ≥ 8 GB VRAM. ≥ 26 GB RAM (+ swap).
- `uv venv --python 3.12 ~/ber/.venv && source ~/ber/.venv/bin/activate && uv pip install -r requirements.txt`
- Put the challenge data at the `paths.raw` location in `configs/default.yaml` (default `~/ber/data/raw/student_resource/dataset`). Intermediate files go to `~/ber/data` and `~/ber/artifacts`. Edit the config to change these.
- The only network access is the one-time download of the pretrained `intfloat/multilingual-e5-small` weights (MIT) from Hugging Face. No other external data or APIs are used.

## Commands (run from this folder, in order)
```bash
bash ber.sh -m pytest -q tests                                   # 32 tests
bash ber.sh -m ber.run --stage prep          --split both        # TSV -> parquet
bash ber.sh -m ber.run --stage folds         --split train       # folds + dev subset
bash ber.sh -m ber.run --stage dict          --split train       # mine token maps from train pairs
bash ber.sh -m ber.run --stage normalize     --split both
bash ber.sh -m ber.run --stage encoder_train --split train       # fine-tune bi-encoder (folds 1-4 only)
bash ber.sh -m ber.run --stage embed         --split both
bash ber.sh -m ber.run --stage block         --split both        # dense kNN; prints holdout recall table
bash ber.sh -m ber.run --stage prune         --split both        # final candidate set (= candidate_pairs.tsv)
bash ber.sh -m ber.run --stage features      --split both
bash ber.sh -m ber.run --stage wordfeat      --split both        # word-identity features (fits tables + embedding ridge on train)
bash ber.sh -m ber.run --stage train         --split train       # stage A + B OOF under simulated test density; holdout report
bash ber.sh -m ber.run --stage cross         --split train       # 2 cross-fitted cross-encoders on the uncertain band; scores train+test
bash ber.sh -m ber.run --stage train_c       --split train       # stage C (B features + cross-encoder logit)
bash ber.sh -m ber.run --stage predict       --split test        # stage A + B on test
bash ber.sh -m ber.run --stage predict_c     --split test        # stage C on test
bash ber.sh -m ber.run --stage export        --split test        # writes output/*.tsv using the best stage; runs the validator
# v5 France refresh (after the above; uses the v4 predictions to mine the geography map)
bash ber.sh -m ber.run --stage geo           --split test        # mine department->region etc. from confident unlabelled pairs
bash ber.sh -m ber.run --stage normalize     --split test        # re-normalize with the geography map
bash ber.sh -m ber.run --stage features      --split test
bash ber.sh -m ber.run --stage wordfeat      --split test
bash ber.sh -m ber.run --stage predict       --split test
bash ber.sh -m ber.run --stage cross_ext     --split both        # widen cross-encoder band to pA in [0.001, 0.999]
bash ber.sh -m ber.run --stage train_c       --split train --set model.train_s1_frac=0.5
bash ber.sh -m ber.run --stage predict_c     --split test
bash ber.sh -m ber.run --stage export        --split test
# v6: self-train the cross-encoders on confident pairs of the unlabelled partition, rescore it, re-export
bash ber.sh -m ber.run --stage adapt         --split test
bash ber.sh -m ber.run --stage predict_c     --split test
bash ber.sh -m ber.run --stage export        --split test
```
The holdout scores and the chosen decision rule are written to `<artifacts>/summary.json`. Per-stage runtimes go to `<artifacts>/runtime.json`.

## Runtimes (RTX 4070 Laptop 8 GB, 16 threads, 26 GB RAM)
| stage | train | test |
|---|---|---|
| prep | 9 s | 6 s |
| dict | 18 s | — |
| normalize | 2.0 min | 2.1 min |
| encoder_train (300k pairs) | 18 min | — |
| embed | 33 min | 29 min |
| block | 34 min | 23 min |
| prune | 6 s | ~10 s |
| features | 10 min | 4.4 min |
| wordfeat | 16 min | 8 min |
| train (A + B, 5 folds each; ~90 min when not sharing the CPU) | 90–130 min | — |
| cross (2 × 24 min training + scoring 1.76M train / 2.16M test pairs) | ~2 h | (included) |
| train_c | 17 min | — |
| predict / predict_c | — | 16 min / 5 min |
| export + validator | — | 1.1 min |
| geo + France refresh + cross_ext + train_c (v5) | | ~65 min |
| adapt + predict_c + export (v6) | | ~28 min |
| **total** | | **≈ 9.5 h** |

## Reproducibility notes
- Seeds are fixed (config `seed: 42`) for folds, sampling, LightGBM and torch.
- GPU fp16 training and embedding are not bit-exact across runs or hardware. A rerun gives the same holdout score to about ±0.0005, but not byte-identical outputs.
- The fine-tuned encoder and cross-encoder weights (~470 MB each) are not shipped. `encoder_train` and `cross` regenerate them from the training data.
- If `libgomp.so.1` is missing (LightGBM) and you can't install `libgomp1`, `ber.sh` falls back to the copy bundled with torch.
