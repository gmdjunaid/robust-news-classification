# Robust News Classification

Text-only fake/real news classification using the ISOT dataset for training and held-out test files for evaluation.

## Data
- Training: `data/training/Fake.csv` (fake=1), `data/training/True.csv` (real=0).
- Tests:
  - `data/test/fake.csv` (all fake) — used to measure false negatives / fake recall.
  - `data/test/WELFake_Dataset_sample_1000.csv` (mixed labeled, source labels 0=fake/1=real; mapped to project convention 1=fake/0=real).

## Models
- Baselines: TF-IDF + Logistic Regression, TF-IDF + Linear SVM.
- Optional: Sentence-embedding classifier (`sentence-transformers`).

## Notebook
- `notebooks/08_main_experiments.ipynb`
  - Trains on full ISOT.
  - Evaluates fake-only test (reports fake recall / false negatives).
  - Evaluates WELFake (Macro-F1, ROC-AUC, PR-AUC, confusion matrix).
  - Embeddings section runs if `sentence-transformers` and model weights are available.

## Running
1. Create/activate a virtual env (example): `python3 -m venv .venv && source .venv/bin/activate`
2. Install deps (example): `pip install --upgrade pip jupyter nbconvert scikit-learn pandas numpy sentence-transformers`
3. From repo root, execute the notebook (will time out without Jupyter):
   ```
   jupyter nbconvert --to notebook --execute notebooks/08_main_experiments.ipynb --output notebooks/08_main_experiments_executed.ipynb
   ```
4. If embeddings aren’t available, expect that section to fail; you can skip or guard it. If you see `huggingface/tokenizers` fork warnings, set `TOKENIZERS_PARALLELISM=false`.

## Notes
- Label convention: project uses 1=fake, 0=real; WELFake source labels are flipped and remapped in the notebook.
- Topic-holdout split is no longer used due to single-class test issues; final flow is full-train + held-out tests above.
- Cross-dataset findings (ISOT → WELFake): models train well on ISOT but perform near-random on WELFake (macro-F1 ≈ 0.14–0.18; embeddings only slightly better). Likely domain/style mismatch; needs further adaptation and preprocessing checks.
- Git hygiene: `.venv/` and other accidental env folders are ignored; keep envs untracked.
