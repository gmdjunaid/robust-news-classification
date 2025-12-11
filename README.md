# Robust News Classification

Robust fake/real news classification trained on ISOT and evaluated on held-out tests, including cross-dataset WELFake. Baselines use TF-IDF; optional embedding models are provided.

## Data
- Training: `data/training/Fake.csv` (1=fake), `data/training/True.csv` (0=real).
- Tests:
  - `data/test/fake.csv` (all fake) — measures false negatives / fake recall.
  - `data/test/WELFake_Dataset_sample_10000.csv` (10k mixed labeled) — external cross-dataset eval; labels already aligned to 1=fake/0=real in the sample.

## Models
- Baselines: TF-IDF + Logistic Regression; TF-IDF + Linear SVM.
- Variants: TF-IDF with simple length features (title/text chars); embedding + LogReg (text-only and with lengths). Length features did not improve results.
- Device: Embedding loader auto-selects CUDA → MPS → CPU; transformer utilities remain but are not used in the notebook (runtime too long).

## Notebook
- `notebooks/08_main_experiments.ipynb`
  - Trains on full ISOT.
  - Evaluates fake-only test (fake recall / false negatives).
  - Evaluates WELFake 10k (Macro-F1, ROC-AUC, PR-AUC, confusion).
  - Embedding sections run if `sentence-transformers` weights are available; transformers removed from the main flow for runtime.

## Running
1. Create/activate env: `python3 -m venv .venv && source .venv/bin/activate`
2. Install deps: `pip install -r requirements.txt`
3. Execute notebook (headless example):
   ```
   jupyter nbconvert --to notebook --execute notebooks/08_main_experiments.ipynb --output notebooks/08_main_experiments_executed.ipynb
   ```
4. If skipping embeddings, you may stop before those cells. For `huggingface/tokenizers` fork warnings, set `TOKENIZERS_PARALLELISM=false`.

## Notes
- Label convention: 1=fake, 0=real across datasets.
- Topic-holdout split was dropped; final flow is full-train + held-out fake-only + WELFake 10k.
- Current results (ISOT → WELFake 10k): TF-IDF LogReg macro-F1 ~0.831 (ROC-AUC ~0.905); TF-IDF SVM similar; fake-only recall up to ~0.947. Embeddings trail; length features hurt recall. Domain/style mismatch remains the main limitation.
- `.venv/` and other env folders should remain untracked.
