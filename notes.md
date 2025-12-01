### Instructions for editing (for humans and AI)

- **Purpose**: This file tracks development-session notes and high-level decisions for the robust news classification project.
- **Heading style**:
  - Use `###` for main sections and `####` for individual dev sessions inside the `Dev session log`.
  - Do not use `#` headings.
- **Bullet style**:
  - Prefer bullets with bold \"pseudo-headings\", e.g. `- **item**: detail`.
  - Keep language concise and neutral.
- **Sections to keep**:
  - `Project summary`
  - `Data organization so far`
  - `Challenges`
  - `Current plan`
  - `Future notes / TODOs`
  - `Dev session log`
- **How to add new notes (humans and AI)**:
  - For new work in a specific session, add a new dated entry under `Dev session log` and describe changes as short bullets.
  - Append new dev sessions to the **bottom** of the log without rewriting or deleting earlier sessions.
  - Only update the high-level sections above if something has materially changed (e.g., project goal, data layout, main plan).
  - When structure needs to evolve, preserve existing sections and add new ones in the same style instead of renaming or removing old ones.

### Dev session log

#### 2025-11-29 – Data layout & large CSV planning

- **Training vs. test data**: Decided how to separate and store CSVs for training (`training-data`) versus testing (`test-data`).
- **File locations**:
  - Training: `training-data/Fake.csv`, `training-data/True.csv`.
  - Test: `test-data/fake.csv`, plus a very large external CSV for additional test examples.
- **Git performance issue**: Noted that large CSV sizes make git push/pull operations slow and cumbersome.
- **Large external CSV**: Identified that the extra test CSV is too large to use fully for quick experiments.
- **Plan for sampling**: Agreed that the first step is to clean and downsample the large external test CSV into a smaller, representative sample to quickly evaluate fake vs. real classification accuracy.

#### 2025-11-29 – `.gitignore` for large test CSV

- **Ignore huge CSV**: Added `test-data/WELFake_Dataset.csv` to `.gitignore` so the very large test CSV is not tracked by git, improving push/pull performance.
- **Temporary workaround**: Treat this ignore rule as a temporary solution while we experiment; we may later replace the raw file with a smaller derived version that can be tracked.
- **Future direction**: Still planning to create a smaller, downsampled version of the large test CSV, since we do not need the full dataset for current accuracy experiments.

#### 2025-11-29 – WELFake sample preparation script

- **Sample script created**: Added `scripts/prepare_welfake_sample.py` that extracts the first 1,000 rows from the large WELFake CSV and saves to `test-data/WELFake_Dataset_sample_1000.csv`.
- **Script location**: Placed in `scripts/` directory for organization and clarity.
- **Documentation**: Script includes comments explaining that it assumes developers have the original large CSV locally, so reviewers can see exactly how the large CSV was processed.

### Project summary

- **Goal**: Build a model to classify news articles as fake or real.

### Project context & proposal

- **Problem framing**: Most fake-news classifiers are evaluated on random train/test splits where topic, source, and writing style are similar, so models can exploit superficial cues (topic, outlet, boilerplate phrases) instead of learning robust signals of misinformation. This project explicitly targets robustness by testing models under topic shifts and across different datasets where shortcut learning is more likely to fail.
- **Why it matters**: Fake news is a moving target—new topics, narratives, and writing styles appear over time. Evaluating only on random splits can overstate reliability because of dataset artifacts and leakage, so we focus on generalization, leakage prevention, and transferability, aligning with course themes on bias, fairness, and methodological rigor.
- **Primary datasets**: Use the ISOT Fake and Real News dataset (~45k labeled articles with `title`, `text`, `subject`, `date`) for training and main experiments, plus a one-time external evaluation on the “Getting Real About Fake News” Kaggle dataset to test cross-dataset transfer. The WELFake sample in `test-data/` is used as additional external test data for quick experimentation.
- **Robustness focus**:
  - **Topic-holdout evaluation**: Train on some subject categories and test on held-out subjects to simulate emerging misinformation topics.
  - **Cross-dataset transfer**: Train on ISOT and evaluate zero-shot on “Getting Real About Fake News” to reveal dataset-specific artifacts vs. generalizable patterns.
  - **Leakage prevention**: Clean and deduplicate data, remove boilerplate and repeated signatures, standardize formatting, and avoid features tied to source identity to reduce label leakage and inflated accuracy.
- **Modeling plan**: Start with TF-IDF features plus Logistic Regression and Linear SVM as interpretable baselines, then add lightweight sentence-embedding models, and finally fine-tune a transformer-based classifier (e.g., DistilBERT) as a modern benchmark.
- **Evaluation plan**: Compare models under both random 80/10/10 splits and topic-holdout splits, and perform cross-dataset tests for zero-shot transfer. Macro-F1 is the primary metric (to balance classes), with PR-AUC and ROC-AUC as secondary metrics for ranking and threshold analysis.
- **Ethics and scope**: Labels may encode curator bias and fake-news detection is context-sensitive, so the project is methodological only (no deployment, no claims about specific outlets). All preprocessing, evaluation choices, and limitations will be documented for transparency and reproducibility.

### Task distribution and status

#### Junaid – Data engineering and baselines

- **Core scripts (planned)**:
  - [ ] `01_preprocessing.py`
    - [ ] `load_isot(fake_path, real_path)`
    - [ ] `clean_text(text)`
    - [ ] `apply_cleaning(df)`
  - [ ] `02_data_splitting.py`
    - [ ] `random_split(df)`
    - [ ] `topic_holdout_split(df, topic_column, heldout_topic)`
  - [ ] `03_baseline_models.py`
    - [ ] `build_tfidf()`
    - [ ] `train_logreg(X_train, y_train)`
    - [ ] `train_svm(X_train, y_train)`
  - [ ] `04_baseline_eval.py`
    - [ ] `evaluate(model, X_test, y_test, model_name="baseline")`

#### Reuben – Advanced models and transfer

- **Core scripts (planned)**:
  - [ ] `05_embeddings_model.py`
    - [ ] `build_embeddings(model_name="all-MiniLM-L6-v2")`
    - [ ] `embed_text(embedder, texts)`
    - [ ] `train_embedding_classifier(emb_train, y_train)`
  - [ ] `06_transformer_model.py`
    - [ ] `build_transformer()`
    - [ ] `tokenize(tokenizer, texts)`
    - [ ] `train_transformer(model, tokenizer, train_texts, train_labels)`
  - [ ] `07_cross_dataset_transfer.py`
    - [ ] `load_kaggle_dataset(path)`
    - [ ] `zero_shot_test(model, X_test, y_test)`

#### Shared – Integration and project setup

- [x] **Decide initial training vs. test data layout**
  - **Status**: Completed in dev log entry `2025-11-29 – Data layout & large CSV planning`.
- [x] **Add `.gitignore` rule to ignore large WELFake CSV**
  - **Status**: Completed in dev log entry `2025-11-29 – .gitignore for large test CSV`.
- [x] **Create script to generate 1,000-row WELFake sample from WELFake dataset**
  - **Status**: Completed in dev log entry `2025-11-29 – WELFake sample preparation script`.
- [ ] **Restructure repository into final `data/`, `src/`, and `notebooks/` layout**
  - **Status**: Planned; not yet reflected in the current file tree as of the latest dev session.
- [ ] **Create `08_main_experiments.ipynb` to tie together all experiments**
  - **Status**: Planned; to be filled jointly by Junaid (baselines) and Reuben (advanced models and transfer).

### Data organization so far

- **Training data location**:
  - `training-data/Fake.csv`
  - `training-data/True.csv`
- **Test data location**:
  - `test-data/fake.csv`
  - `test-data/WELFake_Dataset_sample_1000.csv` (sample of 1,000 rows from the large WELFake dataset, created via `scripts/prepare_welfake_sample.py`)
  - An additional, very large CSV file (`test-data/WELFake_Dataset.csv`) available locally but not tracked in git.
- **Work in progress**: We’ve mainly been deciding where to store training vs. test CSV files and adjusting the project layout to keep this organized.

### Challenges

- **Large CSV sizes** are making git operations (pushing and pulling) slow and a bit cumbersome.
- The **extra test CSV file** is extremely large, which makes it impractical to use in full for quick experiments.

### Current plan

- **Step 1**: Clean and downsample the large external test CSV so that:
  - We keep a smaller, representative sample.
  - We can quickly test how well the model distinguishes fake vs. real news.
- We do **not** need the full dataset yet; this is mainly for an initial accuracy and workflow test.

### Future notes / TODOs

- Use this section to capture next decisions and experiments, for example:
  - Possible sampling strategies for the large test CSV.
  - Notes on model performance and accuracy.
  - Ideas for improving data organization or preprocessing.
  - Decision on whether to keep ignoring the original huge CSV file or replace it with a smaller, tracked subset file.
