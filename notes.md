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

#### 2025-12-01 – Preprocessing and data splitting scripts

- **Preprocessing script created**: Added `scripts/01_preprocessing.py` with functions to load ISOT dataset and clean text data.
  - `load_isot(fake_path, real_path)`: Loads and combines Fake.csv and True.csv files, adds binary labels (1=fake, 0=real), and tracks source file origin.
  - `clean_text(text)`: Cleans individual text strings by removing URLs, email addresses, and normalizing whitespace to reduce noise in news articles.
  - `apply_cleaning(df, text_column='text')`: Applies text cleaning to DataFrame columns, creating cleaned versions while preserving originals.
- **Data splitting script created**: Added `scripts/02_data_splitting.py` with functions for train/test splitting strategies.
  - `random_split(df, test_size=0.2, random_state=42)`: Performs stratified random split to maintain balanced fake/real distribution in train and test sets.
  - `topic_holdout_split(df, topic_column='subject', heldout_topic=None)`: Implements topic-based holdout splitting to evaluate model robustness by testing on completely unseen topics, aligning with project focus on generalization across topics.
- **Documentation**: Both scripts include comprehensive docstrings with purpose, parameters, return values, examples, and error handling documentation to ensure code readability and maintainability.
- **Alignment with project goals**: The topic holdout split function is particularly important for the robustness evaluation strategy, testing whether models can generalize to unseen topics rather than memorizing topic-specific patterns.

#### 2025-12-01 – Baseline models and evaluation scripts

- **Baseline models script created**: Added `scripts/03_baseline_models.py` with functions to build TF-IDF features and train interpretable baseline classifiers.
  - `build_tfidf()`: Creates and configures a TF-IDF vectorizer with sensible defaults (max_features=5000, ngram_range=(1,2)) for news classification feature extraction.
  - `train_logreg(X_train, y_train)`: Trains a Logistic Regression classifier on TF-IDF features, providing interpretable coefficients for understanding word importance patterns.
  - `train_svm(X_train, y_train)`: Trains a Linear SVM classifier on TF-IDF features, offering an alternative learning bias compared to Logistic Regression for model comparison.
- **Evaluation script created**: Added `scripts/04_baseline_eval.py` with comprehensive evaluation function following the project's evaluation plan.
  - `evaluate(model, X_test, y_test, model_name="baseline")`: Evaluates trained models using Macro-F1 as the primary metric (to balance classes), with PR-AUC and ROC-AUC as secondary metrics. Also computes accuracy, precision, recall, confusion matrix, and detailed classification reports for comprehensive analysis.
- **Documentation**: Both scripts include comprehensive docstrings explaining purpose, parameters, return values, examples, and error handling, ensuring code readability and maintainability.
- **Alignment with project goals**: Implements interpretable baseline models (TF-IDF + Logistic Regression/SVM) as specified in the modeling plan, serving as benchmarks against which advanced models will be compared. The evaluation function uses Macro-F1 as the primary metric as specified in the evaluation plan, supporting robust model comparison across different split strategies.

#### 2025-12-01 – Embedding and transformer model modules

- **Embedding-based modeling script created**: Added `scripts/05_embeddings_model.py` with utilities for sentence-embedding experiments.
  - `build_embeddings(model_name="all-MiniLM-L6-v2")`: Loads a SentenceTransformer model for converting news articles into dense embeddings.
  - `embed_text(embedder, texts)`: Encodes cleaned article text (e.g., `text_cleaned`) into NumPy embedding arrays suitable for downstream classifiers.
  - `train_embedding_classifier(emb_train, y_train)`: Trains a Logistic Regression classifier on top of embeddings, using the same label convention (0=real, 1=fake) as the TF-IDF baselines so it can be evaluated with the shared `evaluate` function.
- **Transformer-based modeling script created**: Added `scripts/06_transformer_model.py` with utilities for DistilBERT-style sequence classification.
  - `FakeNewsDataset`: Lightweight Dataset wrapper that tokenizes articles on the fly for Hugging Face Trainer-based fine-tuning.
  - `build_transformer()`, `tokenize(...)`, `train_transformer(...)`: Build, tokenize for, and fine-tune a transformer classifier on ISOT data, again respecting the 0=real, 1=fake label convention.
  - `TransformerSklearnWrapper`: Sklearn-style wrapper exposing `.predict()` and `.predict_proba()` so transformer models can be passed directly into `scripts/04_baseline_eval.py::evaluate`, enabling consistent Macro-F1, PR-AUC, and ROC-AUC comparisons across TF-IDF baselines, embedding models, and transformers.
- **Alignment with project goals**: Implements the advanced modeling components (embeddings + transformer) called for in the proposal while keeping interfaces and label conventions compatible with existing preprocessing, splitting, and evaluation utilities, supporting robust comparisons under random, topic-holdout, and later cross-dataset settings.

#### 2025-12-01 – Cross-dataset transfer utilities

- **Cross-dataset transfer script created**: Added `scripts/07_cross_dataset_transfer.py` with utilities to evaluate ISOT-trained models on external datasets.
  - `load_kaggle_dataset(path, text_column="text", label_column="label", ...)`: Loads an external fake-news CSV (e.g., Kaggle or WELFake-style), maps its labels into the project’s standard convention (0=real, 1=fake), and optionally applies the shared `clean_text` function to create a `text_cleaned` column.
  - `zero_shot_test(model, X_test, y_test, average="macro")`: Runs a trained model (baseline, embedding-based, or transformer wrapper) on the external dataset without any fine-tuning and returns Macro-F1, providing a direct measure of cross-dataset robustness.
- **Alignment with project goals**: Implements the cross-dataset transfer component described in the proposal, enabling zero-shot evaluation on datasets beyond ISOT using the same 0/1 label convention and primary Macro-F1 metric, and keeping interfaces compatible with existing preprocessing and evaluation utilities.

#### 2025-12-01 – Repository restructuring and main experiments notebook

- **Repository restructuring**: Restructured project into final `data/`, `src/`, and `notebooks/` layout as planned.
  - Moved all Python scripts from `scripts/` to `src/` directory.
  - Moved training data from `training-data/` to `data/training/` directory.
  - Moved test data from `test-data/` to `data/test/` directory.
  - Created `notebooks/` directory for Jupyter notebooks.
  - Updated all file paths in scripts (including `prepare_welfake_sample.py`) to reflect new directory structure.
  - Updated `.gitignore` to use new `data/test/WELFake_Dataset.csv` path.
  - Updated example paths in docstrings across all scripts.
- **Main experiments notebook created**: Added `notebooks/08_main_experiments.ipynb` that ties together all experimental components.
  - Comprehensive pipeline covering data loading, preprocessing, splitting (random and topic-holdout), baseline models (TF-IDF + LogReg/SVM), advanced models (embeddings, transformers), evaluation, and cross-dataset transfer.
  - Uses importlib for importing modules with numeric prefixes, ensuring compatibility with the `src/` structure.
  - Includes results summary and robustness analysis comparing models across split strategies.
  - Documents all steps with markdown cells explaining the experimental design and project goals.
- **Alignment with project goals**: The restructuring improves project organization and maintainability, while the main experiments notebook provides a complete, reproducible pipeline for evaluating model robustness under different scenarios (random splits, topic-holdout, cross-dataset transfer) as specified in the evaluation plan.

#### 2025-12-08 – Data split approach revised

- **Initial approach**: Planned random/topic splits on a single combined dataset to test robustness.
- **Issue discovered**: With separate all-fake and all-real training files, the held-out-topic split produced single-class test sets (e.g., politicsNews all real), making Macro-F1/ROC/PR unusable and triggering the two-class guard in `evaluate`.
- **Decision**: Keep `src/02_data_splitting.py` for historical context but ignore it in the final flow. Final evaluation will train on full training data, use a fake-only check for false negatives on `data/test/fake.csv`, and use mixed labeled test sets (e.g., WELFake) for full metrics.
- **Notebook cleanup**: Removed legacy placeholder cells in `08_main_experiments.ipynb` that belonged to the old topic-holdout/cross-dataset sections; kept brief notes instead to reflect the refactor.

#### 2025-12-08 – Notebook refactor for final evaluation flow

- **Training/eval flow**: Train once on full ISOT (Fake + True); no random/topic splits.
- **Test sets**: 
  - `data/test/fake.csv` (all fake) evaluated with a single-class-safe check (fake recall, false negatives).
  - `data/test/WELFake_Dataset_sample_1000.csv` (mixed, labeled) evaluated with full metrics (Macro-F1, ROC-AUC, PR-AUC); labels flipped to our convention (1=fake, 0=real).
- **Models**: TF-IDF + LogReg/SVM; optional sentence-embedding classifier shares the same train/test setup and fake-only check.
- **Vectorizer handling**: Fit TF-IDF once on full train; reuse for both test sets to avoid refitting/overwriting.
- **Cross-dataset**: WELFake eval now serves as the cross-dataset transfer check; legacy placeholder cells removed.

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
  - [x] `01_preprocessing.py`
    - [x] `load_isot(fake_path, real_path)`
    - [x] `clean_text(text)`
    - [x] `apply_cleaning(df)`
  - [x] `02_data_splitting.py`
    - [x] `random_split(df)`
    - [x] `topic_holdout_split(df, topic_column, heldout_topic)`
  - [x] `03_baseline_models.py`
    - [x] `build_tfidf()`
    - [x] `train_logreg(X_train, y_train)`
    - [x] `train_svm(X_train, y_train)`
  - [x] `04_baseline_eval.py`
    - [x] `evaluate(model, X_test, y_test, model_name="baseline")`

#### Reuben – Advanced models and transfer

- **Core scripts (planned)**:
  - [x] `05_embeddings_model.py`
    - [x] `build_embeddings(model_name="all-MiniLM-L6-v2")`
    - [x] `embed_text(embedder, texts)`
    - [x] `train_embedding_classifier(emb_train, y_train)`
  - [x] `06_transformer_model.py`
    - [x] `build_transformer()`
    - [x] `tokenize(tokenizer, texts)`
    - [x] `train_transformer(model, tokenizer, train_texts, train_labels)`
  - [x] `07_cross_dataset_transfer.py`
    - [x] `load_kaggle_dataset(path)`
    - [x] `zero_shot_test(model, X_test, y_test)`

#### Shared – Integration and project setup

- [x] **Decide initial training vs. test data layout**
  - **Status**: Completed in dev log entry `2025-11-29 – Data layout & large CSV planning`.
- [x] **Add `.gitignore` rule to ignore large WELFake CSV**
  - **Status**: Completed in dev log entry `2025-11-29 – .gitignore for large test CSV`.
- [x] **Create script to generate 1,000-row WELFake sample from WELFake dataset**
  - **Status**: Completed in dev log entry `2025-11-29 – WELFake sample preparation script`.
- [x] **Restructure repository into final `data/`, `src/`, and `notebooks/` layout**
  - **Status**: Completed. All scripts moved to `src/`, training data to `data/training/`, test data to `data/test/`, and created `notebooks/` directory. Updated file paths in all scripts and `.gitignore`.
- [x] **Create `08_main_experiments.ipynb` to tie together all experiments**
  - **Status**: Completed. Created comprehensive notebook in `notebooks/08_main_experiments.ipynb` that integrates all components: preprocessing, data splitting (random and topic-holdout), baseline models (TF-IDF + LogReg/SVM), advanced models (embeddings, transformers), evaluation with Macro-F1/ROC-AUC/PR-AUC, and cross-dataset transfer evaluation.

### Data organization so far

- **Training data location**:
  - `data/training/Fake.csv`
  - `data/training/True.csv`
- **Test data location**:
  - `data/test/fake.csv`
  - `data/test/WELFake_Dataset_sample_1000.csv` (sample of 1,000 rows from the large WELFake dataset, created via `src/prepare_welfake_sample.py`)
  - An additional, very large CSV file (`data/test/WELFake_Dataset.csv`) available locally but not tracked in git.
- **Code location**:
  - All Python scripts are in `src/` directory.
  - Main experiments notebook is in `notebooks/08_main_experiments.ipynb`.
- **Work in progress**: We’ve mainly been deciding where to store training vs. test CSV files and adjusting the project layout to keep this organized.

### Challenges

- **Large CSV sizes** are making git operations (pushing and pulling) slow and a bit cumbersome.
- The **extra test CSV file** is extremely large, which makes it impractical to use in full for quick experiments.
- **Topic-holdout single-class issue**: Holding out certain topics (e.g., politicsNews) produced test sets with only one class, making Macro-F1/ROC/PR invalid and triggering the two-class guard. We decided to skip topic holdout for the final flow and instead use fake-only FN checks plus mixed labeled external sets for full metrics.
- **Label conventions differ across datasets**: ISOT uses 1=fake/0=real; WELFake sources use 0=fake/1=real, so labels must be flipped before evaluation to avoid inverted metrics.
- **Embedding dependencies**: Embedding section requires `sentence-transformers` and model weights; needs guarding to avoid breaking the run in constrained environments.

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
