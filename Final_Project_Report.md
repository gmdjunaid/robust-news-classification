% Final Project Report
% Team: Junaid (Data Engineering & Baselines), Reuben (Advanced Models & Transfer)
% Date: December 10, 2025

# Robust News Classification: Evaluating Model Generalization Across Datasets and Topics

## Abstract (150-250 words)

Fake news detection models often achieve high accuracy on held-out test sets from the same dataset, but this performance may not generalize to real-world scenarios where news topics, writing styles, and sources differ. This project investigates the robustness of fake news classifiers by evaluating their performance under distribution shift, specifically through cross-dataset transfer evaluation. We train models on the ISOT Fake and Real News dataset (~45,000 articles) using TF-IDF features with Logistic Regression and Linear SVM baselines, as well as sentence-embedding models. Models achieve 99% accuracy on ISOT but show significant performance degradation when tested on the WELFake dataset, with Macro-F1 scores of 0.83 for TF-IDF models and 0.80 for embedding models. Surprisingly, simple TF-IDF models outperform more sophisticated embedding-based approaches on cross-dataset evaluation. Adding metadata features (title concatenation, text length) did not improve performance and in some cases hurt fake-news recall. Our key finding is that models trained on one dataset learn dataset-specific patterns that do not transfer well to external datasets, highlighting the critical importance of cross-dataset evaluation for real-world deployment. This work demonstrates that standard benchmark accuracy is insufficient for assessing fake news detection systems and validates the need for robustness-focused evaluation strategies.

## 1. Project Definition

- **Problem statement:** Most fake-news classifiers are evaluated on random train/test splits from the same dataset, where topic, source, and writing style distributions are similar. This allows models to exploit superficial cues (topic-specific phrases, outlet characteristics, boilerplate text) rather than learning robust signals of misinformation. We aim to build and evaluate models that can generalize across different topics and datasets, exposing where standard evaluation overstates reliability.

- **Strategic aspects:** Fake news is a moving target—new topics, narratives, and writing styles emerge constantly. Models that fail to generalize beyond their training distribution are ineffective for real-world deployment. Our focus on cross-dataset evaluation and robustness testing aligns with broader themes in machine learning about bias, fairness, and preventing shortcut learning. From a policy perspective, understanding the limitations of fake news detection models is crucial for informed deployment decisions and managing expectations about model reliability.

- **Link to course content:** This project connects to course themes on:
  - **Bias and fairness**: Models may learn dataset artifacts rather than generalizable patterns, leading to biased predictions when deployed on different populations
  - **Evaluation rigor**: Standard random-split evaluation can mask generalization failures, similar to issues discussed regarding test set contamination and data leakage
  - **Distribution shift**: Cross-dataset evaluation directly tests out-of-distribution generalization, a core challenge in real-world ML deployment
  - **Interpretability vs. performance**: We compare interpretable TF-IDF models with black-box embedding models, exploring the trade-off between explainability and performance

## 2. Novelty and Importance

- **Motivation:** Fake news detection is a critical problem with real-world consequences, but the research community's focus on benchmark accuracy often obscures deployment challenges. By explicitly testing cross-dataset transfer and exposing generalization failures, this project provides actionable insights for practitioners and highlights the importance of robustness evaluation.

- **Gaps in current practice:**

  1. **Evaluation methodology**: Most fake news detection papers report accuracy on random splits of a single dataset, which may overstate real-world performance
  2. **Shortcut learning**: Models may exploit topic-specific patterns, source characteristics, or dataset artifacts rather than learning generalizable fake-news signals
  3. **Label leakage prevention**: Many approaches don't explicitly address preventing leakage from source identity, date, or topic information

- **Related work:** Prior work in fake news detection focuses on achieving high accuracy on benchmark datasets (ISOT, LIAR, FakeNewsNet) using methods ranging from traditional ML (TF-IDF + classifiers) to deep learning (CNNs, RNNs, transformers). Few studies explicitly evaluate cross-dataset generalization or test robustness under distribution shift. Our contribution differs by:
  - Explicitly designing experiments for cross-dataset evaluation (ISOT → WELFake)
  - Comparing model performance degradation between in-distribution and out-of-distribution settings
  - Documenting a negative result (embedding models underperforming TF-IDF on cross-dataset tasks) that challenges the assumption that more sophisticated models always generalize better

## 3. Progress and Contribution

### Data

**Sources:**

- **ISOT Fake and Real News Dataset**: Primary training dataset containing ~23,481 fake articles and ~21,417 real articles (44,898 total) with columns: `title`, `text`, `subject`, `date`. Articles span multiple topics (politicsNews, worldnews, News, politics, left-news, Government News, US_News, Middle-east).

- **WELFake Dataset**: External test dataset used for cross-dataset evaluation. We use a 10,000-article sample (`WELFake_Dataset_sample_10000.csv`) created by extracting the last 10,000 rows from the full dataset using `src/prepare_welfake_sample.py`.

**Preprocessing:**
We implement text cleaning functions in `src/01_preprocessing.py` to remove noise and standardize formatting:

```python
def clean_text(text: str) -> str:
    """Remove URLs, email addresses, normalize whitespace."""
    # Remove URLs and emails
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

The preprocessing pipeline creates `text_cleaned` and `title_cleaned` columns while preserving original columns. Label convention: `1 = fake news`, `0 = real news` (consistent across ISOT and WELFake after label mapping investigation).

### Methods

**Baseline Models (TF-IDF-based):**

- **TF-IDF Vectorization** (`src/03_baseline_models.py`): Configurable vectorizer with defaults: `max_features=5000`, `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `stop_words='english'`. Supports unigrams and bigrams for capturing phrase-level patterns.

```python
def build_tfidf(max_features: int = 5000,
                min_df: int = 2,
                max_df: float = 0.95,
                ngram_range: tuple = (1, 2),
                stop_words: Union[str, List[str], None] = 'english') -> TfidfVectorizer:
    """Create TF-IDF vectorizer with sensible defaults for news classification."""
    return TfidfVectorizer(max_features=max_features, min_df=min_df,
                          max_df=max_df, ngram_range=ngram_range, stop_words=stop_words)
```

- **Logistic Regression**: Interpretable linear classifier with L2 regularization (`C=1.0`, `solver='lbfgs'`). Provides coefficient weights that can be analyzed to understand word importance.

- **Linear SVM**: Alternative baseline with different learning bias (`C=1.0`, `dual=False` for sparse TF-IDF features). Robust to high-dimensional sparse features.

**Advanced Models:**

- **Sentence Embeddings** (`src/05_embeddings_model.py`): Uses `all-MiniLM-L6-v2` SentenceTransformer model to generate 384-dimensional dense embeddings. Automatically selects best available device (CUDA → MPS → CPU). Trains Logistic Regression classifier on top of embeddings.

```python
def build_embeddings(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Build sentence-embedding model with automatic device selection."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return SentenceTransformer(model_name, device=device)
```

**Feature Engineering Variants:**

- **Multi-feature TF-IDF**: Concatenates `title_cleaned + text_cleaned` and appends simple metadata features (title length, text length in characters) before TF-IDF vectorization.
- **Embedding + Lengths**: Same length features appended to sentence embeddings before classifier training.

### Experiments

**Experimental Design:**

1. **Training**: Train all models on full ISOT dataset (44,898 articles) without any train/test split.
2. **Evaluation on two test sets**:
   - **Fake-only test** (`data/test/fake.csv`, 12,999 articles): All fake news articles. Evaluates false negative rate and fake recall (how many fake articles are correctly identified as fake).
   - **WELFake test** (`data/test/WELFake_Dataset_sample_10000.csv`, 10,000 articles): Mixed labeled external dataset. Full evaluation with Macro-F1, ROC-AUC, PR-AUC, confusion matrices.

**Baselines:**

- TF-IDF + Logistic Regression (text-only)
- TF-IDF + Linear SVM (text-only)
- TF-IDF + Logistic Regression (multi-feature: title+text+lengths)
- TF-IDF + Linear SVM (multi-feature: title+text+lengths)
- Sentence Embeddings + Logistic Regression (text-only)
- Sentence Embeddings + Logistic Regression (with length features)

**Controls:**

- Same preprocessing pipeline applied to all datasets
- Same label convention (1=fake, 0=real) across ISOT and WELFake
- TF-IDF vectorizer fit once on training data, then reused for all test sets
- All models use same random seed (42) for reproducibility

### Results

**WELFake Evaluation (10,000 mixed labeled samples):**

| Model                                   | Macro-F1   | ROC-AUC | PR-AUC |
| --------------------------------------- | ---------- | ------- | ------ |
| Logistic Regression (TF-IDF, text-only) | **0.8310** | 0.9048  | 0.8781 |
| Linear SVM (TF-IDF, text-only)          | **0.8288** | 0.9033  | 0.8803 |
| Logistic Regression (TF-IDF + lengths)  | 0.8085     | 0.8997  | 0.8972 |
| Linear SVM (TF-IDF + lengths)           | 0.8193     | 0.9142  | 0.9095 |
| Embedding Classifier (text-only)        | 0.8021     | 0.8843  | 0.8752 |
| Embedding + Lengths                     | 0.7924     | 0.8665  | 0.8791 |

**Fake-Only Test (12,999 fake articles, recall metrics):**

| Model                                   | Fake Recall | False Negatives |
| --------------------------------------- | ----------- | --------------- |
| Logistic Regression (TF-IDF, text-only) | 0.9365      | 826             |
| Linear SVM (TF-IDF, text-only)          | **0.9473**  | 685             |
| Logistic Regression (TF-IDF + lengths)  | 0.7246      | 3,580           |
| Linear SVM (TF-IDF + lengths)           | 0.8351      | 2,140           |
| Embedding Classifier (text-only)        | 0.7511      | 3,235           |
| Embedding + Lengths                     | 0.5862      | 5,379           |

**Key Findings:**

1. **Simple models outperform complex ones**: TF-IDF + Linear SVM achieves the best Macro-F1 (0.831) and fake recall (0.947) on cross-dataset evaluation, outperforming sentence embedding models.
2. **Feature engineering hurts performance**: Adding title concatenation and length features reduces Macro-F1 and significantly hurts fake recall (e.g., embedding+lengths drops from 0.75 to 0.59 recall).
3. **Cross-dataset gap**: Despite 99% training accuracy on ISOT, models achieve only ~0.83 Macro-F1 on WELFake, indicating significant distribution shift and generalization challenges.
4. **Inverse transfer observation** (from earlier experiments): Initial experiments with label confusion showed ROC-AUC < 0.5, suggesting models were systematically inverted—patterns learned on ISOT were negatively correlated with WELFake. After label correction, models perform better than random but still show substantial degradation.

### Evaluation

**Metrics:**

- **Primary**: Macro-F1 score (balanced across classes)
- **Secondary**: ROC-AUC (ranking quality), PR-AUC (precision-recall tradeoff)
- **Additional**: Accuracy, precision, recall, confusion matrices, classification reports

**Validation Strategy:**

- No validation set split; train on full ISOT to maximize training data
- Evaluate on completely held-out external dataset (WELFake) to test zero-shot generalization
- Separate fake-only evaluation to specifically measure false negative rate (critical for fake news detection)

**Error Analysis:**
The `evaluate` function in `src/04_baseline_eval.py` provides comprehensive metrics:

```python
def evaluate(model, X_test, y_test, model_name: str = "baseline") -> Dict[str, float]:
    """Comprehensive evaluation with Macro-F1 as primary metric."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # or decision_function for SVM
    metrics = {
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        # ... additional metrics
    }
    return metrics
```

### Strengths and Limitations

**Strengths:**

1. **Robust evaluation framework**: Explicit cross-dataset testing exposes generalization failures that random splits would miss
2. **Reproducible pipeline**: Well-documented code in `src/` modules with comprehensive docstrings
3. **Interpretable baselines**: TF-IDF models provide coefficient weights for understanding word importance
4. **Comprehensive metrics**: Multiple evaluation metrics (Macro-F1, ROC-AUC, PR-AUC) provide different perspectives on performance

**Limitations:**

1. **Single external dataset**: WELFake is the only external test set; results may not generalize to other datasets
2. **Label convention issues**: WELFake documentation claimed 0=fake/1=real but empirical evidence showed 1=fake/0=real, suggesting potential label quality issues
3. **No domain adaptation**: Zero-shot evaluation only; no attempts at fine-tuning or domain adaptation techniques
4. **Limited feature engineering**: Only explored title concatenation and length features; URL-derived features, punctuation ratios, and other metadata not explored
5. **Transformer models excluded**: DistilBERT fine-tuning removed due to impractical training times (15-30+ minutes even with optimizations), limiting model comparison scope

**Assumptions:**

- Labels in ISOT and WELFake are correct after label convention correction
- Text cleaning (URL/email removal, whitespace normalization) doesn't remove critical signals
- 10,000-article WELFake sample is representative of the full dataset

## 4. Changes After Proposal

### Scope Changes

1. **Transformer models removed**: Initially planned to include DistilBERT fine-tuning, but removed from notebook due to training time constraints. Even with optimizations (`max_length=128`, `num_train_epochs=0.5`, `batch_size=32`), transformer training took 15-30+ minutes per run, making iterative experimentation impractical. The transformer module (`src/06_transformer_model.py`) remains in codebase for reference but is not used in main pipeline.

2. **Topic-holdout splits abandoned**: Initially planned topic-based holdout splits (training on some subjects, testing on held-out subjects). Discovered that holding out certain topics (e.g., "politicsNews") produced single-class test sets (all real or all fake), making Macro-F1/ROC-AUC invalid. Decided to use full training set + external test sets instead.

3. **Feature engineering expanded**: Added multi-feature experiments (title+text concatenation, length features) based on TA suggestion, but results showed these features hurt performance rather than help.

4. **WELFake sample size increased**: Changed from 1,000 to 10,000 articles for more reliable cross-dataset evaluation.

### Bottlenecks

1. **Training time**: Transformer fine-tuning proved too slow even with GPU acceleration (MPS on Apple Silicon). This limited model comparison to TF-IDF baselines and embeddings.

2. **Label confusion**: WELFake dataset documentation claimed 0=fake/1=real, but community evidence and empirical testing indicated labels were actually 1=fake/0=real. This required label mapping corrections and reinterpretation of early results showing ROC-AUC < 0.5.

3. **Cross-dataset performance gap**: Models trained on ISOT showed significant degradation on WELFake, but time constraints before oral presentation prevented deeper investigation into domain adaptation or more sophisticated feature engineering.

4. **GPU configuration**: Getting transformers and sentence-embeddings to use Apple Silicon MPS GPU required compatible `accelerate` version installation and explicit device selection logic.

### Mitigations

1. **Focus on baselines and embeddings**: Concentrated experimental effort on TF-IDF models and sentence embeddings, which provide good performance/compute tradeoff.

2. **Automatic device selection**: Implemented automatic GPU selection (CUDA → MPS → CPU) in embedding and transformer modules to maximize performance without manual configuration.

3. **Comprehensive evaluation on external dataset**: Instead of topic-holdout splits, focused on robust cross-dataset evaluation using WELFake, which better tests real-world generalization.

4. **Documentation of limitations**: Explicitly documented transformer removal decision in `notes.md` and updated notebook to reflect current experimental scope.

## 5. Discussion and Conclusion

### Takeaways

1. **Simple models can outperform complex ones**: TF-IDF + Linear SVM achieved the best cross-dataset performance (Macro-F1: 0.831, Fake Recall: 0.947), outperforming sentence embedding models. This challenges the assumption that more sophisticated models always generalize better.

2. **Cross-dataset evaluation is critical**: Models achieving 99% accuracy on ISOT dropped to ~83% Macro-F1 on WELFake, demonstrating that standard random-split evaluation significantly overstates real-world reliability.

3. **Feature engineering isn't always helpful**: Adding title concatenation and length features reduced performance, suggesting that more features don't necessarily improve generalization and may introduce noise or dataset-specific patterns.

4. **Dataset-specific patterns dominate**: The significant performance gap between ISOT and WELFake suggests models learn dataset-specific artifacts (source characteristics, writing style, topic distribution) rather than generalizable fake-news signals.

5. **Negative results are valuable**: The finding that embedding models underperform TF-IDF on cross-dataset tasks is a meaningful negative result that informs future work and challenges common assumptions.

### Implications

**Practical Significance:**

- Practitioners should prioritize cross-dataset evaluation over benchmark accuracy when assessing fake news detection models for deployment
- Simple, interpretable models (TF-IDF + SVM) may be preferable to black-box deep learning models when generalization is the priority
- Feature engineering should be validated on external datasets, not just held-out splits

**Theoretical Significance:**

- Highlights the importance of robustness testing and distribution shift evaluation in ML research
- Demonstrates that shortcut learning (exploiting dataset artifacts) is a significant problem in fake news detection
- Validates the need for evaluation frameworks that test generalization beyond benchmark datasets

### Next Steps

1. **Additional external datasets**: Evaluate models on multiple external datasets (beyond WELFake) to confirm findings and identify dataset-agnostic patterns.

2. **Domain adaptation experiments**: Test fine-tuning approaches (e.g., few-shot learning on WELFake, domain-adversarial training) to improve cross-dataset performance.

3. **Richer feature engineering**: Explore URL-derived features, punctuation/capitalization ratios, readability metrics, and source credibility signals that may improve generalization.

4. **Interpretability analysis**: Analyze TF-IDF coefficients to identify words/phrases that distinguish fake vs. real news across datasets vs. dataset-specific patterns.

5. **Transformer fine-tuning on GPU**: With access to stronger GPU resources, revisit DistilBERT fine-tuning and compare with baseline models.

6. **Label quality investigation**: Investigate WELFake label quality issues and explore semi-supervised learning approaches if labels are noisy.

## 6. References

- **ISOT Fake and Real News Dataset**: Available on Kaggle. Contains ~45k labeled news articles from various topics.
- **WELFake Dataset**: "Getting Real About Fake News" dataset, available on Kaggle. Used for cross-dataset evaluation.
- **Sentence Transformers**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. _Proceedings of EMNLP-IJCNLP_.
- **scikit-learn**: Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. _JMLR_.
- Course materials on bias, fairness, and evaluation methodology.

## Appendix

### Implementation Notes

**Hyperparameters:**

- **TF-IDF**: `max_features=5000`, `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `stop_words='english'`
- **Logistic Regression**: `C=1.0`, `max_iter=1000`, `solver='lbfgs'`, `random_state=42`
- **Linear SVM**: `C=1.0`, `max_iter=1000`, `dual=False`, `random_state=42`
- **Sentence Embeddings**: `model_name='all-MiniLM-L6-v2'`, `batch_size=32`, embedding dimension: 384
- **Embedding Classifier**: `C=1.0`, `max_iter=5000`, `random_state=42`

**Training Details:**

- Training set: Full ISOT dataset (44,898 articles)
- Test sets: WELFake sample (10,000 articles), fake-only test (12,999 articles)
- All models trained on full dataset (no validation split)
- TF-IDF vectorizer fit on training data, then reused for test sets

**Compute Budget:**

- Training: ~1-2 minutes for TF-IDF models, ~5-10 minutes for embedding models (on Apple Silicon MPS)
- Evaluation: < 1 minute per model on test sets
- Total experiment time: ~30-45 minutes for complete pipeline

### Additional Results

**Detailed Confusion Matrices (WELFake, 10k sample):**

Logistic Regression (TF-IDF, text-only):

- True Negatives (Real→Real): ~4,500
- False Positives (Real→Fake): ~344
- False Negatives (Fake→Real): ~515
- True Positives (Fake→Fake): ~4,641

Linear SVM (TF-IDF, text-only):

- True Negatives (Real→Real): ~4,510
- False Positives (Real→Fake): ~334
- False Negatives (Fake→Real): ~560
- True Positives (Fake→Fake): ~4,596

**Training Accuracies:**

- TF-IDF models: ~99% on ISOT training set
- Embedding models: ~96% on ISOT training set

### Reproducibility

**Environment Setup:**

1. Install dependencies: `pip install -r requirements.txt`

   - Key packages: `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.3.0`, `sentence-transformers>=2.2.0`, `torch>=2.0.0`, `jupyter>=1.0.0`

2. Data preparation:

   - Place ISOT dataset in `data/training/`: `Fake.csv`, `True.csv`
   - Run `python src/prepare_welfake_sample.py` to generate WELFake 10k sample
   - Ensure `data/test/fake.csv` exists for fake-only evaluation

3. Run experiments:
   - Open `notebooks/08_main_experiments.ipynb` in Jupyter
   - Execute cells sequentially (all outputs included in notebook)
   - For embedding models, ensure GPU/MPS is available (automatic device selection)

**Random Seeds:**

- All random seeds set to `42` for reproducibility
- TF-IDF, Logistic Regression, Linear SVM, embedding classifier all use `random_state=42`

**Code Organization:**

- Preprocessing: `src/01_preprocessing.py`
- Data splitting (not used in final flow): `src/02_data_splitting.py`
- Baseline models: `src/03_baseline_models.py`
- Evaluation: `src/04_baseline_eval.py`
- Embeddings: `src/05_embeddings_model.py`
- Cross-dataset transfer: `src/07_cross_dataset_transfer.py`
- Main pipeline: `notebooks/08_main_experiments.ipynb`

**Results Log:**

- Comprehensive results documented in `results.md`
- All notebook outputs saved in `notebooks/08_main_experiments.ipynb`
