### Current results (ISOT train → WELFake test, 10k sample)

- **Setup**: Train on full ISOT (Fake + True). Evaluate on two held-out sets:
  - `data/test/fake.csv` (all fake) → fake-only recall / false negatives.
  - `data/test/WELFake_Dataset_sample_10000.csv` (mixed labeled) → full metrics.
- **Label convention**: Project uses 1=fake, 0=real; WELFake labels already match this convention in the sample.

#### WELFake (mixed labeled, 10,000 samples)
- **Logistic Regression (TF-IDF)**: Macro-F1 0.8310; ROC-AUC 0.9048; PR-AUC 0.8781.
- **Linear SVM (TF-IDF)**: Macro-F1 0.8288; ROC-AUC 0.9033; PR-AUC 0.8803.
- **Logistic Regression (TF-IDF + title/text lengths)**: Macro-F1 0.8085; ROC-AUC 0.8997; PR-AUC 0.8972.
- **Linear SVM (TF-IDF + title/text lengths)**: Macro-F1 0.8193; ROC-AUC 0.9142; PR-AUC 0.9095.
- **Embedding classifier (text only, MiniLM + LogReg)**: Macro-F1 0.8021; ROC-AUC 0.8843; PR-AUC 0.8752.
- **Embedding + lengths**: Macro-F1 0.7924; ROC-AUC 0.8665; PR-AUC 0.8791.

#### Fake-only test (all fake, 12,999 samples)
- **LogReg (TF-IDF)**: Fake recall 0.9365 (FN=826).
- **Linear SVM (TF-IDF)**: Fake recall 0.9473 (FN=685).
- **LogReg (TF-IDF + lengths)**: Fake recall 0.7246 (FN=3,580).
- **Linear SVM (TF-IDF + lengths)**: Fake recall 0.8351 (FN=2,140).
- **Embedding (text only)**: Fake recall 0.7511 (FN=3,235).
- **Embedding + lengths**: Fake recall 0.5862 (FN=5,379).

#### Interpretation
- Adding simple length features (title/text char counts) did **not** improve WELFake macro-F1; TF-IDF + lengths slightly underperformed text-only. Fake-only recall also dropped for the multi-feature variants.
- Embeddings remain behind TF-IDF on WELFake and fake-only recall; appending lengths to embeddings hurt further.
- Even with better-than-random discrimination on the 10k sample, cross-dataset robustness is still limited (macro-F1 ≈ 0.79–0.83). Larger gains may require richer feature engineering (URLs, punctuation/ratios, title weighting) or domain adaptation.***
