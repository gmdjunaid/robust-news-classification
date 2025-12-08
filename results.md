### Current results (ISOT train → WELFake test)

- **Setup**: Train on full ISOT (Fake + True). Evaluate on two held-out sets:
  - `data/test/fake.csv` (all fake) → fake-only recall / false negatives.
  - `data/test/WELFake_Dataset_sample_1000.csv` (mixed labeled) → full metrics.
- **Label convention**: Project uses 1=fake, 0=real; WELFake source labels flipped accordingly.

#### WELFake (mixed labeled, 1000 samples)
- **Logistic Regression (TF-IDF)**: Macro-F1 ≈ 0.1379; ROC-AUC ≈ 0.0953; PR-AUC ≈ 0.2914; Accuracy ≈ 0.1550. Heavily misclassifies real as fake.
- **Linear SVM (TF-IDF)**: Macro-F1 ≈ 0.1450; ROC-AUC ≈ 0.0947; PR-AUC ≈ 0.2913; Accuracy ≈ 0.1650. Similar failure mode as LogReg.
- **Embedding classifier (MiniLM embeddings + LogReg)**: Macro-F1 ≈ 0.1833; ROC-AUC ≈ 0.1087; PR-AUC ≈ 0.2940; Accuracy ≈ 0.1880. Slightly better than TF-IDF but still near-random.
- **Confusion patterns**: Models largely predict “fake” for real articles and still miss many fake articles. Overall discrimination is close to random (ROC-AUC ~0.1).

#### Fake-only test (all fake)
- **LogReg (TF-IDF)**: Fake recall ≈ 0.9365 (false negatives ≈ 826 of 12,999).
- **Linear SVM (TF-IDF)**: Fake recall ≈ 0.9473 (false negatives ≈ 685 of 12,999).
- **Embedding model**: Fake recall ≈ 0.7511 (false negatives ≈ 3,235 of 12,999).

#### Interpretation
- Models fit ISOT well (train accuracy ~0.99) but **fail to transfer to WELFake**, indicating strong domain/style mismatch or preprocessing/schema differences.
- Embeddings help slightly but are far from adequate; TF-IDF does better on the fake-only sanity check yet still collapses on WELFake.
- Next steps: verify preprocessing parity and label mapping, consider domain adaptation or fine-tuning on WELFake-like data, and tune thresholds using a WELFake validation split.
