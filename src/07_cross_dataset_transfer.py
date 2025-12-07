"""
Cross-dataset transfer utilities for the robust news classification project.

This module focuses on evaluating how well models trained on the ISOT dataset
transfer to a different fake-news dataset (e.g., the "Getting Real About Fake
News" Kaggle dataset or the WELFake sample). The goal is to test robustness
under distribution shift rather than only on random splits of a single dataset.

It provides:

    - load_kaggle_dataset: Load and clean an external fake-news dataset, mapping
      labels into the project's standard convention (0 = real, 1 = fake).
    - zero_shot_test: Run an ISOT-trained model on the external dataset without
      further fine-tuning and compute Macro-F1 to quantify transfer performance.

These utilities are designed to be flexible so they can work with different
external CSV schemas by specifying the text and label column names, and they
can optionally reuse the `clean_text` function from 01_preprocessing.py.

Typical usage in experiments (conceptual):

    >>> from scripts01_preprocessing import clean_text
    >>> from scripts04_baseline_eval import evaluate
    >>> from scripts07_cross_dataset_transfer import load_kaggle_dataset, zero_shot_test
    >>>
    >>> # Assume `model` was trained on ISOT using TF-IDF, embeddings, or a transformer wrapper
    >>> df_kaggle = load_kaggle_dataset(
    ...     path="data/kaggle_fake_news.csv",
    ...     text_column="text",
    ...     label_column="label",
    ...     text_cleaner=clean_text,
    ... )
    >>>
    >>> X_test = df_kaggle["text_cleaned"].tolist()   # or precomputed features
    >>> y_test = df_kaggle["label"].values
    >>>
    >>> # Simple zero-shot Macro-F1:
    >>> f1_macro = zero_shot_test(model, X_test, y_test)
    >>> print(f"Zero-shot Macro-F1 on Kaggle: {f1_macro:.4f}")
"""

from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def load_kaggle_dataset(
    path: Union[str, Path],
    text_column: str = "text",
    label_column: str = "label",
    fake_values: Tuple[Any, ...] = ("FAKE", 1, "fake", "Fake"),
    real_values: Tuple[Any, ...] = ("REAL", 0, "real", "Real"),
    text_cleaner: Optional[Callable[[str], str]] = None,
) -> pd.DataFrame:
    """
    Load and normalize an external fake-news dataset for cross-dataset testing.

    This function is intended for datasets like the "Getting Real About Fake
    News" Kaggle dataset or WELFake-like CSVs. It:

        - Reads the CSV into a pandas DataFrame.
        - Maps the original label column into the project's standard convention:
              0 = real news
              1 = fake news
        - Optionally applies a text cleaning function (e.g., clean_text from
          01_preprocessing.py) to produce a `text_cleaned` column.

    The function is deliberately flexible: you can customize the text and label
    column names and the set of values that should be treated as "fake" vs.
    "real". This avoids hard-coding assumptions about the external dataset.

    Args:
        path:
            Path to the external CSV file.
        text_column:
            Name of the column containing raw article text. Default is "text".
        label_column:
            Name of the column containing labels in the external dataset.
            Default is "label".
        fake_values:
            Tuple of values in `label_column` that should be mapped to label 1
            (fake news). Default includes common encodings: "FAKE", 1, etc.
        real_values:
            Tuple of values in `label_column` that should be mapped to label 0
            (real news). Default includes common encodings: "REAL", 0, etc.
        text_cleaner:
            Optional callable that takes a raw text string and returns a cleaned
            version (e.g., `clean_text` from 01_preprocessing.py). If provided,
            a `text_cleaned` column will be created. If None, the raw text is
            copied as-is into `text_cleaned`.

    Returns:
        A pandas DataFrame containing:
            - All original columns from the CSV.
            - A new `label` column with 0=real, 1=fake (project convention).
            - A `text_cleaned` column containing cleaned (or raw) text.

    Raises:
        FileNotFoundError:
            If the CSV file does not exist at the given path.
        KeyError:
            If `text_column` or `label_column` is missing from the CSV.
        ValueError:
            If some label values cannot be mapped to 0 or 1 using the provided
            `fake_values` and `real_values`.

    Example:
        >>> from scripts01_preprocessing import clean_text
        >>> df_kaggle = load_kaggle_dataset(
        ...     "data/kaggle_fake_news.csv",
        ...     text_column="text",
        ...     label_column="label",
        ...     text_cleaner=clean_text,
        ... )
        >>> df_kaggle[["text_cleaned", "label"]].head()
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"External dataset not found at: {csv_path}")

    print(f"Loading external fake-news dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Verify required columns exist
    missing_cols = [col for col in (text_column, label_column) if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns in external dataset: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Map raw labels into 0/1 according to the provided value sets.
    def _map_label(value: Any) -> int:
        if value in fake_values:
            return 1
        if value in real_values:
            return 0
        raise ValueError(
            f"Unrecognized label value '{value}' in column '{label_column}'. "
            f"Update fake_values/real_values to cover this case."
        )

    print("Mapping external labels to project convention (0=real, 1=fake)...")
    df = df.copy()
    df["label"] = df[label_column].apply(_map_label)

    # Apply optional text cleaning
    if text_cleaner is not None:
        print(f"Applying text cleaner to column '{text_column}'...")
        df["text_cleaned"] = df[text_column].apply(text_cleaner)
    else:
        print(
            f"No text_cleaner provided. Copying '{text_column}' to 'text_cleaned' "
            "without additional processing."
        )
        df["text_cleaned"] = df[text_column].astype(str)

    print(
        "External dataset loaded and normalized. "
        f"Total articles: {len(df)}, "
        f"Label distribution: {df['label'].value_counts().to_dict()}"
    )

    return df


def zero_shot_test(
    model: Any,
    X_test: Union[Sequence[Any], np.ndarray],
    y_test: Union[Sequence[int], np.ndarray],
    average: str = "macro",
) -> float:
    """
    Evaluate an ISOT-trained model on an external dataset without fine-tuning.

    This function performs a zero-shot evaluation by:

        - Using the provided `model` to generate predictions on `X_test`.
        - Computing the Macro-F1 score between predictions and `y_test`.

    It is intentionally lightweight and model-agnostic:

        - For TF-IDF or embedding-based models, `X_test` should be the
          corresponding feature matrix (e.g., TF-IDF or embeddings).
        - For transformer models wrapped with TransformerSklearnWrapper,
          `X_test` can be a list of raw or cleaned text strings.

    Args:
        model:
            A trained model with a `predict` method that accepts `X_test`
            and returns 0/1 labels (0=real, 1=fake).
        X_test:
            Test features or texts, depending on the model type.
        y_test:
            Ground-truth binary labels for the external dataset
            (0=real, 1=fake), as produced by load_kaggle_dataset().
        average:
            Averaging method for F1 score. Default is "macro", matching the
            project's primary evaluation metric.

    Returns:
        Macro-F1 score (float) for the zero-shot evaluation.

    Example:
        >>> f1_macro = zero_shot_test(model, X_test, y_test)
        >>> print(f"Zero-shot Macro-F1: {f1_macro:.4f}")
    """
    # Convert y_test to NumPy array for consistency
    y_true = np.asarray(y_test)

    print("Running zero-shot evaluation on external dataset...")
    y_pred = model.predict(X_test)

    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    print(f"Zero-shot Macro-F1 ({average}): {f1:.4f}")

    return float(f1)


