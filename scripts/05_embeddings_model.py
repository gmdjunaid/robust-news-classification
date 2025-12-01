"""
Embedding-based modeling utilities for the robust news classification project.

This module provides helper functions to:

    - Build a sentence-embedding model (SentenceTransformer)
    - Convert news article text into dense vector embeddings
    - Train a simple classifier (Logistic Regression) on top of those embeddings

These functions are designed to plug directly into the preprocessing and
splitting utilities in:

    - 01_preprocessing.py   (for loading and cleaning ISOT data)
    - 02_data_splitting.py  (for random and topic-holdout splits)

and to serve as a more powerful, but still relatively lightweight, alternative
to the TF-IDF baselines implemented in 03_baseline_models.py.

Typical usage in experiments:

    >>> from scripts01_preprocessing import load_isot, apply_cleaning
    >>> from scripts02_data_splitting import random_split
    >>> from scripts05_embeddings_model import (
    ...     build_embeddings, embed_text, train_embedding_classifier
    ... )
    >>>
    >>> # Load and preprocess ISOT data
    >>> df = load_isot("training-data/Fake.csv", "training-data/True.csv")
    >>> df = apply_cleaning(df, text_column="text")  # creates 'text_cleaned'
    >>> df_train, df_test = random_split(df)
    >>>
    >>> # Prepare texts and labels
    >>> X_train = df_train["text_cleaned"].tolist()
    >>> X_test = df_test["text_cleaned"].tolist()
    >>> y_train = df_train["label"].values
    >>> y_test = df_test["label"].values
    >>>
    >>> # Build embedder and compute embeddings
    >>> embedder = build_embeddings(model_name="all-MiniLM-L6-v2")
    >>> emb_train = embed_text(embedder, X_train)
    >>> emb_test = embed_text(embedder, X_test)
    >>>
    >>> # Train classifier on top of embeddings
    >>> clf = train_embedding_classifier(emb_train, y_train)
    >>> y_pred = clf.predict(emb_test)
    >>> # Metrics can then be computed with the evaluation utilities.
"""

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


def build_embeddings(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Build and return a SentenceTransformer model for generating text embeddings.

    This function creates a sentence-embedding model (by default, the
    "all-MiniLM-L6-v2" model from sentence-transformers) that can convert
    variable-length news articles into fixed-length dense vectors. These
    embeddings can then be used as features for a downstream classifier.

    Args:
        model_name:
            Hugging Face / sentence-transformers model name to load.
            The default "all-MiniLM-L6-v2" is a compact, general-purpose
            English sentence encoder that balances speed and accuracy.

    Returns:
        A loaded SentenceTransformer instance ready to encode news article text.

    Example:
        >>> embedder = build_embeddings()
        >>> emb = embedder.encode(["Example news article"], show_progress_bar=False)
        >>> emb.shape
        (1, 384)
    """
    print(f"Loading sentence-embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    print("Embedding model loaded successfully.")
    return embedder


def embed_text(
    embedder: SentenceTransformer,
    texts: Sequence[str],
    batch_size: int = 32,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """
    Convert a sequence of news article texts into dense vector embeddings.

    This function wraps SentenceTransformer.encode with sensible defaults for
    batch size and progress reporting. It expects preprocessed text (for
    example, the 'text_cleaned' column produced by apply_cleaning in
    01_preprocessing.py) and returns a NumPy array of embeddings.

    Args:
        embedder:
            A SentenceTransformer instance created by build_embeddings().
        texts:
            Sequence of news article texts to embed. Typically this is a list
            or array of cleaned article bodies, e.g. df_train["text_cleaned"].
        batch_size:
            Number of examples to encode per batch. Larger batches are faster
            but require more GPU/CPU memory. Default is 32.
        show_progress_bar:
            Whether to display a progress bar while encoding. Default is True.

    Returns:
        A 2D NumPy array of shape (n_examples, embedding_dim) containing the
        dense vector representation for each input text.

    Example:
        >>> embedder = build_embeddings()
        >>> texts = ["Fake news example", "Real news example"]
        >>> emb = embed_text(embedder, texts, batch_size=16)
        >>> emb.shape
        (2, 384)
    """
    if not texts:
        # Return an empty (0, dim) array if no texts are provided. We cannot
        # know the dimension without encoding, so we handle this edge case
        # by returning a (0, 0) array and letting the caller decide how to proceed.
        print("Warning: embed_text received an empty list of texts.")
        return np.empty((0, 0), dtype=np.float32)

    print(f"Encoding {len(texts)} texts into embeddings...")
    embeddings = embedder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def train_embedding_classifier(
    emb_train: np.ndarray,
    y_train,
    C: float = 1.0,
    max_iter: int = 5000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier on top of sentence embeddings.

    This function mirrors the baseline train_logreg behavior but assumes that
    the input features are dense embeddings produced by embed_text(). It uses
    the same binary label convention as the rest of the project:

        - 1 = fake news
        - 0 = real news

    Args:
        emb_train:
            2D NumPy array of shape (n_examples, embedding_dim) containing
            training embeddings for news articles.
        y_train:
            1D array-like of length n_examples with binary labels
            (1 = fake, 0 = real). Can be a list, NumPy array, or pandas Series.
        C:
            Inverse of regularization strength for Logistic Regression.
            Smaller values specify stronger regularization. Default is 1.0.
        max_iter:
            Maximum number of iterations for solver convergence. Default is 5000,
            which is higher than the TF-IDF baseline to account for the denser
            feature space.
        random_state:
            Random seed for reproducibility. Default is 42.

    Returns:
        A trained LogisticRegression classifier ready for prediction on
        new embeddings (e.g., test set embeddings).

    Example:
        >>> embedder = build_embeddings()
        >>> emb_train = embed_text(embedder, X_train)
        >>> clf = train_embedding_classifier(emb_train, y_train)
        >>> emb_test = embed_text(embedder, X_test)
        >>> y_pred = clf.predict(emb_test)
    """
    # Convert labels to NumPy array for validation and compatibility
    y_train = np.asarray(y_train)

    # Basic validation: ensure binary labels are 0/1
    unique_labels = np.unique(y_train)
    if not np.array_equal(unique_labels, [0, 1]):
        raise ValueError(
            f"y_train must contain only 0 and 1 labels, got {unique_labels}"
        )

    if emb_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Number of embeddings ({emb_train.shape[0]}) does not match "
            f"number of labels ({y_train.shape[0]})."
        )

    print("Training Logistic Regression classifier on embeddings...")
    print(f"  Training samples: {emb_train.shape[0]}")
    print(f"  Embedding dimension: {emb_train.shape[1]}")
    print(f"  C (regularization): {C}")
    print(f"  max_iter: {max_iter}")

    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(emb_train, y_train)

    train_accuracy = model.score(emb_train, y_train)
    print("Training completed.")
    print(f"  Training accuracy on embeddings: {train_accuracy:.4f}")

    return model


