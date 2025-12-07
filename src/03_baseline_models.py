"""
Baseline model training utilities for the robust news classification project.

This module provides functions to build TF-IDF features and train interpretable
baseline classifiers (Logistic Regression and Linear SVM) for fake news detection.
These baseline models serve as interpretable benchmarks against which more
complex models (embeddings, transformers) will be compared.

The models are designed to work with text data that has been preprocessed
through the preprocessing utilities in 01_preprocessing.py.

Functions:
    build_tfidf: Create and configure a TF-IDF vectorizer for text feature extraction.
    train_logreg: Train a Logistic Regression classifier on TF-IDF features.
    train_svm: Train a Linear SVM classifier on TF-IDF features.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from typing import List, Union
import numpy as np


def build_tfidf(max_features: int = 5000,
                min_df: int = 2,
                max_df: float = 0.95,
                ngram_range: tuple = (1, 2),
                stop_words: Union[str, List[str], None] = 'english',
                random_state: int = 42) -> TfidfVectorizer:
    """
    Create and configure a TF-IDF vectorizer for text feature extraction.
    
    This function builds a TF-IDF (Term Frequency-Inverse Document Frequency)
    vectorizer with sensible defaults for news classification. TF-IDF features
    are interpretable and serve as a strong baseline for text classification,
    allowing the model to learn word importance patterns that distinguish fake
    from real news.
    
    Args:
        max_features: Maximum number of features (vocabulary size) to keep.
                     Default is 5000 to balance performance and interpretability.
        min_df: Minimum document frequency for a term to be included in vocabulary.
                Terms that appear in fewer than min_df documents are ignored.
                Default is 2 to filter out very rare terms.
        max_df: Maximum document frequency threshold. Terms that appear in more
                than max_df proportion of documents are ignored (removes very
                common stopwords and boilerplate). Default is 0.95.
        ngram_range: Tuple (min_n, max_n) for n-gram extraction.
                    (1, 2) means unigrams and bigrams. Default is (1, 2).
        stop_words: Language of stop words to remove, or None to keep all words.
                   Default is 'english' to remove common English stopwords.
        random_state: Random seed for reproducibility (used if any randomness
                     is involved). Default is 42.
    
    Returns:
        A configured TfidfVectorizer instance ready to fit and transform text data.
    
    Example:
        >>> vectorizer = build_tfidf(max_features=5000, ngram_range=(1, 2))
        >>> X_train_tfidf = vectorizer.fit_transform(X_train_text)
        >>> X_test_tfidf = vectorizer.transform(X_test_text)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=stop_words
    )
    
    print(f"Created TF-IDF vectorizer with:")
    print(f"  max_features: {max_features}")
    print(f"  min_df: {min_df}")
    print(f"  max_df: {max_df}")
    print(f"  ngram_range: {ngram_range}")
    print(f"  stop_words: {stop_words}")
    
    return vectorizer


def train_logreg(X_train: Union[List[str], np.ndarray],
                 y_train: Union[List[int], np.ndarray],
                 C: float = 1.0,
                 max_iter: int = 1000,
                 random_state: int = 42,
                 solver: str = 'lbfgs') -> LogisticRegression:
    """
    Train a Logistic Regression classifier on TF-IDF features.
    
    This function trains a Logistic Regression model, which is an interpretable
    baseline classifier suitable for binary fake news detection. Logistic Regression
    provides coefficient weights that can be interpreted to understand which words
    or phrases are most indicative of fake vs. real news.
    
    Args:
        X_train: Training features. Should be TF-IDF transformed text data
                (sparse matrix or dense array) from vectorizer.transform().
        y_train: Training labels. Binary labels where 1 = fake news, 0 = real news.
        C: Inverse of regularization strength. Smaller values specify stronger
           regularization. Default is 1.0.
        max_iter: Maximum number of iterations for solver convergence.
                 Default is 1000.
        random_state: Random seed for reproducibility. Default is 42.
        solver: Algorithm to use for optimization. 'lbfgs' works well for
               small to medium datasets. Default is 'lbfgs'.
    
    Returns:
        A trained LogisticRegression classifier ready for prediction.
    
    Raises:
        ValueError: If X_train and y_train have incompatible shapes or if
                   y_train contains values other than 0 and 1.
    
    Example:
        >>> vectorizer = build_tfidf()
        >>> X_train_tfidf = vectorizer.fit_transform(X_train_text)
        >>> model = train_logreg(X_train_tfidf, y_train)
        >>> predictions = model.predict(X_test_tfidf)
    """
    # Convert to numpy array if needed
    if isinstance(y_train, list):
        y_train = np.array(y_train)
    
    # Validate labels are binary
    unique_labels = np.unique(y_train)
    if not np.array_equal(unique_labels, [0, 1]):
        raise ValueError(f"y_train must contain only 0 and 1 labels, got {unique_labels}")
    
    # Create and train model
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver=solver,
        n_jobs=-1  # Use all available cores
    )
    
    print(f"Training Logistic Regression classifier...")
    print(f"  Training samples: {len(y_train)}")
    print(f"  C (regularization): {C}")
    print(f"  max_iter: {max_iter}")
    print(f"  solver: {solver}")
    
    model.fit(X_train, y_train)
    
    print(f"Training completed.")
    print(f"  Training accuracy: {model.score(X_train, y_train):.4f}")
    
    return model


def train_svm(X_train: Union[List[str], np.ndarray],
              y_train: Union[List[int], np.ndarray],
              C: float = 1.0,
              max_iter: int = 1000,
              random_state: int = 42,
              dual: bool = False) -> LinearSVC:
    """
    Train a Linear SVM classifier on TF-IDF features.
    
    This function trains a Linear Support Vector Machine (SVM), which is another
    interpretable baseline classifier for fake news detection. Linear SVMs are
    robust to high-dimensional features and work well with sparse TF-IDF matrices.
    They provide a different learning bias than Logistic Regression, making them
    useful for model comparison.
    
    Args:
        X_train: Training features. Should be TF-IDF transformed text data
                (sparse matrix or dense array) from vectorizer.transform().
        y_train: Training labels. Binary labels where 1 = fake news, 0 = real news.
        C: Regularization parameter. Smaller values specify stronger regularization.
           Default is 1.0.
        max_iter: Maximum number of iterations for solver convergence.
                 Default is 1000.
        random_state: Random seed for reproducibility. Default is 42.
        dual: Whether to solve the dual optimization problem. False (default) is
             recommended when n_samples > n_features, which is typically the case
             with TF-IDF features.
    
    Returns:
        A trained LinearSVC classifier ready for prediction.
    
    Raises:
        ValueError: If X_train and y_train have incompatible shapes or if
                   y_train contains values other than 0 and 1.
    
    Example:
        >>> vectorizer = build_tfidf()
        >>> X_train_tfidf = vectorizer.fit_transform(X_train_text)
        >>> model = train_svm(X_train_tfidf, y_train)
        >>> predictions = model.predict(X_test_tfidf)
    """
    # Convert to numpy array if needed
    if isinstance(y_train, list):
        y_train = np.array(y_train)
    
    # Validate labels are binary
    unique_labels = np.unique(y_train)
    if not np.array_equal(unique_labels, [0, 1]):
        raise ValueError(f"y_train must contain only 0 and 1 labels, got {unique_labels}")
    
    # Create and train model
    model = LinearSVC(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        dual=dual
    )
    
    print(f"Training Linear SVM classifier...")
    print(f"  Training samples: {len(y_train)}")
    print(f"  C (regularization): {C}")
    print(f"  max_iter: {max_iter}")
    print(f"  dual: {dual}")
    
    model.fit(X_train, y_train)
    
    print(f"Training completed.")
    print(f"  Training accuracy: {model.score(X_train, y_train):.4f}")
    
    return model

