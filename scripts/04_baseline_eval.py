"""
Baseline model evaluation utilities for the robust news classification project.

This module provides comprehensive evaluation functions for baseline models,
calculating multiple metrics to assess model performance on fake news detection.
The evaluation aligns with the project's evaluation plan, using Macro-F1 as the
primary metric (to balance classes) and PR-AUC and ROC-AUC as secondary metrics
for ranking and threshold analysis.

These metrics are crucial for understanding model performance under different
evaluation scenarios (random splits vs. topic-holdout splits) and for comparing
baseline models with more advanced approaches.

Functions:
    evaluate: Comprehensive evaluation of a trained model with multiple metrics.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)
from typing import Union, Dict
import numpy as np


def evaluate(model,
             X_test: Union[list, np.ndarray],
             y_test: Union[list, np.ndarray],
             model_name: str = "baseline") -> Dict[str, float]:
    """
    Comprehensive evaluation of a trained model with multiple metrics.
    
    This function evaluates a trained classifier using the project's evaluation
    plan: Macro-F1 as the primary metric (to balance classes in fake news detection)
    and PR-AUC (Average Precision) and ROC-AUC as secondary metrics for ranking
    and threshold analysis. Additional metrics (accuracy, precision, recall) are
    also computed for comprehensive analysis.
    
    The evaluation aligns with the project's focus on robustness and provides
    interpretable metrics that can be compared across different split strategies
    (random vs. topic-holdout) and different models (baselines vs. advanced models).
    
    Args:
        model: A trained classifier with .predict() and .predict_proba() methods
              (e.g., LogisticRegression, LinearSVC). For models without predict_proba,
              probability estimates are obtained via decision_function.
        X_test: Test features. Should be TF-IDF transformed text data
               (sparse matrix or dense array) matching the format used for training.
        y_test: Test labels. Binary labels where 1 = fake news, 0 = real news.
        model_name: Name of the model for display purposes. Default is "baseline".
    
    Returns:
        A dictionary containing evaluation metrics:
            - 'accuracy': Overall classification accuracy
            - 'precision': Precision score (macro-averaged)
            - 'recall': Recall score (macro-averaged)
            - 'f1_macro': Macro-F1 score (PRIMARY METRIC for this project)
            - 'f1_weighted': Weighted F1 score
            - 'roc_auc': ROC-AUC score (secondary metric)
            - 'pr_auc': PR-AUC (Average Precision) score (secondary metric)
            - 'confusion_matrix': 2x2 confusion matrix as a list of lists
    
    Raises:
        AttributeError: If the model does not have required prediction methods.
        ValueError: If X_test and y_test have incompatible shapes or if
                   y_test contains values other than 0 and 1.
    
    Example:
        >>> vectorizer = build_tfidf()
        >>> X_train_tfidf = vectorizer.fit_transform(X_train_text)
        >>> X_test_tfidf = vectorizer.transform(X_test_text)
        >>> model = train_logreg(X_train_tfidf, y_train)
        >>> metrics = evaluate(model, X_test_tfidf, y_test, model_name="Logistic Regression")
        >>> print(f"Macro-F1: {metrics['f1_macro']:.4f}")
        >>> print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    """
    # Convert to numpy array if needed
    if isinstance(y_test, list):
        y_test = np.array(y_test)
    
    # Validate labels are binary
    unique_labels = np.unique(y_test)
    if not np.array_equal(unique_labels, [0, 1]):
        raise ValueError(f"y_test must contain only 0 and 1 labels, got {unique_labels}")
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    print(f"Test set size: {len(y_test)} samples")
    print(f"Class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get probability scores for AUC metrics
    try:
        # Try to use predict_proba (works for Logistic Regression)
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        # Fall back to decision_function (works for LinearSVC)
        try:
            decision_scores = model.decision_function(X_test)
            # Convert decision scores to probabilities using sigmoid-like transformation
            # This is an approximation; for LinearSVC, we normalize to [0, 1] range
            y_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
        except AttributeError:
            raise AttributeError("Model must have either predict_proba() or decision_function() method")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Calculate AUC metrics
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        # Handle edge case where only one class is present
        roc_auc = float('nan')
        print("Warning: ROC-AUC cannot be calculated (only one class present in y_test)")
    
    try:
        pr_auc = average_precision_score(y_test, y_proba)
    except ValueError:
        pr_auc = float('nan')
        print("Warning: PR-AUC cannot be calculated (only one class present in y_test)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Display results
    print(f"\n{'Metrics':<25} {'Value':<15}")
    print(f"{'-'*40}")
    print(f"{'Accuracy':<25} {accuracy:.4f}")
    print(f"{'Precision (macro)':<25} {precision_macro:.4f}")
    print(f"{'Recall (macro)':<25} {recall_macro:.4f}")
    print(f"{'F1-score (macro)':<25} {f1_macro:.4f}  <-- PRIMARY METRIC")
    print(f"{'F1-score (weighted)':<25} {f1_weighted:.4f}")
    
    if not np.isnan(roc_auc):
        print(f"{'ROC-AUC':<25} {roc_auc:.4f}  <-- Secondary metric")
    else:
        print(f"{'ROC-AUC':<25} N/A")
    
    if not np.isnan(pr_auc):
        print(f"{'PR-AUC (Avg Precision)':<25} {pr_auc:.4f}  <-- Secondary metric")
    else:
        print(f"{'PR-AUC (Avg Precision)':<25} N/A")
    
    print(f"\nConfusion Matrix:")
    print(f"{'':<15} {'Predicted Real':<15} {'Predicted Fake':<15}")
    print(f"{'Actual Real':<15} {cm[0][0]:<15} {cm[0][1]:<15}")
    print(f"{'Actual Fake':<15} {cm[1][0]:<15} {cm[1][1]:<15}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake'], zero_division=0))
    
    print(f"{'='*60}\n")
    
    # Return metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision_macro),
        'recall': float(recall_macro),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else None,
        'pr_auc': float(pr_auc) if not np.isnan(pr_auc) else None,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics
