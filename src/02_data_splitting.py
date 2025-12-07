"""
Data splitting utilities for the robust news classification project.

This module provides functions to split datasets into train/test sets using
different strategies. It supports both random splitting and topic-based holdout
splitting, which is crucial for evaluating model robustness and generalization
across different news topics.

The topic holdout split is particularly important for this project's focus on
robustness, as it tests whether models can generalize to unseen topics rather
than just memorizing topic-specific patterns.

Functions:
    random_split: Split dataset randomly into train and test sets.
    topic_holdout_split: Split dataset by holding out all articles from a specific topic.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def random_split(df: pd.DataFrame, 
                 test_size: float = 0.2,
                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset randomly into train and test sets.
    
    This function performs a standard random split of the dataset, ensuring
    balanced class distribution in both train and test sets. This is useful
    for baseline model evaluation and comparison with topic-based splits.
    
    Args:
        df: DataFrame containing news articles with a 'label' column
            (1 for fake, 0 for real) and other features.
        test_size: Proportion of data to include in the test set (0.0 to 1.0).
                   Default is 0.2 (20% test, 80% train).
        random_state: Random seed for reproducibility. Default is 42.
    
    Returns:
        A tuple of two DataFrames:
            - df_train: Training set DataFrame
            - df_test: Test set DataFrame
    
    Raises:
        KeyError: If 'label' column is missing from the DataFrame.
        ValueError: If test_size is not between 0.0 and 1.0.
    
    Example:
        >>> df = load_isot('data/training/Fake.csv', 'data/training/True.csv')
        >>> df_train, df_test = random_split(df, test_size=0.2, random_state=42)
        >>> print(f"Train: {len(df_train)}, Test: {len(df_test)}")
        Train: 35918, Test: 8980
    """
    # Verify label column exists
    if 'label' not in df.columns:
        raise KeyError("DataFrame must contain a 'label' column for splitting")
    
    # Validate test_size
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")
    
    # Perform stratified split to maintain class balance
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label'],  # Ensure balanced fake/real distribution
        shuffle=True
    )
    
    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # Calculate total for percentage calculations
    total_articles = len(df_train) + len(df_test)
    
    print(f"Random split completed:")
    print(f"  Train set: {len(df_train)} articles ({len(df_train)/total_articles*100:.1f}%)")
    print(f"  Test set: {len(df_test)} articles ({len(df_test)/total_articles*100:.1f}%)")
    print(f"  Train labels: {df_train['label'].value_counts().to_dict()}")
    print(f"  Test labels: {df_test['label'].value_counts().to_dict()}")
    
    return df_train, df_test


def topic_holdout_split(df: pd.DataFrame, 
                        topic_column: str = 'subject',
                        heldout_topic: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by holding out all articles from a specific topic.
    
    This function creates a train/test split where all articles from a specified
    topic are placed in the test set, and all other articles are in the train set.
    This is crucial for evaluating model robustness - it tests whether models can
    generalize to completely unseen topics rather than just memorizing topic-specific
    patterns from training data.
    
    This split strategy aligns with the project's goal of building robust models
    that can classify news across diverse topics, not just those seen during training.
    
    Args:
        df: DataFrame containing news articles with topic information.
            Must contain a 'label' column (1 for fake, 0 for real).
        topic_column: Name of the column containing topic/subject information.
                      Default is 'subject' (as in ISOT dataset).
        heldout_topic: Name of the topic to hold out for testing. If None, the
                      function will select the topic with the most articles as
                      the heldout topic. Default is None.
    
    Returns:
        A tuple of two DataFrames:
            - df_train: Training set DataFrame (all topics except heldout)
            - df_test: Test set DataFrame (only articles from heldout topic)
    
    Raises:
        KeyError: If topic_column or 'label' column is missing from DataFrame.
        ValueError: If heldout_topic is specified but doesn't exist in the data.
    
    Example:
        >>> df = load_isot('data/training/Fake.csv', 'data/training/True.csv')
        >>> # Hold out all 'politicsNews' articles for testing
        >>> df_train, df_test = topic_holdout_split(df, topic_column='subject', 
        ...                                          heldout_topic='politicsNews')
        >>> print(f"Train topics: {df_train['subject'].unique()}")
        >>> print(f"Test topic: {df_test['subject'].unique()}")
    """
    # Verify required columns exist
    if 'label' not in df.columns:
        raise KeyError("DataFrame must contain a 'label' column for splitting")
    
    if topic_column not in df.columns:
        raise KeyError(f"Topic column '{topic_column}' not found in DataFrame. "
                      f"Available columns: {list(df.columns)}")
    
    # Get available topics and their counts
    topic_counts = df[topic_column].value_counts()
    available_topics = topic_counts.index.tolist()
    
    print(f"Available topics in dataset: {available_topics}")
    print(f"Topic distribution:\n{topic_counts}")
    
    # Select heldout topic if not specified
    if heldout_topic is None:
        # Select the topic with the most articles as heldout (most challenging test)
        heldout_topic = topic_counts.index[0]
        print(f"\nNo heldout topic specified. Selecting '{heldout_topic}' "
              f"({topic_counts[heldout_topic]} articles) as heldout topic.")
    else:
        # Verify heldout topic exists
        if heldout_topic not in available_topics:
            raise ValueError(f"Heldout topic '{heldout_topic}' not found in data. "
                           f"Available topics: {available_topics}")
    
    # Split: test set = all articles from heldout topic
    #        train set = all articles from other topics
    mask_heldout = df[topic_column] == heldout_topic
    df_test = df[mask_heldout].copy().reset_index(drop=True)
    df_train = df[~mask_heldout].copy().reset_index(drop=True)
    
    print(f"\nTopic holdout split completed:")
    print(f"  Heldout topic: '{heldout_topic}'")
    print(f"  Train set: {len(df_train)} articles from {len(df_train[topic_column].unique())} topics")
    print(f"  Test set: {len(df_test)} articles from heldout topic '{heldout_topic}'")
    print(f"  Train topics: {sorted(df_train[topic_column].unique().tolist())}")
    print(f"  Train labels: {df_train['label'].value_counts().to_dict()}")
    print(f"  Test labels: {df_test['label'].value_counts().to_dict()}")
    
    return df_train, df_test

