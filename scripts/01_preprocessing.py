"""
Preprocessing utilities for the robust news classification project.

This module provides functions to load the ISOT (Fake and True news) dataset
and apply text cleaning operations to prepare news articles for classification.
These utilities support the project's goal of building a robust model to
classify news articles as fake or real.

Functions:
    load_isot: Load and combine fake and real news CSV files from the ISOT dataset.
    clean_text: Clean a single text string by removing noise and standardizing format.
    apply_cleaning: Apply text cleaning to a DataFrame column and return cleaned data.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Union


def load_isot(fake_path: Union[str, Path], real_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and combine fake and real news CSV files from the ISOT dataset.
    
    This function reads the separate Fake.csv and True.csv files, adds labels
    to distinguish between fake (1) and real (0) articles, and combines them
    into a single DataFrame for preprocessing and modeling.
    
    Args:
        fake_path: Path to the Fake.csv file containing fake news articles.
                   Expected columns: title, text, subject, date
        real_path: Path to the True.csv file containing real news articles.
                   Expected columns: title, text, subject, date
    
    Returns:
        A pandas DataFrame containing:
            - All columns from the original CSVs (title, text, subject, date)
            - A 'label' column: 1 for fake news, 0 for real news
            - A 'source_file' column: 'fake' or 'real' to track origin
    
    Raises:
        FileNotFoundError: If either fake_path or real_path does not exist.
        ValueError: If required columns (title, text) are missing from the CSVs.
    
    Example:
        >>> df = load_isot('training-data/Fake.csv', 'training-data/True.csv')
        >>> print(df['label'].value_counts())
        1    23481  # fake news
        0    21417  # real news
    """
    fake_path = Path(fake_path)
    real_path = Path(real_path)
    
    # Verify files exist
    if not fake_path.exists():
        raise FileNotFoundError(f"Fake news file not found: {fake_path}")
    if not real_path.exists():
        raise FileNotFoundError(f"Real news file not found: {real_path}")
    
    # Load fake news articles
    print(f"Loading fake news from {fake_path}...")
    df_fake = pd.read_csv(fake_path)
    
    # Load real news articles
    print(f"Loading real news from {real_path}...")
    df_real = pd.read_csv(real_path)
    
    # Verify required columns exist
    required_cols = ['title', 'text']
    for col in required_cols:
        if col not in df_fake.columns:
            raise ValueError(f"Required column '{col}' missing from fake news CSV")
        if col not in df_real.columns:
            raise ValueError(f"Required column '{col}' missing from real news CSV")
    
    # Add labels: 1 for fake, 0 for real
    df_fake['label'] = 1
    df_fake['source_file'] = 'fake'
    
    df_real['label'] = 0
    df_real['source_file'] = 'real'
    
    # Combine into single DataFrame
    df_combined = pd.concat([df_fake, df_real], ignore_index=True)
    
    print(f"Loaded {len(df_fake)} fake articles and {len(df_real)} real articles")
    print(f"Total: {len(df_combined)} articles")
    
    return df_combined


def clean_text(text: str) -> str:
    """
    Clean a single text string by removing noise and standardizing format.
    
    This function applies common text preprocessing steps to normalize news
    article text for classification. It removes URLs, extra whitespace, and
    standardizes text formatting while preserving meaningful content.
    
    Args:
        text: Raw text string to clean. Can be None or NaN (will be converted
              to empty string).
    
    Returns:
        Cleaned text string with:
            - URLs removed
            - Multiple spaces collapsed to single spaces
            - Leading/trailing whitespace removed
            - Empty strings if input was None/NaN
    
    Example:
        >>> clean_text("Check out https://example.com   for more info")
        'Check out for more info'
        >>> clean_text("Multiple   spaces   here")
        'Multiple spaces here'
    """
    # Handle None/NaN values
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove URLs (http://, https://, www.)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses (basic pattern)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Replace multiple whitespace characters with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


def apply_cleaning(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Apply text cleaning to a specified column in a DataFrame.
    
    This function takes a DataFrame containing news articles and applies the
    clean_text function to a specified column (typically 'text' or 'title').
    It creates a new column with cleaned text and returns the modified DataFrame.
    
    Args:
        df: DataFrame containing news articles with text columns to clean.
        text_column: Name of the column containing text to clean. Default is 'text'.
                    Can also be 'title' or any other text column.
    
    Returns:
        A new DataFrame with:
            - All original columns
            - A new column named '{text_column}_cleaned' containing cleaned text
            - Original column remains unchanged
    
    Raises:
        KeyError: If the specified text_column does not exist in the DataFrame.
    
    Example:
        >>> df = pd.DataFrame({'text': ['Check https://example.com   out', 'Some text']})
        >>> df_cleaned = apply_cleaning(df, 'text')
        >>> print(df_cleaned['text_cleaned'].iloc[0])
        'Check out'
    """
    # Verify column exists
    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Create cleaned column name
    cleaned_column = f"{text_column}_cleaned"
    
    # Apply cleaning to the specified column
    df_cleaned = df.copy()
    df_cleaned[cleaned_column] = df_cleaned[text_column].apply(clean_text)
    
    print(f"Applied text cleaning to column '{text_column}'")
    print(f"Created new column '{cleaned_column}' with cleaned text")
    
    return df_cleaned
