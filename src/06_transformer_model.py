"""
Transformer-based modeling utilities for the robust news classification project.

This module provides helper functions to:

    - Build a transformer classifier (e.g., DistilBERT) for fake news detection
    - Tokenize news article text into model-ready inputs
    - Fine-tune the transformer on ISOT data using the Hugging Face Trainer API

These utilities are designed to work with the preprocessing and splitting
functions in:

    - 01_preprocessing.py   (for loading and cleaning ISOT data)
    - 02_data_splitting.py  (for random and topic-holdout splits)

and to provide a stronger, modern benchmark compared to TF-IDF baselines and
embedding-based classifiers.

Typical usage in experiments:

    >>> from scripts01_preprocessing import load_isot, apply_cleaning
    >>> from scripts02_data_splitting import random_split
    >>> from scripts06_transformer_model import (
    ...     build_transformer, tokenize, train_transformer
    ... )
    >>>
    >>> # Load and preprocess ISOT data
    >>> df = load_isot("data/training/Fake.csv", "data/training/True.csv")
    >>> df = apply_cleaning(df, text_column="text")  # creates 'text_cleaned'
    >>> df_train, df_val = random_split(df, test_size=0.2)
    >>>
    >>> train_texts = df_train["text_cleaned"].tolist()
    >>> train_labels = df_train["label"].values
    >>>
    >>> model, tokenizer = build_transformer()
    >>> model = train_transformer(
    ...     model=model,
    ...     tokenizer=tokenizer,
    ...     train_texts=train_texts,
    ...     train_labels=train_labels,
    ...     output_dir="models/distilbert_isot"
    ... )
"""

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


@dataclass
class FakeNewsDataset(Dataset):
    """
    Simple Dataset wrapper for transformer fine-tuning on fake news data.

    This dataset expects:

        - texts: sequence of news article strings (already preprocessed, e.g. 'text_cleaned')
        - labels: sequence or array of binary labels (1 = fake, 0 = real)
        - tokenizer: a Hugging Face tokenizer (e.g., DistilBERT tokenizer)

    The dataset lazily tokenizes each example on access, which keeps memory
    usage low and integrates well with the Trainer API.
    """

    texts: Sequence[str]
    labels: Sequence[int]
    tokenizer: Any
    max_length: int = 256

    def __post_init__(self) -> None:
        # Basic validation to keep labels aligned with texts
        if len(self.texts) != len(self.labels):
            raise ValueError(
                f"Number of texts ({len(self.texts)}) does not match "
                f"number of labels ({len(self.labels)})."
            )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = int(self.labels[idx])

        # Tokenize a single example
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # encoded["input_ids"] has shape (1, max_length); squeeze to (max_length,)
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


class TransformerSklearnWrapper:
    """
    Lightweight sklearn-style wrapper around a transformer classifier.

    This wrapper makes a Hugging Face transformer model compatible with the
    evaluation utilities in 04_baseline_eval.py by exposing a similar API:

        - predict(texts): returns hard 0/1 predictions for input texts
        - predict_proba(texts): returns probability estimates for classes
          [P(real), P(fake)] for each input text

    The wrapper takes care of tokenizing raw/preprocessed news article text,
    moving tensors to the correct device (CPU or GPU), and applying softmax to
    convert logits into probabilities. It assumes the model was trained with
    label convention:

        - index 0 -> real news (label 0)
        - index 1 -> fake news (label 1)

    Example:
        >>> from scripts04_baseline_eval import evaluate
        >>> model, tokenizer = build_transformer()
        >>> wrapper = TransformerSklearnWrapper(model, tokenizer)
        >>> metrics = evaluate(wrapper, X_test_texts, y_test, model_name="DistilBERT")
    """

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

        if device is None:
            # Prefer CUDA, then Apple Silicon MPS, then CPU
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Move model to the chosen device and set to eval mode for inference.
        self.model.to(self.device)
        self.model.eval()

    def _encode_texts(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts and move tensors to the correct device.
        """
        if not texts:
            raise ValueError("TransformerSklearnWrapper received an empty list of texts.")

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """
        Return class probabilities for each input text.

        The output is a NumPy array of shape (n_examples, 2) where:

            - column 0: P(real news)
            - column 1: P(fake news)
        """
        self.model.eval()
        encoded = self._encode_texts(texts)

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits  # shape: (batch_size, num_labels)
            probs = F.softmax(logits, dim=-1)  # convert to probabilities

        return probs.detach().cpu().numpy()

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """
        Return hard 0/1 predictions for each input text.

        The predicted label is the argmax over the class probabilities:

            - 0 -> real news
            - 1 -> fake news
        """
        probs = self.predict_proba(texts)
        return probs.argmax(axis=1)


def build_transformer(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Build a transformer classifier and its corresponding tokenizer.

    This function loads a pre-trained transformer model (by default,
    DistilBERT) and configures it for binary sequence classification with
    labels:

        - 1 = fake news
        - 0 = real news

    Args:
        model_name:
            Hugging Face model identifier for the base transformer to load.
            Default is "distilbert-base-uncased".
        num_labels:
            Number of output labels. For this project we use 2 (fake vs real).

    Returns:
        A tuple of (model, tokenizer):
            - model: AutoModelForSequenceClassification ready for fine-tuning.
            - tokenizer: Matching AutoTokenizer instance.

    Example:
        >>> model, tokenizer = build_transformer()
        >>> inputs = tokenizer(
        ...     ["Example news article"],
        ...     padding=True,
        ...     truncation=True,
        ...     max_length=256,
        ...     return_tensors="pt",
        ... )
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits
    """
    print(f"Loading transformer model '{model_name}' for sequence classification...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    print("Transformer model and tokenizer loaded successfully.")
    return model, tokenizer


def tokenize(
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a batch of news article texts for transformer models.

    This convenience function wraps the tokenizer call with consistent
    padding, truncation, and max_length settings so experiments are easy
    to reproduce and compare.

    Args:
        tokenizer:
            A Hugging Face tokenizer returned by build_transformer().
        texts:
            Sequence of raw or preprocessed news article strings to tokenize.
            Typically this will be the 'text_cleaned' column from ISOT data.
        max_length:
            Maximum sequence length (in wordpiece tokens). Longer texts are
            truncated; shorter ones are padded. Default is 256.

    Returns:
        A dictionary of PyTorch tensors suitable for passing directly to a
        transformer model, including keys such as "input_ids" and
        "attention_mask".

    Example:
        >>> model, tokenizer = build_transformer()
        >>> batch = tokenize(tokenizer, ["Example news article"])
        >>> outputs = model(**batch)
    """
    if not texts:
        print("Warning: tokenize received an empty list of texts.")
        return {
            "input_ids": torch.empty((0, max_length), dtype=torch.long),
            "attention_mask": torch.empty((0, max_length), dtype=torch.long),
        }

    print(f"Tokenizing {len(texts)} texts (max_length={max_length})...")
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    print("Tokenization complete.")
    return encoded


def train_transformer(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    output_dir: str = "models/distilbert_isot",
    num_train_epochs: float = 0.5,
    per_device_train_batch_size: int = 32,
    max_length: int = 128,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    logging_steps: int = 100,
) -> AutoModelForSequenceClassification:
    """
    Fine-tune a transformer classifier on ISOT fake news data.

    This function configures a Hugging Face Trainer for supervised fine-tuning
    on binary fake news labels using the standard label convention:

        - 1 = fake news
        - 0 = real news

    It is intentionally minimal and focuses on training only; evaluation and
    hyperparameter search are handled elsewhere (e.g., in notebooks or higher-
    level experiment scripts).

    Args:
        model:
            A transformer classifier created by build_transformer().
        tokenizer:
            Matching tokenizer for the model.
        train_texts:
            Sequence of preprocessed news article texts (e.g., 'text_cleaned').
        train_labels:
            Sequence or array of binary labels (1 = fake, 0 = real).
        output_dir:
            Directory path where model checkpoints and logs will be saved.
        num_train_epochs:
            Number of training epochs. Can be fractional (e.g., 0.5 for half epoch).
            Default is 0.5 for faster training on MPS/CPU.
        per_device_train_batch_size:
            Batch size per device (GPU/CPU) during training.
            Default is 32 for faster training (increase if memory allows).
        max_length:
            Maximum sequence length for tokenization. Shorter sequences (default 128)
            train faster than longer ones (256) with minimal accuracy impact.
        learning_rate:
            Learning rate for the AdamW optimizer.
        weight_decay:
            Weight decay parameter for regularization.
        logging_steps:
            How often to log training metrics (in optimizer steps).

    Returns:
        The fine-tuned transformer model (same instance as the input `model`).

    Example:
        >>> model, tokenizer = build_transformer()
        >>> model = train_transformer(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     train_texts=train_texts,
        ...     train_labels=train_labels,
        ...     output_dir="models/distilbert_isot",
        ...     num_train_epochs=0.5,  # Half epoch for faster training
        ...     max_length=128,  # Shorter sequences for speed
        ...     per_device_train_batch_size=32,  # Larger batches if memory allows
        ... )
    """
    # Convert labels to NumPy array for basic validation
    y = np.asarray(train_labels)
    unique_labels = np.unique(y)
    if not np.array_equal(unique_labels, [0, 1]):
        raise ValueError(
            f"train_labels must contain only 0 and 1, got {unique_labels}"
        )

    if len(train_texts) != len(train_labels):
        raise ValueError(
            f"Number of texts ({len(train_texts)}) does not match "
            f"number of labels ({len(train_labels)})."
        )

    print("Preparing dataset for transformer fine-tuning...")
    train_dataset = FakeNewsDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    print("Configuring TrainingArguments...")
    # Note: older versions of transformers may not support evaluation_strategy/save_strategy/report_to
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        logging_dir=f"{output_dir}/logs",
    )

    print("Initializing Trainer and starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    print("Transformer fine-tuning complete.")

    # The trainer updates the model in place, but we return it for convenience.
    return model


