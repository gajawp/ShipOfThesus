"""
stylometric_features.py
Sentence- and punctuation-level features, and the combined stylometric profile per generation.
Features: Type-Token Ratio (TTR), function-word ratio, average sentence length,
punctuation density, and average word length.
"""

import string

import numpy as np
import pandas as pd
import nltk

from src.features.lexical_features import ttr, function_word_ratio, avg_word_length
from src.utils.config import GENERATIONS
from src.utils.helpers import ensure_nltk

ensure_nltk("punkt_tab")

_PUNCT = set(string.punctuation)

def avg_sentence_length(text):
    """Mean number of words per sentence."""
    try:
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = text.split(".")
    lengths = [len(s.split()) for s in sents if len(s.split()) > 0]
    return float(np.mean(lengths)) if lengths else 0.0


def punctuation_density(text):
    """Punctuation marks per 100 words."""
    words = text.split()
    if not words:
        return 0.0
    puncts = sum(1 for ch in text if ch in _PUNCT)
    return puncts / len(words) * 100


PUNCTUATION_MARKS = [",", ".", ";", ":", "!", "?", "\"", "'", "(", ")", "-", "/"]


def sentence_length_std(sentence_lengths):
    """Standard deviation of words per sentence (rhythm variation)."""
    return float(np.std(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0


def punctuation_profile(text):
    """Relative frequency of each punctuation mark among the marks used."""
    counts = np.array([text.count(m) for m in PUNCTUATION_MARKS], dtype=float)
    return counts / counts.sum() if counts.sum() else counts


# Column order of the feature tables
FEATURES = {
    "ttr": ttr,
    "function_word_ratio": function_word_ratio,
    "avg_sentence_length": avg_sentence_length,
    "punctuation_density": punctuation_density,
    "avg_word_length": avg_word_length,
}


def compute_features_for_texts(texts):
    """Return mean feature values across a list of texts."""
    return {name: round(float(np.mean([fn(t) for t in texts])), 4)
            for name, fn in FEATURES.items()}


def compute_all_features(gens_dict):
    """
    Compute stylometric features for T0–T3.

    Returns
    -------
    pd.DataFrame  columns: [generation, ttr, function_word_ratio,
                             avg_sentence_length, punctuation_density, avg_word_length]
    """
    rows = []
    for gen in GENERATIONS:
        print(f"[features] Computing features for {gen}...")
        feats = compute_features_for_texts(gens_dict[gen])
        feats["generation"] = gen
        rows.append(feats)
    return pd.DataFrame(rows)[["generation"] + list(FEATURES)]
