"""
lexical_features.py
Word-level features: type-token ratio, function-word ratio, average word length.
"""

import numpy as np
import nltk
from nltk.corpus import stopwords

from src.utils.helpers import ensure_nltk

ensure_nltk("punkt_tab", "stopwords")

_FUNCTION_WORDS = set(stopwords.words("english"))


def _tokens(text):
    return nltk.word_tokenize(text.lower())


def _words(text):
    return [t for t in _tokens(text) if t.isalpha()]


def ttr(text):
    """Type-Token Ratio: vocabulary richness."""
    toks = _words(text)
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def function_word_ratio(text):
    """Fraction of tokens that are function words (stopwords)."""
    toks = _words(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in _FUNCTION_WORDS) / len(toks)


def avg_word_length(text):
    """Mean character length of alphabetic tokens."""
    toks = _words(text)
    if not toks:
        return 0.0
    return float(np.mean([len(t) for t in toks]))


# ── Length-robust and profile features (RQ1 decay analysis) ──────

FUNCTION_WORDS = sorted(_FUNCTION_WORDS)


def mattr(words, window=50):
    """
    Moving-average type-token ratio over lower-cased words. Unlike TTR it does not rise
    mechanically when a text gets shorter (Covington & McFall, 2010).
    """
    if len(words) < window:
        return len(set(words)) / len(words) if words else 0.0
    counts, ratios = {}, []
    for w in words[:window]:
        counts[w] = counts.get(w, 0) + 1
    ratios.append(len(counts) / window)
    for i in range(window, len(words)):
        old, new = words[i - window], words[i]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        counts[new] = counts.get(new, 0) + 1
        ratios.append(len(counts) / window)
    return float(np.mean(ratios))


def function_word_profile(words):
    """Relative frequency of each function word among the function words used."""
    counts = np.array([0.0] * len(FUNCTION_WORDS))
    index = {w: i for i, w in enumerate(FUNCTION_WORDS)}
    for w in words:
        if w in index:
            counts[index[w]] += 1
    return counts / counts.sum() if counts.sum() else counts
