"""
syntactic_features.py
Structural "skeleton" features from a spaCy parse (en_core_web_sm), to compare with the
lexical "skin":
  * POS-tag distribution (Universal POS)
  * dependency-label distribution
  * dependency-tree depth (mean and max per sentence) and mean dependency distance
  * Flesch-Kincaid grade level (readability)

Install the model once:  python -m spacy download en_core_web_sm
"""

import re

import numpy as np
import spacy

UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
        "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]

_NLP = None


def get_nlp():
    """spaCy pipeline with only the tagger and parser (NER and lemmas are not needed)."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


def dependency_labels():
    return sorted(get_nlp().get_pipe("parser").labels)


def parse(texts, batch_size=64):
    """Yield spaCy Docs for an iterable of texts."""
    yield from get_nlp().pipe(texts, batch_size=batch_size)


def _distribution(labels, vocabulary):
    counts = np.array([labels.count(v) for v in vocabulary], dtype=float)
    return counts / counts.sum() if counts.sum() else counts


def _depth(token):
    # spaCy creates a new Token object on every access, so compare indices, not identity
    depth = 0
    while token.head.i != token.i:
        token = token.head
        depth += 1
    return depth


def _syllables(word):
    """Vowel-group heuristic, with a silent final 'e'."""
    word = word.lower()
    groups = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and groups > 1:
        groups -= 1
    return max(groups, 1)


def flesch_kincaid_grade(doc):
    words = [t.text for t in doc if t.is_alpha]
    n_sents = max(len(list(doc.sents)), 1)
    if not words:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    return 0.39 * len(words) / n_sents + 11.8 * syllables / len(words) - 15.59


def syntactic_features(doc, dep_vocab):
    """
    Return (scalars, distributions) for one parsed document.

    scalars:       mean_tree_depth, max_tree_depth, mean_dep_distance, fk_grade
    distributions: {"pos": array over UPOS, "dep": array over dep_vocab}
    """
    sent_depths = [max((_depth(t) for t in s), default=0) for s in doc.sents]
    arcs = [abs(t.i - t.head.i) for t in doc if t.head.i != t.i and not t.is_punct]
    scalars = {
        "mean_tree_depth": float(np.mean(sent_depths)) if sent_depths else 0.0,
        "max_tree_depth": float(max(sent_depths, default=0)),
        "mean_dep_distance": float(np.mean(arcs)) if arcs else 0.0,
        "fk_grade": flesch_kincaid_grade(doc),
    }
    distributions = {
        "pos": _distribution([t.pos_ for t in doc], UPOS),
        "dep": _distribution([t.dep_ for t in doc if not t.is_punct], dep_vocab),
    }
    return scalars, distributions
