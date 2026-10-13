"""
profiles.py
Per-document feature profiles for the RQ1 decay analysis, grouped into "planks"
(feature families), and the distance used to compare a paraphrase with its T0.

Each text is parsed once with spaCy; every family is computed from that parse:

  family               features                                   distance
  Word choice          set of lower-cased words                   Jaccard distance
  Function words       relative frequencies of 198 function words Jensen-Shannon
  Punctuation          relative frequencies of 12 marks           Jensen-Shannon
  Lexical richness     MATTR, average word length                 RMS z-score difference
  Sentence shape       mean and std of sentence length            RMS z-score difference
  POS profile          Universal POS distribution                 Jensen-Shannon
  Dependency labels    dependency-label distribution              Jensen-Shannon
  Tree depth           mean/max tree depth, mean dep. distance    RMS z-score difference
  Readability          Flesch-Kincaid grade                       RMS z-score difference
  Meaning              SBERT embedding                            cosine distance
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from src.features.lexical_features import mattr, function_word_profile, FUNCTION_WORDS
from src.features.stylometric_features import (punctuation_profile, sentence_length_std,
                                               PUNCTUATION_MARKS)
from src.features.syntactic_features import UPOS, dependency_labels, parse, syntactic_features

# family -> (kind, columns or column prefix); "words" and "meaning" are handled separately
FAMILIES = {
    "Word choice": ("jaccard", None),
    "Function words": ("js", "fw_"),
    "Punctuation": ("js", "punct_"),
    "Lexical richness": ("z", ["mattr", "avg_word_length"]),
    "Sentence shape": ("z", ["mean_sentence_length", "sentence_length_std"]),
    "POS profile": ("js", "pos_"),
    "Dependency labels": ("js", "dep_"),
    "Tree depth": ("z", ["mean_tree_depth", "max_tree_depth", "mean_dep_distance"]),
    "Readability": ("z", ["fk_grade"]),
    "Meaning (SBERT)": ("cosine", None),
}
STYLE_FAMILIES = [f for f in FAMILIES if f != "Meaning (SBERT)"]
LEXICAL_FAMILIES = ["Word choice", "Function words", "Punctuation", "Lexical richness"]
SYNTACTIC_FAMILIES = ["Sentence shape", "POS profile", "Dependency labels", "Tree depth",
                      "Readability"]


def text_profiles(texts, batch_size=64, log_every=2000, logger=None):
    """Return a DataFrame with one row of scalar and distribution features per text."""
    dep_vocab = dependency_labels()
    rows = []
    for i, (text, doc) in enumerate(zip(texts, parse(texts, batch_size))):
        words = [t.text.lower() for t in doc if t.is_alpha]
        sent_lengths = [sum(1 for t in s if not t.is_punct) for s in doc.sents]
        scalars, dists = syntactic_features(doc, dep_vocab)
        row = {
            "n_words": len(words),
            "mattr": mattr(words),
            "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
            "mean_sentence_length": float(np.mean(sent_lengths)) if sent_lengths else 0.0,
            "sentence_length_std": sentence_length_std(sent_lengths),
            **scalars,
        }
        row.update({f"fw_{w}": v for w, v in zip(FUNCTION_WORDS, function_word_profile(words))})
        row.update({f"punct_{m}": v for m, v in zip(PUNCTUATION_MARKS, punctuation_profile(text))})
        row.update({f"pos_{p}": v for p, v in zip(UPOS, dists["pos"])})
        row.update({f"dep_{d}": v for d, v in zip(dep_vocab, dists["dep"])})
        rows.append(row)
        if logger and (i + 1) % log_every == 0:
            logger.info(f"  parsed {i + 1:,}/{len(texts):,} texts")
    return pd.DataFrame(rows)


def _columns(profiles, spec):
    return [c for c in profiles.columns if c.startswith(spec)] if isinstance(spec, str) else spec


def family_distance(family, a, b, texts_a=None, texts_b=None, emb_a=None, emb_b=None, scale=None):
    """
    Row-wise distance between two aligned profile tables `a` and `b` for one family.
    `scale` maps scalar columns to their standard deviation (z-scoring); texts and
    embeddings are needed for the word-choice and meaning families.
    """
    kind, spec = FAMILIES[family]
    if kind == "jaccard":
        out = []
        for x, y in zip(texts_a, texts_b):
            x, y = set(x.lower().split()), set(y.lower().split())
            out.append(1 - len(x & y) / max(len(x | y), 1))
        return np.array(out)
    if kind == "cosine":
        num = (emb_a * emb_b).sum(axis=1)
        den = np.linalg.norm(emb_a, axis=1) * np.linalg.norm(emb_b, axis=1)
        return 1 - num / np.maximum(den, 1e-12)
    cols = _columns(a, spec)
    x, y = a[cols].to_numpy(float), b[cols].to_numpy(float)
    if kind == "js":
        # an empty profile (e.g. no punctuation) is identical to another empty one and
        # maximally different from a non-empty one
        return np.array([jensenshannon(p, q, base=2) if p.sum() and q.sum()
                         else float(bool(p.sum()) != bool(q.sum())) for p, q in zip(x, y)])
    sd = np.array([scale[c] for c in cols])
    return np.sqrt((((x - y) / sd) ** 2).mean(axis=1))
