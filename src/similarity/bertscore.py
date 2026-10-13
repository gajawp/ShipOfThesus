"""
bertscore.py
BERTScore F1 to T0 (distilbert-base-uncased), raw and rescaled.
"""

import os

import pandas as pd
import bert_score
from bert_score import score as bert_score_fn
from bert_score.utils import model2layers

BERTSCORE_MODEL = "distilbert-base-uncased"


def bertscore_baseline(model_type=BERTSCORE_MODEL):
    """F1 that BERTScore gives unrelated English text (its official rescaling baseline)."""
    path = os.path.join(os.path.dirname(bert_score.__file__), "rescale_baseline", "en",
                        f"{model_type}.tsv")
    return float(pd.read_csv(path).iloc[model2layers[model_type]]["F"])


def rescale_bertscore(f1, model_type=BERTSCORE_MODEL):
    """Map raw F1 so that unrelated text ~ 0 and identical text = 1 (linear, so means commute)."""
    b = bertscore_baseline(model_type)
    return (f1 - b) / (1 - b)


def compute_bertscore(hypotheses, references, batch_size=32):
    """Average BERTScore F1 using distilbert-base-uncased."""
    P, R, F1 = bert_score_fn(
        hypotheses, references,
        model_type=BERTSCORE_MODEL,
        batch_size=batch_size,
        verbose=False,
        device=None
    )
    return float(F1.mean())
