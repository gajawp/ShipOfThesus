"""
bleu_rouge.py
Surface (n-gram overlap) similarity to T0: BLEU-4 and ROUGE-L.
These measure how much wording is kept, not meaning.
"""

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def compute_bleu(hypotheses, references):
    """Sentence BLEU-4 per document (Chen-Cherry smoothing method 4), averaged."""
    smoother = SmoothingFunction().method4
    scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.lower().split()
        ref_tokens = ref.lower().split()
        scores.append(sentence_bleu([ref_tokens], hyp_tokens,
                                    smoothing_function=smoother))
    return float(np.mean(scores))


def compute_rouge_l(hypotheses, references):
    """Average ROUGE-L F1."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, hyp)["rougeL"].fmeasure
              for hyp, ref in zip(hypotheses, references)]
    return float(np.mean(scores))
