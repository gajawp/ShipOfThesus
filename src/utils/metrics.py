"""
metrics.py
All similarity metrics for T1–T3 against the T0 reference, in one table.
"""

import pandas as pd

from src.similarity.bleu_rouge import compute_bleu, compute_rouge_l
from src.similarity.bertscore import compute_bertscore, rescale_bertscore
from src.similarity.sbert_similarity import compute_sbert_cosine
from src.utils.config import GENERATIONS


def compute_all_metrics(gens_dict, use_bertscore=True):
    """
    Compute BLEU, ROUGE-L, BERTScore, SBERT for T1, T2, T3 vs T0.

    Parameters
    ----------
    gens_dict : dict  {"T0": [...], "T1": [...], "T2": [...], "T3": [...]}
    use_bertscore : bool  set False to skip slow BERTScore during debugging

    Returns
    -------
    pd.DataFrame  columns: [generation, bleu, rouge_l, bertscore_f1, bertscore_rescaled,
                            sbert_cosine]
    """
    refs = gens_dict["T0"]
    rows = [{"generation": "T0", "bleu": 1.0, "rouge_l": 1.0,
             "bertscore_f1": 1.0, "bertscore_rescaled": 1.0, "sbert_cosine": 1.0}]

    for gen in GENERATIONS[1:]:
        hyps = gens_dict[gen]
        print(f"[metrics] Computing metrics for {gen}...")
        bleu = compute_bleu(hyps, refs)
        rouge = compute_rouge_l(hyps, refs)
        bert = compute_bertscore(hyps, refs) if use_bertscore else float("nan")
        sbert = compute_sbert_cosine(hyps, refs)
        rows.append({
            "generation": gen,
            "bleu": round(bleu, 4),
            "rouge_l": round(rouge, 4),
            "bertscore_f1": round(bert, 4),
            "bertscore_rescaled": round(rescale_bertscore(bert), 4),
            "sbert_cosine": round(sbert, 4),
        })
        print(f"  BLEU={bleu:.4f}  ROUGE-L={rouge:.4f}  "
              f"BERTScore={bert:.4f}  SBERT={sbert:.4f}")

    return pd.DataFrame(rows)
