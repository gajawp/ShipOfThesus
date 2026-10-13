"""
sbert_similarity.py
Document-level meaning similarity to T0: cosine between SBERT embeddings (all-MiniLM-L6-v2).
"""

from sentence_transformers import SentenceTransformer, util

SBERT_MODEL = "all-MiniLM-L6-v2"
_SBERT = None


def get_sbert():
    global _SBERT
    if _SBERT is None:
        print(f"[metrics] Loading SBERT model ({SBERT_MODEL})...")
        _SBERT = SentenceTransformer(SBERT_MODEL)
    return _SBERT


def compute_sbert_cosine(hypotheses, references, batch_size=64):
    """Average cosine similarity between SBERT embeddings."""
    model = get_sbert()
    emb_hyp = model.encode(hypotheses, batch_size=batch_size,
                           show_progress_bar=False, convert_to_tensor=True)
    emb_ref = model.encode(references, batch_size=batch_size,
                           show_progress_bar=False, convert_to_tensor=True)
    cosines = util.cos_sim(emb_hyp, emb_ref).diagonal()
    return float(cosines.mean())
