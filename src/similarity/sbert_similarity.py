"""
sbert_similarity.py
Document-level meaning similarity to T0: cosine between SBERT embeddings (all-MiniLM-L6-v2).

The model reads at most 256 word pieces, so longer texts are split into chunks of whole
sentences that fit the window, each chunk is embedded, and the document embedding is the
token-weighted mean of its chunk embeddings. Texts that fit in one window are unchanged.
"""

import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_METHOD = f"{SBERT_MODEL}/chunked-mean"   # recorded in caches so a method change invalidates them
_SBERT = None

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def get_sbert():
    global _SBERT
    if _SBERT is None:
        print(f"[metrics] Loading SBERT model ({SBERT_MODEL})...")
        _SBERT = SentenceTransformer(SBERT_MODEL)
    return _SBERT


def chunk_text(text, tokenizer, max_tokens):
    """Split `text` into (chunk, n_tokens) pairs of whole sentences, each <= max_tokens word pieces.
    A single sentence longer than the window is split at word-piece boundaries."""
    n_total = len(tokenizer.tokenize(text))
    if n_total <= max_tokens:
        return [(text, max(n_total, 1))]
    chunks, current, current_len = [], [], 0
    for sentence in _SENTENCE_END.split(text.strip()):
        tokens = tokenizer.tokenize(sentence)
        if not tokens:
            continue
        if len(tokens) > max_tokens:
            if current:
                chunks.append((" ".join(current), current_len))
                current, current_len = [], 0
            for i in range(0, len(tokens), max_tokens):
                piece = tokens[i:i + max_tokens]
                chunks.append((tokenizer.convert_tokens_to_string(piece), len(piece)))
            continue
        if current_len + len(tokens) > max_tokens:
            chunks.append((" ".join(current), current_len))
            current, current_len = [], 0
        current.append(sentence)
        current_len += len(tokens)
    if current:
        chunks.append((" ".join(current), current_len))
    return chunks or [(text, 1)]


def encode_texts(texts, batch_size=64):
    """One embedding per text covering the whole text (chunked when longer than the window)."""
    model = get_sbert()
    max_tokens = model.max_seq_length - 2   # room for [CLS] and [SEP]
    chunk_texts, weights, owner = [], [], []
    for i, text in enumerate(texts):
        for chunk, n_tokens in chunk_text(text, model.tokenizer, max_tokens):
            chunk_texts.append(chunk)
            weights.append(n_tokens)
            owner.append(i)
    chunk_emb = model.encode(chunk_texts, batch_size=batch_size, show_progress_bar=False)
    weights, owner = np.asarray(weights, float), np.asarray(owner)
    emb = np.zeros((len(texts), chunk_emb.shape[1]), dtype=np.float32)
    np.add.at(emb, owner, chunk_emb * weights[:, None])
    return emb / np.bincount(owner, weights=weights, minlength=len(texts))[:, None]


def compute_sbert_cosine(hypotheses, references, batch_size=64):
    """Average cosine similarity between SBERT embeddings."""
    emb_hyp = torch.from_numpy(encode_texts(hypotheses, batch_size))
    emb_ref = torch.from_numpy(encode_texts(references, batch_size))
    cosines = util.pairwise_cos_sim(emb_hyp, emb_ref)
    return float(cosines.mean())
