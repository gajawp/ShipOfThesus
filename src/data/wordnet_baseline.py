"""
wordnet_baseline.py
Rule-based iterative paraphrase simulator using NLTK WordNet synonym replacement.

Phase-1 baseline: demonstrates the methodology pipeline (semantic metrics, feature
extraction, decay curves). The LLM paraphrase chains now come from the Ship of Theseus
corpus (see load_data.py); this simulator is kept as the "wordnet" baseline.
"""

import random
import re
import nltk
from nltk.corpus import wordnet

from src.utils.helpers import ensure_nltk

ensure_nltk("wordnet", "averaged_perceptron_tagger_eng", "punkt_tab", "stopwords")


def _get_wordnet_pos(treebank_tag):
    """Map POS tagger tags to WordNet POS constants."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return None


def _synonym_replace(word, pos):
    """Return a random synonym of word (same POS) if available, else the word itself."""
    synsets = wordnet.synsets(word, pos=pos) if pos else []
    candidates = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower() and " " not in candidate:
                candidates.add(candidate)
    if candidates:
        return random.choice(list(candidates))
    return word


def paraphrase_text(text, swap_rate=0.25, seed=None):
    """
    Paraphrase text by replacing swap_rate fraction of content words with synonyms
    and applying light structural modifications (sentence boundary shifting).

    Parameters
    ----------
    text      : str   input text
    swap_rate : float fraction of eligible tokens to replace (0–1)
    seed      : int   random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    try:
        sentences = nltk.sent_tokenize(text)
        result_sentences = []
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(tokens)
            new_tokens = []
            for word, tag in tagged:
                wn_pos = _get_wordnet_pos(tag)
                if (wn_pos is not None
                        and word.isalpha()
                        and len(word) > 3
                        and random.random() < swap_rate):
                    new_tokens.append(_synonym_replace(word, wn_pos))
                else:
                    new_tokens.append(word)
            # Reconstruct with basic detokenization
            result_sentences.append(_detokenize(new_tokens))
        return " ".join(result_sentences)
    except Exception:
        return text  # fallback: return unchanged


def _detokenize(tokens):
    """Simple detokenizer: rejoin punctuation without leading spaces."""
    text = ""
    for i, tok in enumerate(tokens):
        if tok in {",", ".", "!", "?", ";", ":", "'s", "n't", "'re", "'ve", "'ll", "'d", "'m"}:
            text += tok
        elif i == 0:
            text += tok
        else:
            text += " " + tok
    return text


# Swap rates per generation (cumulative difficulty):
#   T0→T1: gentle (25%), T1→T2: moderate (38%), T2→T3: aggressive (50%)
_GENERATION_RATES = {1: 0.25, 2: 0.38, 3: 0.50}


def generate_generations(texts, n_generations=3, seed=42):
    """
    Given a list of T0 strings, produce T1, T2, T3 iteratively.

    Returns
    -------
    dict  {"T0": [...], "T1": [...], "T2": [...], "T3": [...]}
    """
    gens = {"T0": list(texts)}
    current = list(texts)
    for g in range(1, n_generations + 1):
        rate = _GENERATION_RATES[g]
        paraphrased = [
            paraphrase_text(t, swap_rate=rate, seed=seed + i)
            for i, t in enumerate(current)
        ]
        gens[f"T{g}"] = paraphrased
        current = paraphrased
        print(f"[paraphraser] Generated T{g} ({rate*100:.0f}% swap rate) for {len(current)} docs")
    return gens
