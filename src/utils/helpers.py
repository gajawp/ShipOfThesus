"""
helpers.py
Small shared utilities: logging setup, version names, NLTK resources.
"""

import logging
import os

import nltk

from src.utils.config import LOG_DIR


def setup_logging(name, level=logging.INFO, logfile=None):
    """Log `level` and up to the console and DEBUG to experiments/logs/<logfile>."""
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    handlers = [console]
    if logfile:
        fh = logging.FileHandler(os.path.join(LOG_DIR, logfile), mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        handlers.append(fh)
    logger = logging.getLogger(name)
    logger.handlers = handlers
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def version_name(paraphraser, generation):
    """Map (paraphraser, 'T2') -> 'chatgpt_chatgpt' as named in the raw CSVs; T0 -> 'original'."""
    g = int(generation[1:])
    return "original" if g == 0 else "_".join([paraphraser] * g)


_NLTK_PATHS = {"punkt_tab": "tokenizers/punkt_tab", "stopwords": "corpora/stopwords",
               "wordnet": "corpora/wordnet",
               "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng"}


def ensure_nltk(*resources):
    """Download NLTK resources on first use."""
    for r in resources:
        try:
            nltk.data.find(_NLTK_PATHS[r])
        except LookupError:
            nltk.download(r, quiet=True)
