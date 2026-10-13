"""
load_data.py
Read the raw Ship of Theseus CSVs (Tripto et al., 2024) and the processed Parquet files.

Raw layout:   data/raw/<dataset>/<dataset>_paraphrased.csv  [source, key, text, version_name]
              data/raw/<dataset>/<dataset>_train.csv        [source, key, text]
version_name is "original" (T0) or the paraphraser name repeated once per generation,
e.g. "chatgpt" (T1), "chatgpt_chatgpt" (T2), "chatgpt_chatgpt_chatgpt" (T3).

Processed layout (written by preprocess.py):
              data/processed/t0_human/<dataset>.parquet         [key, source, domain, text]
              data/processed/t{1,2,3}_paraphrased/<dataset>.parquet
                                                        [key, source, domain, paraphraser, text]
              data/processed/train/<dataset>.parquet    [key, source, domain, text, split]
"""

import json
import os

import pandas as pd

from src.utils.config import (RAW_DIR, GENERATION_DIRS, TRAIN_DIR, SPLITS_PATH, DOMAINS,
                              GENERATIONS)


# ── Raw CSVs ─────────────────────────────────────────────────────

def raw_path(dataset, kind="paraphrased"):
    """Path to data/raw/<dataset>/<dataset>_<kind>.csv (kind: paraphrased | train)."""
    path = os.path.join(RAW_DIR, dataset, f"{dataset}_{kind}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - see data/raw/README.md")
    return path


def load_raw(dataset):
    """Load one paraphrased CSV as-is (all sources and versions)."""
    return pd.read_csv(raw_path(dataset, "paraphrased"))


def load_raw_train(dataset):
    """Load one un-paraphrased train CSV as-is: columns [source, key, text]."""
    return pd.read_csv(raw_path(dataset, "train"))


# ── Processed Parquet files ──────────────────────────────────────

def _read_generation(generation, datasets):
    path = GENERATION_DIRS[generation]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} not found - run `python -m src.data.preprocess` first")
    return pd.concat([pd.read_parquet(os.path.join(path, f"{d}.parquet")) for d in datasets],
                     ignore_index=True)


def load_chains(datasets=tuple(DOMAINS), source=None, analysis_sample=False):
    """
    Return one row per document: [key, source, domain, T0, <paraphraser>_T1, ..._T3].

    analysis_sample=True restricts to the fixed baseline sample recorded in
    data/metadata/dataset_splits.json (100 Human documents per domain).
    """
    t0 = _read_generation("T0", datasets).rename(columns={"text": "T0"})
    if analysis_sample:
        with open(SPLITS_PATH) as f:
            sample = json.load(f)["analysis_sample"]
        source = sample["source"]
        keys = [(d, k) for d in datasets for k in sample["keys"].get(d, [])]
        t0 = t0[t0["source"] == source].set_index(["domain", "key"])
        t0 = t0.loc[[(DOMAINS[d], k) for d, k in keys]].reset_index()
    elif source:
        t0 = t0[t0["source"] == source]

    wide = t0[["key", "source", "domain", "T0"]]
    for g in GENERATIONS[1:]:
        gen = _read_generation(g, datasets)
        gen = gen.pivot(index=["domain", "source", "key"], columns="paraphraser", values="text")
        gen.columns = [f"{p}_{g}" for p in gen.columns]
        wide = wide.merge(gen.reset_index(), on=["domain", "source", "key"], how="left")
    return wide.reset_index(drop=True)


def load_train(datasets=tuple(DOMAINS)):
    """Return the attribution training texts: [key, source, domain, text, split]."""
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"{TRAIN_DIR} not found - run `python -m src.data.preprocess` first")
    return pd.concat([pd.read_parquet(os.path.join(TRAIN_DIR, f"{d}.parquet")) for d in datasets],
                     ignore_index=True)


def get_generations(chains, paraphraser):
    """Return {"T0": [...], "T1": [...], "T2": [...], "T3": [...]} for one paraphraser."""
    gens = {"T0": chains["T0"].tolist()}
    for g in GENERATIONS[1:]:
        gens[g] = chains[f"{paraphraser}_{g}"].tolist()
    return gens
