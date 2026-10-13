"""
split_data.py
Leakage-free splits and the fixed analysis sample, recorded in data/metadata/dataset_splits.json.

  * Train files: split train/val by key. Human and LLM texts generated from the same prompt
    share a key, so a row-level split would put near-duplicates on both sides.
  * Paraphrased files: the RQ2 test set. They share no keys with the train files (checked).
  * Analysis sample: n Human documents per domain for the baseline metrics, sampled from
    sorted keys with a fixed seed so every run analyses the same documents.
"""

import json
import logging
import os
import random

from src.utils.config import SPLITS_PATH, DOMAINS, SEED, VAL_RATIO, N_PER_DOMAIN

logger = logging.getLogger("preprocess")


def split_train_by_key(df, val_ratio=VAL_RATIO, seed=SEED):
    """Add a `split` column (train | val) assigned per key."""
    keys = sorted(df["key"].unique())
    random.seed(seed)
    val_keys = set(random.sample(keys, round(val_ratio * len(keys))))
    return df.assign(split=df["key"].map(lambda k: "val" if k in val_keys else "train"))


def check_leakage(chains, train):
    """Return the (domain, key) pairs present in both paraphrased and train data."""
    shared = set(zip(chains.domain, chains.key)) & set(zip(train.domain, train.key))
    if shared:
        logger.warning(f"{len(shared)} keys appear in both paraphrased and train data")
    else:
        logger.info("Leakage check: paraphrased and train data share no keys")
    return shared


def analysis_sample(chains, n_per_domain=N_PER_DOMAIN, source="Human", seed=SEED):
    """Return {dataset: [keys]}: n documents of one source per domain."""
    sample = {}
    for dataset, domain in DOMAINS.items():
        keys = sorted(chains[(chains.domain == domain) & (chains.source == source)]["key"])
        if not keys:
            continue
        random.seed(seed)
        sample[dataset] = random.sample(keys, min(n_per_domain, len(keys)))
    return sample


def write_splits(train, sample, n_per_domain=N_PER_DOMAIN, source="Human", seed=SEED):
    """Save train/val keys per dataset and the analysis sample to dataset_splits.json."""
    domain_to_dataset = {v: k for k, v in DOMAINS.items()}
    splits = {
        "description": {
            "train_files": "data/raw/*/<dataset>_train.csv, split train/val by key (RQ2 training)",
            "paraphrased_files": "data/raw/*/<dataset>_paraphrased.csv, all keys are test (RQ2)",
            "analysis_sample": "documents used for the baseline similarity/stylometry analysis",
        },
        "seed": seed,
        "train_files": {},
        "analysis_sample": {"source": source, "n_per_domain": n_per_domain, "keys": sample},
    }
    for domain, part in train.groupby("domain"):
        by_split = part.drop_duplicates("key").groupby("split")["key"].apply(sorted)
        splits["train_files"][domain_to_dataset[domain]] = {s: list(k) for s, k in by_split.items()}
    os.makedirs(os.path.dirname(SPLITS_PATH), exist_ok=True)
    with open(SPLITS_PATH, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved splits and analysis sample to data/metadata/{os.path.basename(SPLITS_PATH)}")
