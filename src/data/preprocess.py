"""
preprocess.py
Preprocess the Ship of Theseus corpus: all 7 datasets, all 7 text sources.

Inputs: data/raw/<dataset>/<dataset>_paraphrased.csv and <dataset>_train.csv
Outputs:
  data/processed/t0_human/<dataset>.parquet          original texts (T0), all sources
  data/processed/t{1,2,3}_paraphrased/<dataset>.parquet  paraphrases, one row per paraphraser
  data/processed/train/<dataset>.parquet             un-paraphrased texts for RQ2 training
  data/processed/chain_stats.csv                     mean length per domain/source/paraphraser
  data/metadata/dataset_splits.json                  train/val keys + baseline analysis sample
  data/metadata/paraphraser_labels.json              paraphraser and source label ids
  experiments/logs/preprocess.log                    full DEBUG log

Steps (each one is logged with before/after counts):
  1. Load raw CSVs              4. Filter short texts
  2. Clean text                 5. Sanity-check the chains
  3. Align T0–T3 chains         6. Splits, leakage check, statistics, save

Usage (from the repository root):
    python -m src.data.preprocess
    python -m src.data.preprocess --datasets xsum yelp
    python -m src.data.preprocess --show xsum-3981 --paraphraser chatgpt --source Human
"""

import argparse
import html
import json
import logging
import os
import re
import unicodedata

import pandas as pd

from src.data.load_data import load_raw, load_raw_train
from src.data.split_data import split_train_by_key, check_leakage, analysis_sample, write_splits
from src.utils.config import (PROCESSED_DIR, GENERATION_DIRS, TRAIN_DIR,
                              PARAPHRASER_LABELS_PATH, DOMAINS, SOURCES, PARAPHRASERS,
                              EXCLUDED_PARAPHRASERS, GENERATIONS, SEED, MIN_WORDS, N_PER_DOMAIN)
from src.utils.helpers import setup_logging, version_name

logger = logging.getLogger("preprocess")


# ── 2. Text cleaning ─────────────────────────────────────────────

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-", " ": " "})


def clean_text(text):
    """Normalise unicode, quotes and whitespace. Content and punctuation are kept."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTES)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_texts(df, tag):
    """Clean the text column, log what was fixed, and drop empty / duplicate rows."""
    t = df["text"].fillna("").astype(str)
    issues = {
        "leading/trailing whitespace": t.str.match(r"^\s|.*\s$"),
        "newlines": t.str.contains("\n"),
        "repeated spaces": t.str.contains("  "),
        "curly quotes / dashes": t.str.contains("[‘’“”–—]"),
        "HTML entities": t.str.contains(r"&[a-z]+;|&#\d+;"),
    }
    for name, mask in issues.items():
        logger.debug(f"  [{tag}] {name}: {int(mask.sum()):,} rows")
    df = df.assign(text=t.map(clean_text))
    logger.info(f"  [{tag}] cleaned text: {int((df['text'] != t).sum()):,}/{len(df):,} rows modified")

    empty = df["text"] == ""
    if empty.any():
        logger.warning(f"  [{tag}] dropping {int(empty.sum())} empty texts")
        df = df[~empty]
    id_cols = [c for c in ("source", "key", "version_name") if c in df.columns]
    dup = df.duplicated(subset=id_cols)
    if dup.any():
        logger.warning(f"  [{tag}] dropping {int(dup.sum())} duplicate {tuple(id_cols)} rows")
        df = df[~dup]
    return df


def _log_by_source(tag, label, counts):
    logger.info(f"  [{tag}] {label}: " + ", ".join(f"{s} {n}" for s, n in counts.items()))


# ── Paraphrased files → T0–T3 chains ─────────────────────────────

def preprocess_chains(dataset, min_words=MIN_WORDS):
    """Return one row per (source, key): [key, source, domain, T0, <p>_T1, <p>_T2, <p>_T3 ...]."""
    tag = f"{dataset}/paraphrased"
    logger.info(f"── {dataset} ({DOMAINS[dataset]}) - paraphrased " + "─" * 28)

    # 1. Load
    raw = load_raw(dataset)
    logger.info(f"  [{tag}] loaded {len(raw):,} rows, {raw['source'].nunique()} sources, "
                f"{raw['version_name'].nunique()} versions")

    # 2. Clean
    raw = clean_texts(raw, tag)

    # 3. Align: one row per (source, key), one column per version
    wide = raw.pivot(index=["source", "key"], columns="version_name", values="text")
    needed = ["original"] + [version_name(p, g) for p in PARAPHRASERS for g in GENERATIONS[1:]]
    missing = [c for c in needed if c not in wide.columns]
    if missing:
        raise ValueError(f"{dataset}: missing versions {missing}")
    for p in PARAPHRASERS:
        cols = ["original"] + [version_name(p, g) for g in GENERATIONS[1:]]
        logger.debug(f"  [{tag}] complete chains for {p}: "
                     f"{int(wide[cols].notna().all(axis=1).sum())}/{len(wide)}")
    n_before = len(wide)
    wide = wide.dropna(subset=needed)
    logger.info(f"  [{tag}] complete T0-T3 chains for all {len(PARAPHRASERS)} paraphrasers: "
                f"{len(wide):,}/{n_before:,} documents")

    # 4. Length filter on T0
    n_before = len(wide)
    wide = wide[wide["original"].str.split().str.len() >= min_words]
    logger.info(f"  [{tag}] T0 >= {min_words} words: {len(wide):,}/{n_before:,} documents")

    wide = wide.reset_index()
    out = pd.DataFrame({"key": wide["key"], "source": wide["source"],
                        "domain": DOMAINS[dataset], "T0": wide["original"]})
    for p in PARAPHRASERS:
        for g in GENERATIONS[1:]:
            out[f"{p}_{g}"] = wide[version_name(p, g)]
    _log_by_source(tag, "documents per source", out["source"].value_counts())

    # 5. Sanity checks
    sanity_check(out, tag)
    return out


def _overlap(a, b):
    """Jaccard overlap of lower-cased word sets."""
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / max(len(a | b), 1)


def sanity_check(df, tag, min_overlap=0.1):
    """Flag no-op paraphrases and paraphrases that share almost no words with T0."""
    for p in PARAPHRASERS:
        chain = ["T0"] + [f"{p}_{g}" for g in GENERATIONS[1:]]
        unchanged = sum(int((df[a] == df[b]).sum()) for a, b in zip(chain, chain[1:]))
        unrelated = df[f"{p}_T1"].combine(df["T0"], _overlap) < min_overlap
        logger.debug(f"  [{tag}] {p}: {unchanged} unchanged steps, "
                     f"{int(unrelated.sum())} T1 texts with <{min_overlap:.0%} word overlap")
        if unrelated.mean() > 0.5:
            logger.warning(f"  [{tag}] {p}: {unrelated.mean():.0%} of T1 paraphrases share "
                           f"<{min_overlap:.0%} of their words with T0 - likely unrelated text")


# ── Train files → attribution training data ──────────────────────

def preprocess_train(dataset, min_words=MIN_WORDS, seed=SEED):
    """Return [key, source, domain, text, split] with a key-level train/val split."""
    tag = f"{dataset}/train"
    logger.info(f"── {dataset} ({DOMAINS[dataset]}) - train " + "─" * 33)

    raw = load_raw_train(dataset)
    logger.info(f"  [{tag}] loaded {len(raw):,} rows, {raw['key'].nunique():,} keys, "
                f"{raw['source'].nunique()} sources")
    df = clean_texts(raw, tag)

    n_before = len(df)
    df = df[df["text"].str.split().str.len() >= min_words]
    logger.info(f"  [{tag}] text >= {min_words} words: {len(df):,}/{n_before:,} rows")

    df = split_train_by_key(df.assign(domain=DOMAINS[dataset]), seed=seed)
    n_val = df.loc[df.split == "val", "key"].nunique()
    logger.info(f"  [{tag}] split by key: {df['key'].nunique() - n_val} train keys / {n_val} val keys "
                f"({(df.split == 'train').sum():,} / {(df.split == 'val').sum():,} rows)")
    _log_by_source(tag, "rows per source", df["source"].value_counts())
    return df[["key", "source", "domain", "text", "split"]].reset_index(drop=True)


# ── Saving ───────────────────────────────────────────────────────

def save_generations(chains, dataset):
    """Write one Parquet file per generation: t0_human/ and t{1,2,3}_paraphrased/."""
    part = chains[chains["domain"] == DOMAINS[dataset]]
    ids = ["key", "source", "domain"]
    for g, path in GENERATION_DIRS.items():
        os.makedirs(path, exist_ok=True)
        if g == "T0":
            out = part[ids + ["T0"]].rename(columns={"T0": "text"})
        else:
            cols = {f"{p}_{g}": p for p in PARAPHRASERS}
            out = (part[ids + list(cols)].rename(columns=cols)
                   .melt(id_vars=ids, var_name="paraphraser", value_name="text"))
        out.to_parquet(os.path.join(path, f"{dataset}.parquet"), index=False)


def write_paraphraser_labels():
    """Label ids for RQ3 (paraphraser) and RQ2 (source) classifiers."""
    models = {"chatgpt": ("ChatGPT", "default"), "palm": ("PaLM2", "default"),
              "dipper": ("DIPPER", "default"), "dipper(low)": ("DIPPER", "low diversity"),
              "dipper(high)": ("DIPPER", "high diversity"),
              "pegasus(full)": ("Pegasus", "all sentences"),
              "pegasus(slight)": ("Pegasus", "25% of sentences")}
    labels = {
        "paraphrasers": {
            p: {"label_id": i, "model": models[p][0], "setting": models[p][1],
                "version_names": {g: version_name(p, g) for g in GENERATIONS[1:]},
                "excluded": p in EXCLUDED_PARAPHRASERS}
            for i, p in enumerate(PARAPHRASERS)},
        "sources": {s: i for i, s in enumerate(SOURCES)},
        "notes": {"dipper(high)": "excluded: released chains are largely unrelated to their "
                                  "source text (sanity check in preprocess.py)"},
    }
    os.makedirs(os.path.dirname(PARAPHRASER_LABELS_PATH), exist_ok=True)
    with open(PARAPHRASER_LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)


def chain_stats(chains):
    """Mean words per document for every domain, source, paraphraser and generation."""
    rows = []
    for (dom, src), part in chains.groupby(["domain", "source"], sort=False):
        t0_words = part["T0"].str.split().str.len().mean()
        for p in PARAPHRASERS:
            row = {"domain": dom, "source": src, "paraphraser": p, "docs": len(part),
                   "T0_words": t0_words}
            for g in GENERATIONS[1:]:
                row[f"{g}_words"] = part[f"{p}_{g}"].str.split().str.len().mean()
            rows.append(row)
    return pd.DataFrame(rows).round(1)


def log_summary(chains, train):
    """Log a domain x source table of document counts for both outputs."""
    for name, df in [("chains (paraphrased)", chains), ("train", train)]:
        table = pd.crosstab(df["domain"], df["source"], margins=True, margins_name="total")
        logger.info(f"Documents per domain x source - {name}:\n{table.to_string()}")
    t0 = chains[chains.source == "Human"].groupby("domain")["T0"]
    words = t0.apply(lambda s: s.str.split().str.len().mean()).round(1)
    logger.info("Human T0 average length (words): " + ", ".join(f"{d} {w}" for d, w in words.items()))


def show_example(chains, key=None, paraphraser="chatgpt", source="Human", width=300):
    """Print the T0 -> T3 chain for one document."""
    rows = chains[chains["source"] == source]
    row = rows[rows["key"] == key].iloc[0] if key else rows.iloc[0]
    print("\n" + "=" * 70)
    print(f"Document {row['key']} ({row['domain']}, source={row['source']}) "
          f"- paraphraser: {paraphraser}")
    print("=" * 70)
    for g in GENERATIONS:
        text = row["T0"] if g == "T0" else row[f"{paraphraser}_{g}"]
        print(f"\n[{g}] ({len(text.split())} words)\n{text[:width]}{'...' if len(text) > width else ''}")
    print()


def run_preprocessing(datasets=tuple(DOMAINS), min_words=MIN_WORDS, seed=SEED,
                      n_per_domain=N_PER_DOMAIN):
    """Run all steps for both file types and save processed data and metadata."""
    logger.info("=" * 60)
    logger.info(f"Preprocessing Ship of Theseus: datasets={list(datasets)}, "
                f"min_words={min_words}, seed={seed}")
    logger.info("=" * 60)

    chains = pd.concat([preprocess_chains(d, min_words) for d in datasets], ignore_index=True)
    train = pd.concat([preprocess_train(d, min_words, seed) for d in datasets], ignore_index=True)

    # 6. Splits and leakage check
    check_leakage(chains, train)
    sample = analysis_sample(chains, n_per_domain, seed=seed)
    logger.info("Analysis sample: " + ", ".join(f"{d} {len(k)}" for d, k in sample.items())
                + " Human documents")

    logger.info("── Summary " + "─" * 50)
    log_summary(chains, train)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    for d in datasets:
        save_generations(chains, d)
        train[train["domain"] == DOMAINS[d]].to_parquet(os.path.join(TRAIN_DIR, f"{d}.parquet"),
                                                        index=False)
    chain_stats(chains).to_csv(os.path.join(PROCESSED_DIR, "chain_stats.csv"), index=False)
    write_splits(train, sample, n_per_domain, seed=seed)
    write_paraphraser_labels()
    logger.info(f"Saved {len(chains):,} chains to data/processed/t0_human/ and t1-t3_paraphrased/")
    logger.info(f"Saved {len(train):,} training texts to data/processed/train/")
    logger.info("Saved data/processed/chain_stats.csv, data/metadata/*.json; "
                "full log in experiments/logs/preprocess.log")
    return chains, train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DOMAINS), choices=list(DOMAINS),
                        help="Ship of Theseus datasets to preprocess (default: all 7)")
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-per-domain", type=int, default=N_PER_DOMAIN,
                        help="Human documents per domain in the baseline analysis sample")
    parser.add_argument("--show", nargs="?", const="", default=None, metavar="KEY",
                        help="Print the T0 -> T3 chain for a document (default: first one)")
    parser.add_argument("--paraphraser", default="chatgpt", choices=PARAPHRASERS)
    parser.add_argument("--source", default="Human", help="Source of the document to --show")
    parser.add_argument("--verbose", action="store_true", help="Show DEBUG logs on the console")
    args = parser.parse_args()

    setup_logging("preprocess", logging.DEBUG if args.verbose else logging.INFO,
                  logfile="preprocess.log")
    chains, _ = run_preprocessing(args.datasets, args.min_words, args.seed, args.n_per_domain)
    if args.show is not None:
        show_example(chains, args.show or None, args.paraphraser, args.source)
