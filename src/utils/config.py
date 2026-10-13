"""
config.py
Project-wide paths and constants. Every module imports locations from here.
"""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ── Data ─────────────────────────────────────────────────────────
RAW_DIR = os.path.join(ROOT, "data", "raw")              # data/raw/<dataset>/<dataset>_{paraphrased,train}.csv
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
METADATA_DIR = os.path.join(ROOT, "data", "metadata")
GENERATION_DIRS = {                                       # one Parquet file per dataset in each
    "T0": os.path.join(PROCESSED_DIR, "t0_human"),
    "T1": os.path.join(PROCESSED_DIR, "t1_paraphrased"),
    "T2": os.path.join(PROCESSED_DIR, "t2_paraphrased"),
    "T3": os.path.join(PROCESSED_DIR, "t3_paraphrased"),
}
TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
SPLITS_PATH = os.path.join(METADATA_DIR, "dataset_splits.json")
PARAPHRASER_LABELS_PATH = os.path.join(METADATA_DIR, "paraphraser_labels.json")

# ── Outputs ──────────────────────────────────────────────────────
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")
BASELINE_DIR = os.path.join(EXPERIMENTS_DIR, "baseline_results")
STYLOMETRY_DIR = os.path.join(EXPERIMENTS_DIR, "stylometry_results")
RQ1_DIR = os.path.join(STYLOMETRY_DIR, "rq1")                # feature-decay analysis (RQ1)
ATTRIBUTION_DIR = os.path.join(EXPERIMENTS_DIR, "attribution_results")
LOG_DIR = os.path.join(EXPERIMENTS_DIR, "logs")
FIGURES_DIR = os.path.join(ROOT, "figures")
DECAY_FIG_DIR = os.path.join(FIGURES_DIR, "decay_curves")
TSNE_FIG_DIR = os.path.join(FIGURES_DIR, "tsne_plots")
ATTRIBUTION_FIG_DIR = os.path.join(FIGURES_DIR, "attribution_charts")

# ── Corpus ───────────────────────────────────────────────────────
# Dataset name -> domain label used throughout the pipeline
DOMAINS = {
    "xsum": "news",
    "yelp": "review",
    "cmv": "opinion",
    "eli5": "qa",
    "sci_gen": "science",
    "tldr": "summary",
    "wp": "story",
}

SOURCES = ["Human", "OpenAI", "LLAMA", "PaLM", "Tsinghua", "BigScience", "Eleuther-AI"]

PARAPHRASERS = ["chatgpt", "palm", "dipper", "dipper(low)", "dipper(high)",
                "pegasus(full)", "pegasus(slight)"]

# Released chains are largely unrelated to their source (see preprocessing sanity check)
EXCLUDED_PARAPHRASERS = ["dipper(high)"]

GENERATIONS = ["T0", "T1", "T2", "T3"]

# ── Experiment defaults ──────────────────────────────────────────
SEED = 42
MIN_WORDS = 30
VAL_RATIO = 0.15
N_PER_DOMAIN = 100          # Human documents per domain in the baseline analysis
N_BASELINE_PARTNERS = 5     # random same-domain documents per T0 for the "unrelated" baseline
N_BOOTSTRAP = 1000
