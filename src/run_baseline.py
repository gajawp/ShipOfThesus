"""
run_baseline.py
Week 1–4 baseline: load the preprocessed T0–T3 chains → similarity metrics →
stylometric features → figures, one output folder per paraphraser.

Outputs:
  experiments/baseline_results/<paraphraser>/      similarity metrics (overall + per domain)
  experiments/baseline_results/paraphraser_comparison.csv, domain_comparison.csv
  experiments/stylometry_results/<paraphraser>/    stylometric features
  experiments/stylometry_results/feature_comparison.csv
  figures/decay_curves/<paraphraser>/              4 figures per paraphraser
  figures/decay_curves/paraphraser_trajectories.png

Usage (from the repository root, after `python -m src.data.preprocess`):
    python -m src.run_baseline                          # ChatGPT chains, 700 docs
    python -m src.run_baseline --paraphraser dipper     # any paraphraser, or "wordnet"
    python -m src.run_baseline --all                    # every paraphraser + comparison tables
    python -m src.run_baseline --small                  # quick run, writes experiments/small_run/
"""

import argparse
import json
import os

import pandas as pd

from src.data.load_data import load_chains, get_generations
from src.data.wordnet_baseline import generate_generations
from src.features.stylometric_features import compute_all_features
from src.utils.config import (BASELINE_DIR, STYLOMETRY_DIR, DECAY_FIG_DIR, EXPERIMENTS_DIR,
                              PARAPHRASERS, GENERATIONS, SEED)
from src.utils.metrics import compute_all_metrics
from src.visualization.decay_plots import generate_all_figures
from src.visualization.trajectory_plots import plot_paraphraser_trajectories

FULL_DIRS = {"metrics": BASELINE_DIR, "features": STYLOMETRY_DIR, "figures": DECAY_FIG_DIR}
# Test runs go to experiments/small_run/ so they never overwrite full results
_SMALL = os.path.join(EXPERIMENTS_DIR, "small_run")
SMALL_DIRS = {"metrics": os.path.join(_SMALL, "baseline_results"),
              "features": os.path.join(_SMALL, "stylometry_results"),
              "figures": os.path.join(_SMALL, "figures")}


def run_paraphraser(corpus_df, paraphraser, use_bertscore=True, dirs=FULL_DIRS):
    metrics_dir = os.path.join(dirs["metrics"], paraphraser)
    features_dir = os.path.join(dirs["features"], paraphraser)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    domains = corpus_df["domain"].tolist()

    print("\n" + "=" * 60)
    print(f"Paraphraser: {paraphraser}")
    print("=" * 60)

    # ── 1. T0–T3 generations ─────────────────────────────────────
    if paraphraser == "wordnet":
        print("\n[1/4] Simulating WordNet paraphrase generations (T0 -> T3)...")
        gens = generate_generations(corpus_df["T0"].tolist(), n_generations=3, seed=SEED)
    else:
        print("\n[1/4] Loading Ship of Theseus generations (T0 -> T3)...")
        gens = get_generations(corpus_df, paraphraser)

    gen_df = pd.DataFrame({"key": corpus_df["key"], "domain": domains,
                           **{g: gens[g] for g in GENERATIONS}})
    gen_df.to_json(os.path.join(metrics_dir, "all_generations.json"), orient="records", indent=2)
    print(f"  Saved generation data for {len(gen_df)} documents")

    # ── 2. Similarity metrics ─────────────────────────────────────
    print("\n[2/4] Computing similarity metrics (BLEU, ROUGE-L, BERTScore, SBERT)...")
    metrics_df = compute_all_metrics(gens, use_bertscore=use_bertscore)
    metrics_df.to_csv(os.path.join(metrics_dir, "semantic_metrics.csv"), index=False)
    print("\nSemantic Metrics Table:")
    print(metrics_df.to_string(index=False))

    # Per-domain metrics
    domain_rows = []
    for dom in sorted(set(domains)):
        dom_idx = [i for i, d in enumerate(domains) if d == dom]
        dom_gens = {g: [gens[g][i] for i in dom_idx] for g in GENERATIONS}
        dom_metrics = compute_all_metrics(dom_gens, use_bertscore=use_bertscore)
        dom_metrics.to_csv(os.path.join(metrics_dir, f"semantic_metrics_{dom}.csv"), index=False)
        domain_rows.append(dom_metrics.assign(domain=dom))
        print(f"\n  [{dom.upper()} domain metrics]")
        print(dom_metrics.to_string(index=False))

    # ── 3. Stylometric features ───────────────────────────────────
    print("\n[3/4] Computing lexical/stylometric features...")
    features_df = compute_all_features(gens)
    features_df.to_csv(os.path.join(features_dir, "stylometric_features.csv"), index=False)
    print("\nStylometric Features Table:")
    print(features_df.to_string(index=False))

    # ── 4. Figures ───────────────────────────────────────────────
    print("\n[4/4] Generating figures...")
    generate_all_figures(metrics_df, features_df,
                         fig_dir=os.path.join(dirs["figures"], paraphraser))

    # Summary JSON for the paper
    summary = {
        "paraphraser": paraphraser,
        "n_documents": len(domains),
        "domains": {d: domains.count(d) for d in sorted(set(domains))},
        "metrics": metrics_df.to_dict(orient="records"),
        "features": features_df.to_dict(orient="records"),
    }
    with open(os.path.join(metrics_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return metrics_df, features_df, pd.concat(domain_rows)


def _first(df, *cols):
    return df[list(cols) + [c for c in df if c not in cols]]


def run_pipeline(paraphrasers, small=False, use_bertscore=True):
    print("=" * 60)
    print("NLP Project: Style Decay & Semantic Retention Pipeline")
    print("=" * 60)

    print("\nLoading the baseline analysis sample (data/metadata/dataset_splits.json)...")
    corpus_df = load_chains(analysis_sample=True)
    dirs = FULL_DIRS
    if small:
        corpus_df = corpus_df.groupby("domain", sort=False).head(20).reset_index(drop=True)
        dirs = SMALL_DIRS
    print(f"  {len(corpus_df)} documents: {corpus_df.domain.value_counts().to_dict()}")

    all_metrics, all_features, all_domains = [], [], []
    for p in paraphrasers:
        metrics_df, features_df, domain_df = run_paraphraser(corpus_df, p, use_bertscore, dirs)
        all_metrics.append(metrics_df.assign(paraphraser=p))
        all_features.append(features_df.assign(paraphraser=p))
        all_domains.append(domain_df.assign(paraphraser=p))

    if len(paraphrasers) > 1:
        metrics = _first(pd.concat(all_metrics), "paraphraser")
        metrics.to_csv(os.path.join(dirs["metrics"], "paraphraser_comparison.csv"), index=False)
        domains = _first(pd.concat(all_domains), "paraphraser", "domain")
        domains.to_csv(os.path.join(dirs["metrics"], "domain_comparison.csv"), index=False)
        features = _first(pd.concat(all_features), "paraphraser")
        features.to_csv(os.path.join(dirs["features"], "feature_comparison.csv"), index=False)
        plot_paraphraser_trajectories(
            metrics, os.path.join(dirs["figures"], "paraphraser_trajectories.png"))
        print(f"\nComparison tables saved to {os.path.relpath(dirs['metrics'])}/ "
              f"and {os.path.relpath(dirs['features'])}/")

    print("\n" + "=" * 60)
    print(f"Pipeline complete. Results in {os.path.relpath(dirs['metrics'])}/, "
          f"{os.path.relpath(dirs['features'])}/, {os.path.relpath(dirs['figures'])}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paraphraser", default="chatgpt",
                        choices=PARAPHRASERS + ["wordnet"],
                        help="Which paraphrase chain to analyse (wordnet = rule-based baseline)")
    parser.add_argument("--all", action="store_true",
                        help="Run every Ship of Theseus paraphraser plus the wordnet baseline")
    parser.add_argument("--small", action="store_true",
                        help="Quick run with the first 20 docs per domain (for testing)")
    parser.add_argument("--no-bertscore", action="store_true",
                        help="Skip BERTScore computation (faster)")
    args = parser.parse_args()

    paraphrasers = PARAPHRASERS + ["wordnet"] if args.all else [args.paraphraser]
    run_pipeline(paraphrasers, small=args.small, use_bertscore=not args.no_bertscore)
