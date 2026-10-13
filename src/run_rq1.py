"""
run_rq1.py
Weeks 7–8, RQ1: which "planks" are replaced first - style or content?

For every document in the baseline sample (700 Human docs) and every paraphraser, each
feature family (src/features/profiles.py) is compared between T_k and the document's own T0.
To put style and meaning on one scale, each distance is divided by the distance between the
T0 and *unrelated* T0s of the same domain (5 random partners):

    normalised distance  nd = d(T_k, T0) / d(random same-domain T0, T0)
    retention            R  = 1 - mean(nd)     1 = unchanged, 0 = as different as another author

Outputs (experiments/stylometry_results/rq1/):
  text_profiles.parquet, sbert_embeddings.npy   caches (one row per text)
  normalised_distances.parquet                   nd per document x paraphraser x generation x family
  retention.csv, retention_by_domain.csv         R with 95% bootstrap CIs
  decay_ranking.csv                              families ordered by T1 retention, share of loss at T1
  style_vs_meaning_tests.csv                     paired Wilcoxon: style nd > meaning nd
  progression_tests.csv                          paired Wilcoxon: nd(T2) > nd(T1), nd(T3) > nd(T2)
  feature_shifts.csv                             direction of change T0 -> T3 per scalar feature
Figures: figures/decay_curves/<paraphraser>/linguistic_delta.png,
         figures/decay_curves/retention_heatmap.png

Usage (from the repository root, after preprocessing):
    python -m src.run_rq1              # ~10 min the first time (parsing), then cached
    python -m src.run_rq1 --refresh    # recompute the caches
"""

import argparse
import hashlib
import json
import logging
import os
import random

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

from src.data.load_data import load_chains, get_generations
from src.data.wordnet_baseline import generate_generations
from src.features.profiles import FAMILIES, STYLE_FAMILIES, text_profiles, family_distance
from src.similarity.sbert_similarity import get_sbert
from src.utils.config import (RQ1_DIR, BASELINE_DIR, DECAY_FIG_DIR, PARAPHRASERS,
                              EXCLUDED_PARAPHRASERS, GENERATIONS, SEED, N_BASELINE_PARTNERS,
                              N_BOOTSTRAP)
from src.utils.helpers import setup_logging
from src.visualization.decay_plots import plot_linguistic_delta, plot_retention_heatmap

logger = logging.getLogger("rq1")

ALL_PARAPHRASERS = PARAPHRASERS + ["wordnet"]
SCALAR_FEATURES = ["n_words", "mattr", "avg_word_length", "mean_sentence_length",
                   "sentence_length_std", "mean_tree_depth", "max_tree_depth",
                   "mean_dep_distance", "fk_grade", "pos_NOUN", "pos_PROPN", "pos_VERB",
                   "pos_ADJ", "pos_ADV", "pos_PRON", "pos_DET", "pos_ADP"]


# ── 1. Texts ─────────────────────────────────────────────────────

def wordnet_generations(chains):
    """Reuse the baseline's WordNet chains when they cover the same documents."""
    path = os.path.join(BASELINE_DIR, "wordnet", "all_generations.json")
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        if [r["key"] for r in saved] == chains["key"].tolist():
            return {g: [r[g] for r in saved] for g in GENERATIONS}
    logger.info("  generating WordNet chains (not found in baseline results)")
    return generate_generations(chains["T0"].tolist(), n_generations=3, seed=SEED)


def text_table(chains):
    """One row per text: T0 once per document, then T1–T3 for every paraphraser."""
    base = chains[["key", "domain"]]
    frames = [base.assign(paraphraser="original", generation="T0", text=chains["T0"].values)]
    for p in ALL_PARAPHRASERS:
        gens = wordnet_generations(chains) if p == "wordnet" else get_generations(chains, p)
        for g in GENERATIONS[1:]:
            frames.append(base.assign(paraphraser=p, generation=g, text=gens[g]))
    return pd.concat(frames, ignore_index=True)


def _fingerprint(texts):
    return hashlib.sha1("\x00".join(texts).encode()).hexdigest()


# ── 2. Cached profiles and embeddings ────────────────────────────

def load_or_compute(texts_df, refresh=False):
    os.makedirs(RQ1_DIR, exist_ok=True)
    prof_path = os.path.join(RQ1_DIR, "text_profiles.parquet")
    emb_path = os.path.join(RQ1_DIR, "sbert_embeddings.npy")
    meta_path = os.path.join(RQ1_DIR, "cache_fingerprint.txt")
    fp = _fingerprint(texts_df["text"].tolist())
    cached = (not refresh and all(os.path.exists(p) for p in (prof_path, emb_path, meta_path))
              and open(meta_path).read() == fp)
    if cached:
        logger.info("  using cached profiles and embeddings")
        return pd.read_parquet(prof_path), np.load(emb_path)

    logger.info(f"  parsing {len(texts_df):,} texts with spaCy (takes a few minutes)...")
    profiles = text_profiles(texts_df["text"].tolist(), logger=logger)
    logger.info("  encoding texts with SBERT...")
    emb = get_sbert().encode(texts_df["text"].tolist(), batch_size=64, show_progress_bar=False)
    profiles.to_parquet(prof_path, index=False)
    np.save(emb_path, emb)
    with open(meta_path, "w") as f:
        f.write(fp)
    return profiles, emb


# ── 3. Normalised distances ──────────────────────────────────────

def baseline_partners(t0_index, domains, seed=SEED, k=N_BASELINE_PARTNERS):
    """For each T0 row, k random other T0 rows from the same domain."""
    rng = random.Random(seed)
    by_domain = {}
    for i, d in zip(t0_index, domains):
        by_domain.setdefault(d, []).append(i)
    pairs = []
    for i, d in zip(t0_index, domains):
        others = [j for j in by_domain[d] if j != i]
        pairs += [(i, j) for j in rng.sample(others, min(k, len(others)))]
    return pairs


def normalised_distances(texts_df, profiles, emb):
    t0 = texts_df[texts_df.generation == "T0"]
    t0_row = dict(zip(zip(t0.domain, t0.key), t0.index))
    scale = {c: profiles.loc[t0.index, c].std() or 1.0
             for c in profiles.columns if not c.startswith(("fw_", "punct_", "pos_", "dep_"))}

    def dist(family, ia, ib):
        return family_distance(family, profiles.loc[ia].reset_index(drop=True),
                               profiles.loc[ib].reset_index(drop=True),
                               texts_df.loc[ia, "text"].tolist(), texts_df.loc[ib, "text"].tolist(),
                               emb[ia], emb[ib], scale)

    # Unrelated-document baseline, per domain and family
    pairs = baseline_partners(t0.index.tolist(), t0.domain.tolist())
    ia, ib = [a for a, _ in pairs], [b for _, b in pairs]
    base = pd.DataFrame({"domain": texts_df.loc[ia, "domain"].values})
    for f in FAMILIES:
        base[f] = dist(f, ia, ib)
    baseline = base.groupby("domain").mean()
    logger.info("  unrelated-document baseline distance (mean over domains): "
                + ", ".join(f"{f} {baseline[f].mean():.3f}" for f in FAMILIES))

    para = texts_df[texts_df.generation != "T0"]
    ref = [t0_row[(d, k)] for d, k in zip(para.domain, para.key)]
    out = para[["key", "domain", "paraphraser", "generation"]].reset_index(drop=True)
    for f in FAMILIES:
        raw = dist(f, para.index.tolist(), ref)
        out[f] = raw / baseline.loc[out["domain"], f].values
    long = out.melt(id_vars=["key", "domain", "paraphraser", "generation"],
                    var_name="family", value_name="nd")
    return long, baseline.reset_index()


# ── 4. Retention, ranking, tests ─────────────────────────────────

def _bootstrap_ci(values, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def retention_table(nd, by=("paraphraser", "family", "generation")):
    rows = []
    for keys, part in nd.groupby(list(by), sort=False):
        v = part["nd"].to_numpy()
        lo, hi = _bootstrap_ci(v)
        rows.append({**dict(zip(by, keys)), "retention": 1 - v.mean(),
                     "ci_low": 1 - hi, "ci_high": 1 - lo, "n": len(v)})
    return pd.DataFrame(rows)


def decay_ranking(ret):
    wide = ret.pivot_table(index=["paraphraser", "family"], columns="generation",
                           values="retention").reset_index()
    wide["loss_T3"] = 1 - wide["T3"]
    wide["share_of_loss_at_T1"] = (1 - wide["T1"]) / wide["loss_T3"].where(wide["loss_T3"] > 0)
    wide["rank_T1"] = wide.groupby("paraphraser")["T1"].rank(method="first")
    return wide.sort_values(["paraphraser", "rank_T1"])


def _rank_biserial(diff):
    diff = diff[diff != 0]
    if not len(diff):
        return 0.0
    ranks = rankdata(np.abs(diff))
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())


def _holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def _paired_test(x, y, alternative):
    d = x - y
    p = wilcoxon(x, y, alternative=alternative).pvalue if np.any(d != 0) else 1.0
    return float(np.median(d)), _rank_biserial(d), float(p)


def style_vs_meaning_tests(nd):
    """Per paraphraser and generation: is the style family's nd larger than meaning's?"""
    wide = nd.pivot_table(index=["paraphraser", "generation", "domain", "key"],
                          columns="family", values="nd").reset_index()
    rows = []
    for (p, g), part in wide.groupby(["paraphraser", "generation"]):
        for f in STYLE_FAMILIES:
            med, r, pval = _paired_test(part[f].to_numpy(), part["Meaning (SBERT)"].to_numpy(),
                                        "greater")
            rows.append({"paraphraser": p, "generation": g, "family": f,
                         "median_nd_minus_meaning": med, "rank_biserial": r, "p": pval})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby(["paraphraser", "generation"])["p"].transform(_holm)
    return out


def progression_tests(nd):
    """Does each family keep changing after T1? nd(T2) > nd(T1) and nd(T3) > nd(T2)."""
    wide = nd.pivot_table(index=["paraphraser", "family", "domain", "key"],
                          columns="generation", values="nd").reset_index()
    rows = []
    for (p, f), part in wide.groupby(["paraphraser", "family"]):
        for a, b in [("T1", "T2"), ("T2", "T3")]:
            med, r, pval = _paired_test(part[b].to_numpy(), part[a].to_numpy(), "greater")
            rows.append({"paraphraser": p, "family": f, "step": f"{a}->{b}",
                         "median_increase": med, "rank_biserial": r, "p": pval})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("paraphraser")["p"].transform(_holm)
    return out


def feature_shifts(texts_df, profiles):
    """Direction of change T0 -> T3 for each scalar feature (paired over documents)."""
    df = pd.concat([texts_df[["key", "domain", "paraphraser", "generation"]],
                    profiles[SCALAR_FEATURES]], axis=1)
    t0 = df[df.generation == "T0"].set_index(["domain", "key"])
    rows = []
    for p in ALL_PARAPHRASERS:
        t3 = df[(df.paraphraser == p) & (df.generation == "T3")].set_index(["domain", "key"])
        base = t0.loc[t3.index]
        for c in SCALAR_FEATURES:
            x, y = base[c].to_numpy(float), t3[c].to_numpy(float)
            d = y - x
            _, r, pval = _paired_test(y, x, "two-sided")
            rows.append({"paraphraser": p, "feature": c, "mean_T0": x.mean(), "mean_T3": y.mean(),
                         "cohens_dz": d.mean() / d.std() if d.std() else 0.0,
                         "rank_biserial": r, "p": pval})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("paraphraser")["p"].transform(_holm)
    return out


# ── Main ─────────────────────────────────────────────────────────

def run(refresh=False):
    logger.info("=" * 60)
    logger.info("RQ1: feature-family decay vs. meaning decay")
    logger.info("=" * 60)
    chains = load_chains(analysis_sample=True)
    texts_df = text_table(chains)
    logger.info(f"[1/5] {len(chains)} documents, {len(texts_df):,} texts "
                f"(T0 + {len(ALL_PARAPHRASERS)} paraphrasers x T1-T3)")

    logger.info("[2/5] Feature profiles and SBERT embeddings")
    profiles, emb = load_or_compute(texts_df, refresh)

    logger.info("[3/5] Normalised distances to T0")
    nd, baseline = normalised_distances(texts_df, profiles, emb)
    nd.to_parquet(os.path.join(RQ1_DIR, "normalised_distances.parquet"), index=False)
    baseline.to_csv(os.path.join(RQ1_DIR, "unrelated_baseline.csv"), index=False)

    logger.info("[4/5] Retention, ranking and tests")
    ret = retention_table(nd)
    ret.to_csv(os.path.join(RQ1_DIR, "retention.csv"), index=False)
    retention_table(nd, by=("paraphraser", "domain", "family", "generation")).to_csv(
        os.path.join(RQ1_DIR, "retention_by_domain.csv"), index=False)
    ranking = decay_ranking(ret)
    ranking.to_csv(os.path.join(RQ1_DIR, "decay_ranking.csv"), index=False)
    svm = style_vs_meaning_tests(nd)
    svm.to_csv(os.path.join(RQ1_DIR, "style_vs_meaning_tests.csv"), index=False)
    progression_tests(nd).to_csv(os.path.join(RQ1_DIR, "progression_tests.csv"), index=False)
    feature_shifts(texts_df, profiles).to_csv(os.path.join(RQ1_DIR, "feature_shifts.csv"),
                                              index=False)

    for p in ["chatgpt", "palm", "dipper", "wordnet"]:
        r = ranking[ranking.paraphraser == p]
        logger.info(f"  {p}: replaced first -> " + ", ".join(
            f"{f} ({t1:.2f})" for f, t1 in zip(r.family.head(4), r.T1.head(4))))
    sig = svm[(svm.generation == "T1") & (svm.p_holm < 0.05)
              & ~svm.paraphraser.isin(EXCLUDED_PARAPHRASERS)]
    logger.info(f"  style family changes more than meaning at T1 (Holm p<0.05): "
                f"{len(sig)}/{len(STYLE_FAMILIES) * (len(ALL_PARAPHRASERS) - 1)} tests")

    logger.info("[5/5] Figures")
    for p in ALL_PARAPHRASERS:
        plot_linguistic_delta(ret[ret.paraphraser == p], p,
                              os.path.join(DECAY_FIG_DIR, p, "linguistic_delta.png"))
    plot_retention_heatmap(ret[~ret.paraphraser.isin(EXCLUDED_PARAPHRASERS)],
                           os.path.join(DECAY_FIG_DIR, "retention_heatmap.png"))
    logger.info(f"Done. Tables in {os.path.relpath(RQ1_DIR)}/, figures in "
                f"{os.path.relpath(DECAY_FIG_DIR)}/; log in experiments/logs/rq1.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Recompute cached profiles")
    args = parser.parse_args()
    setup_logging("rq1", logfile="rq1.log")
    run(args.refresh)
