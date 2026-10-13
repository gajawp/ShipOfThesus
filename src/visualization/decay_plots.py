"""
decay_plots.py
Per-paraphraser figures: semantic decay, feature evolution, retention heatmap,
semantic vs. stylistic retention. Saved to figures/decay_curves/<paraphraser>/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import DECAY_FIG_DIR

FIG_DIR = DECAY_FIG_DIR

PALETTE = {
    "bleu": "#E63946",
    "rouge_l": "#F4A261",
    "bertscore_f1": "#2A9D8F",
    "sbert_cosine": "#457B9D",
}

METRIC_LABELS = {
    "bleu": "BLEU-4",
    "rouge_l": "ROUGE-L",
    "bertscore_f1": "BERTScore F1",
    "sbert_cosine": "SBERT Cosine",
}

FEATURE_LABELS = {
    "ttr": "Type–Token Ratio",
    "function_word_ratio": "Function Word Ratio",
    "avg_sentence_length": "Avg. Sentence Length",
    "punctuation_density": "Punct. Density (per 100w)",
    "avg_word_length": "Avg. Word Length",
}


def _savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[visualizations] Saved {path}")


def plot_semantic_decay(metrics_df, suffix=""):
    """
    Line plot: semantic metrics (BLEU, ROUGE-L, BERTScore, SBERT) across T0–T3.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    gens = metrics_df["generation"].tolist()
    x = range(len(gens))

    for col, color in PALETTE.items():
        if col in metrics_df.columns:
            ax.plot(x, metrics_df[col], marker="o", linewidth=2.2,
                    markersize=7, color=color, label=METRIC_LABELS[col])

    ax.set_xticks(x)
    ax.set_xticklabels(gens, fontsize=11)
    ax.set_xlabel("Paraphrase Generation", fontsize=12)
    ax.set_ylabel("Score (vs. T0 reference)", fontsize=12)
    ax.set_title("Semantic Metric Decay Across Paraphrase Generations", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(axis="y", alpha=0.35)
    sns.despine(ax=ax)
    _savefig(f"semantic_decay{suffix}.png")


def plot_feature_evolution(features_df, suffix=""):
    """
    2×2 subplots for TTR, function-word ratio, avg sentence length, punct density.
    """
    cols = ["ttr", "function_word_ratio", "avg_sentence_length", "punctuation_density"]
    gens = features_df["generation"].tolist()
    x = range(len(gens))
    colors = ["#E63946", "#2A9D8F", "#457B9D", "#F4A261"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for i, (col, color) in enumerate(zip(cols, colors)):
        if col in features_df.columns:
            axes[i].plot(x, features_df[col], marker="s", linewidth=2.2,
                         markersize=7, color=color)
            axes[i].set_title(FEATURE_LABELS[col], fontsize=11, fontweight="bold")
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(gens, fontsize=10)
            axes[i].grid(axis="y", alpha=0.35)
            sns.despine(ax=axes[i])

    fig.suptitle("Lexical Feature Evolution Across Paraphrase Generations",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _savefig(f"feature_evolution{suffix}.png")


def plot_combined_heatmap(metrics_df, features_df, suffix=""):
    """
    Heatmap of all normalised metrics and features across generations.
    """
    metric_cols = ["bleu", "rouge_l", "bertscore_f1", "sbert_cosine"]
    feat_cols = ["ttr", "function_word_ratio"]

    avail_m = [c for c in metric_cols if c in metrics_df.columns]
    avail_f = [c for c in feat_cols if c in features_df.columns]

    m_part = metrics_df[["generation"] + avail_m].set_index("generation")
    f_part = features_df[["generation"] + avail_f].set_index("generation")
    combined = m_part.join(f_part)

    # Normalise each column to [0,1] relative to T0 baseline
    normalised = combined.div(combined.loc["T0"])

    labels_map = {**METRIC_LABELS, **FEATURE_LABELS}
    normalised.columns = [labels_map.get(c, c) for c in normalised.columns]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    sns.heatmap(normalised.T, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.7, vmax=1.0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Score (normalised to T0)"})
    ax.set_xlabel("Paraphrase Generation", fontsize=11)
    ax.set_title("Metric and Feature Retention Heatmap (T0 = 1.0)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig(f"heatmap{suffix}.png")


def plot_semantic_vs_stylistic(metrics_df, features_df, suffix=""):
    """
    Scatter: semantic retention (SBERT) vs stylistic retention (TTR) per generation.
    """
    if "sbert_cosine" not in metrics_df.columns or "ttr" not in features_df.columns:
        return

    t0_sbert = metrics_df.loc[metrics_df.generation == "T0", "sbert_cosine"].values[0]
    t0_ttr = features_df.loc[features_df.generation == "T0", "ttr"].values[0]

    sbert_ret = metrics_df["sbert_cosine"] / t0_sbert
    ttr_ret = features_df["ttr"] / t0_ttr
    gens = metrics_df["generation"].tolist()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors = ["#264653", "#2A9D8F", "#E9C46A", "#E76F51"]
    for i, (g, sr, tr) in enumerate(zip(gens, sbert_ret, ttr_ret)):
        ax.scatter(sr, tr, s=120, color=colors[i], zorder=5)
        ax.annotate(g, (sr, tr), textcoords="offset points",
                    xytext=(6, 4), fontsize=10)

    ax.plot(sbert_ret, ttr_ret, "--", color="grey", alpha=0.5, linewidth=1)
    ax.set_xlabel("Semantic Retention (SBERT, normalised)", fontsize=11)
    ax.set_ylabel("Stylistic Retention (TTR, normalised)", fontsize=11)
    ax.set_title("Semantic vs. Stylistic Retention Trajectory", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()
    _savefig(f"semantic_vs_stylistic{suffix}.png")


def generate_all_figures(metrics_df, features_df, suffix="", fig_dir=None):
    global FIG_DIR
    if fig_dir:
        FIG_DIR = fig_dir
        os.makedirs(FIG_DIR, exist_ok=True)
    plot_semantic_decay(metrics_df, suffix)
    plot_feature_evolution(features_df, suffix)
    plot_combined_heatmap(metrics_df, features_df, suffix)
    plot_semantic_vs_stylistic(metrics_df, features_df, suffix)
    print("[visualizations] All figures generated.")


# RQ1 figures. Generations form an ordered series: one-hue ordinal ramp (validated).
_GEN_COLORS = {"T1": "#86b6ef", "T2": "#2a78d6", "T3": "#104281"}
_INK, _INK_2, _RULE, _SURFACE = "#0b0b0b", "#52514e", "#b4b2a9", "#fcfcfb"
_NAMES = {"chatgpt": "ChatGPT", "palm": "PaLM2", "dipper": "DIPPER", "dipper(low)": "DIPPER (low)",
          "dipper(high)": "DIPPER (high)", "pegasus(full)": "Pegasus (full)",
          "pegasus(slight)": "Pegasus (slight)", "wordnet": "WordNet (rule-based)"}
_MEANING = "Meaning (SBERT)"


def _style_axes(ax):
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_RULE)
    ax.tick_params(colors=_INK_2, labelsize=8.5, length=0)


def plot_linguistic_delta(ret, paraphraser, out_path):
    """
    Dot plot of retention per feature family at T1, T2, T3, sorted so the families replaced
    first are at the top. 1 = unchanged, 0 = as different as an unrelated document.
    """
    wide = ret.pivot(index="family", columns="generation", values="retention")
    wide = wide.sort_values("T1", ascending=False)          # barh draws bottom-up
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(wide) + 1.4))
    y = np.arange(len(wide))
    lo = min(0.0, float(wide.min().min()) - 0.05)
    ax.axvline(0, color=_RULE, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.axvline(1, color=_RULE, lw=1, zorder=1)
    for i, (fam, row) in enumerate(wide.iterrows()):
        ax.plot([row.min(), row.max()], [i, i], color=_RULE, lw=2, zorder=2,
                solid_capstyle="round")
        for g in ["T1", "T2", "T3"]:
            ax.plot(row[g], i, "o", color=_GEN_COLORS[g], markersize=8, zorder=3,
                    markeredgecolor=_SURFACE, markeredgewidth=1.5,
                    label=g if i == 0 else None)
        ax.annotate(f"{row['T3']:.2f}", (min(row.min(), row["T3"]), i), xytext=(-9, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=7.5,
                    color=_INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(wide.index, fontsize=9, color=_INK)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == _MEANING:
            lbl.set_fontweight("bold")
    ax.set_xlim(lo - 0.08, 1.04)
    ax.set_ylim(-0.7, len(wide) - 0.3)
    ax.set_xlabel("Retention vs. T0   (1 = unchanged · 0 = as different as an unrelated document)",
                  fontsize=9, color=_INK)
    ax.set_title(f"{_NAMES.get(paraphraser, paraphraser)}: which planks are replaced first?",
                 fontsize=11, color=_INK, loc="left")
    ax.grid(axis="x", color="#e8e7e3", lw=0.8)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.legend(title=None, fontsize=8.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=3, handletextpad=0.2, columnspacing=1.2)
    fig.patch.set_facecolor(_SURFACE)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[visualizations] Saved {out_path}")


def plot_retention_heatmap(ret, out_path, generation="T3"):
    """Retention at one generation, feature family x paraphraser, with values printed."""
    sub = ret[ret.generation == generation]
    wide = sub.pivot(index="family", columns="paraphraser", values="retention")
    order = wide.mean(axis=1).sort_values().index           # most-replaced family on top
    cols = [c for c in _NAMES if c in wide.columns]
    wide = wide.loc[order, cols]
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2.6, 0.45 * len(wide) + 1.4))
    # Diverging around 0 ("as different as an unrelated document"): blue = style retained,
    # red = pushed further from T0 than another author's text; equal steps per arm
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "retention", ["#a8302f", "#e34948", "#e87470", "#f2b8b3", "#f0efec",
                      "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    ax.imshow(wide.clip(lower=-1, upper=1).to_numpy(), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(wide.shape[0]):
        for j in range(wide.shape[1]):
            v = wide.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.55 else _INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([_NAMES[c] for c in cols], rotation=30, ha="right", fontsize=8.5,
                       color=_INK)
    ax.set_yticks(range(len(wide)))
    ax.set_yticklabels(wide.index, fontsize=9, color=_INK)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == _MEANING:
            lbl.set_fontweight("bold")
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(wide)), minor=True)
    ax.grid(which="minor", color=_SURFACE, lw=2)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"Retention at {generation}: 1 = unchanged · 0 = as different as an unrelated "
                 f"document · <0 = pushed beyond it", fontsize=10, color=_INK, loc="left")
    fig.patch.set_facecolor(_SURFACE)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[visualizations] Saved {out_path}")
