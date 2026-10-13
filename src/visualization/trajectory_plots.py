"""
trajectory_plots.py
Cross-paraphraser figure: wording kept (BLEU-4) vs. meaning kept (SBERT), T0 -> T3.
Saved to figures/decay_curves/paraphraser_trajectories.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import EXCLUDED_PARAPHRASERS


# Paraphraser comparison: 3 highlighted series (validated all-pairs), the rest in gray
_HIGHLIGHT = {"chatgpt": "#2a78d6", "wordnet": "#eb6834", "dipper": "#1baf7a"}
_CONTEXT_GRAY = "#b4b2a9"
_INK, _INK_2 = "#0b0b0b", "#52514e"
_DISPLAY_NAMES = {"chatgpt": "ChatGPT", "palm": "PaLM2", "dipper": "DIPPER",
                  "dipper(low)": "DIPPER (low)", "pegasus(full)": "Pegasus (full)",
                  "pegasus(slight)": "Pegasus (slight)", "wordnet": "WordNet (rule-based)"}
# Direct-label placement: (generation the label points at, offset in points), chosen so labels
# land in empty space where the paths crowd together; "leader" labels sit at a data position
# and point back to their marker with a thin line
_LABELS = {"chatgpt": ("T3", (-7, -1)), "palm": ("T3", (-5, -9)),
           "dipper(low)": ("T2", "leader", (0.25, 0.955)), "pegasus(slight)": ("T2", (6, -8)),
           "pegasus(full)": ("T3", (6, -5)), "wordnet": ("T3", (7, -2)), "dipper": ("T3", (7, -2))}


def plot_paraphraser_trajectories(comparison_df, out_path, exclude=tuple(EXCLUDED_PARAPHRASERS)):
    """
    Connected scatter: wording kept (BLEU-4) vs meaning kept (SBERT) at T0–T3, one path
    per paraphraser. Points above the diagonal kept more meaning than wording.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], color=_CONTEXT_GRAY, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.725, 0.712, "y = x: wording and meaning lost equally", color=_INK_2,
            fontsize=7.5, ha="left", va="bottom")

    paraphrasers = [p for p in comparison_df["paraphraser"].unique() if p not in exclude]
    # Gray context series first so highlighted paths sit on top
    for p in sorted(paraphrasers, key=lambda p: p in _HIGHLIGHT):
        d = comparison_df[comparison_df["paraphraser"] == p].sort_values("generation")
        color = _HIGHLIGHT.get(p, _CONTEXT_GRAY)
        hi = p in _HIGHLIGHT
        ax.plot(d["bleu"], d["sbert_cosine"], color=color, lw=2 if hi else 1.4,
                marker="o", markersize=6 if hi else 4.5, markeredgecolor="#fcfcfb",
                markeredgewidth=1.2, zorder=3 if hi else 2)
        gen, *place = _LABELS.get(p, ("T3", (6, 0)))
        anchor = d[d["generation"] == gen].iloc[0]
        style = dict(fontsize=8.5, color=_INK if hi else _INK_2,
                     fontweight="bold" if hi else "normal", va="center")
        xy = (anchor["bleu"], anchor["sbert_cosine"])
        if place[0] == "leader":
            ax.annotate(_DISPLAY_NAMES.get(p, p), xy, xytext=place[1], textcoords="data",
                        ha="center", arrowprops=dict(arrowstyle="-", color=_CONTEXT_GRAY,
                                                     lw=0.8, shrinkA=2, shrinkB=4), **style)
        else:
            ax.annotate(_DISPLAY_NAMES.get(p, p), xy, xytext=place[0], textcoords="offset points",
                        ha="right" if place[0][0] < 0 else "left", **style)
        if p == "chatgpt":  # label generations once, on the reference paraphraser
            for _, r in d.iloc[1:].iterrows():
                ax.annotate(r["generation"], (r["bleu"], r["sbert_cosine"]), xytext=(0, 7),
                            textcoords="offset points", fontsize=7.5, color=_INK_2, ha="center")

    ax.annotate("T0 (original)", (1, 1), xytext=(-8, 6), textcoords="offset points",
                fontsize=8, color=_INK_2, ha="right")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0.70, 1.02)
    ax.set_xlabel("Wording kept: BLEU-4 vs. T0", fontsize=10, color=_INK)
    ax.set_ylabel("Meaning kept: SBERT cosine vs. T0", fontsize=10, color=_INK)
    ax.set_title("Paraphrasers replace wording far faster than meaning (T0 → T3)",
                 fontsize=11, color=_INK, loc="left")
    ax.grid(color="#e8e7e3", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_CONTEXT_GRAY)
    ax.tick_params(colors=_INK_2, labelsize=8.5)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[visualizations] Saved {out_path}")
