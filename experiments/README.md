# Experiments

| Folder | Written by | Contents |
|---|---|---|
| `baseline_results/` | `python -m src.run_baseline --all` | Per paraphraser: `semantic_metrics.csv` (overall), `semantic_metrics_<domain>.csv`, `summary.json`, `all_generations.json`. Top level: `paraphraser_comparison.csv` (all paraphrasers × T0–T3) and `domain_comparison.csv` (× domain) |
| `stylometry_results/` | same | Per paraphraser: `stylometric_features.csv`. Top level: `feature_comparison.csv` |
| `stylometry_results/rq1/` | `python -m src.run_rq1` | RQ1 analysis (see below) |
| `attribution_results/` | planned (`src/attribution/`) | RQ2 attribution and RQ3 paraphraser-ID results |
| `logs/` | `python -m src.data.preprocess` | `preprocess.log` (full DEBUG log; not committed) |
| `small_run/` | `--small` test runs | Same layout on 20 docs per domain; not committed |

Current results: 700 Human documents (100 per domain), 7 Ship of Theseus paraphrasers plus the
rule-based WordNet baseline. Metrics compare each T1–T3 with its T0. `dipper(high)` outputs are
largely unrelated to their source and are excluded from interpretation.

## RQ1 outputs (`stylometry_results/rq1/`)

Each document is compared with its own T0 for 10 feature families (`src/features/profiles.py`).
Distances are divided by the distance from T0 to unrelated same-domain documents, so
**retention = 1 − normalised distance**: 1 = unchanged, 0 = as different as another author's
document, < 0 = pushed beyond that.

| File | Contents |
|---|---|
| `retention.csv` | Retention per paraphraser × family × generation, 95% bootstrap CI |
| `retention_by_domain.csv` | Same, per domain |
| `decay_ranking.csv` | Families ordered by T1 retention ("replaced first"), share of the T0→T3 loss at T1 |
| `style_vs_meaning_tests.csv` | Paired Wilcoxon, style distance > meaning distance (Holm-corrected, rank-biserial r) |
| `progression_tests.csv` | Paired Wilcoxon, distance(T2) > distance(T1), distance(T3) > distance(T2) |
| `feature_shifts.csv` | Direction of change T0 → T3 per scalar feature (means, Cohen's d_z, Holm p) |
| `unrelated_baseline.csv` | The unrelated-document distance per domain × family |
| `normalised_distances.parquet` | Per-document normalised distances behind all of the above |
| `text_profiles.parquet`, `sbert_embeddings.npy` | Caches (not committed) |
