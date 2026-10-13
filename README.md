# The Paradox of Authorial Identity

Measuring style decay and semantic retention under iterative LLM paraphrasing, using the
[Ship of Theseus paraphrased corpus](https://github.com/tripto03/Ship_of_theseus_paraphrased_copus)
(Tripto et al., ACL 2024).

Team 5, Northeastern University: Preethi Gajawada, Tanmayee Durga Bulusu, Madhu Babu Cherukuri,
Teja Papadasu.

- **RQ1** Which stylistic features decay before semantic content?
- **RQ2** At which paraphrase generation does authorship attribution fail?
- **RQ3** Do paraphrasers leave a recoverable fingerprint?

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
```

Download the corpus and place its CSVs under `data/raw/<dataset>/` (see
[data/raw/README.md](data/raw/README.md)).

## Run

All commands run from the repository root.

```bash
# 1. Preprocess all 7 datasets (~30 s) -> data/processed/, data/metadata/
.venv/bin/python -m src.data.preprocess --show

# 2. Baseline similarity + stylometry + figures -> experiments/, figures/
.venv/bin/python -m src.run_baseline --small --no-bertscore   # quick check (~2 min)
.venv/bin/python -m src.run_baseline                          # ChatGPT chains, 700 docs (~10 min)
.venv/bin/python -m src.run_baseline --all                    # all paraphrasers + WordNet (~1 h)

# 3. RQ1: feature-family decay vs. meaning (~6 min first time, then cached)
.venv/bin/python -m src.run_rq1

# 4. Explore results
.venv/bin/jupyter notebook notebooks/
```

| Option | Effect |
|---|---|
| `preprocess --show xsum-3981 --paraphraser dipper` | Print one document's T0 → T3 chain |
| `preprocess --datasets xsum yelp` | Preprocess a subset of datasets |
| `preprocess --verbose` | Show DEBUG logs on the console |
| `run_baseline --paraphraser palm` | One paraphraser: `chatgpt`, `palm`, `dipper`, `dipper(low)`, `dipper(high)`, `pegasus(full)`, `pegasus(slight)`, `wordnet` |

## Repository structure

```
├── data/
│   ├── raw/<dataset>/            corpus CSVs (not committed)
│   ├── processed/                t0_human/, t1–t3_paraphrased/, train/  (Parquet, regenerable)
│   └── metadata/                 dataset_splits.json, paraphraser_labels.json
├── notebooks/                    01 exploration → 05 attribution
├── src/
│   ├── data/                     load_data, preprocess, split_data, wordnet_baseline
│   ├── features/                 lexical, stylometric, syntactic, profiles, semantic*
│   ├── similarity/               bleu_rouge, bertscore, sbert_similarity
│   ├── attribution/              train_classifier*, evaluate_decay*, paraphraser_id*
│   ├── visualization/            decay_plots, trajectory_plots, tsne_umap*
│   ├── utils/                    config (paths, constants), metrics, helpers
│   ├── run_baseline.py           Weeks 1–4: similarity + stylometry baseline
│   └── run_rq1.py                Weeks 7–8: RQ1 feature-family decay analysis
├── experiments/                  baseline_results/, stylometry_results/, attribution_results/
├── figures/                      decay_curves/, tsne_plots/, attribution_charts/
├── streamlit_app/                decay dashboard* (optional)
├── paper/                        acl202X.tex, sections/, references.bib
└── docs/                         project_proposal, weekly_progress, final_checklist, progress_week8.pptx
```
`*` = placeholder for later weeks (raises `NotImplementedError`).

## Data

7 datasets (xsum, yelp, cmv, eli5, sci_gen, tldr, wp) × 7 sources that wrote T0 (Human + 6 LLMs)
× 7 paraphraser configurations (ChatGPT, PaLM2, DIPPER default/low/high, Pegasus full/slight),
each applied three times (T0 → T1 → T2 → T3).

Preprocessing cleans text (HTML entities, Unicode, quotes, whitespace), aligns complete T0–T3
chains for all paraphrasers, drops texts under 30 words, flags degenerate paraphrases
(`dipper(high)` outputs are largely unrelated to their source), splits the train files by key and
checks that train and paraphrased files share no documents. Output: 16,386 chains and 19,200
training texts; the baseline analyses a fixed sample of 100 Human documents per domain.
See [data/processed/README.md](data/processed/README.md).

## Results

| Path | Contents |
|---|---|
| `experiments/baseline_results/paraphraser_comparison.csv` | BLEU-4, ROUGE-L, BERTScore (raw, rescaled), SBERT; every paraphraser × T0–T3 |
| `experiments/baseline_results/domain_comparison.csv` | Same, per domain |
| `experiments/stylometry_results/feature_comparison.csv` | Stylometric features; every paraphraser × T0–T3 |
| `figures/decay_curves/paraphraser_trajectories.png` | Wording kept vs. meaning kept, T0 → T3, all paraphrasers |
| `figures/decay_curves/<paraphraser>/` | Decay curve, feature evolution, heatmap, semantic vs. stylistic, `linguistic_delta.png` (RQ1) |
| `experiments/stylometry_results/rq1/` | RQ1: retention per feature family (with CIs), decay ranking, statistical tests, feature shifts |
| `figures/decay_curves/retention_heatmap.png` | RQ1: retention at T3, feature family × paraphraser |

**RQ1 (preliminary):** style is replaced long before meaning. After one LLM pass, lexical and
syntactic feature families have moved 55–90% of the way to an unrelated author's style while
79–88% of the meaning is retained; the syntactic skeleton goes as fast as the lexical skin.

See [experiments/README.md](experiments/README.md) and [docs/weekly_progress.md](docs/weekly_progress.md).

## Paper

`paper/acl202X.tex` includes `sections/*.tex` and reads figures from `../figures/` (when
uploading to Overleaf, upload `figures/` alongside `paper/`). It currently uses the ACM `sigconf`
class; the final submission must be converted to the ACL template.
