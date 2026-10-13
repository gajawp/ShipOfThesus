# Notebooks

Run from this folder after `python -m src.data.preprocess` (and `python -m src.run_baseline --all`
for 02–04). Each notebook adds the repository root to `sys.path` and imports from `src/`.

| Notebook | Contents | Status |
|---|---|---|
| `01_data_exploration.ipynb` | Raw and processed counts, one T0 → T3 chain, one SBERT score | Week 2 checkpoint ✓ |
| `02_baseline_similarity.ipynb` | BLEU / ROUGE-L / BERTScore / SBERT tables, trajectory figure | Week 4 checkpoint ✓ |
| `03_stylometry_analysis.ipynb` | Stylometric features T0 → T3; RQ1 retention, decay ranking, tests, feature shifts | Week 8 checkpoint ✓ |
| `04_semantic_similarity.ipynb` | SBERT and rescaled BERTScore by domain | ✓ |
| `05_attribution_models.ipynb` | RQ2/RQ3 training data overview | Planned (Weeks 9–12) |
