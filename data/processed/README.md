# Processed data

Written by `python -m src.data.preprocess` (≈30 s); regenerate rather than edit. Details of
every step are in `experiments/logs/preprocess.log`.

| Path | Columns | Contents |
|---|---|---|
| `t0_human/<dataset>.parquet` | `key, source, domain, text` | Original texts (T0). Includes LLM-authored originals, tagged by `source`; filter `source == "Human"` for human T0 |
| `t1_paraphrased/<dataset>.parquet` | `key, source, domain, paraphraser, text` | First paraphrase of each T0, one row per paraphraser |
| `t2_paraphrased/`, `t3_paraphrased/` | same | Second and third paraphrase |
| `train/<dataset>.parquet` | `key, source, domain, text, split` | Un-paraphrased texts for RQ2 training; `split` = train / val by key |
| `chain_stats.csv` | `domain, source, paraphraser, docs, T0_words … T3_words` | Mean document length per generation |

Totals: 16,386 aligned chains (2,269 Human) across 7 datasets and 7 sources; 19,200 training
texts. Only documents with a complete T0–T3 chain for **all seven** paraphrasers and at least
30 words at T0 are kept.

Load them with `src.data.load_data`:

```python
from src.data.load_data import load_chains, load_train, get_generations
chains = load_chains(analysis_sample=True)   # the 700 Human docs used in the baseline
gens = get_generations(chains, "chatgpt")    # {"T0": [...], "T1": [...], ...}
```

Splits and the analysis sample are recorded in `data/metadata/dataset_splits.json`; label ids
for paraphrasers and sources in `data/metadata/paraphraser_labels.json`.
