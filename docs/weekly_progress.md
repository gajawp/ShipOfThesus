# Weekly progress

## Weeks 1–2: Data ingestion
- Downloaded the Ship of Theseus corpus: 7 datasets × (paraphrased + train) CSVs → `data/raw/`.
- Verified the schema: `source, key, text, version_name`; 7 sources, 7 paraphraser
  configurations × T1–T3.
- Loaded the data, displayed T0 → T3 for a document and computed SBERT similarity
  (`notebooks/01_data_exploration.ipynb`).

## Weeks 3–4: Preprocessing and baseline similarity
- **Preprocessing** (`src/data/preprocess.py`) for all 7 datasets and sources: text cleaning,
  T0–T3 chain alignment (complete chains for all 7 paraphrasers), 30-word filter, sanity checks,
  key-level train/val split of the train files, leakage check (paraphrased and train files share
  no keys). Output: 16,386 chains and 19,200 training texts.
- **Data issue found:** `dipper(high)` chains are largely unrelated to their source (54–95% of T1
  outputs share <10% of words with T0 in 5 of 7 domains); excluded from interpretation.
- **Baseline** (`src/run_baseline.py --all`) on 700 Human documents (100 per domain): BLEU-4,
  ROUGE-L, BERTScore (raw and rescaled), SBERT; five stylometric features; per-paraphraser and
  cross-paraphraser figures.
- **Initial observations:**
  - Wording is replaced far faster than meaning: ChatGPT T0→T3 loses 86% of BLEU but 12% of SBERT.
  - Most of the loss happens at T1 for ChatGPT and PaLM2 (80% of the SBERT loss); DIPPER and
    WordNet lose meaning more gradually.
  - The rule-based WordNet baseline keeps more wording but less meaning than LLM paraphrasers.
  - Science/summary/news text keeps the most meaning; stories and opinions the least.
  - Each paraphraser shifts style in its own direction (ChatGPT: longer words, fewer function
    words; DIPPER: the reverse; Pegasus-full: shorter sentences), an early sign for RQ3.
- Report updated (`paper/`); repository reorganised into the course structure.

## Weeks 5–6: Stylometric features
- Five stylometric features (TTR, function-word ratio, sentence length, punctuation density,
  word length) across T0–T3 for every paraphraser; decay plots per paraphraser.

## Weeks 7–8: Syntactic features and RQ1
- **Syntactic features** (`src/features/syntactic_features.py`, spaCy `en_core_web_sm`): Universal
  POS distribution, dependency-label distribution, tree depth, dependency distance,
  Flesch–Kincaid grade.
- **New lexical features:** MATTR (length-robust TTR), sentence-length standard deviation,
  function-word and punctuation profiles.
- **Per-document decay analysis** (`src/run_rq1.py`): 10 feature families compared with each
  document's own T0; distances normalised by the distance to unrelated same-domain documents,
  giving retention (1 = unchanged, 0 = another author). 95% bootstrap CIs, paired Wilcoxon tests
  with Holm correction, rank-biserial effect sizes.
- **Preliminary RQ1 answer:**
  - Style is replaced long before meaning: for every LLM paraphraser, meaning is the most retained
    family (ChatGPT T3: meaning 0.85 vs. style ≤ 0.34); in 61/63 tests at T1 the style change is
    significantly larger than the meaning change (exceptions: two WordNet families).
  - The syntactic skeleton goes as fast as the lexical skin (ChatGPT 0.14 vs. 0.15; PaLM2 0.30 vs.
    0.33); Pegasus-full restructures most (syntactic retention −0.53).
  - 82–92% of each style family's loss happens at T1 for ChatGPT and PaLM2; conservative
    settings and WordNet decay more gradually.
  - Several features are overwritten, not just erased (retention < 0): ChatGPT raises the reading
    grade 9.8 → 12.4 and word length 4.68 → 5.21; DIPPER moves the other way, a lead for RQ3.
  - MATTR confirms ChatGPT's vocabulary diversification is real, not a length artefact.
- Figures: `figures/decay_curves/<paraphraser>/linguistic_delta.png`,
  `figures/decay_curves/retention_heatmap.png`. Paper section "RQ1: Which Planks Are Replaced
  First?" added.

## Next (Weeks 9–10): RQ2
- Train stylometric SVM and RoBERTa attribution models on `data/processed/train/`.
- Evaluate on T0–T3 per paraphraser; locate the "point of no return".
