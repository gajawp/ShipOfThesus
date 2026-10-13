# Project Proposal: The Paradox of Authorial Identity

*Measuring Style Decay and Semantic Retention Under Iterative LLM Paraphrasing*

Team 5, Northeastern University: Preethi Gajawada, Tanmayee Durga Bulusu, Madhu Babu Cherukuri,
Teja Papadasu. Original: [project_proposal.pdf](project_proposal.pdf),
slides: [project_proposal.pptx](project_proposal.pptx).

## 1. Problem description

A machine paraphrase preserves *what* a text says while rewriting *how* it says it. Applied
repeatedly, this poses the Ship of Theseus question for authorship: after a document has been
paraphrased three times by a large language model, whose text is it? Authorship attribution,
machine-generated text (MGT) detection, plagiarism analysis and forensic provenance all assume a
stable stylistic signal survives in a document; paraphrasing cheaply defeats that assumption.

We measure identity decay quantitatively rather than as a single pass/fail outcome, tracking
generation by generation which linguistic properties are destroyed first and which persist.

- **RQ1** Which features decay before semantic content does?
- **RQ2** At which generation does attribution become unreliable?
- **RQ3** Does the paraphraser leave a recoverable fingerprint, so that authorial identity is not
  merely erased but replaced?

Contribution: a decay curve separating human-author signal loss from model-fingerprint gain.

## 2. Background

- **Iterative paraphrasing and authorship** (Tripto et al., ACL 2024): four paraphrasers
  (ChatGPT, PaLM2, Pegasus, DIPPER) applied iteratively over seven datasets; attribution degrades,
  but only aggregate accuracy is reported, never which features decay.
- **Paraphrase as an evasion attack** (Krishna et al., NeurIPS 2023): DIPPER, an 11B
  discourse-level paraphraser; one pass drives DetectGPT and GPTZero toward chance.
- **Zero-shot MGT detection** (Mitchell et al., ICML 2023): DetectGPT; needs white-box
  likelihoods and collapses under paraphrase.
- **Attribution across neural generators** (Uchendu et al., EMNLP 2020): generators are separable,
  but only single-pass text is tested.
- **Content-independent author representations** (Rivera-Soto et al., EMNLP 2021): embeddings
  transfer across domains but stay sensitive to topic overlap.
- **Measuring semantic retention** (Zhang et al., ICLR 2020): BERTScore tracks adequacy far better
  than BLEU but is blind to style, so we pair it with explicit stylometry.

## 3. Dataset and repository

The Ship of Theseus paraphrased corpus: seven source datasets × four paraphrasers, each sample
linked to its originating document across generations T0 (original) through T3.
Repository: <https://github.com/gajawp/ShipOfThesus>.

## 4. Project plan and milestones

| Week | Activity and deliverable |
|---|---|
| 3 | Acquire corpus; verify schema; parse T0–T3 lineage |
| 4 | Preprocessing pipeline; leakage-free document-level splits |
| 5 | Semantic baselines (BLEU, ROUGE-L, BERTScore, SBERT). **M1: data pipeline** |
| 6 | Lexical features: type–token ratio, function words, punctuation, length |
| 7 | Syntactic/readability features: POS n-grams, dependency patterns, Flesch–Kincaid |
| 8 | Decay curves; style-vs-semantics tests. **M2: content audit (RQ1)** |
| 9 | Attribution baselines on T0 only: stylometry + SVM, fine-tuned RoBERTa |
| 10 | Evaluate T0-trained models on T1–T3; per-generation accuracy decay |
| 11 | Locate "point of no return"; error analysis. **M3: breakdown (RQ2)** |
| 12 | Paraphraser classifier on held-out data. **M4: fingerprints (RQ3)** |
| 13 | Identity-trajectory visualisations, Streamlit dashboard, ablations |
| 14 | Final ACL-format paper; repo docs; presentation. **M5: deliverables** |

Gates throughout: no cross-split document leakage; macro-F1 per classifier; fingerprints
validated on unseen data.
