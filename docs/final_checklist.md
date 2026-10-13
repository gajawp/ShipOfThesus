# Final checklist

## What success looks like
- [ ] Clear research questions explicitly answered with evidence
- [x] Well-justified feature choices and evaluation metrics (similarity, lexical, syntactic)
- [ ] At least one strong visualisation per research question (RQ1: trajectory plot, linguistic delta, retention heatmap ✓; RQ2, RQ3 pending)
- [x] Reproducible code with a clean repository structure
- [ ] A coherent ACL-style paper with interpretable results (currently ACM `sigconf`; convert to ACL)

## Weekly checkpoints
- [x] **Week 2:** data loads; T0 → T3 displayed for a document; one similarity metric computed
- [x] **Week 4:** baseline BLEU/ROUGE/BERTScore across T0–T3; initial observations documented
- [x] **Week 6:** at least five stylometric features extracted; one decay plot generated
- [x] **Week 8:** feature trends analysed; preliminary answers to RQ1 drafted
- [ ] **Week 10:** authorship attribution model trained on T0 and evaluated on T1–T3
- [ ] **Week 12:** all experiments complete; figures finalised
- [ ] **Week 14:** paper polished, code reproducible, final submission ready

## Canonical workflow
- [x] Load and inspect the Ship of Theseus corpus
- [x] Compute baseline lexical and semantic similarity
- [x] Extract stylometric and syntactic features
- [x] Analyse decay across T0 → T3
- [ ] Train attribution and paraphraser-identification models
- [ ] Visualise identity drift
- [ ] Write and refine the research paper

## Pitfalls to avoid
- [x] Training and testing on the same paraphrase level: train files and paraphrased files share no keys; splits are by key
- [x] Using BLEU as a semantic metric: BLEU/ROUGE reported as wording-kept only; meaning from BERTScore/SBERT
- [ ] Over-interpreting t-SNE without quantitative support: pair each plot with a cluster metric
- [ ] Reporting accuracy without explanation: report macro-F1 with error analysis
- [x] Missing a limitations section: present in the paper; keep updating

## Final submission
- [ ] All three research questions answered
- [ ] Each figure referenced and explained in text (true for the current paper; recheck at the end)
- [x] Code runs end-to-end from README
- [ ] Results interpreted, not just reported
- [ ] Limitations and future work included (draft present)

`[~]` = partly done.
