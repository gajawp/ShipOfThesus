# Raw data: Ship of Theseus paraphrased corpus

Source: <https://github.com/tripto03/Ship_of_theseus_paraphrased_copus>
(Tripto et al., *A Ship of Theseus: Curious Cases of Paraphrasing in LLM-Generated Texts*, ACL 2024).
The original repository README is kept as [CORPUS_README.md](CORPUS_README.md).

The CSVs are not committed (≈520 MB). To set up, download the repository as a zip and place
each dataset's two files in its own folder:

```
data/raw/
├── xsum/     xsum_paraphrased.csv     xsum_train.csv      (news)
├── yelp/     yelp_paraphrased.csv     yelp_train.csv      (review)
├── cmv/      cmv_paraphrased.csv      cmv_train.csv       (opinion)
├── eli5/     eli5_paraphrased.csv     eli5_train.csv      (qa)
├── sci_gen/  sci_gen_paraphrased.csv  sci_gen_train.csv   (science)
├── tldr/     tldr_paraphrased.csv     tldr_train.csv      (summary)
└── wp/       wp_paraphrased.csv       wp_train.csv        (story)
```

| File | Columns | Contents |
|---|---|---|
| `<dataset>_paraphrased.csv` | `source, key, text, version_name` | Each original (`version_name = original`, T0) plus 21 paraphrases: 7 paraphrasers × T1–T3 (`chatgpt`, `chatgpt_chatgpt`, `chatgpt_chatgpt_chatgpt`, …) |
| `<dataset>_train.csv` | `source, key, text` | Un-paraphrased texts; no keys in common with the paraphrased file |

`source` is who wrote T0: `Human` or one of six LLMs (OpenAI, LLAMA, PaLM, Tsinghua,
BigScience, Eleuther-AI), which were prompted with the first 30 tokens of the matching human
text; they share its `key`. Paraphrasers: chatgpt, palm, dipper, dipper(low), dipper(high),
pegasus(full), pegasus(slight).
