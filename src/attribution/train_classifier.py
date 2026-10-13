"""
train_classifier.py  [planned: Weeks 9-10, RQ2]
Train authorship-attribution models (7 sources: Human + 6 LLMs) on
data/processed/train/ using the key-level train/val split:
  * stylometric features + SVM
  * fine-tuned RoBERTa
Report macro-F1 on the validation split. Save models to experiments/attribution_results/.
"""


def train(model="svm"):
    raise NotImplementedError("Planned for Weeks 9-10")
