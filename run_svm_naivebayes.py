"""
run_svm_naivebayes.py
=======================
Reproduces the SVM and Naive Bayes results (2-classifier comparison) on
the 2-class (positive/negative) aligned Indonesian, English, and Malay
datasets. CPU only -- no GPU or internet access required.

Usage:
    python run_svm_naivebayes.py

Output:
    ../results/hasil_SVM_NaiveBayes.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run(train_texts, train_labels, test_texts, test_labels, max_features=1500):
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test = le.transform(test_labels)
    classes = list(le.classes_)

    vec = TfidfVectorizer(max_features=max_features)
    Xtr = vec.fit_transform(train_texts).toarray()
    Xte = vec.transform(test_texts).toarray()

    models = {
        "SVM": SVC(kernel="rbf", random_state=42),
        "NaiveBayes": MultinomialNB(),
    }
    results = {}
    for name, clf in models.items():
        clf.fit(Xtr, y_train)
        preds = clf.predict(Xte)
        prec, rec, f1, support = precision_recall_fscore_support(
            y_test, preds, labels=[0, 1], zero_division=0
        )
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1_macro": round(f1_score(y_test, preds, average="macro"), 4),
            "per_class": {
                classes[0]: {"precision": round(prec[0], 4), "recall": round(rec[0], 4), "f1": round(f1[0], 4), "support": int(support[0])},
                classes[1]: {"precision": round(prec[1], 4), "recall": round(rec[1], 4), "f1": round(f1[1], 4), "support": int(support[1])},
            },
            "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
        }
    return results


def main():
    all_results = {}

    idtr = pd.concat([pd.read_csv(DATA_DIR / "indotrain_2class.csv"), pd.read_csv(DATA_DIR / "indovalid_2class.csv")])
    idte = pd.read_csv(DATA_DIR / "indotest_2class.csv")
    all_results["Indonesian"] = run(idtr["text"].tolist(), idtr["label"].tolist(), idte["text"].tolist(), idte["label"].tolist())

    entr = pd.concat([pd.read_csv(DATA_DIR / "inggristrain_2class.csv"), pd.read_csv(DATA_DIR / "inggrisvalid_2class.csv")])
    ente = pd.read_csv(DATA_DIR / "inggristest_2class.csv")
    all_results["English"] = run(entr["text"].tolist(), entr["label"].tolist(), ente["text"].tolist(), ente["label"].tolist())

    mal = pd.read_csv(DATA_DIR / "melayu_2class.csv")
    mtr, mte = train_test_split(mal, test_size=0.2, stratify=mal["label"], random_state=42)
    all_results["Malay"] = run(mtr["text"].tolist(), mtr["label"].tolist(), mte["text"].tolist(), mte["label"].tolist())

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "hasil_SVM_NaiveBayes.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(json.dumps(all_results, indent=2))
    print(f"\n✅ Results written to {out_path}")


if __name__ == "__main__":
    main()
