# SVM and Naive Bayes: Trilingual Sentiment Classification Results

Reproducibility package for the 2-classifier comparison (Support Vector Machine and Naive Bayes) on trilingual (Indonesian, English, Malay) binary sentiment classification.

## Structure

```
├── data/          7 CSV files, 2-class (positive/negative) aligned dataset
├── code/
│   └── run_svm_naivebayes.py    Trains/evaluates SVM and Naive Bayes. CPU only.
└── results/
    └── hasil_SVM_NaiveBayes.json
```

## How to Reproduce

```bash
cd code
pip install pandas scikit-learn numpy
python run_svm_naivebayes.py
```

## Verification Log

This script was independently re-executed from a clean environment. All 6 output values (2 classifiers × 3 languages: accuracy and F1-macro) were checked against previously reported figures and matched to 4 decimal places.

## Results Summary

| Language | Model | Accuracy | F1-Macro |
|---|---|---|---|
| Indonesian | SVM | 0.8849 | 0.8847 |
| Indonesian | Naive Bayes | 0.8454 | 0.8449 |
| English | SVM | 0.8618 | 0.8612 |
| English | Naive Bayes | 0.8717 | 0.8716 |
| Malay | SVM | 0.7537 | 0.6897 |
| Malay | Naive Bayes | 0.7211 | 0.6149 |

## Companion Repository

A related package extends this 2-classifier comparison with Logistic Regression as a third comparator, matching the IJAIN paper's scope. It is available at:
https://github.com/yanimaulita26/Sentimen-SVM-NB-LogReg-id-en-ma

## This Repository

https://github.com/yanimaulita26/Sentimen-SVM-NB-id-en-ma

