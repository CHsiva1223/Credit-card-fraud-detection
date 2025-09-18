# Credit Card Fraud Detection

> A reproducible project for detecting fraudulent credit card transactions using supervised machine learning.

---

## Project overview

This repository contains code, notebooks, and supporting files for building, evaluating, and deploying models that detect credit card fraud. The goal is to produce a high-quality, explainable model pipeline that handles heavily imbalanced data and provides reliable evaluation metrics for production decision-making.

Key features

* End-to-end pipeline: data ingestion → preprocessing → model training → evaluation → inference
* Handling class imbalance (resampling, class weights, threshold tuning)
* Multiple model baselines (Logistic Regression, Random Forest, XGBoost, simple NN)
* Model evaluation with ROC-AUC, PR-AUC, precision/recall and confusion matrices
* Reproducible Jupyter notebook with visualizations and experiments

---

## Dataset

Use a public credit card fraud dataset (for example, Kaggle's *Credit Card Fraud Detection* dataset). Typical dataset characteristics:

* Tabular data where each row is a transaction
* Features may be anonymized (e.g., `V1..V28` from PCA), plus `Amount` and `Time`
* Target column: `Class` or `is_fraud` (1 = fraud, 0 = legitimate)
* Extremely imbalanced (fraud is usually < 1% of transactions)

> **Note:** Do not use production or sensitive customer data in this repository. If you evaluate on private data, follow your organization’s data governance rules.

---

## Getting started

### Prerequisites

* Python 3.8+
* Recommended: create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

`requirements.txt` should include typical packages such as:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyterlab
xgboost
imbalanced-learn
joblib
```

### File structure

```
README.md
requirements.txt
notebooks/
  └── 01_exploratory_analysis.ipynb
  └── 02_modeling.ipynb
src/
  └── data.py          # data loading & preprocessing
  └── features.py      # feature engineering
  └── train.py         # training script
  └── predict.py       # inference script
  └── evaluate.py      # evaluation utilities
models/
  └── best_model.joblib
reports/
  └── figures/
```

---

## Quick usage

### Run the Jupyter notebooks

Open `notebooks/01_exploratory_analysis.ipynb` first to understand the data, then `02_modeling.ipynb` to reproduce experiments.

```bash
jupyter lab
```

### Train model from CLI

A minimal training command (example):

```bash
python src/train.py --data data/creditcard.csv --output models/best_model.joblib --model xgboost
```

`train.py` responsibilities:

* load dataset
* train/test split with stratification
* scale/transform features
* handle imbalance (e.g., `SMOTE` or class weights)
* train model(s) and save the best one

### Run inference

```bash
python src/predict.py --model models/best_model.joblib --input data/sample_transactions.csv --output results/predictions.csv
```

---

## Modeling & evaluation

Recommended modeling workflow:

1. Baseline: Logistic Regression with class weights
2. Tree-based models: Random Forest / XGBoost with calibrated probabilities
3. Threshold tuning on validation set using precision-recall trade-off
4. Evaluate using: ROC-AUC, PR-AUC (recommended for imbalanced tasks), precision\@k, recall, F1, confusion matrix

Example evaluation code (conceptual):

```python
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

y_proba = model.predict_proba(X_val)[:,1]
print('ROC AUC:', roc_auc_score(y_val, y_proba))
print('PR AUC:', average_precision_score(y_val, y_proba))
```

---

## Reproducibility

* Use fixed random seeds for splits and model training
* Log experiments (use simple CSV logs or tools like MLflow)
* Save preprocessing pipeline (e.g., `sklearn.pipeline.Pipeline`) together with the model

---

## Tips for production

* Calibrate probabilities (Platt scaling / isotonic) when using non-probabilistic decision rules
* Monitor data drift and model performance over time
* Use human-in-the-loop review for flagged transactions until the model is trusted
* Set up rejection/override rules for high-value transactions

---

## License & contact

This project is released under the MIT License.

For questions or contributions, open an issue or contact the maintainer.

---

## Acknowledgements

Inspired by many public tutorials and notebooks on credit card fraud detection. Use responsibly and respect privacy and compliance requirements when handling real transaction data.
