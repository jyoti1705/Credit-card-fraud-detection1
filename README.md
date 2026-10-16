# Credit Card Fraud Detection

**Live app:** https://jyoti1705-credit-card-fraud-detection1-app-7nbhne.streamlit.app/

A machine learning model that detects fraudulent credit card transactions, deployed as an interactive Streamlit app.

## Problem
Fraud is extremely rare (0.17% of transactions), so accuracy is misleading: a model that always predicts "not fraud" scores 99.8%. I used precision, recall and PR-AUC instead.

## Dataset
[Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions, 492 frauds (0.17%)
- After removing 1,081 duplicates: 283,726 rows, 473 frauds
- Features: `Time`, `Amount`, and `V1`-`V28` (anonymized PCA components)

## Method
1. Removed duplicates before splitting, so no row appears in both train and test
2. Stratified 75/25 train-test split (the test set has 70,932 transactions, including 118 frauds)
3. Scaled features using the training set only (no data leakage)
4. Compared three models:
   - Logistic Regression (undersampling 1:1)
   - Random Forest (undersampling 1:10)
   - XGBoost (full data, no undersampling)
5. Chose the decision threshold on a separate validation split, not on the test set

## Results (test set)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression (undersample 1:1) | 0.034 | 0.907 | 0.065 | 0.505 |
| Random Forest (undersample 1:10) | 0.579 | 0.805 | 0.674 | 0.682 |
| XGBoost (default threshold 0.5) | 0.947 | 0.754 | 0.840 | 0.832 |

Logistic Regression and Random Forest have almost the same ROC-AUC (0.974), but their PR-AUC differs a lot, so PR-AUC is the more reliable metric here.

**Final model: XGBoost with a tuned threshold of about 0.07** (chosen on the validation set to reach 80% recall):

| Precision | Recall | F1 | Frauds caught | False alarms |
|---|---|---|---|---|
| 0.81 | 0.80 | 0.80 | 94 of 118 | 22 (out of 70,814 normal transactions) |

Pushing recall higher (around 85%) sharply increases false alarms, so I chose 80% as a balanced operating point.

## Try the app
1. Open the live app
2. Upload a CSV with the columns `Time`, `V1`-`V28`, `Amount` (a `Class` column is optional)
3. The app shows the flagged transactions and their fraud probability

`sample_transactions.csv` is included for a demo (500 rows, 30 frauds). It is sampled from the original data, including rows seen during training, so its results are optimistic. The numbers in the Results section are the real test results.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack
Python, Pandas, NumPy, Matplotlib, Scikit-learn, imbalanced-learn, XGBoost, Joblib, Streamlit

## Limitations and next steps
- `V1`-`V28` are anonymized, so fraud patterns cannot be explained in business terms
- Only 473 fraud cases exist and the test set has 118, so results can vary between splits
- XGBoost parameters were set manually, with no hyperparameter search
- Next: cross-validation, SHAP explanations, and an API endpoint

## Files
- `credit_card_fraud_detection.ipynb`: full analysis and model training
- `app.py`: Streamlit app
- `fraud_model.joblib`: saved scaler, XGBoost model and threshold
- `sample_transactions.csv`: demo data
- `requirements.txt`: dependencies with pinned versions
