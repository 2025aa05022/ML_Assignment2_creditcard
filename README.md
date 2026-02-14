# ML_Assignment2_creditcard

BITS Pilani — M.Tech (AIML) — Machine Learning Assignment 2  
Credit Card Fraud Detection

**Student:** Alokamaya Routray  
**ID:** 2025aa05022  
**Email:** 2025aa05022@wilp.bits-pilani.ac.in  
**Submission date:** 14-February-2026

---

## 📋 Assignment Overview

This project implements a pipeline for detecting credit-card fraud using six classification models, evaluates them on six metrics, and provides a Streamlit dashboard for exploration and batch prediction.

- Models: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, Random Forest, Gradient Boosting
- Metrics: Accuracy, AUC, Precision, Recall, F1, Matthews Correlation Coefficient (MCC)
- Includes: notebook, trained artifacts, sample test data, Streamlit app

---

## Dataset

Source: Kaggle — Credit Card Fraud Detection  
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Key points:
- 284,807 transactions, 30 features (V1–V28 are PCA components, plus Time and Amount).
- Extremely imbalanced: ~0.17% fraud.

Note: Original dataset (creditcard.csv) is not included in the repo for size/privacy reasons.

---

## Methods

- Undersampled majority class to create a balanced training set (fraud + matched legitimate).
- StandardScaler applied where needed (Logistic Regression, KNN).
- Stratified train/test split (80/20).
- Models trained with sklearn; artifacts saved with joblib for deployment.

---

## Results (test set)

| Model                   | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-------------------------|----------:|-------:|----------:|-------:|-------:|-------:|
| Logistic Regression     | 0.9340    | 0.9840 | 0.9474   | 0.9184 | 0.9326 | 0.8684 |
| Decision Tree           | 0.9137    | 0.9134 | 0.9010   | 0.9286 | 0.9146 | 0.8278 |
| K-Nearest Neighbor      | 0.8985    | 0.9517 | 0.9432   | 0.8469 | 0.8925 | 0.8010 |
| Naive Bayes (Gaussian)  | 0.8376    | 0.9555 | 0.9714   | 0.6939 | 0.8095 | 0.7038 |
| Random Forest           | 0.9492    | 0.9783 | 0.9783   | 0.9184 | 0.9474 | 0.9001 |
| Gradient Boosting       | 0.9188    | 0.9407 | 0.9100   | 0.9286 | 0.9192 | 0.8377 |

Best overall (balanced accuracy + MCC): Random Forest.

---

## Files & structure

```
ML_Assignment2_CreditCardFraud/
├── ML_Assignment2_CreditCardFraud.ipynb    # analysis & training
├── app.py                                  # Streamlit dashboard
├── requirements.txt                        # dependencies
├── README.md                               # this file
├── creditcard.csv                          # dataset (not included)
├── results.csv                             # consolidated metrics table
├── test_data_sample.csv                    # sample test rows used by the app
└── models/
    ├── LogisticRegression.pkl
    ├── DecisionTree.pkl
    ├── KNN.pkl
    ├── NaiveBayes.pkl
    ├── RandomForest.pkl
    ├── GradientBoosting.pkl
    └── Scaler.pkl
```

---


## How to run

1. Open terminal in project root:
   ```
   cd /Users/username/Documents/ML_Assignment_2/ML_Assignment2_creditcard
   ```

2. Create and activate venv (macOS):
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

Open http://localhost:8501 in your browser.

---


## Environment:
- **Python Version**: 3.14.1
- **Virtual Environment**: .venv
- **Platform**: macOS or Windows

---

## 📊 Results Summary

- **Best Model**: Random Forest with 94.92% accuracy
- **Most Interpretable**: Decision Tree and Logistic Regression
- **Fastest Training**: Naive Bayes and Logistic Regression
- **Best AUC Score**: Logistic Regression (0.984)
- **Highest Precision**: Naive Bayes (0.971)
- **Best MCC Score**: Random Forest (0.900)

---

## 🔗 Links & Resources

- **GitHub Repository**: [(https://github.com/2025aa05022/ML_Assignment2_creditcard.git)]
- **Live Application**: [(https://creditcard2025aa05022.streamlit.app/)]
- **Dataset Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Streamlit Documentation**: [streamlit.io](https://streamlit.io)

---

**BITS Pilani - Machine Learning Assignment 2**  
**Credit Card Fraud Detection System**  
**Submitted by: [Alokamaya Routray]**  
**ID: 2025aa05022**