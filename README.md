# Customer Churn Prediction

## Problem Statement
Predict whether a telecom customer will churn based on their tenure,subscription services, billing information and contract details.

## Dataset 
- Telco Customer Churn dataset
- 7043 samples, 21 features
- Target Variable: Churn(Yes/No)
- Class Imbalance: ~26% churn for minority class 

## Approach
- Performed EDA to identify churn patterns with respect to different features
- Preprocessed data: checked null values and duplicated rows,encoded categorical binary valued features to 0/1, 
encoded categorical multi-valued features using one-hot encoding
- Models trained and evaluated:
  ○ Baseline XGBOOST Classifier
- Handled class imbalance using:
  -XGBOOST + Threshold
  -XGBOOST + SMOTE
  -XGBOOST + ROSE
  -XGBOOST + RUS
  -XGBOOST tuned using RandomizedSearchCV + RUS
  -XGBOOST tuned using RandomizedSearchCV + Threshold
- Evaluation Metric: Recall,Precision,F1-Score

## Results

- Model Performance(for class 1 only): 
  -Baseline XGBOOST: Recall:0.51 Precicion:0.59 F1-Score:0.55
  -XGBOOST + Threshold: Recall:0.72 Precicion:0.48 F1-Score:0.58
  -XGBOOST + SMOTE:  Recall:0.58 Precicion:0.53 F1-Score:0.55
  -XGBOOST + ROSE:   Recall:0.65 Precicion:0.53 F1-Score:0.58
  -XGBOOST + RUS:    Recall:0.77 Precicion:0.47 F1-Score:0.59
  -XGBOOST tuned using RandomizedSearchCV + RUS: Recall:0.81 Precicion:0.49 F1-Score:0.61
  -XGBOOST tuned using RandomizedSearchCV + Threshold: Recall:0.82 Precicion:0.49 F1-Score:0.61
  
- Best Model: XGBOOST tuned using RandomizedSearchCV + Threshold
- Recall was prioritized as missing a churner is costlier than a false alarm
- XGBOOST tuned using RandomizedSearchCV + Threshold gave the best recall than the baseline XGBOOST model by an increase of ~61%  
 
## How to Run
- pip install -r requirements.txt
- Open CustomerChurn.ipynb and run all cells.

## Tech Stack
Python, scikit-learn, XGBoost,imblearn, pandas, matplotlib, seaborn