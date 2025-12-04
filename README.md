

# 🚀 Credit Default Prediction – AIHack 2025 (AIFUL Japan FinTech Challenge)
### Ranked 4th out of 60 Teams | CatBoost + XGBoost Hybrid Risk Modeling

This repository contains a complete end-to-end ML pipeline for predicting **12-month credit default probability** using advanced feature engineering and ensemble modeling.  
The project was built for the **AIHack 2025 FinTech Challenge (AIFUL Japan)**.

---


---

## 🎯 Objective











Build a **robust & high-AUC credit risk model** using:

- Demographics  
- Employment stability  
- Financial indicators  
- Digital footprint behavior  
- Regional + temporal patterns  

Model outputs probability of **Default within 12 months**.

---

## 🧪 Dataset Overview

- 80,000+ customers  
- 50+ raw columns  
- Strong class imbalance (**~10% defaulters**)  
- Weak raw correlations → heavy feature engineering required  
- Contains timestamps, region IDs, financial indicators, family + employment info  

---

## 🔧 Preprocessing & Feature Engineering

### Key Steps:
- Cleaning missing & inconsistent values  
- Parsing dates (DOB, Application Date)  
- Creating more than **100 engineered features**, including:

#### Financial Features
- DTI (Debt-to-Income Ratio)  
- Loan Intensity  
- Income per dependent  
- Rent burden ratios  

#### Stability Features
- Employment years  
- Residence years  
- Employment-to-age ratio  

#### Risk/Behavioral Features
- Loan discrepancy metrics (declared vs actual)  
- High-risk employment types  
- Late-night / weekend application patterns  

#### Encodings
- JIS Region prefix encoding  
- Cleaned categorical mappings  
- Label encoding for XGBoost  
- Native handling for CatBoost  

#### Skewness Handling
- PowerTransformer (Yeo-Johnson)  
- Log transforms for income, loan amounts  

---

## 🤖 Modeling Approach

### ✔ **CatBoost Classifier (Primary Model)**
- Best performance on categorical + noisy data  
- 20+ hyperparameter experiments  
- Tuned:
  - depth  
  - learning rate  
  - L2 regularization  
  - min_data_in_leaf  
- Most stable AUC across folds  

### ✔ **XGBoost Classifier (Support Model)**
- Heavy regularization to avoid overfitting  
- Label-encoded features  
- Tuned:
  - max_depth  
  - min_child_weight  
  - gamma  
  - subsample / colsample  
  - L1 & L2 penalties  

### ❌ **Why LightGBM Was Not Used**
- Unstable validation curves  
- Overfitted on rare categories  
- Lower AUC on multiple CV folds  

---

## 🔥 Ensemble Strategy

Final prediction =  


0.75 × CatBoost + 0.25 × XGBoost


Reason:
- CatBoost → high stability  
- XGBoost → good variance & complementary patterns  

Blending improved final AUC reliably.

---

## 📈 Evaluation Metrics

Used the following robust evaluation setup:

- **AUC-ROC** (main metric)  
- **Stratified K-Fold CV**  
- **OOF (Out-of-Fold) Predictions**  
- Feature importance + stability checks  

This ensured **zero data leakage** and **no overfitting**.

---

## 🧑‍🤝‍🧑 Team Contributions

| Name | Role | Contribution |
|------|------|--------------|
| **Ansh Attre** | Preprocessing I | Raw data cleaning, imbalance analysis, EDA, skew detection |
| **Pragyansh Saxena** | Preprocessing II | Feature engineering, risk factor analysis, categorical fixing, JIS region encoding |
| **Jitendra Dawar** | Modelling I | Baseline models, CV structure, CatBoost experimentation, overfitting diagnostics |
| **Anshu Kumar Mandal** | Modelling II | Advanced XGBoost training, label encoding, numeric stabilization, final model blending |

---

## ▶️ How to Run

### Install dependencies  


pip install -r requirements.txt


### Run preprocessing  


python src/preprocessing.py


### Train models  


python src/model_training.py


### Generate final blended predictions  


python src/blending.py


---

## 🏆 Results

- **High and stable AUC across all folds**  
- Final blended model significantly outperformed individual models  
- Ranked **Top 4 out of 60 teams** in AIHack 2025  

---

## 📬 Contact

**Anshu Kumar Mandal**  
 

