#!/usr/bin/env python3
# hybrid_reliable_god.py
# Robust, ready-to-run version for VS Code / Mac
# Fixes: XGBoost early-stopping compatibility, safer preprocessing, no Colab deps.

import os
import time
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif

from catboost import CatBoostClassifier, Pool
import xgboost as xgb_lib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------- CONFIG ----------------------

TRAIN_FILE = os.path.join("train.csv")
TEST_FILE = os.path.join("test.csv")
TARGET = "Default 12 Flag"
ID_COL = "ID"
N_SPLITS = 5
RANDOM_STATE = 42
TOP_K_GOD_FEATURES = 40   # number of god features to add to safe set (tune me)
# ----------------------------------------------------

def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

print("Loading data...")
train_df = safe_read_csv(TRAIN_FILE)
test_df = safe_read_csv(TEST_FILE)
print(f"Train: {train_df.shape} | Test: {test_df.shape}")
print(f"Default rate: {train_df[TARGET].mean():.4f}")

# ---------------- SAFE FEATURES function ----------------
def create_safe_features(df):
    df = df.copy()
    # JIS cleaning
    df['JIS_str'] = df.get('JIS Address Code', pd.Series(dtype=str)).fillna(-999).astype(str)
    df['JIS_Prefix_2'] = df['JIS_str'].str[:2]
    df['JIS_Prefix_3'] = df['JIS_str'].str[:3]

    # Dates
    df['Application Date'] = pd.to_datetime(df.get('Application Date', pd.NaT), format='%Y/%m/%d', errors='coerce')
    df['Date of Birth'] = pd.to_datetime(df.get('Date of Birth', pd.NaT), format='%Y/%m/%d', errors='coerce')

    df['Age'] = (df['Application Date'] - df['Date of Birth']).dt.days / 365.25
    df['App_Month'] = df['Application Date'].dt.month
    df['App_DayOfWeek'] = df['Application Date'].dt.dayofweek
    df['App_Quarter'] = df['Application Date'].dt.quarter
    df['App_Hour'] = df.get('Application Time', 0) // 10000

    # Basic flags
    df['Is_Weekend'] = (df['App_DayOfWeek'] >= 5).astype(int)
    df['Is_BusinessHours'] = ((df['App_Hour'] >= 9) & (df['App_Hour'] <= 17)).astype(int)
    df['Is_LateNight'] = ((df['App_Hour'] >= 22) | (df['App_Hour'] <= 5)).astype(int)

    # Fraud / honesty
    df['Amount of Unsecured Loans'] = df.get('Amount of Unsecured Loans', 0)
    df['Declared Amount of Unsecured Loans'] = df.get('Declared Amount of Unsecured Loans', 0)
    df['Declared Number of Unsecured Loans'] = df.get('Declared Number of Unsecured Loans', 0)
    df['Number of Unsecured Loans'] = df.get('Number of Unsecured Loans', 0)

    df['Loan_Amount_Gap'] = df['Declared Amount of Unsecured Loans'] - df['Amount of Unsecured Loans']
    df['Loan_Count_Gap'] = df['Declared Number of Unsecured Loans'] - df['Number of Unsecured Loans']
    df['Abs_Amount_Gap'] = df['Loan_Amount_Gap'].abs()
    df['Hidden_Loans'] = (df['Loan_Count_Gap'] < 0).astype(int)
    df['Severe_Underreporting'] = ((df['Loan_Amount_Gap'] < 0) & (df['Abs_Amount_Gap'] >= 200000)).astype(int)
    df['Honesty_Score'] = np.where(df['Declared Amount of Unsecured Loans'] > 0,
                                  1 - np.clip(df['Abs_Amount_Gap'] / (df['Declared Amount of Unsecured Loans'] + 1), 0, 1), 1)
    df['Perfect_Match'] = ((df['Loan_Amount_Gap'] == 0) & (df['Loan_Count_Gap'] == 0)).astype(int)

    # Financial
    df['Total Annual Income'] = df.get('Total Annual Income', 0)
    df['Amount of Unsecured Loans'] = df['Amount of Unsecured Loans'].fillna(0)
    df['Application Limit Amount(Desired)'] = df.get('Application Limit Amount(Desired)', 0)
    df['Rent Burden Amount'] = df.get('Rent Burden Amount', 0)

    df['Income_log'] = np.log1p(df['Total Annual Income'])
    df['Existing_Loan_log'] = np.log1p(df['Amount of Unsecured Loans'])
    df['Desired_Loan_log'] = np.log1p(df['Application Limit Amount(Desired)'])
    df['DTI_Total'] = (df['Amount of Unsecured Loans'] + df['Application Limit Amount(Desired)']) / (df['Total Annual Income'] + 1)
    df['DTI_WithRent_Annual'] = (df['Amount of Unsecured Loans'] + df['Rent Burden Amount'] * 12) / (df['Total Annual Income'] + 1)
    df['Income_per_Dependent'] = df['Total Annual Income'] / (df.get('Number of Dependents', 0) + 1)
    df['Avg_Existing_Loan'] = df['Amount of Unsecured Loans'] / (df.get('Number of Unsecured Loans', 0) + 1)
    df['Loan_Intensity'] = df.get('Number of Unsecured Loans', 0) / (df['Age'].fillna(0) + 1)

    # Stability
    df['Employment_Years'] = df.get('Duration of Employment at Company (Months)', 0) / 12
    df['Residence_Years'] = df.get('Duration of Residence (Months)', 0) / 12
    df['Employment_to_Age'] = df['Employment_Years'] / (df['Age'].fillna(0) + 1)
    df['Combined_Stability'] = (df['Employment_Years'] + df['Residence_Years']) / 2
    df['Is_New_Job'] = (df['Employment_Years'] <= 1).astype(int)

    # Housing & family
    df['Is_Homeowner'] = df.get('Residence Type', -999).isin([1,2,8,9]).astype(int) if 'Residence Type' in df else 0
    df['Is_Renter'] = df.get('Residence Type', -999).isin([4,5,6,7]).astype(int) if 'Residence Type' in df else 0
    df['Is_Regular_Employee'] = (df.get('Employment Status Type', -999) == 1).astype(int) if 'Employment Status Type' in df else 0
    df['Is_Part_Time'] = (df.get('Employment Type', -999) == 4).astype(int) if 'Employment Type' in df else 0
    df['Is_Married'] = (df.get('Single/Married Status', -999) == 2).astype(int) if 'Single/Married Status' in df else 0
    df['Has_Dependents'] = (df.get('Number of Dependents', 0) > 0).astype(int)

    # Interactions limited (safe)
    df['Age_Income'] = df['Age'].fillna(0) * df['Income_log']
    df['Age_DTI'] = df['Age'].fillna(0) * df['DTI_Total']
    df['Income_Dependents'] = df['Income_log'] * (df.get('Number of Dependents', 0) + 1)

    # Drop raw date columns to avoid leakage
    df = df.drop(columns=['Application Date', 'Date of Birth', 'JIS_str'], errors='ignore')
    return df

# ---------------- GOD features (heavy) function (compact) ----------------
def create_god_features(df):
    df = df.copy()
    df['Application Date'] = pd.to_datetime(df.get('Application Date', pd.NaT), format='%Y/%m/%d', errors='coerce')
    df['Date of Birth'] = pd.to_datetime(df.get('Date of Birth', pd.NaT), format='%Y/%m/%d', errors='coerce')
    df['Age'] = (df['Application Date'] - df['Date of Birth']).dt.days / 365.25
    df['App_Month'] = df['Application Date'].dt.month
    df['App_DayOfWeek'] = df['Application Date'].dt.dayofweek
    df['App_Hour'] = df.get('Application Time', 0) // 10000

    df['Total Annual Income'] = df.get('Total Annual Income', 0)
    df['Amount of Unsecured Loans'] = df.get('Amount of Unsecured Loans', 0)
    df['Declared Amount of Unsecured Loans'] = df.get('Declared Amount of Unsecured Loans', 0)
    df['Application Limit Amount(Desired)'] = df.get('Application Limit Amount(Desired)', 0)

    df['Income_log'] = np.log1p(df['Total Annual Income'])
    df['Amount_log'] = np.log1p(df['Amount of Unsecured Loans'])
    df['Declared_log'] = np.log1p(df['Declared Amount of Unsecured Loans'])
    df['Desired_log'] = np.log1p(df['Application Limit Amount(Desired)'])

    df['DTI_Total'] = (df['Amount of Unsecured Loans'] + df['Application Limit Amount(Desired)']) / (df['Total Annual Income'] + 1)
    df['DTI_Declared'] = df['Declared Amount of Unsecured Loans'] / (df['Total Annual Income'] + 1)

    df['Income_Percentile'] = df['Total Annual Income'].rank(pct=True)
    df['Loan_Percentile'] = df['Application Limit Amount(Desired)'].rank(pct=True)
    df['Age_Sq'] = df['Age'] ** 2
    df['Age_Sqrt'] = np.sqrt(np.clip(df['Age'].fillna(0), 0, None))

    df['Loan_Amount_Gap'] = df['Declared Amount of Unsecured Loans'] - df['Amount of Unsecured Loans']
    df['Abs_Amount_Gap'] = df['Loan_Amount_Gap'].abs()
    df['Honesty_Score'] = np.where(df['Declared Amount of Unsecured Loans'] > 0,
                                  1 - np.clip(df['Abs_Amount_Gap'] / (df['Declared Amount of Unsecured Loans'] + 1), 0, 1), 1)

    df['Simple_Risk'] = ((df['DTI_Total'] > 0.4).astype(int) +
                         (df['Is_New_Job'] if 'Is_New_Job' in df.columns else 0))

    # safe KMeans-style cluster proxy
    try:
        arr = np.vstack([df['Income_log'].fillna(0), df['Amount_log'].fillna(0)]).T
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['income_loan_cluster'] = km.fit_predict(np.nan_to_num(arr))
    except Exception:
        df['income_loan_cluster'] = 0

    df = df.drop(columns=['Application Date', 'Date of Birth'], errors='ignore')
    return df

# ---------------- APPLY SAFE + GOD (for selection) ----------------
print("Creating SAFE features (train + test)...")
train_safe = create_safe_features(train_df)
test_safe = create_safe_features(test_df)

print("Creating GOD features (train + test) for signal selection...")
train_god_all = create_god_features(train_df)
test_god_all = create_god_features(test_df)

# Align god columns
train_god_all = train_god_all.reindex(columns=[c for c in train_god_all.columns], fill_value=np.nan)
test_god_all = test_god_all.reindex(columns=train_god_all.columns, fill_value=np.nan)

# ---------------- ADVERSARIAL VALIDATION (as feature) ----------------
print("Running Adversarial Validation (fast LightGBM) to create av_score feature...")
av_X = pd.concat([
    train_safe.drop(columns=[TARGET], errors='ignore'),
    test_safe.drop(columns=[ID_COL], errors='ignore')
], axis=0, ignore_index=True)
av_y = np.array([0]*len(train_safe) + [1]*len(test_safe))

# Select simple subset for AV training
av_cols = []
for c in av_X.columns:
    if av_X[c].dtype.kind in 'bifc' or av_X[c].nunique() < 200:
        av_cols.append(c)
av_X_sel = av_X[av_cols].fillna(-999)

# label encode low-cardinality cats
for c in av_X_sel.columns:
    if av_X_sel[c].dtype == object or av_X_sel[c].nunique() < 200:
        av_X_sel[c] = av_X_sel[c].astype(str)
        le = LabelEncoder()
        av_X_sel[c] = le.fit_transform(av_X_sel[c].fillna("-999"))

av_model = LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1)
av_model.fit(av_X_sel, av_y)
av_preds_all = av_model.predict_proba(av_X_sel)[:, 1]
av_auc = roc_auc_score(av_y, av_preds_all)
print(f"AV AUC (train+test): {av_auc:.4f}")

# put av_score back
train_safe['av_score'] = av_preds_all[:len(train_safe)]
test_safe['av_score'] = av_preds_all[len(train_safe):len(train_safe)+len(test_safe)]

# ---------------- SELECT TOP GOD FEATURES by mutual information ----------------
print("Computing mutual information of god-features and selecting top ones...")
candidate_cols = []
for c in train_god_all.columns:
    if c in [TARGET, ID_COL]:
        continue
    if train_god_all[c].dtype.kind in 'bifc' or train_god_all[c].nunique() < 200:
        candidate_cols.append(c)

mi_X = train_god_all[candidate_cols].fillna(-999)
for c in mi_X.columns:
    if mi_X[c].dtype == object:
        mi_X[c] = LabelEncoder().fit_transform(mi_X[c].astype(str).fillna("-999"))

mi = mutual_info_classif(mi_X, train_df[TARGET].values, random_state=RANDOM_STATE)
mi_ser = pd.Series(mi, index=mi_X.columns).sort_values(ascending=False)
top_god = list(mi_ser.head(TOP_K_GOD_FEATURES).index)
print(f"Selected top {len(top_god)} god features: {top_god[:10]} ...")

for c in top_god:
    train_safe[c] = train_god_all[c].values
    test_safe[c] = test_god_all[c].values

# ---------------- PREP final datasets ----------------
print("Preparing final datasets...")
X = train_safe.drop(columns=[TARGET], errors='ignore')
X_test = test_safe.drop(columns=[ID_COL], errors='ignore')
y = train_df[TARGET].copy()
test_ids = test_df[ID_COL].copy()

# ---------------- FEATURE REGULARIZATION ----------------
print("Feature regularization: drop near-constant and highly correlated features...")

# 1) low variance
vt = VarianceThreshold(threshold=1e-6)
numeric_for_vt = X.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_for_vt) > 0:
    vt.fit(X[numeric_for_vt].fillna(-999))
    keep_num = [c for c, keep in zip(numeric_for_vt, vt.get_support()) if keep]
else:
    keep_num = []

other_cols = [c for c in X.columns if c not in numeric_for_vt]
keep_cols = keep_num + other_cols
X = X[keep_cols].copy()
X_test = X_test.reindex(columns=X.columns, fill_value=-999)

# 2) drop high correlation among numeric features (corr > 0.98)
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) >= 2:
    corr = X[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [c for c in upper.columns if any(upper[c] > 0.98)]
    if drop_corr:
        print(f"Dropping {len(drop_corr)} highly correlated numeric features")
        X = X.drop(columns=drop_corr, errors='ignore')
        X_test = X_test.drop(columns=drop_corr, errors='ignore')

print(f"Final feature count: {X.shape[1]}")

# ---------------- PREPROCESS: categorical handling & power transform ----------------
print("Encoding categorical features and applying PowerTransformer for skewed numeric...")

cat_features = [c for c in X.columns if (X[c].dtype == object) or (X[c].nunique() < 200 and X[c].dtype.kind not in 'fi')]
cat_features = [c for c in cat_features if c in X.columns]
num_features = [c for c in X.columns if c not in cat_features]

for c in cat_features:
    X[c] = X[c].fillna("-999").astype(str)
    X_test[c] = X_test[c].fillna("-999").astype(str)

X[num_features] = X[num_features].fillna(-999)
X_test[num_features] = X_test[num_features].fillna(-999)

skewed = [c for c in ['Total Annual Income','Amount of Unsecured Loans','Application Limit Amount(Desired)','Rent Burden Amount'] if c in num_features]
if skewed:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    X_sk = X[skewed].replace(-999, 0)
    Xt_sk = X_test[skewed].replace(-999, 0)
    try:
        Xt_trans = pt.fit_transform(X_sk)
        X_test_trans = pt.transform(Xt_sk)
        for i,c in enumerate(skewed):
            X[c + '_power'] = Xt_trans[:, i]
            X_test[c + '_power'] = X_test_trans[:, i]
            num_features.append(c + '_power')
        print(f"Applied PowerTransformer to {len(skewed)} features")
    except Exception:
        print("PowerTransformer failed, continuing without it.")

# Label-encoders for XGBoost (pre-fit)
label_encoders = {}
for c in cat_features:
    le = LabelEncoder()
    le.fit(pd.concat([X[c], X_test[c]]).astype(str))
    X[c] = le.transform(X[c].astype(str))
    X_test[c] = le.transform(X_test[c].astype(str))
    label_encoders[c] = le

X = X.replace([np.inf, -np.inf], -999).fillna(-999)
X_test = X_test.replace([np.inf, -np.inf], -999).fillna(-999)

# ---------------- TRAINING: CatBoost + XGBoost with cautious regularization ----------------
print("Training models with Stratified K-Fold CV...")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

cb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))
cb_test = np.zeros(len(X_test))
xgb_test = np.zeros(len(X_test))

cb_scores = []
xgb_scores = []

fold = 0
for train_idx, val_idx in skf.split(X, y):
    fold += 1
    print(f"\n--- Fold {fold}/{N_SPLITS} ---")
    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

    # CatBoost
    cb_params = dict(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=10,
        min_data_in_leaf=25,
        bagging_temperature=0.7,
        random_strength=0.6,
        eval_metric='AUC',
        random_seed=RANDOM_STATE + fold,
        early_stopping_rounds=150,
        verbose=0,
        thread_count=-1
    )
    cb = CatBoostClassifier(**cb_params)
    pool_tr = Pool(X_tr, y_tr, cat_features=[c for c in cat_features if c in X_tr.columns])
    pool_val = Pool(X_val, y_val, cat_features=[c for c in cat_features if c in X_tr.columns])
    cb.fit(pool_tr, eval_set=pool_val)
    cb_pred_val = cb.predict_proba(X_val)[:, 1]
    cb_oof[val_idx] = cb_pred_val
    cb_test += cb.predict_proba(X_test)[:, 1] / N_SPLITS
    cb_fold_auc = roc_auc_score(y_val, cb_pred_val)
    cb_scores.append(cb_fold_auc)
    print(f"CatBoost fold AUC: {cb_fold_auc:.5f} (trees: {cb.tree_count_})")

    # XGBoost
    X_tr_xgb, X_val_xgb, X_test_xgb = X_tr.copy(), X_val.copy(), X_test.copy()
    xgb = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=7,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.8,
        reg_lambda=1.2,
        gamma=0.05,
        random_state=RANDOM_STATE + fold,
        eval_metric='auc',
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0
    )

    # XGBoost early stopping compatibility: try sklearn param, else use callbacks
    ES = 150
    try:
        # Try typical sklearn wrapper param (works on recent xgboost versions)
        xgb.fit(
            X_tr_xgb, y_tr,
            eval_set=[(X_val_xgb, y_val)],
            early_stopping_rounds=ES,
            verbose=False
        )
    except TypeError:
        # Fallback: use xgboost callback early stopping (older/newer API)
        try:
            cb_ = xgb_lib.callback.EarlyStopping(rounds=ES, save_best=True)
            xgb.fit(
                X_tr_xgb, y_tr,
                eval_set=[(X_val_xgb, y_val)],
                callbacks=[cb_],
                verbose=False
            )
        except Exception:
            # Final fallback: fit without early stopping
            xgb.fit(X_tr_xgb, y_tr, verbose=False)

    xgb_pred_val = xgb.predict_proba(X_val_xgb)[:, 1]
    xgb_oof[val_idx] = xgb_pred_val
    xgb_test += xgb.predict_proba(X_test_xgb)[:, 1] / N_SPLITS
    xgb_fold_auc = roc_auc_score(y_val, xgb_pred_val)
    xgb_scores.append(xgb_fold_auc)
    best_it = getattr(xgb, "best_iteration", None)
    print(f"XGBoost fold AUC: {xgb_fold_auc:.5f} (best_it: {best_it})")

# OOF AUCs
cb_oof_auc = roc_auc_score(y, cb_oof)
xgb_oof_auc = roc_auc_score(y, xgb_oof)
print("\nModel OOF AUCs:")
print(f" CatBoost OOF AUC: {cb_oof_auc:.5f} (mean folds {np.mean(cb_scores):.5f})")
print(f" XGBoost OOF AUC:  {xgb_oof_auc:.5f} (mean folds {np.mean(xgb_scores):.5f})")

# ---------------- STACKING (regularized meta) ----------------
print("Training stacked meta-model (LogisticRegression) on OOF predictions...")

meta_X = np.vstack([cb_oof, xgb_oof]).T
meta_test = np.vstack([cb_test, xgb_test]).T

meta_clf = LogisticRegression(C=0.5, solver='lbfgs', max_iter=2000, random_state=RANDOM_STATE)
meta_clf.fit(meta_X, y)
meta_oof_pred = meta_clf.predict_proba(meta_X)[:, 1]
meta_oof_auc = roc_auc_score(y, meta_oof_pred)
print(f"Meta OOF AUC (before calibration): {meta_oof_auc:.5f}")

# Calibrate meta with sigmoid (Platt)
calibrator = CalibratedClassifierCV(estimator=meta_clf, cv='prefit', method='sigmoid')
calibrator.fit(meta_X, y)
final_test_preds = calibrator.predict_proba(meta_test)[:, 1]
final_oof_calibrated = calibrator.predict_proba(meta_X)[:, 1]
final_oof_auc = roc_auc_score(y, final_oof_calibrated)
print(f"Meta OOF AUC (after calibration): {final_oof_auc:.5f}")

# ---------------- CREATE SUBMISSION ----------------
submission = pd.DataFrame({ID_COL: test_ids, TARGET: final_test_preds})
submission[TARGET] = submission[TARGET].clip(0, 1)
out_name = f"HYBRID_reliable_god_top{TOP_K_GOD_FEATURES}_fold{N_SPLITS}_auc{final_oof_auc:.5f}.csv"
submission.to_csv(out_name, index=False)
print(f"Saved submission: {out_name}")
print(f"Submission stats: mean={submission[TARGET].mean():.5f} std={submission[TARGET].std():.5f}")

# ---------------- SAVE MODELS ----------------
print("Saving models...")
joblib.dump(cb, "model_catboost.pkl")
joblib.dump(xgb, "model_xgboost.pkl")
joblib.dump(meta_clf, "model_meta_lr.pkl")
joblib.dump(calibrator, "model_meta_calibrator.pkl")
print("Models saved: model_catboost.pkl, model_xgboost.pkl, model_meta_lr.pkl, model_meta_calibrator.pkl")

print("\nDone. Final OOF AUC (meta, calibrated):", final_oof_auc)
