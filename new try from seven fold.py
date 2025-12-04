# Ultimate upgraded pipeline (v10.0) - Feature selection, stacking, calibration, clustering
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ----------------- USER CONFIG -----------------
N_SPLITS = 7                 # use 7 folds for stability (change if slow)
RANDOM_STATE = 42
DATA_PATH = ""               # path prefix, e.g., "data/"
KEEP_TOP_RATIO = 0.70        # after importance scoring keep top 70% features (tuneable)
CALIBRATION_CV = 3           # cv for calibration step
CLUSTERS = 10                # KMeans clusters for financial features
# ------------------------------------------------

print("="*80)
print("🔥🔥🔥 ULTIMATE RELIABLE PIPELINE (v10.0) 🔥🔥🔥")
print("="*80)

# ----------------- STEP 1: LOAD DATA -----------------
print("\n[STEP 1] Loading data...")
train_df = pd.read_csv(f"{DATA_PATH}train.csv")
test_df = pd.read_csv(f"{DATA_PATH}test.csv")
print(f"✓ Train: {train_df.shape}, Test: {test_df.shape}")
print(f"✓ Default rate (train): {train_df['Default 12 Flag'].mean():.4f}")

# ----------------- STEP 2: FEATURE ENGINEERING (safe features) -----------------
print("\n[STEP 2] Feature engineering (safe set + JIS + clusters later)...")
def create_safe_features(df):
    df = df.copy()
    # JIS cleaning
    df['JIS_str'] = df['JIS Address Code'].fillna(-999).astype(str)
    df['JIS_Prefix_2'] = df['JIS_str'].str[:2]
    df['JIS_Prefix_3'] = df['JIS_str'].str[:3]
    # Dates
    df['Application Date'] = pd.to_datetime(df['Application Date'], format='%Y/%m/%d', errors='coerce')
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], format='%Y/%m/%d', errors='coerce')
    df['Age'] = (df['Application Date'] - df['Date of Birth']).dt.days / 365.25
    df['Age'] = df['Age'].replace([np.inf, -np.inf], np.nan)
    df['App_Month'] = df['Application Date'].dt.month
    df['App_DayOfWeek'] = df['Application Date'].dt.dayofweek
    df['App_Quarter'] = df['Application Date'].dt.quarter
    df['App_Hour'] = df['Application Time'] // 10000
    # Time flags
    df['Is_Weekend'] = (df['App_DayOfWeek'] >= 5).astype(int)
    df['Is_BusinessHours'] = ((df['App_Hour'] >= 9) & (df['App_Hour'] <= 17)).astype(int)
    df['Is_LateNight'] = ((df['App_Hour'] >= 22) | (df['App_Hour'] <= 5)).astype(int)
    # Fraud / honesty
    df['Loan_Amount_Gap'] = df['Declared Amount of Unsecured Loans'] - df['Amount of Unsecured Loans']
    df['Loan_Count_Gap'] = df['Declared Number of Unsecured Loans'] - df['Number of Unsecured Loans']
    df['Abs_Amount_Gap'] = df['Loan_Amount_Gap'].abs()
    df['Hidden_Loans'] = (df['Loan_Count_Gap'] < 0).astype(int)
    df['Severe_Underreporting'] = ((df['Loan_Amount_Gap'] < 0) & (df['Abs_Amount_Gap'] >= 200000)).astype(int)
    df['Honesty_Score'] = np.where(
        df['Declared Amount of Unsecured Loans'] > 0,
        1 - np.clip(df['Abs_Amount_Gap'] / (df['Declared Amount of Unsecured Loans'] + 1), 0, 1),
        1
    )
    df['Perfect_Match'] = ((df['Loan_Amount_Gap'] == 0) & (df['Loan_Count_Gap'] == 0)).astype(int)
    # Financials
    df['Income_log'] = np.log1p(df['Total Annual Income'].fillna(0))
    df['Existing_Loan_log'] = np.log1p(df['Amount of Unsecured Loans'].fillna(0))
    df['Desired_Loan_log'] = np.log1p(df['Application Limit Amount(Desired)'].fillna(0))
    df['DTI_Total'] = (df['Amount of Unsecured Loans'].fillna(0) + df['Application Limit Amount(Desired)'].fillna(0)) / (df['Total Annual Income'].fillna(0) + 1)
    df['DTI_WithRent_Annual'] = (df['Amount of Unsecured Loans'].fillna(0) + df['Rent Burden Amount'].fillna(0) * 12) / (df['Total Annual Income'].fillna(0) + 1)
    df['Income_per_Dependent'] = df['Total Annual Income'].fillna(0) / (df['Number of Dependents'].fillna(0) + 1)
    df['Avg_Existing_Loan'] = df['Amount of Unsecured Loans'].fillna(0) / (df['Number of Unsecured Loans'].fillna(0) + 1)
    df['Loan_Intensity'] = df['Number of Unsecured Loans'].fillna(0) / (df['Age'].fillna(0) + 1)
    # thresholds
    df['DTI_Critical'] = (df['DTI_Total'] > 0.6).astype(int)
    df['Has_Many_Loans'] = (df['Number of Unsecured Loans'].fillna(0) >= 3).astype(int)
    df['Loan_Free'] = (df['Number of Unsecured Loans'].fillna(0) == 0).astype(int)
    # Stability
    df['Employment_Years'] = df['Duration of Employment at Company (Months)'].fillna(0) / 12
    df['Residence_Years'] = df['Duration of Residence (Months)'].fillna(0) / 12
    df['Employment_to_Age'] = df['Employment_Years'] / (df['Age'].fillna(0) + 1)
    df['Combined_Stability'] = (df['Employment_Years'] + df['Residence_Years']) / 2
    df['Is_New_Job'] = (df['Employment_Years'] <= 1).astype(int)
    # Housing / employment / family
    df['Is_Homeowner'] = df['Residence Type'].isin([1,2,8,9]).astype(int)
    df['Is_Renter'] = df['Residence Type'].isin([4,5,6,7]).astype(int)
    df['Is_Regular_Employee'] = (df['Employment Status Type'] == 1).astype(int)
    df['Is_Public_Sector'] = (df['Company Size Category'] == 1).astype(int)
    df['Is_Large_Company'] = df['Company Size Category'].isin([1,2,3,4]).astype(int)
    df['Is_Part_Time'] = (df['Employment Type'] == 4).astype(int)
    df['Is_Married'] = (df['Single/Married Status'] == 2).astype(int)
    df['Has_Dependents'] = (df['Number of Dependents'].fillna(0) > 0).astype(int)
    df['Large_Family'] = (df['Number of Dependents'].fillna(0) >= 3).astype(int)
    # Age
    df['Age_Squared'] = df['Age'].fillna(0) ** 2
    df['Is_Very_Young'] = (df['Age'].fillna(0) < 25).astype(int)
    # Interactions
    df['Age_Income'] = df['Age'].fillna(0) * df['Income_log']
    df['Age_DTI'] = df['Age'].fillna(0) * df['DTI_Total'].fillna(0)
    df['Income_Dependents'] = df['Income_log'] * (df['Number of Dependents'].fillna(0) + 1)
    # Composite simple scores
    df['Financial_Risk_Score'] = (df['DTI_Critical'] * 4 + (df['DTI_Total'] > 0.4).astype(int) * 2 + df['Has_Many_Loans'] * 2)
    df['Stability_Risk_Score'] = (df['Is_New_Job'] * 2 + df['Is_Part_Time'] * 2 + (1 - df['Is_Homeowner']) * 1)
    df['Net_Risk_Score'] = df['Financial_Risk_Score'] + df['Stability_Risk_Score'] - (df['Is_Large_Company'] * 2)
    # Drop raw dates helper
    df = df.drop(columns=['Application Date','Date of Birth','JIS_str'], errors='ignore')
    return df

train_features = create_safe_features(train_df)
test_features = create_safe_features(test_df)
print(f"✓ features created. Train feature columns: {train_features.shape[1]}")

# ----------------- STEP 3: KMEANS CLUSTERING (financial segments) -----------------
print("\n[STEP 3] Creating KMeans clusters over financial features...")
fin_feats = ['Total Annual Income','Amount of Unsecured Loans','Declared Amount of Unsecured Loans','Application Limit Amount(Desired)','Rent Burden Amount']
fin_feats = [f for f in fin_feats if f in train_features.columns]
# fillna temporarily for clustering
clust_df = pd.concat([train_features[fin_feats].fillna(0), test_features[fin_feats].fillna(0)], axis=0)
# small check: if all zeros for a column, remove
clust_df = clust_df.loc[:, (clust_df != 0).any(axis=0)]
if clust_df.shape[1] == 0:
    print("→ No financial features suitable for clustering.")
else:
    k = min(CLUSTERS, max(2, int(np.sqrt(len(clust_df))//1)))
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    clust_labels = kmeans.fit_predict(clust_df.fillna(0))
    train_features['fin_cluster'] = clust_labels[:len(train_features)]
    test_features['fin_cluster'] = clust_labels[len(train_features):]
    print(f"✓ KMeans clusters created: {k} clusters (feature 'fin_cluster')")

# ----------------- STEP 4: ADVERSARIAL VALIDATION -----------------
print("\n[STEP 4] Adversarial validation (helps detect distribution shift + feature av_score)")
av_X = pd.concat([train_features.drop(columns=['Default 12 Flag','ID'], errors='ignore'), test_features.drop(columns=['ID'], errors='ignore')], axis=0, ignore_index=True)
av_y = np.array([0]*len(train_features) + [1]*len(test_features))
# simple cat list for AV
cat_features_av = [c for c in ['Major Media Code','Internet Details','Reception Type Category','Gender','Single/Married Status','Residence Type','Name Type','Family Composition Type','Living Arrangement Type','Insurance Job Type','Employment Type','Employment Status Type','Industry Type','Company Size Category','JIS Address Code','JIS_Prefix_2','JIS_Prefix_3','fin_cluster'] if c in av_X.columns]
for c in cat_features_av:
    av_X[c] = av_X[c].fillna(-999).astype(str).astype('category')
numeric_av = [c for c in av_X.columns if c not in cat_features_av]
av_X[numeric_av] = av_X[numeric_av].fillna(-999)
av_model = LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
av_model.fit(av_X, av_y, callbacks=[log_evaluation(0)])
av_preds = av_model.predict_proba(av_X)[:,1]
av_auc = roc_auc_score(av_y, av_preds)
print(f"✓ Adversarial Validation AUC: {av_auc:.5f}")
train_features['av_score'] = av_preds[:len(train_features)]
test_features['av_score'] = av_preds[len(train_features):]

# ----------------- STEP 5: PREPARE MODELING MATRICES -----------------
print("\n[STEP 5] Preparing modeling matrices, handling missing, encoding categories...")

y = train_features['Default 12 Flag']
X = train_features.drop(columns=['Default 12 Flag','ID'], errors='ignore').copy()
X_test = test_features.drop(columns=['ID'], errors='ignore').copy()
# maintain same columns
X_test = X_test.reindex(columns=X.columns, fill_value=np.nan)

# categorical final list
cat_features = [c for c in ['Major Media Code','Internet Details','Reception Type Category','Gender','Single/Married Status','Residence Type','Name Type','Family Composition Type','Living Arrangement Type','Insurance Job Type','Employment Type','Employment Status Type','Industry Type','Company Size Category','JIS Address Code','App_Month','App_DayOfWeek','App_Quarter','JIS_Prefix_2','JIS_Prefix_3','fin_cluster'] if c in X.columns]

# fill missing
for c in cat_features:
    X[c] = X[c].fillna(-999).astype(str)
    X_test[c] = X_test[c].fillna(-999).astype(str)
numeric_cols = [c for c in X.columns if c not in cat_features]
X[numeric_cols] = X[numeric_cols].fillna(-999)
X_test[numeric_cols] = X_test[numeric_cols].fillna(-999)
X = X.replace([np.inf, -np.inf], -999)
X_test = X_test.replace([np.inf, -np.inf], -999)

print(f"✓ Feature count: {X.shape[1]} (categorical: {len(cat_features)}, numeric: {len(numeric_cols)})")

# ----------------- STEP 6: POWER TRANSFORM SKEWED NUMERICS -----------------
print("\n[STEP 6] Power-transforming skewed numeric features...")
skewed_features = [f for f in ['Total Annual Income','Amount of Unsecured Loans','Application Limit Amount(Desired)','Rent Burden Amount'] if f in numeric_cols]
if skewed_features:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    X_sk = X[skewed_features].replace(-999, 0)
    Xt_sk = X_test[skewed_features].replace(-999, 0)
    Xt = pt.fit_transform(X_sk)
    Xt_test = pt.transform(Xt_sk)
    for i,f in enumerate(skewed_features):
        X[f + '_power'] = Xt[:,i]
        X_test[f + '_power'] = Xt_test[:,i]
    # extend numeric_cols
    numeric_cols.extend([f + '_power' for f in skewed_features])
    print(f"  ✓ transformed: {skewed_features}")

# ----------------- STEP 7: FIRST-PASS FEATURE IMPORTANCE (CatBoost) -----------------
# We'll get per-fold CatBoost importances then keep top features (importance-based selection)
print("\n[STEP 7] First-pass CatBoost importances (to select top features)...")
first_fold = 3  # quick lower it if too slow; used for verbosity control
skf_imp = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
feat_importance_df = pd.DataFrame({'feature': X.columns})
feat_importance_df.set_index('feature', inplace=True)

cb_imp_oof = np.zeros(len(X))
fold_count = 0
for fold, (tr_idx, val_idx) in enumerate(skf_imp.split(X, y)):
    fold_count += 1
    X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx].copy()
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    train_pool = Pool(X_tr, y_tr, cat_features=[c for c in cat_features if c in X_tr.columns])
    val_pool = Pool(X_val, y_val, cat_features=[c for c in cat_features if c in X_val.columns])
    cb = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=10,
        random_seed=RANDOM_STATE + fold,
        early_stopping_rounds=100,
        verbose=0
    )
    cb.fit(train_pool, eval_set=val_pool)
    cb_imp_oof[val_idx] = cb.predict_proba(X_val)[:,1]
    imp = cb.get_feature_importance(train_pool)
    # join
    feat_importance_df[f'fold_{fold+1}'] = imp
    print(f"  Fold {fold+1} CatBoost AUC: {roc_auc_score(y_val, cb_imp_oof[val_idx]):.6f}")

# aggregate importance
feat_importance_df['mean_importance'] = feat_importance_df.mean(axis=1)
feat_importance_df = feat_importance_df.sort_values('mean_importance', ascending=False)
print("Top 10 features by CatBoost importance (first-pass):")
print(feat_importance_df['mean_importance'].head(10))

# decide threshold: keep top KEEP_TOP_RATIO fraction
num_keep = max(50, int(len(feat_importance_df) * KEEP_TOP_RATIO))
keep_features = feat_importance_df.index[:num_keep].tolist()
print(f"✓ Keeping top {num_keep} features ({KEEP_TOP_RATIO*100:.0f}% or min50) for final training")

# Prune X and X_test to keep_features
X_sel = X[keep_features].copy()
X_test_sel = X_test[keep_features].copy()

# Update categorical and numeric lists for selected features
cat_features_sel = [c for c in cat_features if c in X_sel.columns]
numeric_cols_sel = [c for c in X_sel.columns if c not in cat_features_sel]
print(f"✓ Selected features: {len(X_sel.columns)} (cat: {len(cat_features_sel)}, num: {len(numeric_cols_sel)})")

# ----------------- STEP 8: LABEL ENCODERS for XGBoost -----------------
print("\n[STEP 8] Preparing label encoders for XGBoost...")
label_encoders = {}
for col in cat_features_sel:
    le = LabelEncoder()
    le.fit(pd.concat([X_sel[col], X_test_sel[col]]).astype(str).fillna('-999'))
    label_encoders[col] = le
    # transform now to consistent types for models that need numeric labels later if needed
    X_sel[col] = X_sel[col].astype(str)
    X_test_sel[col] = X_test_sel[col].astype(str)

# ----------------- STEP 9: L1 TRAINING (base models) - CB, XGB, LGB -----------------
print("\n[STEP 9] Training base models in CV (CatBoost, XGBoost, LightGBM). Collect OOF preds for stacking.")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# OOF and test predictions for each base model
cb_oof = np.zeros(len(X_sel))
xgb_oof = np.zeros(len(X_sel))
lgb_oof = np.zeros(len(X_sel))

cb_test = np.zeros(len(X_test_sel))
xgb_test = np.zeros(len(X_test_sel))
lgb_test = np.zeros(len(X_test_sel))

# optionally track model importances
cb_importances = []
lgb_importances = []
xgb_importances = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, y)):
    print(f"\n--- FOLD {fold+1}/{N_SPLITS} ---")
    X_tr, y_tr = X_sel.iloc[tr_idx].copy(), y.iloc[tr_idx].copy()
    X_val, y_val = X_sel.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    # ---------------- CatBoost ----------------
    print("Training CatBoost (regularized + bagging)...")
    train_pool = Pool(X_tr, y_tr, cat_features=[c for c in cat_features_sel if c in X_tr.columns])
    val_pool = Pool(X_val, y_val, cat_features=[c for c in cat_features_sel if c in X_val.columns])
    cb = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=20,
        bagging_temperature=0.8,
        random_strength=1.0,
        eval_metric='AUC',
        random_seed=RANDOM_STATE + fold,
        early_stopping_rounds=200,
        verbose=0,
        thread_count=-1
    )
    cb.fit(train_pool, eval_set=val_pool)
    cb_val_pred = cb.predict_proba(X_val)[:,1]
    cb_oof[val_idx] = cb_val_pred
    cb_test += cb.predict_proba(X_test_sel)[:,1] / N_SPLITS
    cb_score = roc_auc_score(y_val, cb_val_pred)
    cb_importances.append(pd.Series(cb.get_feature_importance(train_pool), index=X_sel.columns))
    print(f"CatBoost fold {fold+1} AUC: {cb_score:.6f} (trees: {cb.tree_count_})")
    # ---------------- LightGBM ----------------
    print("Training LightGBM (robust)...")
    X_tr_lgb = X_tr.copy(); X_val_lgb = X_val.copy(); X_test_lgb = X_test_sel.copy()
    for c in cat_features_sel:
        if c in X_tr_lgb.columns:
            X_tr_lgb[c] = X_tr_lgb[c].astype('category')
            X_val_lgb[c] = X_val_lgb[c].astype('category')
            X_test_lgb[c] = X_test_lgb[c].astype('category')
    lgb = LGBMClassifier(
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=9,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbosity=-1
    )
    lgb.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_val_lgb, y_val)],
        eval_metric='auc',
        callbacks=[early_stopping(200), log_evaluation(0)]
    )
    lgb_val_pred = lgb.predict_proba(X_val_lgb)[:,1]
    lgb_oof[val_idx] = lgb_val_pred
    lgb_test += lgb.predict_proba(X_test_lgb)[:,1] / N_SPLITS
    lgb_score = roc_auc_score(y_val, lgb_val_pred)
    lgb_importances.append(pd.Series(lgb.feature_importances_, index=X_sel.columns))
    print(f"LightGBM fold {fold+1} AUC: {lgb_score:.6f}")
    # ---------------- XGBoost ----------------
    print("Training XGBoost (regularized)...")
    X_tr_xgb = X_tr.copy(); X_val_xgb = X_val.copy(); X_test_xgb = X_test_sel.copy()
    for c in cat_features_sel:
        if c in X_tr_xgb.columns:
            le = label_encoders[c]
            X_tr_xgb[c] = le.transform(X_tr_xgb[c].astype(str))
            X_val_xgb[c] = le.transform(X_val_xgb[c].astype(str))
            X_test_xgb[c] = le.transform(X_test_xgb[c].astype(str))
    xgb = XGBClassifier(
        n_estimators=4000,
        learning_rate=0.02,
        max_depth=7,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        gamma=0.1,
        random_state=RANDOM_STATE + fold,
        use_label_encoder=False,
        eval_metric='auc',
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_val_xgb, y_val)], verbose=False)
    xgb_val_pred = xgb.predict_proba(X_val_xgb)[:,1]
    xgb_oof[val_idx] = xgb_val_pred
    xgb_test += xgb.predict_proba(X_test_xgb)[:,1] / N_SPLITS
    try:
        xgb_imp = pd.Series(xgb.feature_importances_, index=X_sel.columns)
    except:
        xgb_imp = pd.Series(0, index=X_sel.columns)
    xgb_importances.append(xgb_imp)
    xgb_score = roc_auc_score(y_val, xgb_val_pred)
    print(f"XGBoost fold {fold+1} AUC: {xgb_score:.6f}")

# Report base model oof AUCs
print("\nBase model OOF AUCs:")
print(f"  CatBoost OOF AUC: {roc_auc_score(y, cb_oof):.6f}")
print(f"  LightGBM OOF AUC: {roc_auc_score(y, lgb_oof):.6f}")
print(f"  XGBoost OOF AUC: {roc_auc_score(y, xgb_oof):.6f}")

# ----------------- STEP 10: STACKING (meta-model) -----------------
print("\n[STEP 10] Building meta-model (logistic) on base OOF preds + few strong features")
# choose a handful of original high-signal features to include in meta (e.g., Net_Risk_Score, DTI_Total, av_score)
meta_features_to_add = []
for cand in ['Net_Risk_Score','DTI_Total','av_score','Income_log','Is_Homeowner']:
    if cand in X_sel.columns:
        meta_features_to_add.append(cand)
print(f"Meta will use additional features: {meta_features_to_add}")

# Build meta training set from OOF preds
meta_X = np.vstack([cb_oof, lgb_oof, xgb_oof]).T
if len(meta_features_to_add) > 0:
    meta_X = np.hstack([meta_X, X_sel[meta_features_to_add].values])
# Also prepare meta test set
meta_test = np.vstack([cb_test, lgb_test, xgb_test]).T
if len(meta_features_to_add) > 0:
    meta_test = np.hstack([meta_test, X_test_sel[meta_features_to_add].values])

# scale meta features (important for LR)
scaler_meta = StandardScaler()
meta_X = scaler_meta.fit_transform(meta_X)
meta_test = scaler_meta.transform(meta_test)

# Fit a logistic regression meta-model with calibration
print("Training meta-model (LogisticRegression)...")
meta_base = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0, random_state=RANDOM_STATE)
# We will use CalibratedClassifierCV with isotonic calibration for better probability scaling
calibrator = CalibratedClassifierCV(estimator=meta_base, method='isotonic', cv=CALIBRATION_CV)
calibrator.fit(meta_X, y)
meta_oof_pred = calibrator.predict_proba(meta_X)[:,1]
meta_test_pred = calibrator.predict_proba(meta_test)[:,1]

meta_oof_auc = roc_auc_score(y, meta_oof_pred)
print(f"✓ Meta-model OOF AUC (after calibration): {meta_oof_auc:.6f}")

# ----------------- STEP 11: FINAL ENSEMBLE & SUBMISSION -----------------
print("\n[STEP 11] Final ensemble & submission creation")
# We can combine meta_test_pred with a small direct average of the best base (CatBoost) for robustness
final_pred = meta_test_pred  # meta calibrated probabilities
# Optionally blend small CatBoost weight for stability
final_pred = 0.85*final_pred + 0.15*cb_test

# clip
final_pred = np.clip(final_pred, 0, 1)

# Create submission
submission = pd.DataFrame({'ID': test_df['ID'], 'Default 12 Flag': final_pred})
filename = f'submission_v10_meta_calibrated_auc_est.csv'
submission.to_csv(filename, index=False)
print(f"✅ Submission saved as: {filename}")
print(f"  mean: {final_pred.mean():.6f} | std: {final_pred.std():.6f}")
print("\nAll done. Submit the generated CSV to the competition.")

# ----------------- END -----------------
print("\n" + "="*80)
print("🏁 ULTIMATE PIPELINE v10.0 COMPLETED")
print("="*80)
