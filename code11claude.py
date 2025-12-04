import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.utils import check_random_state
from scipy.stats import rankdata

import warnings
warnings.filterwarnings('ignore')

# ================================
# Configs
N_SPLITS = 7
SEED = 42
DATA_PATH = ""

# ================================
# Data Load
train = pd.read_csv(f"{DATA_PATH}train.csv")
test = pd.read_csv(f"{DATA_PATH}test.csv")

# ================================
# Feature Engineering Function
def create_features(df):
    df = df.copy()
    # Temporal
    df['Application Date'] = pd.to_datetime(df['Application Date'], format='%Y/%m/%d', errors='coerce')
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], format='%Y/%m/%d', errors='coerce')
    df['Age'] = (df['Application Date'] - df['Date of Birth']).dt.days / 365.25
    df['App_Month'] = df['Application Date'].dt.month
    df['App_DayOfWeek'] = df['Application Date'].dt.dayofweek
    df['App_Quarter'] = df['Application Date'].dt.quarter
    df['App_Hour'] = df['Application Time'] // 10000
    df['Is_Weekend'] = (df['App_DayOfWeek'] >= 5).astype(int)
    # Fraud/risk metrics & ratios
    df['Loan_Amount_Gap'] = df['Declared Amount of Unsecured Loans'] - df['Amount of Unsecured Loans']
    df['Abs_Amount_Gap'] = np.abs(df['Loan_Amount_Gap'])
    df['Loan_Count_Gap'] = df['Declared Number of Unsecured Loans'] - df['Number of Unsecured Loans']
    df['DTI_Total'] = (df['Amount of Unsecured Loans'] + df['Application Limit Amount(Desired)']) / (df['Total Annual Income'] + 1)
    df['Income_log'] = np.log1p(df['Total Annual Income'])
    df['Employment_Years'] = df['Duration of Employment at Company (Months)'] / 12
    df['Residence_Years'] = df['Duration of Residence (Months)'] / 12
    # Categorical processing
    df['JIS_Prefix_2'] = df['JIS Address Code'].fillna(-999).astype(str).str[:2]
    return df

train = create_features(train)
test = create_features(test)

# ================================
# Categorical features to encode (edit as needed)
cat_cols = [
    'Major Media Code', 'Internet Details', 'Reception Type Category',
    'Gender', 'Single/Married Status', 'Residence Type', 'Name Type',
    'Family Composition Type', 'Living Arrangement Type', 'Insurance Job Type',
    'Employment Type', 'Employment Status Type', 'Industry Type',
    'Company Size Category', 'JIS Address Code', 'JIS_Prefix_2', 
    'App_Month', 'App_DayOfWeek', 'App_Quarter'
]

cat_cols = [c for c in cat_cols if c in train.columns]

num_cols = [c for c in train.columns if c not in ['ID', 'Default 12 Flag', 'Application Date', 'Date of Birth'] + cat_cols]

# ================================
# ------------- KFold Target Encoding --------------
def add_target_encoding(train, test, cols, target_col, n_splits=7, seed=42):
    rng = check_random_state(seed)
    train = train.copy()
    test = test.copy()
    for col in cols:
        oof = np.zeros(len(train))
        test_col_mean = np.zeros(len(test))
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, val_idx in kf.split(train, train[target_col]):
            map_means = train.iloc[tr_idx].groupby(col)[target_col].mean()
            oof[val_idx] = train.iloc[val_idx][col].map(map_means).fillna(train[target_col].mean())
        train[f"{col}_te"] = oof
        # For test set, map using all train (no leakage)
        global_map = train.groupby(col)[target_col].mean()
        test_col_mean = test[col].map(global_map).fillna(train[target_col].mean())
        test[f"{col}_te"] = test_col_mean
    return train, test

target_enc_cols = ['JIS Address Code', 'Industry Type', 'Employment Type']
if 'JIS Address Code' in train.columns:
    train, test = add_target_encoding(train, test, target_enc_cols, 'Default 12 Flag', n_splits=N_SPLITS, seed=SEED)

# ================================
# Label Encoders for trees
def fit_label_encoding(train_df, test_df, categorical):
    encoders = {}
    for col in categorical:
        le = LabelEncoder()
        le.fit(pd.concat([train_df[col], test_df[col]]).astype(str))
        encoders[col] = le
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
    return train_df, test_df, encoders

train_xgb, test_xgb, label_encoders = fit_label_encoding(train.copy(), test.copy(), cat_cols)

# ================================
# Features, Targets, IDs
y = train['Default 12 Flag']
train_ids = train['ID']
test_ids = test['ID']
train_X = train.drop(['Default 12 Flag', 'ID', 'Application Date', 'Date of Birth'], axis=1)
test_X = test.drop(['ID', 'Application Date', 'Date of Birth'], axis=1)
train_X_xgb = train_xgb.drop(['Default 12 Flag', 'ID', 'Application Date', 'Date of Birth'], axis=1)
test_X_xgb = test_xgb.drop(['ID', 'Application Date', 'Date of Birth'], axis=1)

# ================================
# Cross-Validation Setup
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
cb_oof = np.zeros(len(train_X))
lgb_oof = np.zeros(len(train_X))
xgb_oof = np.zeros(len(train_X))

cb_preds = np.zeros(len(test_X))
lgb_preds = np.zeros(len(test_X))
xgb_preds = np.zeros(len(test_X))

print("\n========= CV Loop =========")
for fold, (tr_idx, va_idx) in enumerate(skf.split(train_X, y)):
    print(f"\n=== Fold {fold+1}/{N_SPLITS} ===")
    X_tr, X_val = train_X.iloc[tr_idx], train_X.iloc[va_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[va_idx]
    X_tr_xgb, X_val_xgb = train_X_xgb.iloc[tr_idx], train_X_xgb.iloc[va_idx]
    # CatBoost
    cb = CatBoostClassifier(
        iterations=3500,
        learning_rate=0.017,
        depth=8,
        l2_leaf_reg=12,
        min_data_in_leaf=40,
        random_seed=SEED+fold,
        eval_metric='AUC',
        early_stopping_rounds=120,
        cat_features=[X_tr.columns.get_loc(col) for col in cat_cols if col in X_tr.columns],
        verbose=0
    )
    cb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    cb_oof[va_idx] = cb.predict_proba(X_val)[:, 1]
    cb_preds += cb.predict_proba(test_X)[:, 1] / N_SPLITS
    # LightGBM
    lgb = LGBMClassifier(
        n_estimators=3500,
        learning_rate=0.017,
        max_depth=8,
        num_leaves=56,
        subsample=0.76,
        colsample_bytree=0.79,
        reg_alpha=0.65,
        reg_lambda=0.65,
        random_state=SEED+fold,
        n_jobs=-1
    )
    X_tr_lgb, X_val_lgb = X_tr.copy(), X_val.copy()
    for c in cat_cols: X_tr_lgb[c] = X_tr_lgb[c].astype("category"); X_val_lgb[c] = X_val_lgb[c].astype("category")
    lgb.fit(X_tr_lgb, y_tr, eval_set=[(X_val_lgb, y_val)], eval_metric='auc', early_stopping_rounds=120, verbose=0)
    lgb_oof[va_idx] = lgb.predict_proba(X_val_lgb)[:, 1]
    lgb_preds += lgb.predict_proba(test_X)[:, 1]/ N_SPLITS
    # XGBoost
    xgb = XGBClassifier(
        n_estimators=3500,
        learning_rate=0.017,
        max_depth=8,
        min_child_weight=11,
        subsample=0.79,
        colsample_bytree=0.78,
        reg_alpha=1.11,
        reg_lambda=1.11,
        gamma=0.1,
        random_state=SEED+fold,
        eval_metric='auc',
        early_stopping_rounds=120,
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_val_xgb, y_val)], verbose=False)
    xgb_oof[va_idx] = xgb.predict_proba(X_val_xgb)[:, 1]
    xgb_preds += xgb.predict_proba(test_X_xgb)[:, 1]/ N_SPLITS

# ================================
# Blending (Meta-Stack)
stack_X = np.column_stack([cb_oof, lgb_oof, xgb_oof])
stack_X_test = np.column_stack([cb_preds, lgb_preds, xgb_preds])
meta = LogisticRegression(max_iter=1000)
meta.fit(stack_X, y)
meta_preds = meta.predict_proba(stack_X_test)[:,1]

final_preds = 0.67*cb_preds + 0.13*lgb_preds + 0.13*xgb_preds + 0.07*meta_preds

oof_auc = roc_auc_score(y, final_preds[rankdata(stack_X.mean(axis=1), method="average").astype(int)-1])
print(f"\nFinal Out-of-Fold ROC AUC: {oof_auc:.5f}")

# ================================
# Submission
sub = pd.DataFrame({'ID': test_ids, 'Default 12 Flag': np.clip(final_preds, 0, 1)})
sub.to_csv('ultimate_levelup_submission.csv', index=False)
print("Submission saved as 'ultimate_levelup_submission.csv'")
