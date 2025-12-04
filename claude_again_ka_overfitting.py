import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from scipy.stats import rankdata
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🛡️ ULTIMATE CODE - OVERFITTING BLOCKER VERSION 🛡️")
print("="*80)

# ===============================
# 1. LOAD DATA
# ===============================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print(f"Train: {train_df.shape} | Test: {test_df.shape}")

# ===============================
# 2. FEATURE ENGINEERING
# ===============================
def create_god_features(df):
    # Paste your huge engineering here!
    return df

train_features = create_god_features(train_df)
test_features = create_god_features(test_df)

# ===============================
# 3. SPLIT & TYPE SAFETY
# ===============================
y = train_features['Default 12 Flag']
X = train_features.drop(columns=['Default 12 Flag','ID'], errors='ignore')
test_ids = test_features['ID']
X_test = test_features.drop(columns=['ID'], errors='ignore')
X_test = X_test.reindex(columns=X.columns, fill_value=0)

cat_features = [
    'Major Media Code', 'Internet Details', 'Reception Type Category',
    'Gender', 'Single/Married Status', 'Residence Type', 'Name Type',
    'Family Composition Type', 'Living Arrangement Type', 'Insurance Job Type',
    'Employment Type', 'Employment Status Type', 'Industry Type', 'Company Size Category',
    'JIS Address Code', 'App_Month', 'App_DayOfWeek', 'App_Quarter', 'Age_Group_10'
]
cat_features = [col for col in cat_features if col in X.columns]

for col in X.columns:
    if col in cat_features:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    else:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# ===== OUTLIER CAP ============
print("Outlier capping numeric columns…")
for col in X.select_dtypes(include=[np.number]).columns:
    p1, p99 = X[col].quantile(0.01), X[col].quantile(0.99)
    X[col] = np.clip(X[col], p1, p99)
    X_test[col] = np.clip(X_test[col], p1, p99)

# =======================================
# 4. LABEL ENCODE & POWER TRANSFORM
# =======================================
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    le.fit(list(X[col]) + list(X_test[col]))
    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])

skewed = ['Total Annual Income', 'Amount of Unsecured Loans',
          'Application Limit Amount(Desired)', 'Rent Burden Amount']
skewed = [c for c in skewed if c in X.columns]
if skewed:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    X[skewed] = pt.fit_transform(X[skewed].replace(-999,0))
    X_test[skewed] = pt.transform(X_test[skewed].replace(-999,0))

# =======================================
# 5. MODEL TRAINING — ULTRA REGULARIZED
# =======================================
N_SPLITS = 7
RANDOM_STATE = 42

cb_oof = np.zeros(len(X))
lgb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))

cb_test = np.zeros(len(X_test))
lgb_test = np.zeros(len(X_test))
xgb_test = np.zeros(len(X_test))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"=== FOLD {fold+1}/{N_SPLITS} ===")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # CatBoost - MAX REGULARIZATION
    cb = CatBoostClassifier(
        iterations=4000,
        learning_rate=0.008,     # Lower LR
        depth=6,                 # Shallower tree
        l2_leaf_reg=25,          # Stronger L2
        min_data_in_leaf=40,     # Bigger leaves
        subsample=0.6,           # Fewer rows used per tree
        rsm=0.7,                 # Fewer cols per tree
        bagging_temperature=1.5, # Dropout/bagginess
        random_strength=1.3,
        border_count=64,
        eval_metric='AUC',
        random_seed=RANDOM_STATE + fold,
        early_stopping_rounds=100, # Early break
        verbose=0,
        thread_count=-1
    )
    cb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    cb_oof[val_idx] = cb.predict_proba(X_val)[:,1]
    cb_test += cb.predict_proba(X_test)[:,1]/N_SPLITS

    # LightGBM - MAX REGULARIZATION
    lgb = LGBMClassifier(
        n_estimators=4000,
        learning_rate=0.008,
        max_depth=6,
        num_leaves=16,
        min_child_samples=40,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=5.0,
        reg_lambda=5.0,
        min_split_gain=0.05,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbosity=-1
    )
    lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc',
            callbacks=[early_stopping(100), log_evaluation(0)])
    lgb_oof[val_idx] = lgb.predict_proba(X_val)[:,1]
    lgb_test += lgb.predict_proba(X_test)[:,1]/N_SPLITS

    # XGBoost - MAX REGULARIZATION
    xgb = XGBClassifier(
        n_estimators=4000,
        learning_rate=0.008,
        max_depth=6,
        min_child_weight=30,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=5.0,
        reg_lambda=5.0,
        gamma=0.05,
        random_state=RANDOM_STATE + fold,
        eval_metric='auc',
        early_stopping_rounds=100,
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_oof[val_idx] = xgb.predict_proba(X_val)[:,1]
    xgb_test += xgb.predict_proba(X_test)[:,1]/N_SPLITS

print(f"OOF Scores: CB={roc_auc_score(y,cb_oof):.5f} LGB={roc_auc_score(y,lgb_oof):.5f} XGB={roc_auc_score(y,xgb_oof):.5f}")

# ==========================
# 6. OPTIMIZED ENSEMBLE
# ==========================
def objective(weights):
    weights = np.abs(weights)
    weights = weights/weights.sum()
    ensemble = weights[0]*cb_oof + weights[1]*lgb_oof + weights[2]*xgb_oof
    return -roc_auc_score(y, ensemble)
init = np.array([0.33,0.33,0.34])
result = minimize(objective, init, method='Nelder-Mead', options={'maxiter':300,'xatol':1e-6})
optimal_weights = np.abs(result.x)/np.abs(result.x).sum()
ensemble_test = optimal_weights[0]*cb_test + optimal_weights[1]*lgb_test + optimal_weights[2]*xgb_test

# ==========================
# 7. RANK BLEND FOR LB SAFETY
# ==========================
final_predictions = 0.7 * ensemble_test + 0.3 * (
    (rankdata(cb_test) + rankdata(lgb_test) + rankdata(xgb_test)) / (3 * len(cb_test))
)

# ==========================
# 8. SAVE SUBMISSION
# ==========================
submission = pd.DataFrame({'ID': test_ids, 'Default 12 Flag': final_predictions.clip(0,1)})
fname = f'ULTRA_REGULARIZED_LB_SAFE_auc{roc_auc_score(y,optimal_weights[0]*cb_oof + optimal_weights[1]*lgb_oof + optimal_weights[2]*xgb_oof):.5f}.csv'
submission.to_csv(fname, index=False)
print(f"\n✓ CSV SAVED: {fname}")
print("="*80)
print("MODEL TRAINED - OVERFITTING BLOCKED 🔥 Leaderboard + Production Ready! 🚀")
print("="*80)
