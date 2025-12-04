
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, log_evaluation

warnings.filterwarnings("ignore")

print("="*90)
print("🔥🔥🔥 RELIABLE STACKED ENSEMBLE (v10) — 7-Fold | CB + XGB + LGBM + Meta-LR 🔥🔥🔥")
print("="*90)

# ============================================================================
# CONFIG
# ============================================================================
N_SPLITS = 7                # 7 folds for a more stable estimate
RANDOM_STATE = 42
DATA_PATH = ""              # e.g. "data/"
OUTPUT_PREFIX = "RELIABLE_STACK_v10"

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/9] Loading data...")
train_df = pd.read_csv(f"{DATA_PATH}train.csv")
test_df  = pd.read_csv(f"{DATA_PATH}test.csv")
print(f"✓ Train: {train_df.shape} | Test: {test_df.shape}")
if "Default 12 Flag" in train_df.columns:
    print(f"✓ Default rate: {train_df['Default 12 Flag'].mean():.4f}")
else:
    raise ValueError("Target column 'Default 12 Flag' not found in train.csv")

# ============================================================================
# STEP 2: SAFE FEATURE ENGINEERING (same core as v9, extended-safe)
# ============================================================================
print("\n[2/9] Creating SAFE features (~120+)...")

def create_safe_features(df):
    df = df.copy()

    # --- JIS ADDRESS CODE CLEANING ---
    df['JIS_str'] = df['JIS Address Code'].fillna(-999).astype(str)
    df['JIS_Prefix_2'] = df['JIS_str'].str[:2]  # State
    df['JIS_Prefix_3'] = df['JIS_str'].str[:3]  # District

    # --- TEMPORAL FEATURES ---
    df['Application Date'] = pd.to_datetime(df['Application Date'], format='%Y/%m/%d', errors='coerce')
    df['Date of Birth']    = pd.to_datetime(df['Date of Birth'],    format='%Y/%m/%d', errors='coerce')
    df['Age'] = (df['Application Date'] - df['Date of Birth']).dt.days / 365.25
    df['App_Month']    = df['Application Date'].dt.month
    df['App_DayOfWeek']= df['Application Date'].dt.dayofweek
    df['App_Quarter']  = df['Application Date'].dt.quarter

    df['App_Hour'] = (df['Application Time'] // 10000).astype(float)

    df['Is_Weekend']       = (df['App_DayOfWeek'] >= 5).astype(int)
    df['Is_BusinessHours'] = ((df['App_Hour'] >= 9) & (df['App_Hour'] <= 17)).astype(int)
    df['Is_LateNight']     = ((df['App_Hour'] >= 22) | (df['App_Hour'] <= 5)).astype(int)

    # --- CONSISTENCY / FRAUD SIGNALS ---
    df['Loan_Amount_Gap'] = df['Declared Amount of Unsecured Loans'] - df['Amount of Unsecured Loans']
    df['Loan_Count_Gap']  = df['Declared Number of Unsecured Loans'] - df['Number of Unsecured Loans']
    df['Abs_Amount_Gap']  = np.abs(df['Loan_Amount_Gap'])

    df['Hidden_Loans']          = (df['Loan_Count_Gap'] < 0).astype(int)
    df['Severe_Underreporting'] = ((df['Loan_Amount_Gap'] < 0) & (df['Abs_Amount_Gap'] >= 200000)).astype(int)

    df['Honesty_Score'] = np.where(
        df['Declared Amount of Unsecured Loans'] > 0,
        1 - np.clip(df['Abs_Amount_Gap'] / (df['Declared Amount of Unsecured Loans'] + 1), 0, 1),
        1
    )
    df['Perfect_Match'] = ((df['Loan_Amount_Gap'] == 0) & (df['Loan_Count_Gap'] == 0)).astype(int)

    # --- FINANCIAL HEALTH ---
    df['Income_log']         = np.log1p(df['Total Annual Income'])
    df['Existing_Loan_log']  = np.log1p(df['Amount of Unsecured Loans'])
    df['Desired_Loan_log']   = np.log1p(df['Application Limit Amount(Desired)'])

    df['DTI_Total'] = (df['Amount of Unsecured Loans'] + df['Application Limit Amount(Desired)']) / (df['Total Annual Income'] + 1)
    df['DTI_WithRent_Annual'] = (df['Amount of Unsecured Loans'] + df['Rent Burden Amount'] * 12) / (df['Total Annual Income'] + 1)

    df['Income_per_Dependent'] = df['Total Annual Income'] / (df['Number of Dependents'] + 1)

    df['Avg_Existing_Loan'] = df['Amount of Unsecured Loans'] / (df['Number of Unsecured Loans'] + 1)
    df['Loan_Intensity']    = df['Number of Unsecured Loans'] / (df['Age'] + 1)

    df['DTI_Critical']  = (df['DTI_Total'] > 0.6).astype(int)
    df['Has_Many_Loans']= (df['Number of Unsecured Loans'] >= 3).astype(int)
    df['Loan_Free']     = (df['Number of Unsecured Loans'] == 0).astype(int)

    # --- STABILITY INDICATORS ---
    df['Employment_Years'] = df['Duration of Employment at Company (Months)'] / 12.0
    df['Residence_Years']  = df['Duration of Residence (Months)'] / 12.0
    df['Employment_to_Age']= df['Employment_Years'] / (df['Age'] + 1)
    df['Combined_Stability']= (df['Employment_Years'] + df['Residence_Years']) / 2.0
    df['Is_New_Job']       = (df['Employment_Years'] <= 1).astype(int)

    # --- HOUSING / EMPLOYMENT / FAMILY FLAGS ---
    df['Is_Homeowner']   = df['Residence Type'].isin([1,2,8,9]).astype(int)
    df['Is_Renter']      = df['Residence Type'].isin([4,5,6,7]).astype(int)
    df['Is_Regular_Employee'] = (df['Employment Status Type'] == 1).astype(int)
    df['Is_Public_Sector']    = (df['Company Size Category'] == 1).astype(int)
    df['Is_Large_Company']    = df['Company Size Category'].isin([1,2,3,4]).astype(int)
    df['Is_Part_Time']        = (df['Employment Type'] == 4).astype(int)

    df['Is_Married']     = (df['Single/Married Status'] == 2).astype(int)
    df['Has_Dependents'] = (df['Number of Dependents'] > 0).astype(int)
    df['Large_Family']   = (df['Number of Dependents'] >= 3).astype(int)

    # --- AGE & INTERACTIONS ---
    df['Age_Squared'] = df['Age'] ** 2
    df['Is_Very_Young'] = (df['Age'] < 25).astype(int)

    df['Age_Income']         = df['Age'] * df['Income_log']
    df['Age_DTI']            = df['Age'] * df['DTI_Total']
    df['Income_Dependents']  = df['Income_log'] * (df['Number of Dependents'] + 1)

    # --- COMPOSITE SCORES ---
    df['Financial_Risk_Score'] = (
        df['DTI_Critical'] * 4 +
        (df['DTI_Total'] > 0.4).astype(int) * 2 +
        df['Has_Many_Loans'] * 2
    )
    df['Stability_Risk_Score'] = (
        df['Is_New_Job'] * 2 +
        df['Is_Part_Time'] * 2 +
        (1 - df['Is_Homeowner']) * 1
    )
    df['Net_Risk_Score'] = df['Financial_Risk_Score'] + df['Stability_Risk_Score'] - (df['Is_Large_Company'] * 2)

    # Drop raw date columns (we used their info)
    df = df.drop(columns=['Application Date', 'Date of Birth', 'JIS_str'], errors='ignore')
    return df

train_features = create_safe_features(train_df)
test_features  = create_safe_features(test_df)

print(f"✓ Train features: {train_features.shape[1]}  |  New features: {train_features.shape[1] - train_df.shape[1]}")

# ============================================================================
# STEP 3: ADVERSARIAL VALIDATION (as a feature)
# ============================================================================
print("\n[3/9] Adversarial validation score (used as feature)...")
av_X = pd.concat([
    train_features.drop(['Default 12 Flag', 'ID'], axis=1, errors='ignore'),
    test_features.drop(['ID'], axis=1, errors='ignore')
], axis=0, ignore_index=True)
av_y = np.array([0]*len(train_features) + [1]*len(test_features))

cat_features_av = [
    'Major Media Code', 'Internet Details', 'Reception Type Category', 'Gender',
    'Single/Married Status', 'Residence Type', 'Name Type', 'Family Composition Type',
    'Living Arrangement Type', 'Insurance Job Type', 'Employment Type',
    'Employment Status Type', 'Industry Type', 'Company Size Category',
    'JIS Address Code', 'JIS_Prefix_2', 'JIS_Prefix_3'
]
cat_features_av = [c for c in cat_features_av if c in av_X.columns]

# Cast categories
for c in cat_features_av:
    av_X[c] = av_X[c].fillna(-999).astype(str).astype('category')
num_cols_av = [c for c in av_X.columns if c not in cat_features_av]
av_X[num_cols_av] = av_X[num_cols_av].fillna(-999)

av_model = LGBMClassifier(n_estimators=500, learning_rate=0.05, n_jobs=-1, verbosity=-1)
av_model.fit(av_X, av_y, callbacks=[log_evaluation(0)])
av_preds = av_model.predict_proba(av_X)[:, 1]
av_auc = roc_auc_score(av_y, av_preds)
print(f"✓ AV AUC: {av_auc:.5f} (higher means stronger train/test drift)")

train_features['av_score'] = av_preds[:len(train_features)]
test_features['av_score']  = av_preds[len(train_features):]

# ============================================================================
# STEP 4: FINAL DATA MATRICES
# ============================================================================
print("\n[4/9] Preparing matrices...")
y = train_features['Default 12 Flag'].values
X = train_features.drop(columns=['Default 12 Flag', 'ID'], errors='ignore')

test_ids = test_features['ID'].copy()
X_test = test_features.drop(columns=['ID'], errors='ignore').reindex(columns=X.columns, fill_value=0)

cat_features = [
    'Major Media Code', 'Internet Details', 'Reception Type Category', 'Gender',
    'Single/Married Status', 'Residence Type', 'Name Type', 'Family Composition Type',
    'Living Arrangement Type', 'Insurance Job Type', 'Employment Type',
    'Employment Status Type', 'Industry Type', 'Company Size Category', 'JIS Address Code',
    'App_Month', 'App_DayOfWeek', 'App_Quarter', 'JIS_Prefix_2', 'JIS_Prefix_3'
]
cat_features = [c for c in cat_features if c in X.columns]

# Fill missing
for c in cat_features:
    X[c] = X[c].fillna(-999).astype(str)
    X_test[c] = X_test[c].fillna(-999).astype(str)

num_cols = [c for c in X.columns if c not in cat_features]
X[num_cols] = X[num_cols].fillna(-999)
X_test[num_cols] = X_test[num_cols].fillna(-999)

X = X.replace([np.inf, -np.inf], -999)
X_test = X_test.replace([np.inf, -np.inf], -999)

print(f"✓ Final feature count: {X.shape[1]} | Categorical: {len(cat_features)}")

# ============================================================================
# STEP 5: POWER TRANSFORM FOR SKEW
# ============================================================================
print("\n[5/9] PowerTransform (Yeo-Johnson) on skewed features...")
skewed_features = ['Total Annual Income', 'Amount of Unsecured Loans', 'Application Limit Amount(Desired)', 'Rent Burden Amount']
skewed_features = [f for f in skewed_features if f in num_cols]

if len(skewed_features) > 0:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    X_sk = X[skewed_features].replace(-999, 0)
    XT_sk = X_test[skewed_features].replace(-999, 0)
    X_sk_t  = pt.fit_transform(X_sk)
    XT_sk_t = pt.transform(XT_sk)
    for i, col in enumerate(skewed_features):
        X[f'{col}_power'] = X_sk_t[:, i]
        X_test[f'{col}_power'] = XT_sk_t[:, i]
    num_cols += [f'{c}_power' for c in skewed_features]
    print(f"  ✓ Transformed {len(skewed_features)} features")

# ============================================================================
# STEP 6: LABEL ENCODERS for XGBoost (CatBoost/LGBM handle cat as strings)
# ============================================================================
print("\n[6/9] Preparing label encoders for XGBoost...")
label_encoders = {}
for c in cat_features:
    le = LabelEncoder()
    all_vals = pd.concat([X[c], X_test[c]]).astype(str).fillna("NaN").values
    le.fit(all_vals)
    label_encoders[c] = le

def encode_for_xgb(df):
    df_enc = df.copy()
    for c in cat_features:
        df_enc[c] = label_encoders[c].transform(df_enc[c].astype(str))
    return df_enc

# ============================================================================
# STEP 7: 7-FOLD TRAINING — CatBoost + XGBoost + LightGBM
# ============================================================================
print("\n[7/9] Training 7-Fold models (CB + XGB + LGBM)...")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_cb  = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))

pred_cb  = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_lgb = np.zeros(len(X_test))

scores_cb, scores_xgb, scores_lgb = [], [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n{'='*78}\nFOLD {fold}/{N_SPLITS}\n{'='*78}")
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[va_idx], y[va_idx]

    # 1) CatBoost
    print("→ [1/3] CatBoost...")
    tr_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    va_pool = Pool(X_va, y_va, cat_features=cat_features)
    cb = CatBoostClassifier(
        iterations=4000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=15,
        min_data_in_leaf=30,
        eval_metric='AUC',
        random_seed=RANDOM_STATE + fold,
        early_stopping_rounds=200,
        verbose=0,
        thread_count=-1
    )
    cb.fit(tr_pool, eval_set=va_pool)
    oof_cb[va_idx] = cb.predict_proba(X_va)[:, 1]
    pred_cb += cb.predict_proba(X_test)[:, 1] / N_SPLITS
    auc_cb = roc_auc_score(y_va, oof_cb[va_idx])
    scores_cb.append(auc_cb)
    print(f"   ✓ AUC: {auc_cb:.6f}  |  Trees: {cb.tree_count_}")

    # 2) XGBoost
    print("→ [2/3] XGBoost...")
    X_tr_xgb = encode_for_xgb(X_tr)
    X_va_xgb = encode_for_xgb(X_va)
    X_te_xgb = encode_for_xgb(X_test)

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
        eval_metric='auc',
        early_stopping_rounds=200,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    oof_xgb[va_idx] = xgb.predict_proba(X_va_xgb)[:, 1]
    pred_xgb += xgb.predict_proba(X_te_xgb)[:, 1] / N_SPLITS
    auc_xgb = roc_auc_score(y_va, oof_xgb[va_idx])
    scores_xgb.append(auc_xgb)
    print(f"   ✓ AUC: {auc_xgb:.6f}  |  Best iters: {xgb.best_iteration}")

    # 3) LightGBM
    print("→ [3/3] LightGBM...")
    lgb = LGBMClassifier(
        n_estimators=4000,
        learning_rate=0.02,
        max_depth=-1,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective='binary',
        random_state=RANDOM_STATE + fold,
        n_jobs=-1
    )
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[log_evaluation(0)])
    oof_lgb[va_idx] = lgb.predict_proba(X_va)[:, 1]
    pred_lgb += lgb.predict_proba(X_test)[:, 1] / N_SPLITS
    auc_lgb = roc_auc_score(y_va, oof_lgb[va_idx])
    scores_lgb.append(auc_lgb)
    print(f"   ✓ AUC: {auc_lgb:.6f}")

print("\n📊 Level-1 OOF AUCs:")
print(f"  CatBoost : {roc_auc_score(y, oof_cb):.6f} | mean fold: {np.mean(scores_cb):.6f}")
print(f"  XGBoost  : {roc_auc_score(y, oof_xgb):.6f} | mean fold: {np.mean(scores_xgb):.6f}")
print(f"  LightGBM : {roc_auc_score(y, oof_lgb):.6f} | mean fold: {np.mean(scores_lgb):.6f}")

# ============================================================================
# STEP 8: BLENDING (weight search) + STACKING (meta LR)
# ============================================================================
print("\n[8/9] Finding best 3-model blend and training meta-model...")

# --- (A) GRID-SEARCH BLEND (simple convex weights) ---
def best_blend(oof_list, y, step=0.05):
    best_auc, best_w = -1.0, None
    cb, xg, lg = oof_list
    for w_cb in np.arange(0.0, 1.0 + 1e-9, step):
        for w_xg in np.arange(0.0, 1.0 - w_cb + 1e-9, step):
            w_lg = 1.0 - w_cb - w_xg
            if w_lg < -1e-9: 
                continue
            oof_blend = w_cb*cb + w_xg*xg + w_lg*lg
            auc = roc_auc_score(y, oof_blend)
            if auc > best_auc:
                best_auc, best_w = auc, (w_cb, w_xg, w_lg)
    return best_auc, best_w

blend_auc, blend_w = best_blend([oof_cb, oof_xgb, oof_lgb], y, step=0.05)
print(f"✓ Best blend AUC: {blend_auc:.6f} at weights (CB, XGB, LGB) = {blend_w}")

# Apply best weights to test preds
w_cb, w_xg, w_lg = blend_w
test_blend = w_cb*pred_cb + w_xg*pred_xgb + w_lg*pred_lgb
oof_blend = w_cb*oof_cb + w_xg*oof_xgb + w_lg*oof_lgb

# --- (B) META-STACK (Logistic Regression on OOFs) ---
stack_train = np.vstack([oof_cb, oof_xgb, oof_lgb]).T
stack_test  = np.vstack([pred_cb, pred_xgb, pred_lgb]).T

skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
meta_oof = np.zeros(len(y))
meta_pred = np.zeros(len(X_test))

for f, (tr, va) in enumerate(skf_meta.split(stack_train, y), 1):
    meta = LogisticRegression(max_iter=200, solver='lbfgs')
    meta.fit(stack_train[tr], y[tr])
    meta_oof[va] = meta.predict_proba(stack_train[va])[:, 1]
    meta_pred += meta.predict_proba(stack_test)[:, 1] / skf_meta.n_splits

meta_auc = roc_auc_score(y, meta_oof)
print(f"✓ Meta-model (LR) OOF AUC: {meta_auc:.6f}")

# Choose the better approach by OOF AUC
use_meta = meta_auc > blend_auc
final_oof = meta_oof if use_meta else oof_blend
final_test = meta_pred if use_meta else test_blend
chosen = "Meta-Stack (LR)" if use_meta else "Best 3-Model Blend"
print(f"🎯 Selected: {chosen}  |  OOF AUC: {max(meta_auc, blend_auc):.6f}")

# ============================================================================
# STEP 9: CREATE SUBMISSION
# ============================================================================
print("\n[9/9] Saving submission...")
sub = pd.DataFrame({
    "ID": test_ids,
    "Default 12 Flag": np.clip(final_test, 0, 1)
})
sub_name = f"{OUTPUT_PREFIX}_{chosen.replace(' ', '')}_OOF{max(meta_auc, blend_auc):.5f}.csv"
sub.to_csv(sub_name, index=False)

print(f"✅ Submission saved: {sub_name}")
print(f"   Mean: {sub['Default 12 Flag'].mean():.6f} | Std: {sub['Default 12 Flag'].std():.6f}")
print("\n🏁 Done.")
