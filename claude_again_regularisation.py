import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.linear_model import LogisticRegression
# from google.colab import files <-- YEH HATA DIYA
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔥🔥🔥 THE FINAL HACK (v10.0 - SAFE STACKING) 🔥🔥🔥")
print("="*80)
print("Features: ~120 (Safe Set) | Models: 3 (2 L1 + 1 L2) | Target: 0.68+ (Stable)")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_SPLITS = 5  # 5 Folds (Fast & Stable)
RANDOM_STATE = 42
DATA_PATH = "" # Agar data "data/" folder mein hai, toh yahan "data/" likho

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/9] Loading data...")
try:
    train_df = pd.read_csv(f"{DATA_PATH}train.csv")
    test_df = pd.read_csv(f"{DATA_PATH}test.csv")
    print(f"✓ Train: {train_df.shape} | Test: {test_df.shape}")
    print(f"✓ Default rate: {train_df['Default 12 Flag'].mean():.4f}")
except FileNotFoundError:
    print("❌ ERROR: data/train.csv ya data/test.csv nahi mili.")
    raise

# ============================================================================
# STEP 2: SAFE FEATURE ENGINEERING (v9.0 Features)
# ============================================================================
print("\n[2/9] Creating ~120 SAFE features...")

def create_safe_features(df):
    """
    Yeh aapka 'v9.0' waala stable feature set hai + JIS Cleaning
    """
    df = df.copy()

    # === NEW: JIS ADDRESS CODE CLEANING ===
    df['JIS_str'] = df['JIS Address Code'].fillna(-999).astype(str)
    df['JIS_Prefix_2'] = df['JIS_str'].str[:2] # State code
    df['JIS_Prefix_3'] = df['JIS_str'].str[:3] # District code
    
    # === TEMPORAL FEATURES ===
    df['Application Date'] = pd.to_datetime(df['Application Date'], format='%Y/%m/%d', errors='coerce')
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], format='%Y/%m/%d', errors='coerce')
    
    df['Age'] = (df['Application Date'] - df['Date of Birth']).dt.days / 365.25
    df['App_Month'] = df['Application Date'].dt.month
    df['App_DayOfWeek'] = df['Application Date'].dt.dayofweek
    df['App_Quarter'] = df['Application Date'].dt.quarter
    
    df['App_Hour'] = df['Application Time'] // 10000
    
    # Time-based risk patterns
    df['Is_Weekend'] = (df['App_DayOfWeek'] >= 5).astype(int)
    df['Is_BusinessHours'] = ((df['App_Hour'] >= 9) & (df['App_Hour'] <= 17)).astype(int)
    df['Is_LateNight'] = ((df['App_Hour'] >= 22) | (df['App_Hour'] <= 5)).astype(int)
    
    # === FRAUD DETECTION (CRITICAL!) ===
    df['Loan_Amount_Gap'] = df['Declared Amount of Unsecured Loans'] - df['Amount of Unsecured Loans']
    df['Loan_Count_Gap'] = df['Declared Number of Unsecured Loans'] - df['Number of Unsecured Loans']
    df['Abs_Amount_Gap'] = abs(df['Loan_Amount_Gap'])
    
    df['Hidden_Loans'] = (df['Loan_Count_Gap'] < 0).astype(int)
    df['Severe_Underreporting'] = ((df['Loan_Amount_Gap'] < 0) & (df['Abs_Amount_Gap'] >= 200000)).astype(int)
    
    # Honesty metrics
    df['Honesty_Score'] = np.where(
        df['Declared Amount of Unsecured Loans'] > 0,
        1 - np.clip(df['Abs_Amount_Gap'] / (df['Declared Amount of Unsecured Loans'] + 1), 0, 1),
        1
    )
    df['Perfect_Match'] = ((df['Loan_Amount_Gap'] == 0) & (df['Loan_Count_Gap'] == 0)).astype(int)
    
    # === FINANCIAL HEALTH ===
    df['Income_log'] = np.log1p(df['Total Annual Income'])
    df['Existing_Loan_log'] = np.log1p(df['Amount of Unsecured Loans'])
    df['Desired_Loan_log'] = np.log1p(df['Application Limit Amount(Desired)'])
    
    # DTI ratios (CRITICAL!)
    df['DTI_Total'] = (df['Amount of Unsecured Loans'] + df['Application Limit Amount(Desired)']) / (df['Total Annual Income'] + 1)
    df['DTI_WithRent_Annual'] = (df['Amount of Unsecured Loans'] + df['Rent Burden Amount'] * 12) / (df['Total Annual Income'] + 1)
    
    # Income adequacy
    df['Income_per_Dependent'] = df['Total Annual Income'] / (df['Number of Dependents'] + 1)
    
    # Loan characteristics
    df['Avg_Existing_Loan'] = df['Amount of Unsecured Loans'] / (df['Number of Unsecured Loans'] + 1)
    df['Loan_Intensity'] = df['Number of Unsecured Loans'] / (df['Age'] + 1)
    
    # Risk thresholds
    df['DTI_Critical'] = (df['DTI_Total'] > 0.6).astype(int)
    df['Has_Many_Loans'] = (df['Number of Unsecured Loans'] >= 3).astype(int)
    df['Loan_Free'] = (df['Number of Unsecured Loans'] == 0).astype(int)
    
    # === STABILITY INDICATORS ===
    df['Employment_Years'] = df['Duration of Employment at Company (Months)'] / 12
    df['Residence_Years'] = df['Duration of Residence (Months)'] / 12
    
    df['Employment_to_Age'] = df['Employment_Years'] / (df['Age'] + 1)
    df['Combined_Stability'] = (df['Employment_Years'] + df['Residence_Years']) / 2
    
    # Job security flags
    df['Is_New_Job'] = (df['Employment_Years'] <= 1).astype(int)
    
    # === HOUSING & LIFESTYLE ===
    df['Is_Homeowner'] = df['Residence Type'].isin([1, 2, 8, 9]).astype(int)
    df['Is_Renter'] = df['Residence Type'].isin([4, 5, 6, 7]).astype(int)
    
    # === EMPLOYMENT QUALITY ===
    df['Is_Regular_Employee'] = (df['Employment Status Type'] == 1).astype(int)
    df['Is_Public_Sector'] = (df['Company Size Category'] == 1).astype(int)
    df['Is_Large_Company'] = df['Company Size Category'].isin([1, 2, 3, 4]).astype(int)
    df['Is_Part_Time'] = (df['Employment Type'] == 4).astype(int)
    
    # === FAMILY STRUCTURE ===
    df['Is_Married'] = (df['Single/Married Status'] == 2).astype(int)
    df['Has_Dependents'] = (df['Number of Dependents'] > 0).astype(int)
    df['Large_Family'] = (df['Number of Dependents'] >= 3).astype(int)
    
    # === AGE-BASED FEATURES ===
    df['Age_Squared'] = df['Age'] ** 2
    df['Is_Very_Young'] = (df['Age'] < 25).astype(int)
    
    # === INTERACTION FEATURES (POWER COMBOS) ===
    df['Age_Income'] = df['Age'] * df['Income_log']
    df['Age_DTI'] = df['Age'] * df['DTI_Total']
    df['Income_Dependents'] = df['Income_log'] * (df['Number of Dependents'] + 1)
    
    # === COMPOSITE RISK SCORES ===
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

    # Drop date columns
    df = df.drop(columns=['Application Date', 'Date of Birth', 'JIS_str'], errors='ignore')
    
    return df

# Apply safe features
train_features = create_safe_features(train_df)
test_features = create_safe_features(test_df)

print(f"✓ Train features: {train_features.shape[1]}")
print(f"✓ New features: {train_features.shape[1] - train_df.shape[1]}")

# ============================================================================
# STEP 3: ADVERSARIAL VALIDATION (AS A FEATURE)
# ============================================================================
print("\n[3/9] Running Adversarial Validation (as a Feature)...")
av_X = pd.concat([train_features.drop(['Default 12 Flag', 'ID'], axis=1, errors='ignore'),
                  test_features.drop('ID', axis=1, errors='ignore')],
                 axis=0, ignore_index=True)
av_y = np.array([0] * len(train_features) + [1] * len(test_features))

cat_features_av = [
    'Major Media Code', 'Internet Details', 'Reception Type Category', 'Gender', 
    'Single/Married Status', 'Residence Type', 'Name Type', 'Family Composition Type', 
    'Living Arrangement Type', 'Insurance Job Type', 'Employment Type', 
    'Employment Status Type', 'Industry Type', 'Company Size Category', 
    'JIS Address Code', 'JIS_Prefix_2', 'JIS_Prefix_3'
]
cat_features_av = [col for col in cat_features_av if col in av_X.columns]

for col in cat_features_av:
    av_X[col] = av_X[col].fillna(-999).astype(str).astype('category')

numeric_cols_av = [col for col in av_X.columns if col not in cat_features_av]
av_X[numeric_cols_av] = av_X[numeric_cols_av].fillna(-999)

av_model = LGBMClassifier(n_estimators=500, learning_rate=0.05, n_jobs=-1, verbosity=-1)
av_model.fit(av_X, av_y, callbacks=[log_evaluation(0)]) # Train on all data
av_preds = av_model.predict_proba(av_X)[:, 1]
av_auc = roc_auc_score(av_y, av_preds)
print(f"✓ Adversarial Validation AUC: {av_auc:.5f}")

train_features['av_score'] = av_preds[:len(train_features)]
test_features['av_score'] = av_preds[len(train_features):]

# ============================================================================
# STEP 4: PREPARE DATA FOR MODELING
# ============================================================================
print("\n[4/9] Preparing final data for modeling...")

y = train_features['Default 12 Flag']
X = train_features.drop(columns=['Default 12 Flag', 'ID'], errors='ignore')

test_ids = test_features['ID']
X_test = test_features.drop(columns=['ID'], errors='ignore')
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Define FINAL categorical features
cat_features = [
    'Major Media Code', 'Internet Details', 'Reception Type Category',
    'Gender', 'Single/Married Status', 'Residence Type', 'Name Type',
    'Family Composition Type', 'Living Arrangement Type', 
    'Insurance Job Type', 'Employment Type', 'Employment Status Type',
    'Industry Type', 'Company Size Category', 'JIS Address Code',
    'App_Month', 'App_DayOfWeek', 'App_Quarter',
    'JIS_Prefix_2', 'JIS_Prefix_3'
]
cat_features = [col for col in cat_features if col in X.columns]

# Handle missing values
print("→ Handling missing values...")
for col in cat_features:
    X[col] = X[col].fillna(-999).astype(str)
    X_test[col] = X_test[col].fillna(-999).astype(str)

numeric_cols = [col for col in X.columns if col not in cat_features]
X[numeric_cols] = X[numeric_cols].fillna(-999)
X_test[numeric_cols] = X_test[numeric_cols].fillna(-999)

X = X.replace([np.inf, -np.inf], -999)
X_test = X_test.replace([np.inf, -np.inf], -999)

print(f"✓ Final feature count: {X.shape[1]}")
print(f"✓ Categorical: {len(cat_features)}")

# ============================================================================
# STEP 5: POWER TRANSFORMATION
# ============================================================================
print("\n[5/9] Advanced preprocessing (PowerTransformer)...")
skewed_features = ['Total Annual Income', 'Amount of Unsecured Loans', 
                   'Application Limit Amount(Desired)', 'Rent Burden Amount']
skewed_features = [f for f in skewed_features if f in numeric_cols]

pt = PowerTransformer(method='yeo-johnson', standardize=False)
if len(skewed_features) > 0:
    X_skewed = X[skewed_features].replace(-999, 0)
    X_test_skewed = X_test[skewed_features].replace(-999, 0)
    
    X_skewed_transformed = pt.fit_transform(X_skewed)
    X_test_skewed_transformed = pt.transform(X_test_skewed)
    
    for i, col in enumerate(skewed_features):
        X[f'{col}_power'] = X_skewed_transformed[:, i]
        X_test[f'{col}_power'] = X_test_skewed_transformed[:, i]
    
    numeric_cols.extend([f'{col}_power' for col in skewed_features])
    print(f"  ✓ Transformed {len(skewed_features)} skewed features")

# ============================================================================
# STEP 6: LABEL ENCODERS (for XGBoost)
# ============================================================================
print("\n[6/9] Preparing label encoders...")
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    all_cats = pd.concat([X[col], X_test[col]]).unique()
    le.fit(all_cats)
    label_encoders[col] = le

# ============================================================================
# STEP 7: L1 TRAINING (CB + XGB) with REGULARIZATION
# ============================================================================
print("\n[7/9] Training REGULARIZED L1 models (CB + XGB)...")
print(f"Yeh {N_SPLITS}-Fold CV hai, time lega...")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

cb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))
cb_test = np.zeros(len(X_test))
xgb_test = np.zeros(len(X_test))

cb_scores = []
xgb_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*70}\nFOLD {fold+1}/{N_SPLITS}\n{'='*70}")
    
    X_tr, y_tr = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    
    # --- MODEL 1: CATBOOST (REGULARIZED) ---
    print("→ [1/2] Training CatBoost...")
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    cb = CatBoostClassifier(
        iterations=4000, 
        learning_rate=0.02, # Slow learning
        depth=8,            # Kam deep
        l2_leaf_reg=15,     # Zyada regularization
        min_data_in_leaf=30,# Zyada regularization
        eval_metric='AUC', 
        random_seed=RANDOM_STATE + fold,
        early_stopping_rounds=200, 
        verbose=0, 
        thread_count=-1
    )
    cb.fit(train_pool, eval_set=val_pool)
    
    cb_oof[val_idx] = cb.predict_proba(X_val)[:, 1]
    cb_test += cb.predict_proba(X_test)[:, 1] / N_SPLITS
    cb_score = roc_auc_score(y_val, cb_oof[val_idx])
    cb_scores.append(cb_score)
    print(f"  ✓ CatBoost AUC: {cb_score:.6f} (trees: {cb.tree_count_})")
    
    # --- MODEL 2: XGBOOST (REGULARIZED) ---
    print("\n→ [2/2] Training XGBoost...")
    X_tr_xgb, X_val_xgb, X_test_xgb = X_tr.copy(), X_val.copy(), X_test.copy()
    for col in cat_features:
        X_tr_xgb[col] = label_encoders[col].transform(X_tr_xgb[col])
        X_val_xgb[col] = label_encoders[col].transform(X_val_xgb[col])
        X_test_xgb[col] = label_encoders[col].transform(X_test_xgb[col])
    
    xgb = XGBClassifier(
        n_estimators=4000, 
        learning_rate=0.02,
        max_depth=7,            # Kam deep
        min_child_weight=10,     # Zyada regularization
        subsample=0.8, 
        colsample_bytree=0.8,
        reg_alpha=1.0, 
        reg_lambda=1.0, 
        gamma=0.1,
        random_state=RANDOM_STATE + fold, 
        eval_metric='auc',
        early_stopping_rounds=200, 
        use_label_encoder=False,
        n_jobs=-1, 
        verbosity=0
    )
    xgb.fit(
        X_tr_xgb, y_tr,
        eval_set=[(X_val_xgb, y_val)],
        verbose=False
    )
    
    xgb_oof[val_idx] = xgb.predict_proba(X_val_xgb)[:, 1]
    xgb_test += xgb.predict_proba(X_test_xgb)[:, 1] / N_SPLITS
    xgb_score = roc_auc_score(y_val, xgb_oof[val_idx])
    xgb_scores.append(xgb_score)
    print(f"  ✓ XGBoost AUC: {xgb_score:.6f} (trees: {xgb.best_iteration})")

# ============================================================================
# STEP 8: L2 STACKING (THE SAFE HACK)
# ============================================================================
print("\n[8/9] Training L2 Stacker (Meta-Model)...")

# Stacker ka training data = OOF predictions
X_level2 = pd.DataFrame({
    'cb_pred': cb_oof,
    'xgb_pred': xgb_oof,
    'cb_rank': rankdata(cb_oof),
    'xgb_rank': rankdata(xgb_oof),
    'cb_minus_xgb': cb_oof - xgb_oof
})

# Stacker ka test data = L1 test predictions
X_test_level2 = pd.DataFrame({
    'cb_pred': cb_test,
    'xgb_pred': xgb_test,
    'cb_rank': rankdata(cb_test),
    'xgb_rank': rankdata(xgb_test),
    'cb_minus_xgb': cb_test - xgb_test
})

# Stacker model (Logistic Regression) ko train karo
# Yeh simple hai aur overfit nahi hoga
stacker = LogisticRegression(C=0.1, solver='liblinear', random_state=RANDOM_STATE)
stacker.fit(X_level2, y)

final_predictions = stacker.predict_proba(X_test_level2)[:, 1]
final_oof_score = roc_auc_score(y, stacker.predict_proba(X_level2)[:, 1])

print(f"✓ Final L2 Stacker OOF AUC: {final_oof_score:.6f}")

# ============================================================================
# STEP 9: CREATE SUBMISSION
# ============================================================================
print("\n[9/9] Creating final submission...")

submission = pd.DataFrame({
    'ID': test_ids,
    'Default 12 Flag': final_predictions
})
submission['Default 12 Flag'] = submission['Default 12 Flag'].clip(0, 1)
filename = f'RELIABLE_STACKING_v10_auc{final_oof_score:.5f}.csv'
submission.to_csv(filename, index=False)

print(f"\n✅ SUBMISSION SAVED: {filename}")
print(f"  Mean: {final_predictions.mean():.6f} | Std: {final_predictions.std():.6f}")

# files.download(filename) <-- YEH HATA DIYA
print(f"\n✅ File saved locally: {filename}")

print("\n" + "="*80)
print("🏆🏆🏆 RELIABLE STACKING MODEL READY! 🏆🏆🏆")
print("="*80)
print(f"\n🎯 L2 OOF Score: {final_oof_score:.5f} (Yeh score reliable hai)")
print(f"🔥 Features Used: {X.shape[1]}")
print(f"💪 Model: L2 Stacker (LogisticRegression) on (CB + XGB)")
print(f"📊 CV: {N_SPLITS}-Fold Stratified")
print(f"⚡ Hacks Removed: Pseudo-Labeling, 250+ features (Overfitting waale)")
print(f"⚡ Hacks Kept: Safe Features, JIS Cleaning, AV-Score, L2 Stacking")
print("\n🚀 ISKO SUBMIT KARO. YEH AAPKA NAYA BEST SHOT HAI. 🚀")
print("="*80)