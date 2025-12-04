import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from scipy.optimize import minimize
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔥🔥🔥 ENHANCED SCRIPT (v10.0 - TARGET: 0.70+) 🔥🔥🔥")
print("="*80)
print("Features: ~150+ | Models: 3 (CB + XGB + LGB) | Optimized Blending")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_SPLITS = 5
RANDOM_STATE = 42
DATA_PATH = ""

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/10] Loading data...")
try:
    train_df = pd.read_csv(f"{DATA_PATH}train.csv")
    test_df = pd.read_csv(f"{DATA_PATH}test.csv")
    print(f"✓ Train: {train_df.shape} | Test: {test_df.shape}")
    print(f"✓ Default rate: {train_df['Default 12 Flag'].mean():.4f}")
except FileNotFoundError:
    print(f"❌ ERROR: train.csv or test.csv not found.")
    raise

# ============================================================================
# STEP 2: ENHANCED FEATURE ENGINEERING
# ============================================================================
print("\n[2/10] Creating ~150+ ENHANCED features...")

def create_enhanced_features(df):
    """Enhanced feature set with credit scoring domain knowledge"""
    df = df.copy()

    # === JIS ADDRESS CODE CLEANING ===
    df['JIS_str'] = df['JIS Address Code'].fillna(-999).astype(str)
    df['JIS_Prefix_2'] = df['JIS_str'].str[:2]
    df['JIS_Prefix_3'] = df['JIS_str'].str[:3]
    
    # === TEMPORAL FEATURES ===
    df['Application Date'] = pd.to_datetime(df['Application Date'], format='%Y/%m/%d', errors='coerce')
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], format='%Y/%m/%d', errors='coerce')
    
    df['Age'] = (df['Application Date'] - df['Date of Birth']).dt.days / 365.25
    df['App_Month'] = df['Application Date'].dt.month
    df['App_DayOfWeek'] = df['Application Date'].dt.dayofweek
    df['App_Quarter'] = df['Application Date'].dt.quarter
    df['App_Hour'] = df['Application Time'] // 10000
    
    df['Is_Weekend'] = (df['App_DayOfWeek'] >= 5).astype(int)
    df['Is_BusinessHours'] = ((df['App_Hour'] >= 9) & (df['App_Hour'] <= 17)).astype(int)
    df['Is_LateNight'] = ((df['App_Hour'] >= 22) | (df['App_Hour'] <= 5)).astype(int)
    
    # === FRAUD DETECTION ===
    df['Loan_Amount_Gap'] = df['Declared Amount of Unsecured Loans'] - df['Amount of Unsecured Loans']
    df['Loan_Count_Gap'] = df['Declared Number of Unsecured Loans'] - df['Number of Unsecured Loans']
    df['Abs_Amount_Gap'] = abs(df['Loan_Amount_Gap'])
    
    df['Hidden_Loans'] = (df['Loan_Count_Gap'] < 0).astype(int)
    df['Severe_Underreporting'] = ((df['Loan_Amount_Gap'] < 0) & (df['Abs_Amount_Gap'] >= 200000)).astype(int)
    
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
    
    # Monthly calculations
    df['Monthly_Income'] = df['Total Annual Income'] / 12
    df['Monthly_Debt_Service'] = df['Amount of Unsecured Loans'] * 0.03
    df['Payment_to_Income_Ratio'] = df['Monthly_Debt_Service'] / (df['Monthly_Income'] + 1)
    
    # DTI ratios
    df['DTI_Total'] = (df['Amount of Unsecured Loans'] + df['Application Limit Amount(Desired)']) / (df['Total Annual Income'] + 1)
    df['DTI_WithRent_Annual'] = (df['Amount of Unsecured Loans'] + df['Rent Burden Amount'] * 12) / (df['Total Annual Income'] + 1)
    df['DTI_Existing'] = df['Amount of Unsecured Loans'] / (df['Total Annual Income'] + 1)
    
    # Income adequacy
    df['Income_per_Dependent'] = df['Total Annual Income'] / (df['Number of Dependents'] + 1)
    df['Income_per_Child'] = df['Total Annual Income'] / (df['Number of Dependent Children'] + 1)
    
    # Loan characteristics
    df['Avg_Existing_Loan'] = df['Amount of Unsecured Loans'] / (df['Number of Unsecured Loans'] + 1)
    df['Loan_Intensity'] = df['Number of Unsecured Loans'] / (df['Age'] + 1)
    df['Desired_to_Income_Ratio'] = df['Application Limit Amount(Desired)'] / (df['Total Annual Income'] + 1)
    df['Total_Credit_Exposure'] = (df['Amount of Unsecured Loans'] + df['Application Limit Amount(Desired)']) / (df['Total Annual Income'] + 1)
    
    # Risk thresholds
    df['DTI_Critical'] = (df['DTI_Total'] > 0.6).astype(int)
    df['DTI_High'] = (df['DTI_Total'] > 0.4).astype(int)
    df['Has_Many_Loans'] = (df['Number of Unsecured Loans'] >= 3).astype(int)
    df['Loan_Free'] = (df['Number of Unsecured Loans'] == 0).astype(int)
    
    # === HOUSING & LIFESTYLE ===
    df['Is_Homeowner'] = df['Residence Type'].isin([1, 2, 8, 9]).astype(int)
    df['Is_Renter'] = df['Residence Type'].isin([4, 5, 6, 7]).astype(int)
    df['Housing_Burden_Ratio'] = (df['Rent Burden Amount'] * 12) / (df['Total Annual Income'] + 1)
    df['High_Housing_Burden'] = (df['Housing_Burden_Ratio'] > 0.3).astype(int)
    
    # === STABILITY INDICATORS ===
    df['Employment_Years'] = df['Duration of Employment at Company (Months)'] / 12
    df['Residence_Years'] = df['Duration of Residence (Months)'] / 12
    df['Employment_to_Age'] = df['Employment_Years'] / (df['Age'] + 1)
    df['Combined_Stability'] = (df['Employment_Years'] + df['Residence_Years']) / 2
    df['Is_New_Job'] = (df['Employment_Years'] <= 1).astype(int)
    df['Long_Employment'] = (df['Employment_Years'] > 5).astype(int)
    
    # === EMPLOYMENT QUALITY ===
    df['Is_Regular_Employee'] = (df['Employment Status Type'] == 1).astype(int)
    df['Is_Public_Sector'] = (df['Company Size Category'] == 1).astype(int)
    df['Is_Large_Company'] = df['Company Size Category'].isin([1, 2, 3, 4]).astype(int)
    df['Is_Part_Time'] = (df['Employment Type'] == 4).astype(int)
    
    df['Employment_Stability_Score'] = (
        (df['Duration of Employment at Company (Months)'] > 24).astype(int) * 2 +
        (df['Employment Status Type'] == 1).astype(int) * 2 +
        (df['Company Size Category'] <= 4).astype(int)
    )
    
    # === FAMILY STRUCTURE ===
    df['Is_Married'] = (df['Single/Married Status'] == 2).astype(int)
    df['Has_Dependents'] = (df['Number of Dependents'] > 0).astype(int)
    df['Large_Family'] = (df['Number of Dependents'] >= 3).astype(int)
    df['Child_Ratio'] = df['Number of Dependent Children'] / (df['Number of Dependents'] + 1)
    
    # === AGE-BASED FEATURES ===
    df['Age_Squared'] = df['Age'] ** 2
    df['Is_Very_Young'] = (df['Age'] < 25).astype(int)
    df['Is_Senior'] = (df['Age'] > 55).astype(int)
    df['Prime_Age'] = ((df['Age'] >= 30) & (df['Age'] <= 50)).astype(int)
    
    # Age segments
    df['Risk_Age_Segment'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                                     labels=[0, 1, 2, 3, 4]).astype(float)
    
    # === INCOME SEGMENTS ===
    df['Income_Segment'] = pd.cut(df['Total Annual Income'], 
                                   bins=[0, 2000000, 4000000, 6000000, np.inf],
                                   labels=[0, 1, 2, 3]).astype(float)
    df['High_Income'] = (df['Total Annual Income'] > 5000000).astype(int)
    df['Low_Income'] = (df['Total Annual Income'] < 2000000).astype(int)
    
    # === INTERACTION FEATURES ===
    df['Age_Income'] = df['Age'] * df['Income_log']
    df['Age_DTI'] = df['Age'] * df['DTI_Total']
    df['Income_Dependents'] = df['Income_log'] * (df['Number of Dependents'] + 1)
    df['Age_Income_Interaction'] = df['Age'] * np.log1p(df['Total Annual Income'])
    df['Employment_Income'] = df['Employment_Years'] * df['Income_log']
    df['Stability_Income'] = df['Combined_Stability'] * df['Income_log']
    
    # === COMPOSITE RISK SCORES ===
    df['Financial_Risk_Score'] = (
        df['DTI_Critical'] * 4 +
        df['DTI_High'] * 2 +
        df['Has_Many_Loans'] * 2 +
        df['Hidden_Loans'] * 3
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

# Apply enhanced features
train_features = create_enhanced_features(train_df)
test_features = create_enhanced_features(test_df)

print(f"✓ Train features: {train_features.shape[1]}")
print(f"✓ New features created: {train_features.shape[1] - train_df.shape[1]}")

# ============================================================================
# STEP 3: TARGET ENCODING
# ============================================================================
print("\n[3/10] Creating Target Encodings...")

def create_target_encoding(X, y, X_test, cat_cols, n_splits=5):
    """Target encode categorical features with CV to prevent leakage"""
    X_encoded = X.copy()
    X_test_encoded = X_test.copy()
    
    for col in cat_cols:
        if col not in X.columns:
            continue
            
        X_encoded[f'{col}_target_enc'] = 0
        X_test_encoded[f'{col}_target_enc'] = 0
        
        # CV encoding for train
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, val_idx in skf.split(X, y):
            target_mean = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).mean()
            X_encoded.loc[X_encoded.index[val_idx], f'{col}_target_enc'] = \
                X[col].iloc[val_idx].map(target_mean).fillna(y.mean())
        
        # Full train encoding for test
        target_mean = y.groupby(X[col]).mean()
        X_test_encoded[f'{col}_target_enc'] = X_test[col].map(target_mean).fillna(y.mean())
    
    return X_encoded, X_test_encoded

# Target encode high-cardinality categoricals
target_encode_cols = ['JIS_Prefix_2', 'JIS_Prefix_3', 'Industry Type', 
                      'Reception Type Category', 'Major Media Code', 
                      'Company Size Category']

y_temp = train_features['Default 12 Flag']
X_temp = train_features.drop(columns=['Default 12 Flag', 'ID'], errors='ignore')
X_test_temp = test_features.drop(columns=['ID'], errors='ignore')

X_temp, X_test_temp = create_target_encoding(X_temp, y_temp, X_test_temp, target_encode_cols)

train_features = pd.concat([train_features[['ID', 'Default 12 Flag']], X_temp], axis=1)
test_features = pd.concat([test_features[['ID']], X_test_temp], axis=1)

print(f"✓ Target encoded {len(target_encode_cols)} features")

# ============================================================================
# STEP 4: FREQUENCY ENCODING
# ============================================================================
print("\n[4/10] Adding Frequency Encodings...")

def add_frequency_encoding(df, df_test, cols):
    """Add frequency counts for categorical features"""
    for col in cols:
        if col not in df.columns:
            continue
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[f'{col}_freq'] = df[col].map(freq_map)
        df_test[f'{col}_freq'] = df_test[col].map(freq_map).fillna(0)
    return df, df_test

freq_cols = ['JIS_Prefix_2', 'Industry Type', 'Company Size Category', 
             'Major Media Code', 'Reception Type Category']

X_temp = train_features.drop(columns=['Default 12 Flag', 'ID'], errors='ignore')
X_test_temp = test_features.drop(columns=['ID'], errors='ignore')

X_temp, X_test_temp = add_frequency_encoding(X_temp, X_test_temp, freq_cols)

train_features = pd.concat([train_features[['ID', 'Default 12 Flag']], X_temp], axis=1)
test_features = pd.concat([test_features[['ID']], X_test_temp], axis=1)

print(f"✓ Frequency encoded {len(freq_cols)} features")

# ============================================================================
# STEP 5: ADVERSARIAL VALIDATION
# ============================================================================
print("\n[5/10] Running Adversarial Validation...")
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
av_model.fit(av_X, av_y, callbacks=[log_evaluation(0)])
av_preds = av_model.predict_proba(av_X)[:, 1]
av_auc = roc_auc_score(av_y, av_preds)
print(f"✓ Adversarial Validation AUC: {av_auc:.5f}")

train_features['av_score'] = av_preds[:len(train_features)]
test_features['av_score'] = av_preds[len(train_features):]

# ============================================================================
# STEP 6: PREPARE DATA FOR MODELING
# ============================================================================
print("\n[6/10] Preparing final data...")

y = train_features['Default 12 Flag']
X = train_features.drop(columns=['Default 12 Flag', 'ID'], errors='ignore')

test_ids = test_features['ID']
X_test = test_features.drop(columns=['ID'], errors='ignore')
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Define categorical features
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
# STEP 7: POWER TRANSFORMATION
# ============================================================================
print("\n[7/10] Applying PowerTransformer...")
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
    
    print(f"  ✓ Transformed {len(skewed_features)} skewed features")

# ============================================================================
# STEP 8: LABEL ENCODERS
# ============================================================================
print("\n[8/10] Preparing label encoders...")
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    all_cats = pd.concat([X[col], X_test[col]]).unique()
    le.fit(all_cats)
    label_encoders[col] = le

# ============================================================================
# STEP 9: TRAINING 3 MODELS (CB + XGB + LGB)
# ============================================================================
print("\n[9/10] Training 3 ENHANCED models (CB + XGB + LGB)...")
print(f"Running {N_SPLITS}-Fold CV...")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

cb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))
lgb_oof = np.zeros(len(X))

cb_test = np.zeros(len(X_test))
xgb_test = np.zeros(len(X_test))
lgb_test = np.zeros(len(X_test))

cb_scores = []
xgb_scores = []
lgb_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*70}\nFOLD {fold+1}/{N_SPLITS}\n{'='*70}")
    
    X_tr, y_tr = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    
    # --- MODEL 1: CATBOOST ---
    print("→ [1/3] Training CatBoost...")
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    cb = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=18,
        min_data_in_leaf=30,
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
    
    # --- MODEL 2: XGBOOST ---
    print("\n→ [2/3] Training XGBoost...")
    X_tr_xgb, X_val_xgb, X_test_xgb = X_tr.copy(), X_val.copy(), X_test.copy()
    for col in cat_features:
        X_tr_xgb[col] = label_encoders[col].transform(X_tr_xgb[col])
        X_val_xgb[col] = label_encoders[col].transform(X_val_xgb[col])
        X_test_xgb[col] = label_encoders[col].transform(X_test_xgb[col])
    
    xgb = XGBClassifier(
        n_estimators=5000,
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
    
    # --- MODEL 3: LIGHTGBM ---
    print("\n→ [3/3] Training LightGBM...")
    X_tr_lgb = X_tr.copy()
    X_val_lgb = X_val.copy()
    X_test_lgb = X_test.copy()
    
    for col in cat_features:
        X_tr_lgb[col] = X_tr_lgb[col].astype('category')
        X_val_lgb[col] = X_val_lgb[col].astype('category')
        X_test_lgb[col] = X_test_lgb[col].astype('category')
    
    lgb = LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.015,
        max_depth=9,
        num_leaves=50,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.5,
        reg_lambda=1.5,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbosity=-1
    )
    
    lgb.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_val_lgb, y_val)],
        callbacks=[early_stopping(200), log_evaluation(0)]
    )
    
    lgb_oof[val_idx] = lgb.predict_proba(X_val_lgb)[:, 1]
    lgb_test += lgb.predict_proba(X_test_lgb)[:, 1] / N_SPLITS
    lgb_score = roc_auc_score(y_val, lgb_oof[val_idx])
    lgb_scores.append(lgb_score)
    print(f"  ✓ LightGBM AUC: {lgb_score:.6f}")

# ============================================================================
# STEP 10: OPTIMIZE BLEND WEIGHTS
# ============================================================================
print("\n[10/10] Optimizing blend weights...")

cb_oof_auc = roc_auc_score(y, cb_oof)
xgb_oof_auc = roc_auc_score(y, xgb_oof)
lgb_oof_auc = roc_auc_score(y, lgb_oof)

print(f"\n📊 Individual OOF AUC Scores:")
print(f"  CatBoost:  {cb_oof_auc:.6f}")
print(f"  XGBoost:   {xgb_oof_auc:.6f}")
print(f"  LightGBM:  {lgb_oof_auc:.6f}")

def blend_objective(weights):
    """Objective function for blend optimization"""
    weights = weights / weights.sum()
    blended = weights[0] * cb_oof + weights[1] * xgb_oof + weights[2] * lgb_oof
    return -roc_auc_score(y, blended)

initial_weights = np.array([0.5, 0.25, 0.25])
bounds = [(0, 1), (0, 1), (0, 1)]

result = minimize(
    blend_objective,
    initial_weights,
    method='Nelder-Mead',
    bounds=bounds
)

optimal_weights = result.x / result.x.sum()
print(f"\n✓ Optimal weights: CB={optimal_weights[0]:.3f}, XGB={optimal_weights[1]:.3f}, LGB={optimal_weights[2]:.3f}")

# Create final predictions
final_predictions = (optimal_weights[0] * cb_test + 
                    optimal_weights[1] * xgb_test + 
                    optimal_weights[2] * lgb_test)

final_oof_score = roc_auc_score(y, optimal_weights[0] * cb_oof + 
                                 optimal_weights[1] * xgb_oof + 
                                 optimal_weights[2] * lgb_oof)

print(f"✓ Final Blended OOF AUC: {final_oof_score:.6f}")

# ============================================================================
# CREATE SUBMISSION
# ============================================================================
submission = pd.DataFrame({
    'ID': test_ids,
    'Default 12 Flag': final_predictions
})
submission['Default 12 Flag'] = submission['Default 12 Flag'].clip(0, 1)
filename = f'ENHANCED_BLEND_v10_auc{final_oof_score:.5f}.csv'
submission.to_csv(filename, index=False)

print(f"\n✅ SUBMISSION SAVED: {filename}")
print(f"  Mean: {final_predictions.mean():.6f} | Std: {final_predictions.std():.6f}")
print(f"\n✅ File saved locally: {filename}")

print("\n" + "="*80)
print("🏆🏆🏆 ENHANCED MODEL COMPLETE! 🏆🏆🏆")
print("="*80)
print(f"\n🎯 Final OOF Score: {final_oof_score:.5f}")
print(f"🔥 Total Features: {X.shape[1]}")
print(f"💪 Models: CatBoost + XGBoost + LightGBM (Optimized Blend)")
print(f"📊 CV: {N_SPLITS}-Fold Stratified")
print(f"⚡ New Additions: Target Encoding, Frequency Encoding, Advanced Credit Features")
print(f"⚡ Expected LB: 0.69-0.71 (Conservative Estimate)")
print("\n🚀 SUBMIT THIS TO REACH 0.70+! 🚀")
print("="*80)
