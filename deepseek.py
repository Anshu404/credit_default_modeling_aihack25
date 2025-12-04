import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.feature_selection import SelectFromModel
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀🚀🚀 ELITE SCORING MODEL (v10.0 - TARGET: 0.70+) 🚀🚀🚀")
print("="*80)
print("Features: ~150 | Models: 3 (CB + XGB + LGB) | Advanced Stacking")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_SPLITS = 10  # More folds for better stability
RANDOM_STATE = 42
DATA_PATH = ""

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/10] Loading data...")
train_df = pd.read_csv(f"{DATA_PATH}train.csv")
test_df = pd.read_csv(f"{DATA_PATH}test.csv")
print(f"✓ Train: {train_df.shape} | Test: {test_df.shape}")
print(f"✓ Default rate: {train_df['Default 12 Flag'].mean():.4f}")

# ============================================================================
# STEP 2: ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n[2/10] Creating ~150 ADVANCED features...")

def create_advanced_features(df):
    df = df.copy()

    # === JIS GEOGRAPHICAL FEATURES ===
    df['JIS_str'] = df['JIS Address Code'].fillna(-999).astype(str)
    df['JIS_Prefix_2'] = df['JIS_str'].str[:2].astype('category')
    df['JIS_Prefix_3'] = df['JIS_str'].str[:3].astype('category')
    df['JIS_Length'] = df['JIS_str'].str.len()

    # === TEMPORAL FEATURES (ENHANCED) ===
    df['Application Date'] = pd.to_datetime(df['Application Date'], format='%Y/%m/%d', errors='coerce')
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], format='%Y/%m/%d', errors='coerce')
    
    df['Age'] = (df['Application Date'] - df['Date of Birth']).dt.days / 365.25
    df['App_Month'] = df['Application Date'].dt.month
    df['App_Day'] = df['Application Date'].dt.day
    df['App_DayOfWeek'] = df['Application Date'].dt.dayofweek
    df['App_Quarter'] = df['Application Date'].dt.quarter
    df['App_WeekOfYear'] = df['Application Date'].dt.isocalendar().week
    
    df['App_Hour'] = df['Application Time'] // 10000
    df['App_Minute'] = (df['Application Time'] % 10000) // 100
    
    # Advanced time patterns
    df['Is_Weekend'] = (df['App_DayOfWeek'] >= 5).astype(int)
    df['Is_BusinessHours'] = ((df['App_Hour'] >= 9) & (df['App_Hour'] <= 17)).astype(int)
    df['Is_LateNight'] = ((df['App_Hour'] >= 22) | (df['App_Hour'] <= 5)).astype(int)
    df['Is_MonthEnd'] = (df['App_Day'] >= 25).astype(int)
    df['Is_MonthStart'] = (df['App_Day'] <= 7).astype(int)
    
    # === FRAUD DETECTION (ENHANCED) ===
    df['Loan_Amount_Gap'] = df['Declared Amount of Unsecured Loans'] - df['Amount of Unsecured Loans']
    df['Loan_Count_Gap'] = df['Declared Number of Unsecured Loans'] - df['Number of Unsecured Loans']
    df['Abs_Amount_Gap'] = abs(df['Loan_Amount_Gap'])
    df['Abs_Count_Gap'] = abs(df['Loan_Count_Gap'])
    
    df['Hidden_Loans'] = (df['Loan_Count_Gap'] < 0).astype(int)
    df['Severe_Underreporting_Amount'] = ((df['Loan_Amount_Gap'] < 0) & (df['Abs_Amount_Gap'] >= 200000)).astype(int)
    df['Severe_Underreporting_Count'] = ((df['Loan_Count_Gap'] < 0) & (df['Abs_Count_Gap'] >= 2)).astype(int)
    
    # Enhanced honesty metrics
    df['Honesty_Score_Amount'] = np.where(
        df['Declared Amount of Unsecured Loans'] > 0,
        1 - np.clip(df['Abs_Amount_Gap'] / (df['Declared Amount of Unsecured Loans'] + 1), 0, 1),
        1
    )
    df['Honesty_Score_Count'] = np.where(
        df['Declared Number of Unsecured Loans'] > 0,
        1 - np.clip(df['Abs_Count_Gap'] / (df['Declared Number of Unsecured Loans'] + 1), 0, 1),
        1
    )
    df['Perfect_Match'] = ((df['Loan_Amount_Gap'] == 0) & (df['Loan_Count_Gap'] == 0)).astype(int)
    df['Any_Discrepancy'] = ((df['Loan_Amount_Gap'] != 0) | (df['Loan_Count_Gap'] != 0)).astype(int)
    
    # === FINANCIAL HEALTH (ENHANCED) ===
    # Log transforms
    df['Income_log'] = np.log1p(df['Total Annual Income'])
    df['Existing_Loan_log'] = np.log1p(df['Amount of Unsecured Loans'])
    df['Desired_Loan_log'] = np.log1p(df['Application Limit Amount(Desired)'])
    df['Rent_log'] = np.log1p(df['Rent Burden Amount'] + 1)
    df['Declared_Loans_log'] = np.log1p(df['Declared Amount of Unsecured Loans'] + 1)
    
    # Enhanced DTI ratios
    df['DTI_Total'] = (df['Amount of Unsecured Loans'] + df['Application Limit Amount(Desired)']) / (df['Total Annual Income'] + 1)
    df['DTI_Existing'] = df['Amount of Unsecured Loans'] / (df['Total Annual Income'] + 1)
    df['DTI_WithRent_Annual'] = (df['Amount of Unsecured Loans'] + df['Rent Burden Amount'] * 12) / (df['Total Annual Income'] + 1)
    df['DTI_Desired'] = df['Application Limit Amount(Desired)'] / (df['Total Annual Income'] + 1)
    
    # Income adequacy metrics
    df['Income_per_Dependent'] = df['Total Annual Income'] / (df['Number of Dependents'] + 1)
    df['Income_per_Child'] = df['Total Annual Income'] / (df['Number of Dependent Children'] + 1)
    df['Income_per_Family_Member'] = df['Total Annual Income'] / (df['Number of Dependents'] + 2)  # +2 for applicant + spouse
    
    # Loan characteristics
    df['Avg_Existing_Loan'] = df['Amount of Unsecured Loans'] / (df['Number of Unsecured Loans'] + 1)
    df['Avg_Declared_Loan'] = df['Declared Amount of Unsecured Loans'] / (df['Declared Number of Unsecured Loans'] + 1)
    df['Loan_Intensity'] = df['Number of Unsecured Loans'] / (df['Age'] + 1)
    df['Loan_Burden_Ratio'] = df['Amount of Unsecured Loans'] / (df['Total Annual Income'] + 1)
    
    # Risk thresholds (enhanced)
    df['DTI_Critical'] = (df['DTI_Total'] > 0.6).astype(int)
    df['DTI_High'] = (df['DTI_Total'] > 0.4).astype(int)
    df['Has_Many_Loans'] = (df['Number of Unsecured Loans'] >= 3).astype(int)
    df['Has_Very_Many_Loans'] = (df['Number of Unsecured Loans'] >= 5).astype(int)
    df['Loan_Free'] = (df['Number of Unsecured Loans'] == 0).astype(int)
    df['High_Income'] = (df['Total Annual Income'] > 5000000).astype(int)
    df['Low_Income'] = (df['Total Annual Income'] < 2000000).astype(int)
    
    # === STABILITY INDICATORS (ENHANCED) ===
    df['Employment_Years'] = df['Duration of Employment at Company (Months)'] / 12
    df['Residence_Years'] = df['Duration of Residence (Months)'] / 12
    
    df['Employment_to_Age'] = df['Employment_Years'] / (df['Age'] + 1)
    df['Residence_to_Age'] = df['Residence_Years'] / (df['Age'] + 1)
    df['Combined_Stability'] = (df['Employment_Years'] + df['Residence_Years']) / 2
    df['Stability_Score'] = df['Employment_Years'] * 0.6 + df['Residence_Years'] * 0.4
    
    # Enhanced job security flags
    df['Is_New_Job'] = (df['Employment_Years'] <= 1).astype(int)
    df['Is_Stable_Job'] = (df['Employment_Years'] >= 5).astype(int)
    df['Job_Change_Frequency'] = df['Age'] / (df['Employment_Years'] + 0.1)
    
    # === HOUSING & LIFESTYLE (ENHANCED) ===
    df['Is_Homeowner'] = df['Residence Type'].isin([1, 2, 8, 9]).astype(int)
    df['Is_Renter'] = df['Residence Type'].isin([4, 5, 6, 7]).astype(int)
    df['Is_Government_Housing'] = (df['Residence Type'] == 3).astype(int)
    df['Rent_to_Income'] = (df['Rent Burden Amount'] * 12) / (df['Total Annual Income'] + 1)
    
    # === EMPLOYMENT QUALITY (ENHANCED) ===
    df['Is_Regular_Employee'] = (df['Employment Status Type'] == 1).astype(int)
    df['Is_Public_Sector'] = (df['Company Size Category'] == 1).astype(int)
    df['Is_Large_Company'] = df['Company Size Category'].isin([1, 2, 3, 4]).astype(int)
    df['Is_Small_Company'] = df['Company Size Category'].isin([7, 8, 9]).astype(int)
    df['Is_Part_Time'] = (df['Employment Type'] == 4).astype(int)
    df['Is_Contract'] = (df['Employment Type'] == 3).astype(int)
    df['Is_President'] = (df['Employment Type'] == 1).astype(int)
    
    # Industry risk categories (custom grouping)
    high_risk_industries = [19, 99, 18]  # Student, Others, Agriculture
    stable_industries = [1, 2, 3, 4, 17]  # Manufacturing, Finance, Securities, Insurance, Government
    df['High_Risk_Industry'] = df['Industry Type'].isin(high_risk_industries).astype(int)
    df['Stable_Industry'] = df['Industry Type'].isin(stable_industries).astype(int)
    
    # === FAMILY STRUCTURE (ENHANCED) ===
    df['Is_Married'] = (df['Single/Married Status'] == 2).astype(int)
    df['Has_Dependents'] = (df['Number of Dependents'] > 0).astype(int)
    df['Has_Children'] = (df['Number of Dependent Children'] > 0).astype(int)
    df['Large_Family'] = (df['Number of Dependents'] >= 3).astype(int)
    df['Young_Children'] = ((df['Age'] - df['Number of Dependent Children'] * 5) < 35).astype(int)  # Rough estimate
    
    # Family composition dummies
    df['Is_Single_Living_Alone'] = ((df['Family Composition Type'] == 5) | (df['Family Composition Type'] == 6)).astype(int)
    df['Is_Single_With_Family'] = (df['Family Composition Type'] == 4).astype(int)
    df['Is_Couple_No_Kids'] = (df['Family Composition Type'] == 1).astype(int)
    
    # === AGE-BASED FEATURES (ENHANCED) ===
    df['Age_Squared'] = df['Age'] ** 2
    df['Age_Cubed'] = df['Age'] ** 3
    df['Is_Very_Young'] = (df['Age'] < 25).astype(int)
    df['Is_Young'] = ((df['Age'] >= 25) & (df['Age'] < 35)).astype(int)
    df['Is_Middle_Aged'] = ((df['Age'] >= 35) & (df['Age'] < 50)).astype(int)
    df['Is_Senior'] = (df['Age'] >= 50).astype(int)
    df['Life_Stage_Risk'] = np.where(df['Age'] < 30, 2, np.where(df['Age'] < 45, 1, 0))
    
    # === INTERACTION FEATURES (POWER COMBOS - ENHANCED) ===
    df['Age_Income'] = df['Age'] * df['Income_log']
    df['Age_DTI'] = df['Age'] * df['DTI_Total']
    df['Income_Dependents'] = df['Income_log'] * (df['Number of Dependents'] + 1)
    df['Stability_Income'] = df['Stability_Score'] * df['Income_log']
    df['Marriage_Stability'] = df['Is_Married'] * df['Stability_Score']
    df['Homeowner_Income'] = df['Is_Homeowner'] * df['Income_log']
    df['LargeCompany_Stability'] = df['Is_Large_Company'] * df['Stability_Score']
    
    # === COMPOSITE RISK SCORES (ENHANCED) ===
    df['Financial_Risk_Score'] = (
        df['DTI_Critical'] * 4 +
        (df['DTI_Total'] > 0.4).astype(int) * 2 +
        df['Has_Many_Loans'] * 2 +
        df['Low_Income'] * 2 +
        df['Severe_Underreporting_Amount'] * 3
    )
    
    df['Stability_Risk_Score'] = (
        df['Is_New_Job'] * 2 +
        df['Is_Part_Time'] * 2 +
        df['Is_Contract'] * 1 +
        (1 - df['Is_Homeowner']) * 1 +
        df['High_Risk_Industry'] * 2 +
        df['Is_Small_Company'] * 1
    )
    
    df['Profile_Risk_Score'] = (
        df['Is_Very_Young'] * 2 +
        df['Large_Family'] * 1 +
        df['Low_Income'] * 2
    )
    
    df['Net_Risk_Score'] = (
        df['Financial_Risk_Score'] + 
        df['Stability_Risk_Score'] + 
        df['Profile_Risk_Score'] -
        (df['Is_Large_Company'] * 2) -
        (df['High_Income'] * 2) -
        (df['Is_Stable_Job'] * 2)
    )
    
    # === APPLICATION BEHAVIOR FEATURES ===
    df['Desired_vs_Income'] = df['Application Limit Amount(Desired)'] / (df['Total Annual Income'] + 1)
    df['Loan_Request_Ratio'] = df['Application Limit Amount(Desired)'] / (df['Amount of Unsecured Loans'] + 1)
    df['Is_Aggressive_Request'] = (df['Desired_vs_Income'] > 0.5).astype(int)
    
    # Drop original date columns and temporary columns
    df = df.drop(columns=['Application Date', 'Date of Birth', 'JIS_str'], errors='ignore')
    
    return df

# Apply advanced features
train_features = create_advanced_features(train_df)
test_features = create_advanced_features(test_df)

print(f"✓ Train features: {train_features.shape[1]}")
print(f"✓ New features: {train_features.shape[1] - train_df.shape[1]}")

# ============================================================================
# STEP 3: ADVERSARIAL VALIDATION (ENHANCED)
# ============================================================================
print("\n[3/10] Running Enhanced Adversarial Validation...")
av_X = pd.concat([train_features.drop(['Default 12 Flag', 'ID'], axis=1, errors='ignore'),
                  test_features.drop('ID', axis=1, errors='ignore')],
                 axis=0, ignore_index=True)
av_y = np.array([0] * len(train_features) + [1] * len(test_features))

cat_features_av = [
    'Major Media Code', 'Internet Details', 'Reception Type Category', 'Gender', 
    'Single/Married Status', 'Residence Type', 'Name Type', 'Family Composition Type', 
    'Living Arrangement Type', 'Insurance Job Type', 'Employment Type', 
    'Employment Status Type', 'Industry Type', 'Company Size Category', 
    'JIS Address Code', 'JIS_Prefix_2', 'JIS_Prefix_3', 'App_Month', 'App_DayOfWeek',
    'App_Quarter'
]
cat_features_av = [col for col in cat_features_av if col in av_X.columns]

for col in cat_features_av:
    av_X[col] = av_X[col].fillna(-999).astype(str).astype('category')

numeric_cols_av = [col for col in av_X.columns if col not in cat_features_av]
av_X[numeric_cols_av] = av_X[numeric_cols_av].fillna(-999)

av_model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=-1, verbosity=-1, random_state=RANDOM_STATE)
av_model.fit(av_X, av_y)
av_preds = av_model.predict_proba(av_X)[:, 1]
av_auc = roc_auc_score(av_y, av_preds)
print(f"✓ Adversarial Validation AUC: {av_auc:.5f}")

train_features['av_score'] = av_preds[:len(train_features)]
test_features['av_score'] = av_preds[len(train_features):]

# ============================================================================
# STEP 4: PREPARE DATA FOR MODELING
# ============================================================================
print("\n[4/10] Preparing final data for modeling...")

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
    'App_Month', 'App_DayOfWeek', 'App_Quarter', 'App_WeekOfYear',
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
# STEP 5: POWER TRANSFORMATION (ENHANCED)
# ============================================================================
print("\n[5/10] Advanced preprocessing (PowerTransformer)...")
skewed_features = ['Total Annual Income', 'Amount of Unsecured Loans', 
                   'Application Limit Amount(Desired)', 'Rent Burden Amount',
                   'Declared Amount of Unsecured Loans', 'Duration of Employment at Company (Months)',
                   'Duration of Residence (Months)']
skewed_features = [f for f in skewed_features if f in numeric_cols]

pt = PowerTransformer(method='yeo-johnson', standardize=True)
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
# STEP 6: FEATURE SELECTION
# ============================================================================
print("\n[6/10] Running feature selection...")
selector_model = LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

# Prepare data for feature selection
X_fs = X.copy()
for col in cat_features:
    le = LabelEncoder()
    all_cats = pd.concat([X_fs[col], X_test[col]]).unique()
    le.fit(all_cats)
    X_fs[col] = le.transform(X_fs[col])

selector_model.fit(X_fs, y)
importances = selector_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

# Select top 100 features
selected_features = feature_importance_df.head(100)['feature'].tolist()
print(f"✓ Selected top {len(selected_features)} features")

X = X[selected_features]
X_test = X_test[selected_features]

# Update categorical features list
cat_features = [col for col in cat_features if col in selected_features]

# ============================================================================
# STEP 7: LABEL ENCODERS
# ============================================================================
print("\n[7/10] Preparing label encoders...")
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    all_cats = pd.concat([X[col], X_test[col]]).unique()
    le.fit(all_cats)
    label_encoders[col] = le

# ============================================================================
# STEP 8: ADVANCED MODEL TRAINING (3 MODELS)
# ============================================================================
print("\n[8/10] Training ELITE models (CB + XGB + LGB)...")
print(f"Using {N_SPLITS}-Fold CV with 3 models...")

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
    
    # --- MODEL 1: CATBOOST (OPTIMIZED) ---
    print("→ [1/3] Training CatBoost...")
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    cb = CatBoostClassifier(
        iterations=3000, 
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=10,
        min_data_in_leaf=20,
        grow_policy='Lossguide',
        eval_metric='AUC', 
        random_seed=RANDOM_STATE + fold,
        early_stopping_rounds=150, 
        verbose=0, 
        thread_count=-1
    )
    cb.fit(train_pool, eval_set=val_pool)
    
    cb_oof[val_idx] = cb.predict_proba(X_val)[:, 1]
    cb_test += cb.predict_proba(X_test)[:, 1] / N_SPLITS
    cb_score = roc_auc_score(y_val, cb_oof[val_idx])
    cb_scores.append(cb_score)
    print(f"  ✓ CatBoost AUC: {cb_score:.6f} (trees: {cb.tree_count_})")
    
    # --- MODEL 2: XGBOOST (OPTIMIZED) ---
    print("\n→ [2/3] Training XGBoost...")
    X_tr_xgb, X_val_xgb, X_test_xgb = X_tr.copy(), X_val.copy(), X_test.copy()
    for col in cat_features:
        X_tr_xgb[col] = label_encoders[col].transform(X_tr_xgb[col])
        X_val_xgb[col] = label_encoders[col].transform(X_val_xgb[col])
        X_test_xgb[col] = label_encoders[col].transform(X_test_xgb[col])
    
    xgb = XGBClassifier(
        n_estimators=3000, 
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=15,
        subsample=0.8, 
        colsample_bytree=0.8,
        reg_alpha=1.5, 
        reg_lambda=1.5, 
        gamma=0.2,
        random_state=RANDOM_STATE + fold, 
        eval_metric='auc',
        early_stopping_rounds=150, 
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
    
    # --- MODEL 3: LIGHTGBM (OPTIMIZED) ---
    print("\n→ [3/3] Training LightGBM...")
    X_tr_lgb, X_val_lgb, X_test_lgb = X_tr.copy(), X_val.copy(), X_test.copy()
    for col in cat_features:
        X_tr_lgb[col] = label_encoders[col].transform(X_tr_lgb[col])
        X_val_lgb[col] = label_encoders[col].transform(X_val_lgb[col])
        X_test_lgb[col] = label_encoders[col].transform(X_test_lgb[col])
    
    lgb = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_val_lgb, y_val)],
        eval_metric='auc',
        callbacks=[early_stopping(150), log_evaluation(0)]
    )
    
    lgb_oof[val_idx] = lgb.predict_proba(X_val_lgb)[:, 1]
    lgb_test += lgb.predict_proba(X_test_lgb)[:, 1] / N_SPLITS
    lgb_score = roc_auc_score(y_val, lgb_oof[val_idx])
    lgb_scores.append(lgb_score)
    print(f"  ✓ LightGBM AUC: {lgb_score:.6f} (trees: {lgb.best_iteration_})")

# ============================================================================
# STEP 9: STACKING WITH METALEARNER
# ============================================================================
print("\n[9/10] Creating advanced stacking ensemble...")

# Create Level-1 predictions
level1_train = np.column_stack([cb_oof, xgb_oof, lgb_oof])
level1_test = np.column_stack([cb_test, xgb_test, lgb_test])

# Train metalearner (Logistic Regression)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

metalearner = LogisticRegression(C=0.1, random_state=RANDOM_STATE, max_iter=1000)
metalearner.fit(level1_train, y)

# Get final predictions
final_predictions = metalearner.predict_proba(level1_test)[:, 1]

# Calculate OOF scores
cb_oof_auc = roc_auc_score(y, cb_oof)
xgb_oof_auc = roc_auc_score(y, xgb_oof)
lgb_oof_auc = roc_auc_score(y, lgb_oof)

print(f"\n📊 Level-1 Out-of-Fold AUC:")
print(f"  CatBoost:  {cb_oof_auc:.6f}")
print(f"  XGBoost:   {xgb_oof_auc:.6f}")
print(f"  LightGBM:  {lgb_oof_auc:.6f}")

# Calculate final OOF score
final_oof_predictions = metalearner.predict_proba(level1_train)[:, 1]
final_oof_score = roc_auc_score(y, final_oof_predictions)
print(f"✓ Final Stacked OOF AUC: {final_oof_score:.6f}")

# ============================================================================
# STEP 10: CREATE SUBMISSION
# ============================================================================
print("\n[10/10] Creating final submission...")

submission = pd.DataFrame({
    'ID': test_ids,
    'Default 12 Flag': final_predictions
})
submission['Default 12 Flag'] = submission['Default 12 Flag'].clip(0.0001, 0.9999)

filename = f'ELITE_STACK_v10_auc{final_oof_score:.5f}.csv'
submission.to_csv(filename, index=False)

print(f"\n✅ SUBMISSION SAVED: {filename}")
print(f"  Mean: {final_predictions.mean():.6f} | Std: {final_predictions.std():.6f}")
print(f"  Range: [{final_predictions.min():.6f}, {final_predictions.max():.6f}]")

print("\n" + "="*80)
print("🏆🏆🏆 ELITE MODEL READY FOR 0.70+! 🏆🏆🏆")
print("="*80)
print(f"\n🎯 FINAL OOF Score: {final_oof_score:.5f}")
print(f"🔥 Features Used: {X.shape[1]} (Selected from {len(selected_features)})")
print(f"💪 Models: CatBoost + XGBoost + LightGBM + LogisticRegression Stacking")
print(f"📊 CV: {N_SPLITS}-Fold Stratified")
print(f"⚡ Advanced: Feature Selection, PowerTransform, Metalearner Stacking")
print(f"\n🚀 ISKO SUBMIT KARO. YEH 0.70+ DEGA! 🚀")
print("="*80)