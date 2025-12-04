import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import rankdata

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔥🔥🔥 FINAL BOSS SCRIPT (v8.0) 🔥🔥🔥")
print("="*80)
print("Features: 250+ | Models: 4 (3 L1 + 1 L2) | Hacks: Outliers + Stacking v2")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_SPLITS = 7  # 7 Folds for stable OOF
RANDOM_STATE = 42
N_CLUSTERS = 10 # Customer segments
OUTLIER_CAP = 0.99 # 99th percentile pe cap karenge

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/11] Loading data...")
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    print(f"✓ Train: {train_df.shape} | Test: {test_df.shape}")
    print(f"✓ Default rate: {train_df['Default 12 Flag'].mean():.4f}")
except FileNotFoundError:
    print("❌ ERROR: data/train.csv ya data/test.csv nahi mili.")
    raise

# ============================================================================
# STEP 2: GOD-LEVEL FEATURE ENGINEERING (200+ FEATURES)
# ============================================================================
print("\n[2/11] Creating 200+ god-level features...")

def create_god_features(df):
    """
    God-Level Function + JIS Cleaning
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
    df['App_WeekOfYear'] = df['Application Date'].dt.isocalendar().week
    df['App_DayOfYear'] = df['Application Date'].dt.dayofyear
    
    df['App_Hour'] = df['Application Time'] // 10000
    
    # Time-based risk patterns
    df['Is_Weekend'] = (df['App_DayOfWeek'] >= 5).astype(int)
    df['Is_BusinessHours'] = ((df['App_Hour'] >= 9) & (df['App_Hour'] <= 17)).astype(int)
    df['Is_LateNight'] = ((df['App_Hour'] >= 22) | (df['App_Hour'] <= 5)).astype(int)
    df['Is_MonthEnd'] = (df['Application Date'].dt.day >= 25).astype(int)
    
    # === FRAUD DETECTION (CRITICAL!) ===
    df['Loan_Amount_Gap'] = df['Declared Amount of Unsecured Loans'] - df['Amount of Unsecured Loans']
    df['Loan_Count_Gap'] = df['Declared Number of Unsecured Loans'] - df['Number of Unsecured Loans']
    df['Abs_Amount_Gap'] = abs(df['Loan_Amount_Gap'])
    df['Abs_Count_Gap'] = abs(df['Loan_Count_Gap'])
    
    df['Hidden_Loans'] = (df['Loan_Count_Gap'] < 0).astype(int)
    df['Severe_Underreporting'] = ((df['Loan_Amount_Gap'] < 0) & (df['Abs_Amount_Gap'] >= 200000)).astype(int)
    
    # Honesty metrics
    df['Honesty_Score'] = np.where(
        df['Declared Amount of Unsecured Loans'] > 0,
        1 - np.clip(df['Abs_Amount_Gap'] / (df['Declared Amount of Unsecured Loans'] + 1), 0, 1),
        1
    )
    df['Perfect_Match'] = ((df['Loan_Amount_Gap'] == 0) & (df['Loan_Count_Gap'] == 0)).astype(int)
    df['Has_Any_Discrepancy'] = ((df['Abs_Amount_Gap'] > 0) | (df['Abs_Count_Gap'] > 0)).astype(int)
    
    # === FINANCIAL HEALTH (ULTRA DETAILED) ===
    df['Income_log'] = np.log1p(df['Total Annual Income'])
    df['Existing_Loan_log'] = np.log1p(df['Amount of Unsecured Loans'])
    df['Desired_Loan_log'] = np.log1p(df['Application Limit Amount(Desired)'])
    df['Rent_log'] = np.log1p(df['Rent Burden Amount'])
    
    # DTI ratios (CRITICAL!)
    df['DTI_Existing'] = df['Amount of Unsecured Loans'] / (df['Total Annual Income'] + 1)
    df['DTI_Desired'] = df['Application Limit Amount(Desired)'] / (df['Total Annual Income'] + 1)
    df['DTI_Total'] = (df['Amount of Unsecured Loans'] + df['Application Limit Amount(Desired)']) / (df['Total Annual Income'] + 1)
    df['DTI_WithRent_Annual'] = (df['Amount of Unsecured Loans'] + df['Rent Burden Amount'] * 12) / (df['Total Annual Income'] + 1)
    
    # Income adequacy
    df['Income_per_Dependent'] = df['Total Annual Income'] / (df['Number of Dependents'] + 1)
    df['Income_per_Child'] = df['Total Annual Income'] / (df['Number of Dependent Children'] + 1)
    
    # Loan characteristics
    df['Avg_Existing_Loan'] = df['Amount of Unsecured Loans'] / (df['Number of Unsecured Loans'] + 1)
    df['Desired_vs_Existing_Ratio'] = df['Application Limit Amount(Desired)'] / (df['Amount of Unsecured Loans'] + 1)
    df['Loan_Intensity'] = df['Number of Unsecured Loans'] / (df['Age'] + 1)
    
    # Risk thresholds
    df['DTI_High'] = ((df['DTI_Total'] > 0.4) & (df['DTI_Total'] <= 0.6)).astype(int)
    df['DTI_Critical'] = (df['DTI_Total'] > 0.6).astype(int)
    df['Has_Many_Loans'] = (df['Number of Unsecured Loans'] >= 3).astype(int)
    df['Loan_Free'] = (df['Number of Unsecured Loans'] == 0).astype(int)
    
    # Financial freedom
    df['Free_Income_Annual'] = df['Total Annual Income'] - (df['Amount of Unsecured Loans'] * 0.15 + df['Rent Burden Amount'] * 12)
    df['Free_Income_Ratio'] = df['Free_Income_Annual'] / (df['Total Annual Income'] + 1)
    df['Is_Financially_Stressed'] = (df['Free_Income_Ratio'] < 0.3).astype(int)
    
    # === STABILITY INDICATORS ===
    df['Employment_Years'] = df['Duration of Employment at Company (Months)'] / 12
    df['Residence_Years'] = df['Duration of Residence (Months)'] / 12
    
    df['Employment_to_Age'] = df['Employment_Years'] / (df['Age'] + 1)
    df['Residence_to_Age'] = df['Residence_Years'] / (df['Age'] + 1)
    df['Combined_Stability'] = (df['Employment_Years'] + df['Residence_Years']) / 2
    
    # Job security flags
    df['Is_New_Job'] = (df['Employment_Years'] <= 1).astype(int)
    df['Is_Long_Tenure'] = (df['Employment_Years'] > 60).astype(int)
    
    # Residence stability
    df['Is_Recent_Move'] = (df['Residence_Years'] <= 1).astype(int)
    
    # === HOUSING & LIFESTYLE ===
    df['Is_Homeowner'] = df['Residence Type'].isin([1, 2, 8, 9]).astype(int)
    df['Has_Mortgage'] = df['Residence Type'].isin([2, 9]).astype(int)
    df['Is_Renter'] = df['Residence Type'].isin([4, 5, 6, 7]).astype(int)
    
    df['Rent_to_Income'] = df['Rent Burden Amount'] / (df['Total Annual Income'] + 1)
    df['Rent_Burden_High'] = (df['Rent_to_Income'] > 0.3).astype(int)
    
    # === EMPLOYMENT QUALITY ===
    df['Is_Regular_Employee'] = (df['Employment Status Type'] == 1).astype(int)
    df['Is_Public_Sector'] = (df['Company Size Category'] == 1).astype(int)
    df['Is_Large_Company'] = df['Company Size Category'].isin([1, 2, 3, 4]).astype(int)
    df['Is_Small_Company'] = df['Company Size Category'].isin([7, 8, 9]).astype(int)
    df['Is_Part_Time'] = (df['Employment Type'] == 4).astype(int)
    df['Is_Student'] = (df['Industry Type'] == 19).astype(int)
    
    # Insurance type
    df['Has_Company_Insurance'] = df['Insurance Job Type'].isin([1, 3]).astype(int)
    
    # === FAMILY STRUCTURE ===
    df['Is_Married'] = (df['Single/Married Status'] == 2).astype(int)
    df['Has_Dependents'] = (df['Number of Dependents'] > 0).astype(int)
    df['Has_Children'] = (df['Number of Dependent Children'] > 0).astype(int)
    df['Large_Family'] = (df['Number of Dependents'] >= 3).astype(int)
    df['Is_Single_Parent'] = ((df['Is_Married'] == 0) & (df['Has_Children'] == 1)).astype(int)
    df['Adult_Dependents'] = df['Number of Dependents'] - df['Number of Dependent Children']
    
    # === DIGITAL BEHAVIOR ===
    df['Is_Mobile_App'] = df['Reception Type Category'].isin([1701, 1801]).astype(int)
    df['Is_Paid_Search'] = (df['Internet Details'] == 4).astype(int)
    
    # === AGE-BASED FEATURES ===
    df['Age_Squared'] = df['Age'] ** 2
    df['Is_Very_Young'] = (df['Age'] < 25).astype(int)
    df['Is_Senior'] = (df['Age'] >= 60).astype(int)
    df['Years_to_Retirement'] = np.maximum(65 - df['Age'], 0)
    
    # === INTERACTION FEATURES (POWER COMBOS) ===
    df['Age_Income'] = df['Age'] * df['Income_log']
    df['Age_DTI'] = df['Age'] * df['DTI_Total']
    df['Income_Dependents'] = df['Income_log'] * (df['Number of Dependents'] + 1)
    df['Income_Employment'] = df['Income_log'] * (df['Employment_Years'] + 1)
    df['Stability_Income'] = df['Combined_Stability'] * df['Income_log']
    df['DTI_Dependents'] = df['DTI_Total'] * (df['Number of Dependents'] + 1)
    
    # === COMPOSITE RISK SCORES ===
    df['Fraud_Risk_Score'] = (
        df['Severe_Underreporting'] * 5 +
        df['Hidden_Loans'] * 2 +
        (1 - df['Honesty_Score']) * 3
    )
    df['Financial_Risk_Score'] = (
        df['DTI_Critical'] * 4 +
        df['DTI_High'] * 2 +
        df['Has_Many_Loans'] * 2 +
        df['Is_Financially_Stressed'] * 2
    )
    df['Employment_Risk_Score'] = (
        df['Is_New_Job'] * 2 +
        df['Is_Part_Time'] * 2 +
        df['Is_Student'] * 3 +
        df['Is_Small_Company'] * 1
    )
    df['Life_Risk_Score'] = (
        df['Is_Very_Young'] * 2 +
        df['Is_Single_Parent'] * 2 +
        df['Large_Family'] * 1 +
        (1 - df['Is_Homeowner']) * 1
    )
    df['Total_Risk_Score'] = (
        df['Fraud_Risk_Score'] * 2.0 +
        df['Financial_Risk_Score'] * 1.5 +
        df['Employment_Risk_Score'] * 1.2 +
        df['Life_Risk_Score'] * 1.0
    )
    
    df['Protection_Score'] = (
        df['Is_Homeowner'] * 2 +
        df['Is_Long_Tenure'] * 2 +
        df['Is_Public_Sector'] * 2 +
        df['Is_Large_Company'] * 1 +
        df['Loan_Free'] * 3 +
        df['Perfect_Match'] * 2
    )
    df['Net_Risk_Score'] = df['Total_Risk_Score'] - df['Protection_Score']
    
    # === STATISTICAL FEATURES ===
    df['Income_Percentile'] = df['Total Annual Income'].rank(pct=True)
    df['Loan_Percentile'] = df['Application Limit Amount(Desired)'].rank(pct=True)
    df['DTI_Percentile'] = df['DTI_Total'].rank(pct=True)
    
    # Drop date columns
    df = df.drop(columns=['Application Date', 'Date of Birth', 'JIS_str'], errors='ignore')
    
    return df

# Apply god-level features
train_features = create_god_features(train_df)
test_features = create_god_features(test_df)

print(f"✓ Train features: {train_features.shape[1]}")
print(f"✓ New features: {train_features.shape[1] - train_df.shape[1]}")

# ============================================================================
# STEP 3: META-HACK #1 - ADVERSARIAL VALIDATION
# ============================================================================
print("\n[3/11] Running Adversarial Validation (Meta-Hack)...")

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

print(f"✓ Adversarial Validation complete.")

# Add AV score as a new feature
train_features['av_score'] = av_preds[:len(train_features)]
test_features['av_score'] = av_preds[len(train_features):]

# ============================================================================
# STEP 4: OUTLIER CLEANING & FLAGGING (NEW!)
# ============================================================================
print("\n[4/11] Cleaning & Flagging Outliers...")

outlier_cols = [
    'Total Annual Income', 'Amount of Unsecured Loans', 'Declared Amount of Unsecured Loans',
    'Application Limit Amount(Desired)', 'Rent Burden Amount', 'Age', 'Employment_Years',
    'Residence_Years', 'Total_Risk_Score', 'Net_Risk_Score'
]
outlier_cols = [col for col in outlier_cols if col in train_features.columns]

for col in outlier_cols:
    low_cap = train_features[col].quantile(1.0 - OUTLIER_CAP)
    high_cap = train_features[col].quantile(OUTLIER_CAP)
    
    # Create flags
    train_features[f'{col}_is_outlier_high'] = (train_features[col] > high_cap).astype(int)
    train_features[f'{col}_is_outlier_low'] = (train_features[col] < low_cap).astype(int)
    test_features[f'{col}_is_outlier_high'] = (test_features[col] > high_cap).astype(int)
    test_features[f'{col}_is_outlier_low'] = (test_features[col] < low_cap).astype(int)
    
    # Cap (Winsorize) the features
    train_features[col] = train_features[col].clip(low_cap, high_cap)
    test_features[col] = test_features[col].clip(low_cap, high_cap)

print(f"✓ Outliers flagged & capped for {len(outlier_cols)} features.")

# ============================================================================
# STEP 5: META-HACK #2 - KMEANS CLUSTERING
# ============================================================================
print("\n[5/11] Creating KMeans Cluster feature (Meta-Hack)...")

cluster_features = [
    'Age', 'Income_log', 'DTI_Total', 'Net_Risk_Score', 
    'Combined_Stability', 'Loan_Intensity', 'av_score'
]
cluster_features = [col for col in cluster_features if col in train_features.columns]

X_cluster = pd.concat([train_features[cluster_features], test_features[cluster_features]], axis=0)
X_cluster = X_cluster.fillna(-999).replace([np.inf, -np.inf], -999)

scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster_scaled)

train_features['Cluster'] = cluster_labels[:len(train_features)]
test_features['Cluster'] = cluster_labels[len(train_features):]
print(f"✓ Cluster feature created ({N_CLUSTERS} clusters).")

# ============================================================================
# STEP 6: TARGET ENCODING (on new features)
# ============================================================================
print("\n[6/11] Creating Target Encodings...")

y = train_features['Default 12 Flag']
X = train_features.drop(columns=['Default 12 Flag', 'ID'], errors='ignore')
X_test = test_features.drop(columns=['ID'], errors='ignore')

te_features = ['JIS_Prefix_2', 'JIS_Prefix_3', 'Industry Type', 'Cluster']
te_features = [col for col in te_features if col in X.columns]

target_encodings = {}

for col in te_features:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)
    
    skf_te = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    col_encodings = np.zeros(len(X))
    
    for train_idx, val_idx in skf_te.split(X, y):
        encoding_map = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).mean().to_dict()
        overall_mean = y.iloc[train_idx].mean()
        col_encodings[val_idx] = X[col].iloc[val_idx].map(encoding_map).fillna(overall_mean)
    
    encoding_map_full = y.groupby(X[col]).mean().to_dict()
    encoding_map_full['__OVERALL__'] = y.mean()
    target_encodings[col] = encoding_map_full
    
    X[f'{col}_te'] = col_encodings
    X_test[f'{col}_te'] = X_test[col].map(encoding_map_full).fillna(encoding_map_full['__OVERALL__'])

print(f"✓ Created {len(target_encodings)} target encodings.")

# ============================================================================
# STEP 7: PREPARE FINAL DATA
# ============================================================================
print("\n[7/11] Preparing final data for modeling...")

test_ids = test_features['ID']
X = X.reindex(columns=X_test.columns.union(X.columns), fill_value=0)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

cat_features = [
    'Major Media Code', 'Internet Details', 'Reception Type Category',
    'Gender', 'Single/Married Status', 'Residence Type', 'Name Type',
    'Family Composition Type', 'Living Arrangement Type', 
    'Insurance Job Type', 'Employment Type', 'Employment Status Type',
    'Industry Type', 'Company Size Category', 'JIS Address Code',
    'App_Month', 'App_DayOfWeek', 'App_Quarter',
    'JIS_Prefix_2', 'JIS_Prefix_3', 'Cluster'
]
cat_features = [col for col in cat_features if col in X.columns]

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
# STEP 8: L1 TRAINING (CATBOOST ONLY)
# ============================================================================
print("\n[8/11] Training LEVEL-1 CatBoost model...")
print(f"Yeh {N_SPLITS}-Fold CV hai, time lega...")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

cb_oof = np.zeros(len(X))
cb_test = np.zeros(len(X_test))
cb_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*70}\nFOLD {fold+1}/{N_SPLITS}\n{'='*70}")
    
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    cb = CatBoostClassifier(
        iterations=5000, 
        learning_rate=0.018, 
        depth=10,
        l2_leaf_reg=7, 
        min_data_in_leaf=15,
        bagging_temperature=0.8, 
        random_strength=0.8,
        eval_metric='AUC', 
        random_seed=RANDOM_STATE + fold,
        early_stopping_rounds=250, 
        verbose=0, 
        thread_count=-1
    )
    cb.fit(train_pool, eval_set=val_pool)
    
    cb_oof[val_idx] = cb.predict_proba(X_val)[:, 1]
    cb_test += cb.predict_proba(X_test)[:, 1] / N_SPLITS
    cb_score = roc_auc_score(y_val, cb_oof[val_idx])
    cb_scores.append(cb_score)
    print(f"  ✓ CatBoost AUC: {cb_score:.6f} (trees: {cb.tree_count_})")

cb_oof_auc = roc_auc_score(y, cb_oof)
print(f"\n📊 FINAL L1 OOF AUC: {cb_oof_auc:.6f}")

# ============================================================================
# STEP 9: PSEUDO-LABELING (THE HACK)
# ============================================================================
print("\n[9/11] Applying Pseudo-Labeling Hack...")

pseudo_idx_high = np.where(cb_test > PL_THRESH_HIGH)[0]
pseudo_idx_low = np.where(cb_test < PL_THRESH_LOW)[0]

print(f"  ✓ Found {len(pseudo_idx_high)} high-confidence 'Default' (1) samples")
print(f"  ✓ Found {len(pseudo_idx_low)} high-confidence 'No-Default' (0) samples")

X_pseudo_high = X_test.iloc[pseudo_idx_high]
y_pseudo_high = pd.Series(np.ones(len(pseudo_idx_high)), index=X_pseudo_high.index)

X_pseudo_low = X_test.iloc[pseudo_idx_low]
y_pseudo_low = pd.Series(np.zeros(len(pseudo_idx_low)), index=X_pseudo_low.index)

X_final_train = pd.concat([X, X_pseudo_high, X_pseudo_low], axis=0)
y_final_train = pd.concat([y, y_pseudo_high, y_pseudo_low], axis=0)

print(f"  ✓ New training data shape: {X_final_train.shape}")

# ============================================================================
# STEP 10: FINAL MODEL TRAINING
# ============================================================================
print("\n[10/11] Training FINAL model on Original + Pseudo-Labels...")

final_params = {
    'iterations': 5000,
    'learning_rate': 0.015,
    'depth': 10,
    'l2_leaf_reg': 7,
    'eval_metric': 'AUC',
    'random_seed': RANDOM_STATE,
    'verbose': 500,
    'thread_count': -1
}

final_model = CatBoostClassifier(**final_params)
final_pool = Pool(X_final_train, y_final_train, cat_features=cat_features)

final_model.fit(
    final_pool,
    eval_set=final_pool.split(0.05), 
    early_stopping_rounds=200 
)

print("✓ Final model training complete.")

# ============================================================================
# STEP 11: CREATE SUBMISSION
# ============================================================================
print("\n[11/11] Creating final submission...")

final_predictions = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'ID': test_ids,
    'Default 12 Flag': final_predictions
})

filename = f'GOD_TIER_CATBOOST_v8_auc{cb_oof_auc:.5f}.csv'
submission.to_csv(filename, index=False)

print(f"\n✅ SUBMISSION SAVED: {filename}")
print(f"  Mean pred: {final_predictions.mean():.6f}")
print(f"  Std pred:  {final_predictions.std():.6f}")

files.download(filename)

print("\n" + "="*80)
print("🏆🏆🏆 DUNIYA KA BEST CATBOOST MODEL READY! 🏆🏆🏆")
print("="*80)
print(f"\n🎯 L1 OOF Score: {cb_oof_auc:.5f} (Iske aas-paas LB score hoga)")
print(f"🔥 Features Used: {X.shape[1]}")
print(f"💪 Model: CatBoost ONLY")
print(f"📊 CV: {N_SPLITS}-Fold Stratified")
print(f"⚡ Meta-Hacks:")
print("  ✓ 250+ 'God-Tier' Features")
print("  ✓ Outlier Capping & Flagging (NEW!)")
print("  ✓ JIS Address Code Cleaning")
print("  ✓ Adversarial Validation (av_score) Feature")
print("  ✓ KMeans Cluster Feature")
print("  ✓ Target Encoding")
print("  ✓ Pseudo-Labeling (Final Hack)")
print("\n🚀 AB ISSE SUBMIT KARO. YEH AAPKA BEST SHOT HAI. 🚀")
print("="*80)