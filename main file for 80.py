import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import rankdata
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔥🔥🔥 ULTIMATE WINNING MODEL - COMPLETE VERSION 🔥🔥🔥")
print("="*80)
print("Target: 0.80+ AUC | All Advanced Techniques Included")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_SPLITS = 12
RANDOM_STATE = 44
USE_NEURAL_NET = True
USE_META_LEARNING = True

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/10] Loading data...")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"✓ Train: {train_df.shape} | Test: {test_df.shape}")
print(f"✓ Default rate: {train_df['Default 12 Flag'].mean():.4f}")

# ============================================================================
# STEP 2: MEGA FEATURE ENGINEERING
# ============================================================================
print("\n[2/10] Creating MEGA feature set...")

def create_mega_features(df, target_encodings=None):
    """COMPLETE feature engineering with ALL techniques"""
    df = df.copy()
    
    # === TEMPORAL ===
    df['Application Date'] = pd.to_datetime(df['Application Date'], format='%Y/%m/%d', errors='coerce')
    df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], format='%Y/%m/%d', errors='coerce')
    
    df['Age'] = (df['Application Date'] - df['Date of Birth']).dt.days / 365.25
    df['App_Year'] = df['Application Date'].dt.year
    df['App_Month'] = df['Application Date'].dt.month
    df['App_Day'] = df['Application Date'].dt.day
    df['App_DayOfWeek'] = df['Application Date'].dt.dayofweek
    df['App_Quarter'] = df['Application Date'].dt.quarter
    df['App_WeekOfYear'] = df['Application Date'].dt.isocalendar().week
    
    df['App_Hour'] = df['Application Time'] // 10000
    df['App_Minute'] = (df['Application Time'] % 10000) // 100
    
    # Time patterns
    df['Is_Weekend'] = (df['App_DayOfWeek'] >= 5).astype(int)
    df['Is_BusinessHours'] = ((df['App_Hour'] >= 9) & (df['App_Hour'] <= 17)).astype(int)
    df['Is_LateNight'] = ((df['App_Hour'] >= 22) | (df['App_Hour'] <= 5)).astype(int)
    df['Is_OfficeHours'] = ((df['Is_BusinessHours'] == 1) & (df['Is_Weekend'] == 0)).astype(int)
    df['Is_MonthEnd'] = (df['App_Day'] >= 25).astype(int)
    df['Is_MonthStart'] = (df['App_Day'] <= 5).astype(int)
    
    # Japanese fiscal patterns
    df['Is_Fiscal_Year_End'] = (df['App_Month'] == 3).astype(int)
    df['Is_Bonus_Month'] = df['App_Month'].isin([6, 12]).astype(int)
    
    # === FRAUD DETECTION (CRITICAL!) ===
    df['Loan_Amount_Gap'] = df['Declared Amount of Unsecured Loans'] - df['Amount of Unsecured Loans']
    df['Loan_Count_Gap'] = df['Declared Number of Unsecured Loans'] - df['Number of Unsecured Loans']
    df['Abs_Amount_Gap'] = abs(df['Loan_Amount_Gap'])
    df['Abs_Count_Gap'] = abs(df['Loan_Count_Gap'])
    
    df['Hidden_Loans'] = (df['Loan_Count_Gap'] < 0).astype(int)
    df['Hidden_Amount'] = (df['Loan_Amount_Gap'] < 0).astype(int)
    df['Severe_Underreporting'] = ((df['Loan_Amount_Gap'] < 0) & (df['Abs_Amount_Gap'] >= 200000)).astype(int)
    df['Major_Underreporting'] = ((df['Loan_Amount_Gap'] < 0) & (df['Abs_Amount_Gap'] >= 50000) & (df['Abs_Amount_Gap'] < 200000)).astype(int)
    
    # Honesty scores
    df['Honesty_Score'] = np.where(
        df['Declared Amount of Unsecured Loans'] > 0,
        1 - np.clip(df['Abs_Amount_Gap'] / (df['Declared Amount of Unsecured Loans'] + 1), 0, 1),
        1
    )
    df['Count_Honesty'] = np.where(
        df['Declared Number of Unsecured Loans'] > 0,
        1 - np.clip(df['Abs_Count_Gap'] / (df['Declared Number of Unsecured Loans'] + 1), 0, 1),
        1
    )
    df['Combined_Honesty'] = (df['Honesty_Score'] + df['Count_Honesty']) / 2
    df['Perfect_Match'] = ((df['Loan_Amount_Gap'] == 0) & (df['Loan_Count_Gap'] == 0)).astype(int)
    
    # === FINANCIAL HEALTH ===
    df['Income_log'] = np.log1p(df['Total Annual Income'])
    df['Existing_Loan_log'] = np.log1p(df['Amount of Unsecured Loans'])
    df['Desired_Loan_log'] = np.log1p(df['Application Limit Amount(Desired)'])
    df['Rent_log'] = np.log1p(df['Rent Burden Amount'])
    
    # DTI ratios (SUPER IMPORTANT!)
    df['DTI_Existing'] = df['Amount of Unsecured Loans'] / (df['Total Annual Income'] + 1)
    df['DTI_Desired'] = df['Application Limit Amount(Desired)'] / (df['Total Annual Income'] + 1)
    df['DTI_Total'] = (df['Amount of Unsecured Loans'] + df['Application Limit Amount(Desired)']) / (df['Total Annual Income'] + 1)
    df['DTI_WithRent'] = (df['Amount of Unsecured Loans'] + df['Rent Burden Amount'] * 12) / (df['Total Annual Income'] + 1)
    
    df['Income_per_Dependent'] = df['Total Annual Income'] / (df['Number of Dependents'] + 1)
    df['Monthly_Income'] = df['Total Annual Income'] / 12
    df['Avg_Existing_Loan'] = df['Amount of Unsecured Loans'] / (df['Number of Unsecured Loans'] + 1)
    
    # Risk categories
    df['DTI_Critical'] = (df['DTI_Total'] > 0.6).astype(int)
    df['DTI_High'] = ((df['DTI_Total'] > 0.4) & (df['DTI_Total'] <= 0.6)).astype(int)
    df['DTI_Safe'] = (df['DTI_Total'] <= 0.3).astype(int)
    df['Has_Multiple_Loans'] = (df['Number of Unsecured Loans'] >= 2).astype(int)
    df['Has_Many_Loans'] = (df['Number of Unsecured Loans'] >= 3).astype(int)
    df['Loan_Free'] = (df['Number of Unsecured Loans'] == 0).astype(int)
    
    df['Free_Income_Annual'] = df['Total Annual Income'] - (df['Amount of Unsecured Loans'] * 0.15 + df['Rent Burden Amount'] * 12)
    df['Free_Income_Ratio'] = df['Free_Income_Annual'] / (df['Total Annual Income'] + 1)
    df['Is_Financially_Stressed'] = (df['Free_Income_Ratio'] < 0.3).astype(int)
    
    # === STABILITY ===
    df['Employment_Years'] = df['Duration of Employment at Company (Months)'] / 12
    df['Residence_Years'] = df['Duration of Residence (Months)'] / 12
    df['Employment_to_Age'] = df['Employment_Years'] / (df['Age'] + 1)
    df['Residence_to_Age'] = df['Residence_Years'] / (df['Age'] + 1)
    df['Combined_Stability'] = (df['Employment_Years'] + df['Residence_Years']) / 2
    
    df['Is_New_Job'] = (df['Employment_Years'] <= 1).astype(int)
    df['Is_Long_Tenure'] = (df['Employment_Years'] > 5).astype(int)
    df['Is_Very_Long_Tenure'] = (df['Employment_Years'] > 10).astype(int)
    df['Is_Recent_Move'] = (df['Residence_Years'] <= 1).astype(int)
    df['Frequent_Mover'] = ((df['Age'] > 25) & (df['Residence_Years'] < 2)).astype(int)
    
    # Japanese culture: job hopping penalty
    df['Job_Hopper_Penalty'] = ((df['Age'] > 30) & (df['Employment_Years'] < 3)).astype(int)
    
    # === HOUSING ===
    df['Is_Homeowner'] = df['Residence Type'].isin([1, 2, 8, 9]).astype(int)
    df['Has_Mortgage'] = df['Residence Type'].isin([2, 9]).astype(int)
    df['Is_Renter'] = df['Residence Type'].isin([4, 5, 6, 7]).astype(int)
    df['Has_Own_Home_Free'] = df['Residence Type'].isin([1, 8]).astype(int)
    df['Rent_to_Income'] = df['Rent Burden Amount'] / ((df['Total Annual Income'] / 12) + 1)
    df['Rent_Burden_High'] = (df['Rent_to_Income'] > 0.3).astype(int)
    
    # Japanese culture: homeownership trust boost
    df['Home_Trust_Boost'] = df['Is_Homeowner'] * df['Residence_Years'] * 0.1
    
    # === EMPLOYMENT ===
    df['Is_Regular_Employee'] = (df['Employment Status Type'] == 1).astype(int)
    df['Is_Public_Sector'] = (df['Company Size Category'] == 1).astype(int)
    df['Is_Listed_Company'] = (df['Company Size Category'] == 2).astype(int)
    df['Is_Large_Company'] = df['Company Size Category'].isin([1, 2, 3, 4]).astype(int)
    df['Is_Small_Company'] = df['Company Size Category'].isin([7, 8, 9]).astype(int)
    df['Is_Part_Time'] = (df['Employment Type'] == 4).astype(int)
    df['Is_Employee'] = (df['Employment Type'] == 2).astype(int)
    df['Is_Student'] = (df['Industry Type'] == 19).astype(int)
    df['Is_Financial'] = df['Industry Type'].isin([2, 3, 4]).astype(int)
    df['Is_Stable_Industry'] = df['Industry Type'].isin([1, 2, 5, 15, 16, 17]).astype(int)
    
    # Elite company (Japanese prestige)
    df['Elite_Company'] = (
        (df['Company Size Category'].isin([1, 2])) & 
        (df['Industry Type'].isin([1, 2, 15, 17]))
    ).astype(int)
    
    # === FAMILY ===
    df['Is_Married'] = (df['Single/Married Status'] == 2).astype(int)
    df['Has_Dependents'] = (df['Number of Dependents'] > 0).astype(int)
    df['Has_Children'] = (df['Number of Dependent Children'] > 0).astype(int)
    df['Large_Family'] = (df['Number of Dependents'] >= 3).astype(int)
    df['Is_Single_Parent'] = ((df['Is_Married'] == 0) & (df['Has_Children'] == 1)).astype(int)
    df['Children_Ratio'] = df['Number of Dependent Children'] / (df['Number of Dependents'] + 1)
    
    # === DIGITAL BEHAVIOR ===
    df['Is_Mobile_App'] = df['Reception Type Category'].isin([1701, 1801]).astype(int)
    df['Is_Internet'] = (df['Major Media Code'] == 11).astype(int)
    df['Is_PC'] = (df['Reception Type Category'] == 502).astype(int)
    df['Digital_Savvy'] = (df['Is_Internet'] + df['Is_Mobile_App']).astype(int)
    
    # === AGE GROUPS ===
    df['Is_Young'] = (df['Age'] < 35).astype(int)
    df['Is_Prime_Age'] = ((df['Age'] >= 35) & (df['Age'] < 50)).astype(int)
    df['Is_Mature'] = ((df['Age'] >= 50) & (df['Age'] < 60)).astype(int)
    df['Is_Senior'] = (df['Age'] >= 60).astype(int)
    df['Age_Group_10'] = (df['Age'] // 10).astype(int)
    
    # === REGIONAL (JIS Codes) ===
    df['Prefecture'] = df['JIS Address Code'].astype(str).str[:2]
    urban_codes = ['13', '27', '23', '14', '28']  # Tokyo, Osaka, Aichi, Kanagawa, Hyogo
    df['Is_Urban'] = df['Prefecture'].isin(urban_codes).astype(int)
    
    # === COMPOSITE RISK SCORES ===
    fraud_risk = (
        df['Severe_Underreporting'] * 5 +
        df['Major_Underreporting'] * 3 +
        df['Hidden_Loans'] * 2 +
        (1 - df['Combined_Honesty']) * 3
    )
    df['Fraud_Risk_Score'] = fraud_risk
    
    financial_risk = (
        df['DTI_Critical'] * 4 +
        df['DTI_High'] * 2 +
        df['Has_Multiple_Loans'] * 2 +
        df['Is_Financially_Stressed'] * 2 +
        df['Rent_Burden_High'] * 1
    )
    df['Financial_Risk_Score'] = financial_risk
    
    employment_risk = (
        df['Is_New_Job'] * 2 +
        df['Is_Part_Time'] * 2 +
        df['Is_Student'] * 3 +
        df['Is_Small_Company'] * 1 +
        df['Job_Hopper_Penalty'] * 2
    )
    df['Employment_Risk_Score'] = employment_risk
    
    stability_risk = (
        df['Is_Recent_Move'] * 1 +
        df['Frequent_Mover'] * 2 +
        (1 - df['Is_Homeowner']) * 1
    )
    df['Stability_Risk_Score'] = stability_risk
    
    df['Total_Risk_Score'] = (
        fraud_risk * 2.5 +
        financial_risk * 2.0 +
        employment_risk * 1.5 +
        stability_risk * 1.0
    )
    
    protection = (
        df['Is_Homeowner'] * 2 +
        df['Is_Long_Tenure'] * 2 +
        df['Is_Public_Sector'] * 2 +
        df['Elite_Company'] * 2 +
        df['Loan_Free'] * 3 +
        df['Perfect_Match'] * 2 +
        (df['Combined_Honesty'] > 0.9).astype(int) * 2
    )
    df['Protection_Score'] = protection
    df['Net_Risk_Score'] = df['Total_Risk_Score'] - df['Protection_Score']
    
    # === KEY INTERACTIONS ===
    df['Age_Income'] = df['Age'] * df['Income_log']
    df['Age_DTI'] = df['Age'] * df['DTI_Total']
    df['Income_DTI'] = df['Income_log'] * df['DTI_Total']
    df['Stability_Income'] = df['Combined_Stability'] * df['Income_log']
    df['Honesty_DTI'] = df['Combined_Honesty'] * (1 - df['DTI_Total'])
    df['Age_Employment'] = df['Age'] * df['Employment_Years']
    df['Income_Dependents'] = df['Income_log'] * (df['Number of Dependents'] + 1)
    df['DTI_Dependents'] = df['DTI_Total'] * (df['Number of Dependents'] + 1)
    
    # === PERCENTILES ===
    df['Income_Percentile'] = df['Total Annual Income'].rank(pct=True)
    df['DTI_Percentile'] = df['DTI_Total'].rank(pct=True)
    df['Risk_Percentile'] = df['Total_Risk_Score'].rank(pct=True)
    df['Age_Percentile'] = df['Age'].rank(pct=True)
    
    df['Is_High_Income'] = (df['Income_Percentile'] > 0.75).astype(int)
    df['Is_Low_Income'] = (df['Income_Percentile'] < 0.25).astype(int)
    
    # === POLYNOMIAL FEATURES ===
    df['DTI_squared'] = df['DTI_Total'] ** 2
    df['DTI_cubed'] = df['DTI_Total'] ** 3
    df['Age_squared'] = df['Age'] ** 2
    df['Income_squared'] = df['Income_log'] ** 2
    df['Honesty_squared'] = df['Combined_Honesty'] ** 2
    df['Risk_squared'] = df['Total_Risk_Score'] ** 2
    
    # === TARGET ENCODING ===
    if target_encodings is not None:
        for col, encoding_map in target_encodings.items():
            if col in df.columns:
                df[f'{col}_target_enc'] = df[col].map(encoding_map).fillna(encoding_map.get('__OVERALL__', 0.5))
    
    # Clean up
    df = df.drop(columns=['Application Date', 'Date of Birth', 'Prefecture'], errors='ignore')
    
    return df

# ============================================================================
# STEP 3: PREPARE FEATURES
# ============================================================================
print("\n[3/10] Preparing features...")

train_features = create_mega_features(train_df)
test_features = create_mega_features(test_df)

y = train_features['Default 12 Flag']
X = train_features.drop(columns=['Default 12 Flag', 'ID'], errors='ignore')
test_ids = test_features['ID']
X_test = test_features.drop(columns=['ID'], errors='ignore')
X_test = X_test.reindex(columns=X.columns, fill_value=0)

print(f"✓ Base features: {X.shape[1]}")

# Define categorical features
cat_features = [
    'Major Media Code', 'Internet Details', 'Reception Type Category',
    'Gender', 'Single/Married Status', 'Residence Type', 'Name Type',
    'Family Composition Type', 'Living Arrangement Type',
    'Insurance Job Type', 'Employment Type', 'Employment Status Type',
    'Industry Type', 'Company Size Category', 'JIS Address Code',
    'App_Month', 'App_DayOfWeek', 'App_Quarter', 'Age_Group_10'
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

# ============================================================================
# STEP 4: TARGET ENCODING
# ============================================================================
print("\n[4/10] Creating target encodings...")

def create_target_encodings(X, y, cat_cols, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    target_encodings = {}
    
    for col in cat_cols:
        col_encodings = np.zeros(len(X))
        
        for train_idx, val_idx in skf.split(X, y):
            encoding_map = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).mean().to_dict()
            overall_mean = y.iloc[train_idx].mean()
            col_encodings[val_idx] = X[col].iloc[val_idx].map(encoding_map).fillna(overall_mean)
        
        encoding_map = y.groupby(X[col]).mean().to_dict()
        encoding_map['__OVERALL__'] = y.mean()
        target_encodings[col] = encoding_map
        X[f'{col}_target_enc'] = col_encodings
    
    return target_encodings

# Target encode top categorical features
target_encodings = create_target_encodings(X, y, cat_features[:12])
print(f"✓ Created {len(target_encodings)} target encodings")

for col, encoding_map in target_encodings.items():
    X_test[f'{col}_target_enc'] = X_test[col].map(encoding_map).fillna(encoding_map['__OVERALL__'])

numeric_cols = [col for col in X.columns if col not in cat_features]

# ============================================================================
# STEP 5: POWER TRANSFORMATION
# ============================================================================
print("\n[5/10] Power transformations...")

skewed_features = ['Total Annual Income', 'Amount of Unsecured Loans',
                   'Application Limit Amount(Desired)', 'Rent Burden Amount']
skewed_features = [f for f in skewed_features if f in X.columns]

pt = PowerTransformer(method='yeo-johnson', standardize=False)
if len(skewed_features) > 0:
    X_skewed = X[skewed_features].replace(-999, 0)
    X_test_skewed = X_test[skewed_features].replace(-999, 0)
    
    X_skewed_transformed = pt.fit_transform(X_skewed)
    X_test_skewed_transformed = pt.transform(X_test_skewed)
    
    for i, col in enumerate(skewed_features):
        X[f'{col}_power'] = X_skewed_transformed[:, i]
        X_test[f'{col}_power'] = X_test_skewed_transformed[:, i]

print(f"✓ Total features: {X.shape[1]}")

# ============================================================================
# STEP 6: LABEL ENCODERS
# ============================================================================
print("\n[6/10] Preparing label encoders...")

label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    all_cats = pd.concat([X[col], X_test[col]]).unique()
    le.fit(all_cats)
    label_encoders[col] = le

# ============================================================================
# STEP 7: MAIN TRAINING LOOP
# ============================================================================
print("\n[7/10] Training 4-model ensemble with 10-fold CV...")

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

cb_oof = np.zeros(len(X))
lgb_oof = np.zeros(len(X))
xgb_oof = np.zeros(len(X))
nn_oof = np.zeros(len(X))

cb_test = np.zeros(len(X_test))
lgb_test = np.zeros(len(X_test))
xgb_test = np.zeros(len(X_test))
nn_test = np.zeros(len(X_test))

cb_scores = []
lgb_scores = []
xgb_scores = []
nn_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*70}")
    print(f"FOLD {fold+1}/{N_SPLITS}")
    print(f"{'='*70}")
    
    X_tr, y_tr = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    
    print(f"Train: {len(X_tr)} | Val: {len(X_val)} | Default: {y_tr.mean():.4f}")
    
    # === CATBOOST ===
    print("\n→ [1/4] CatBoost...")
    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    cb = CatBoostClassifier(
        iterations=6000,
        learning_rate=0.012,
        depth=11,
        l2_leaf_reg=9,
        min_data_in_leaf=10,
        bagging_temperature=0.6,
        random_strength=0.6,
        border_count=254,
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
    print(f"  ✓ AUC: {cb_score:.6f} | Trees: {cb.tree_count_}")
    
    # === LIGHTGBM ===
    print("\n→ [2/4] LightGBM...")
    X_tr_lgb, X_val_lgb, X_test_lgb = X_tr.copy(), X_val.copy(), X_test.copy()
    
    for col in cat_features:
        X_tr_lgb[col] = X_tr_lgb[col].astype('category')
        X_val_lgb[col] = X_val_lgb[col].astype('category')
        X_test_lgb[col] = X_test_lgb[col].astype('category')
    
    lgb = LGBMClassifier(
        n_estimators=6000,
        learning_rate=0.012,
        max_depth=11,
        num_leaves=120,
        min_child_samples=12,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=2.0,
        reg_lambda=2.0,
        min_split_gain=0.01,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbosity=-1
    )
    
    from lightgbm import early_stopping, log_evaluation
    lgb.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_val_lgb, y_val)],
        eval_metric='auc',
        callbacks=[early_stopping(250), log_evaluation(0)]
    )
    
    lgb_oof[val_idx] = lgb.predict_proba(X_val_lgb)[:, 1]
    lgb_test += lgb.predict_proba(X_test_lgb)[:, 1] / N_SPLITS
    lgb_score = roc_auc_score(y_val, lgb_oof[val_idx])
    lgb_scores.append(lgb_score)
    print(f"  ✓ AUC: {lgb_score:.6f}")
    
    # === XGBOOST ===
    print("\n→ [3/4] XGBoost...")
    X_tr_xgb, X_val_xgb, X_test_xgb = X_tr.copy(), X_val.copy(), X_test.copy()
    
    for col in cat_features:
        X_tr_xgb[col] = label_encoders[col].transform(X_tr_xgb[col])
        X_val_xgb[col] = label_encoders[col].transform(X_val_xgb[col])
        X_test_xgb[col] = label_encoders[col].transform(X_test_xgb[col])
    
    xgb = XGBClassifier(
        n_estimators=6000,
        learning_rate=0.012,
        max_depth=11,
        min_child_weight=1,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=2.0,
        reg_lambda=2.0,
        gamma=0.01,
        random_state=RANDOM_STATE + fold,
        eval_metric='auc',
        early_stopping_rounds=250,
        n_jobs=-1,
        verbosity=0
    )
    
    xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_val_xgb, y_val)], verbose=False)
    
    xgb_oof[val_idx] = xgb.predict_proba(X_val_xgb)[:, 1]
    xgb_test += xgb.predict_proba(X_test_xgb)[:, 1] / N_SPLITS
    xgb_score = roc_auc_score(y_val, xgb_oof[val_idx])
    xgb_scores.append(xgb_score)
    print(f"  ✓ AUC: {xgb_score:.6f}")
    
    # === NEURAL NETWORK ===
    if USE_NEURAL_NET:
        print("\n→ [4/4] Neural Network...")
        X_tr_nn, X_val_nn, X_test_nn = X_tr.copy(), X_val.copy(), X_test.copy()
        
        for col in cat_features:
            X_tr_nn[col] = label_encoders[col].transform(X_tr_nn[col])
            X_val_nn[col] = label_encoders[col].transform(X_val_nn[col])
            X_test_nn[col] = label_encoders[col].transform(X_test_nn[col])
        
        scaler = StandardScaler()
        X_tr_nn_scaled = scaler.fit_transform(X_tr_nn)
        X_val_nn_scaled = scaler.transform(X_val_nn)
        X_test_nn_scaled = scaler.transform(X_test_nn)
        
        nn = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=150,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_STATE + fold,
            verbose=False
        )
        nn.fit(X_tr_nn_scaled, y_tr)
        
        nn_oof[val_idx] = nn.predict_proba(X_val_nn_scaled)[:, 1]
        nn_test += nn.predict_proba(X_test_nn_scaled)[:, 1] / N_SPLITS
        nn_score = roc_auc_score(y_val, nn_oof[val_idx])
        nn_scores.append(nn_score)
        print(f"  ✓ AUC: {nn_score:.6f}")
    
    print(f"\n  Fold {fold+1} Summary:")
    print(f"    Best: {max(cb_score, lgb_score, xgb_score, nn_score):.6f}")
    print(f"    Avg:  {np.mean([cb_score, lgb_score, xgb_score, nn_score]):.6f}")

# ============================================================================
# STEP 8: META-LEARNING (STACKING)
# ============================================================================
print("\n[8/10] Training meta-learner...")

if USE_META_LEARNING:
    # Create meta-features
    meta_train = pd.DataFrame({
        'cb': cb_oof,
        'lgb': lgb_oof,
        'xgb': xgb_oof,
        'nn': nn_oof,
        'cb_rank': rankdata(cb_oof) / len(cb_oof),
        'lgb_rank': rankdata(lgb_oof) / len(lgb_oof),
        'xgb_rank': rankdata(xgb_oof) / len(xgb_oof),
        'nn_rank': rankdata(nn_oof) / len(nn_oof),
        'mean_pred': (cb_oof + lgb_oof + xgb_oof + nn_oof) / 4,
        'std_pred': np.std([cb_oof, lgb_oof, xgb_oof, nn_oof], axis=0),
        'max_pred': np.max([cb_oof, lgb_oof, xgb_oof, nn_oof], axis=0),
        'min_pred': np.min([cb_oof, lgb_oof, xgb_oof, nn_oof], axis=0),
        'cb_lgb_diff': abs(cb_oof - lgb_oof),
        'cb_xgb_diff': abs(cb_oof - xgb_oof),
        'lgb_xgb_diff': abs(lgb_oof - xgb_oof),
    })
    
    meta_test = pd.DataFrame({
        'cb': cb_test,
        'lgb': lgb_test,
        'xgb': xgb_test,
        'nn': nn_test,
        'cb_rank': rankdata(cb_test) / len(cb_test),
        'lgb_rank': rankdata(lgb_test) / len(lgb_test),
        'xgb_rank': rankdata(xgb_test) / len(xgb_test),
        'nn_rank': rankdata(nn_test) / len(nn_test),
        'mean_pred': (cb_test + lgb_test + xgb_test + nn_test) / 4,
        'std_pred': np.std([cb_test, lgb_test, xgb_test, nn_test], axis=0),
        'max_pred': np.max([cb_test, lgb_test, xgb_test, nn_test], axis=0),
        'min_pred': np.min([cb_test, lgb_test, xgb_test, nn_test], axis=0),
        'cb_lgb_diff': abs(cb_test - lgb_test),
        'cb_xgb_diff': abs(cb_test - xgb_test),
        'lgb_xgb_diff': abs(lgb_test - xgb_test),
    })
    
    # Train meta-model
    meta_model = LogisticRegression(
        C=0.1,
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver='lbfgs'
    )
    meta_model.fit(meta_train, y)
    
    meta_oof = meta_model.predict_proba(meta_train)[:, 1]
    meta_test_pred = meta_model.predict_proba(meta_test)[:, 1]
    
    meta_auc = roc_auc_score(y, meta_oof)
    print(f"✓ Meta-learner AUC: {meta_auc:.6f}")
    print(f"  Feature weights: {dict(zip(meta_train.columns[:4], meta_model.coef_[0][:4]))}")

# ============================================================================
# STEP 9: ANALYZE & CREATE OPTIMAL ENSEMBLE
# ============================================================================
print("\n[9/10] Creating optimal ensemble...")

cb_auc = roc_auc_score(y, cb_oof)
lgb_auc = roc_auc_score(y, lgb_oof)
xgb_auc = roc_auc_score(y, xgb_oof)
nn_auc = roc_auc_score(y, nn_oof)

print(f"\n📊 Individual Model Performance:")
print(f"  CatBoost:  {cb_auc:.6f} (±{np.std(cb_scores):.6f})")
print(f"  LightGBM:  {lgb_auc:.6f} (±{np.std(lgb_scores):.6f})")
print(f"  XGBoost:   {xgb_auc:.6f} (±{np.std(xgb_scores):.6f})")
print(f"  Neural:    {nn_auc:.6f} (±{np.std(nn_scores):.6f})")

# Performance-based weights
total = cb_auc + lgb_auc + xgb_auc + nn_auc
w_cb = cb_auc / total
w_lgb = lgb_auc / total
w_xgb = xgb_auc / total
w_nn = nn_auc / total

print(f"\n🎯 Optimal Weights:")
print(f"  CB: {w_cb:.4f} | LGB: {w_lgb:.4f} | XGB: {w_xgb:.4f} | NN: {w_nn:.4f}")

# Create multiple ensemble strategies
ensemble1_oof = w_cb * cb_oof + w_lgb * lgb_oof + w_xgb * xgb_oof + w_nn * nn_oof
ensemble1_test = w_cb * cb_test + w_lgb * lgb_test + w_xgb * xgb_test + w_nn * nn_test

# Rank-based ensemble
ensemble2_oof = (
    rankdata(cb_oof) + rankdata(lgb_oof) + rankdata(xgb_oof) + rankdata(nn_oof)
) / (4 * len(cb_oof))
ensemble2_test = (
    rankdata(cb_test) + rankdata(lgb_test) + rankdata(xgb_test) + rankdata(nn_test)
) / (4 * len(cb_test))

# Power ensemble
power = 1.2
ensemble3_oof = (
    cb_oof**power * w_cb + lgb_oof**power * w_lgb + 
    xgb_oof**power * w_xgb + nn_oof**power * w_nn
) / (w_cb + w_lgb + w_xgb + w_nn)
ensemble3_test = (
    cb_test**power * w_cb + lgb_test**power * w_lgb + 
    xgb_test**power * w_xgb + nn_test**power * w_nn
) / (w_cb + w_lgb + w_xgb + w_nn)

# Harmonic mean ensemble
epsilon = 1e-10
ensemble4_oof = 4 / (
    1/(cb_oof + epsilon) + 1/(lgb_oof + epsilon) + 
    1/(xgb_oof + epsilon) + 1/(nn_oof + epsilon)
)
ensemble4_test = 4 / (
    1/(cb_test + epsilon) + 1/(lgb_test + epsilon) + 
    1/(xgb_test + epsilon) + 1/(nn_test + epsilon)
)

print(f"\n📈 Ensemble Strategies:")
print(f"  Weighted Avg:  {roc_auc_score(y, ensemble1_oof):.6f}")
print(f"  Rank-Based:    {roc_auc_score(y, ensemble2_oof):.6f}")
print(f"  Power (1.2):   {roc_auc_score(y, ensemble3_oof):.6f}")
print(f"  Harmonic:      {roc_auc_score(y, ensemble4_oof):.6f}")
if USE_META_LEARNING:
    print(f"  Meta-Learner:  {meta_auc:.6f}")

# Choose best
ensembles = {
    'weighted': (ensemble1_oof, ensemble1_test, roc_auc_score(y, ensemble1_oof)),
    'rank': (ensemble2_oof, ensemble2_test, roc_auc_score(y, ensemble2_oof)),
    'power': (ensemble3_oof, ensemble3_test, roc_auc_score(y, ensemble3_oof)),
    'harmonic': (ensemble4_oof, ensemble4_test, roc_auc_score(y, ensemble4_oof)),
}

if USE_META_LEARNING:
    ensembles['meta'] = (meta_oof, meta_test_pred, meta_auc)

best_name = max(ensembles, key=lambda k: ensembles[k][2])
best_oof, best_test, best_score = ensembles[best_name]

print(f"\n🏆 BEST ENSEMBLE: {best_name.upper()} | AUC: {best_score:.6f}")

# Create blend of top 2 ensembles
sorted_ensembles = sorted(ensembles.items(), key=lambda x: x[1][2], reverse=True)
blend_oof = (sorted_ensembles[0][1][0] * 0.6 + sorted_ensembles[1][1][0] * 0.4)
blend_test = (sorted_ensembles[0][1][1] * 0.6 + sorted_ensembles[1][1][1] * 0.4)
blend_score = roc_auc_score(y, blend_oof)

print(f"\n🎨 BLEND (60-40): {sorted_ensembles[0][0]} + {sorted_ensembles[1][0]}")
print(f"   Blend AUC: {blend_score:.6f}")

# Choose final
if blend_score > best_score:
    final_oof = blend_oof
    final_test = blend_test
    final_score = blend_score
    final_name = f"blend_{sorted_ensembles[0][0]}_{sorted_ensembles[1][0]}"
else:
    final_oof = best_oof
    final_test = best_test
    final_score = best_score
    final_name = best_name

# ============================================================================
# STEP 10: CREATE SUBMISSION
# ============================================================================
print("\n[10/10] Creating submission...")

submission = pd.DataFrame({
    'ID': test_ids,
    'Default 12 Flag': final_test
})

submission['Default 12 Flag'] = submission['Default 12 Flag'].clip(0, 1)

filename = f'ULTIMATE_winner_auc{final_score:.4f}_{final_name}.csv'
submission.to_csv(filename, index=False)

print(f"\n✅ SUBMISSION SAVED: {filename}")
print(f"\n" + "="*80)
print(f"🏆🏆🏆 FINAL RESULTS 🏆🏆🏆")
print("="*80)
print(f"\n📊 Cross-Validation Performance:")
print(f"   OOF AUC:           {final_score:.6f}")
print(f"   Strategy:          {final_name}")
print(f"   Total Features:    {X.shape[1]}")
print(f"   CV Folds:          {N_SPLITS}")
print(f"   Models:            4 (CB + LGB + XGB + NN)")

print(f"\n📈 Model Correlations:")
print(f"   CB-LGB:   {np.corrcoef(cb_oof, lgb_oof)[0,1]:.4f}")
print(f"   CB-XGB:   {np.corrcoef(cb_oof, xgb_oof)[0,1]:.4f}")
print(f"   LGB-XGB:  {np.corrcoef(lgb_oof, xgb_oof)[0,1]:.4f}")
print(f"   CB-NN:    {np.corrcoef(cb_oof, nn_oof)[0,1]:.4f}")

print(f"\n📉 Test Prediction Statistics:")
print(f"   Mean:     {final_test.mean():.6f}")
print(f"   Std:      {final_test.std():.6f}")
print(f"   Min:      {final_test.min():.6f}")
print(f"   Max:      {final_test.max():.6f}")
print(f"   Median:   {np.median(final_test):.6f}")
print(f"   25th:     {np.percentile(final_test, 25):.6f}")
print(f"   75th:     {np.percentile(final_test, 75):.6f}")

print(f"\n🔥 Prediction Distribution:")
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(final_test, bins=bins)
for i in range(len(bins)-1):
    pct = hist[i] / len(final_test) * 100
    bar = '█' * int(pct / 2)
    print(f"   {bins[i]:.1f}-{bins[i+1]:.1f}: {bar} {pct:5.2f}%")

print(f"\n📋 Sample Predictions:")
print(submission.head(15).to_string(index=False))

print(f"\n💡 Key Success Factors:")
print(f"   ✓ {X.shape[1]} engineered features")
print(f"   ✓ Fraud detection (CRITICAL!)")
print(f"   ✓ Multiple DTI ratios")
print(f"   ✓ Japanese cultural factors")
print(f"   ✓ Target encoding")
print(f"   ✓ Power transformations")
print(f"   ✓ 4-model diversity")
print(f"   ✓ {N_SPLITS}-fold stratified CV")
print(f"   ✓ Meta-learning stacking")
print(f"   ✓ Multiple ensemble strategies")

print(f"\n🎯 Expected Leaderboard Position:")
if final_score >= 0.80:
    print(f"   🥇 TOP 3 - PRIZE MONEY ZONE!")
elif final_score >= 0.79:
    print(f"   🥈 TOP 10 - VERY STRONG!")
elif final_score >= 0.78:
    print(f"   🥉 TOP 20 - GOOD SCORE!")
else:
    print(f"   📊 Competitive - Keep improving!")

print(f"\n🚀 NEXT STEPS TO WIN:")
print(f"   1. Submit this baseline")
print(f"   2. Check leaderboard position")
print(f"   3. If top 10: Implement pseudo-labeling")
print(f"   4. Try Optuna hyperparameter tuning")
print(f"   5. Add adversarial validation")
print(f"   6. Ensemble multiple submissions")

print(f"\n" + "="*80)
print(f"💰 AB JAO AUR ₹2,65,000 JEET KE AAO! 💰")
print("="*80)

try:
    from google.colab import files
    files.download(filename)
    print(f"\n✅ File downloaded successfully!")
except:
    print(f"\n✅ File saved: {filename}")
    print(f"   (Download manually if not using Colab)")

print(f"\n🙏 BEST OF LUCK BHAI! YOU GOT THIS! 🙏")