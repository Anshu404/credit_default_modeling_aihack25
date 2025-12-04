
import pandas as pd, numpy as np, warnings, gc, os, sys, random
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

SEED         = 42
N_SPLITS     = 5
DATA_PATH    = ""              # update if needed
PRUNE_TOP_N  = 240             # keep N best features after CV
np.random.seed(SEED); random.seed(SEED)

###############################################################################
# 1. LOAD
###############################################################################
train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
print(f"train {train_df.shape} | test {test_df.shape}")

###############################################################################
# 2. FEATURE ENGINEERING (same ‘safe’ logic, + K-fold target-encoding helper)
###############################################################################
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ---- dates --------------------------------------------------------------
    df['Application Date'] = pd.to_datetime(df['Application Date'], format='%Y/%m/%d')
    df['Date of Birth']    = pd.to_datetime(df['Date of Birth'],    format='%Y/%m/%d')
    df['Age']              = (df['Application Date']-df['Date of Birth']).dt.days/365.25
    df['App_Month']        = df['Application Date'].dt.month
    df['App_DOW']          = df['Application Date'].dt.dayofweek
    df['App_Hour']         = df['Application Time']//10000
    # ---- gaps ---------------------------------------------------------------
    df['amt_gap']  = df['Declared Amount of Unsecured Loans']-df['Amount of Unsecured Loans']
    df['cnt_gap']  = df['Declared Number of Unsecured Loans']-df['Number of Unsecured Loans']
    # ---- financial ratios ---------------------------------------------------
    df['income_log'] = np.log1p(df['Total Annual Income'])
    df['dti']        = (df['Amount of Unsecured Loans']+
                        df['Application Limit Amount(Desired)'])/(df['Total Annual Income']+1)
    df['rent_ratio'] = (df['Rent Burden Amount']*12)/(df['Total Annual Income']+1)
    # ---- other easy flags ---------------------------------------------------
    df['is_home']   = df['Residence Type'].isin([1,2,8,9]).astype(int)
    df['large_co']  = df['Company Size Category'].isin([1,2,3]).astype(int)
    df['is_male']   = (df['Gender']==1).astype(int)
    # ---- JIS helper strings -------------------------------------------------
    df['JIS_str']      = df['JIS Address Code'].fillna(-999).astype(int).astype(str)
    df['JIS_pref2']    = df['JIS_str'].str[:2]
    df['JIS_pref3']    = df['JIS_str'].str[:3]
    # drop raw dates
    df = df.drop(columns=['Application Date','Date of Birth','JIS_str'])
    return df

train_f = create_features(train_df)
test_f  = create_features(test_df)

###############################################################################
# 3. K-fold TARGET ENCODING of “JIS Address Code” (very high cardinality)
###############################################################################
def add_target_encoding(train, test, target, col, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    full_te = pd.Series(index=train.index, dtype=float)
    for tr_idx, val_idx in skf.split(train, target):
        means = train.loc[tr_idx].groupby(col)[target.name].mean()
        full_te[val_idx] = train.loc[val_idx, col].map(means)
    global_means = train.groupby(col)[target.name].mean()
    train[f'{col}_te'] = full_te.fillna(target.mean())
    test[f'{col}_te']  = test[col].map(global_means).fillna(target.mean())

y = train_f['Default 12 Flag']
add_target_encoding(train_f, test_f, y, 'JIS Address Code', n_splits=N_SPLITS, seed=SEED)

###############################################################################
# 4. SPLIT FEATURES INTO CAT/NUM, POWER-TRANSFORM SOME NUMERICS
###############################################################################
cat_cols = [
    'Major Media Code','Internet Details','Reception Type Category','Gender',
    'Single/Married Status','Residence Type','Name Type','Family Composition Type',
    'Living Arrangement Type','Insurance Job Type','Employment Type',
    'Employment Status Type','Industry Type','Company Size Category',
    'JIS Address Code','JIS_pref2','JIS_pref3','App_Month','App_DOW'
]
cat_cols = [c for c in cat_cols if c in train_f.columns]
num_cols = [c for c in train_f.columns if c not in cat_cols+['Default 12 Flag','ID']]

# power-transform skewed
skew_cols = ['Total Annual Income','Amount of Unsecured Loans',
             'Application Limit Amount(Desired)','Rent Burden Amount']
pt = PowerTransformer(method='yeo-johnson')
train_f[skew_cols] = pt.fit_transform(train_f[skew_cols].fillna(0))
test_f[skew_cols]  = pt.transform(test_f[skew_cols].fillna(0))

# missing handling
train_f[cat_cols] = train_f[cat_cols].fillna('-999').astype(str)
test_f[cat_cols]  = test_f[cat_cols].fillna('-999').astype(str)
train_f[num_cols] = train_f[num_cols].fillna(-999)
test_f[num_cols]  = test_f[num_cols].fillna(-999)

X = train_f.drop(columns=['Default 12 Flag','ID'])
X_test = test_f.drop(columns=['ID'])

###############################################################################
# 5. CV – FIT CATBOOST + XGBOOST   (label-encoders per fold, no leakage)
###############################################################################
skf      = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_cb   = np.zeros(len(X));     oof_xgb  = np.zeros(len(X))
pred_cb  = np.zeros(len(X_test)); pred_xgb = np.zeros(len(X_test))
feat_importance = pd.DataFrame({'feature':X.columns})

for fold,(tr,val) in enumerate(skf.split(X,y),1):
    X_tr, y_tr = X.iloc[tr], y.iloc[tr]
    X_val, y_val = X.iloc[val], y.iloc[val]

    # ------- CatBoost (native cat handling) ----------------------------------
    pool_tr  = Pool(X_tr, y_tr, cat_features=[X.columns.get_loc(c) for c in cat_cols])
    pool_val = Pool(X_val, y_val, cat_features=[X.columns.get_loc(c) for c in cat_cols])
    cb = CatBoostClassifier(
        iterations=5000, learning_rate=0.02, depth=8, l2_leaf_reg=12,
        loss_function='Logloss', eval_metric='AUC',
        random_seed=SEED+fold, early_stopping_rounds=200, verbose=False,
        subsample=0.85, colsample_bylevel=0.85)
    cb.fit(pool_tr, eval_set=pool_val)
    oof_cb[val]  = cb.predict_proba(X_val)[:,1]
    pred_cb     += cb.predict_proba(X_test)[:,1]/N_SPLITS
    feat_importance[f'fold{fold}'] = cb.get_feature_importance()

    # ------- XGB (train-only encoders) ---------------------------------------
    X_tr_le  = X_tr.copy(); X_val_le = X_val.copy(); X_test_le = X_test.copy()
    le_dict = {}
    for c in cat_cols:
        le = LabelEncoder().fit(X_tr_le[c])
        X_tr_le[c]  = le.transform(X_tr_le[c])
        X_val_le[c] = le.transform(X_val_le[c])
        X_test_le[c]= le.transform(X_test_le[c])
        le_dict[c] = le
    xgb = XGBClassifier(
        n_estimators=5000, learning_rate=0.02, max_depth=7,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, gamma=0.15,
        eval_metric='auc', random_state=SEED+fold, n_jobs=-1, early_stopping_rounds=200)
    xgb.fit(X_tr_le, y_tr, eval_set=[(X_val_le,y_val)], verbose=False)
    oof_xgb[val] = xgb.predict_proba(X_val_le)[:,1]
    pred_xgb    += xgb.predict_proba(X_test_le)[:,1]/N_SPLITS

    print(f"fold {fold}:  CB {roc_auc_score(y_val,oof_cb[val]):.4f} | "
          f"XGB {roc_auc_score(y_val,oof_xgb[val]):.4f}")

###############################################################################
# 6. STACKING – logistic regression on OOF preds
###############################################################################
stack_tr   = np.vstack([oof_cb,  oof_xgb ]).T
stack_test = np.vstack([pred_cb, pred_xgb]).T
meta = LogisticRegression(max_iter=1000, solver='lbfgs')
meta.fit(stack_tr, y)
stack_oof   = meta.predict_proba(stack_tr)[:,1]
stack_pred  = meta.predict_proba(stack_test)[:,1]
print("\nOOF AUC  >  CB:",roc_auc_score(y,oof_cb),
      "| XGB:",roc_auc_score(y,oof_xgb),
      "| STACK:",roc_auc_score(y,stack_oof))

###############################################################################
# 7. FEATURE PRUNING – keep best N (CatBoost mean importance)
###############################################################################
feat_importance['mean'] = feat_importance[[c for c in feat_importance.columns
                                           if c.startswith('fold')]].mean(axis=1)
top_feats = feat_importance.sort_values('mean',ascending=False).head(PRUNE_TOP_N)['feature']
X_prune       = X[top_feats];     X_test_prune  = X_test[top_feats]

###############################################################################
# 8. FINAL CATBOOST on pruned features (train all data)
###############################################################################
full_pool = Pool(X_prune, y,
                 cat_features=[X_prune.columns.get_loc(c) for c in cat_cols if c in X_prune.columns])
cb_final = CatBoostClassifier(
    iterations=int(cb.get_best_iteration()*1.1),
    learning_rate=0.02, depth=8, l2_leaf_reg=12,
    loss_function='Logloss', eval_metric='AUC',
    random_seed=SEED, verbose=False)
cb_final.fit(full_pool)
final_pred = 0.6*cb_final.predict_proba(X_test_prune)[:,1] + 0.4*stack_pred

###############################################################################
# 9. SUBMISSION
###############################################################################
sub = pd.DataFrame({'ID':test_df['ID'],'Default 12 Flag':np.clip(final_pred,0,1)})
fname = f"STACKED_0p70_attempt_seed{SEED}.csv"
sub.to_csv(fname,index=False)
print(f"\nSaved {fname}  |  mean={sub['Default 12 Flag'].mean():.4f}")
