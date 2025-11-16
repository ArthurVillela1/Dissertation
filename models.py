import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, precision_recall_curve
)
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")

## ================================================== Data Handling and Threshold definition ================================================== ##

df = pd.read_excel("Data.xlsx")
print(df)

data1 = df[['R_12-18M', 'T10Y2Y', 'T10Y3M', 'BaaSpread', 'PERatioS&P']].dropna()
y = data1['R_12-18M']
X = data1[['T10Y2Y']].copy()   # main slope spec here

feature_cols = X.columns.tolist()
X["Inverted"] = (X.iloc[:, 0] < 0).astype(int)
threshold = 0.5

n_samples = len(X)
dates = pd.date_range(start='1980-11-01', periods=n_samples, freq='MS')

## ================================================== Cross Validation Functions ================================================== ##

def expanding_window_split(X, n_splits=5):
    n_samples = len(X)
    splits = []
    blocks = np.array_split(np.arange(n_samples), n_splits + 2)
    for i in range(n_splits):
        train_blocks = blocks[: i + 2]
        if len(train_blocks) == 0:
            continue
        train_idx = np.concatenate(train_blocks)
        test_idx = blocks[i + 2]
        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits

expanding_splits = expanding_window_split(X, n_splits=3)

def display_fold_info(X, splits):
    print("=" * 70)
    print(f"Cross-validation Fold Structure")
    print(f"Total data points: {len(X)}")
    if hasattr(X, 'index') and hasattr(X.index, 'strftime'):
        dates_local = X.index
    else:
        dates_local = pd.date_range(start='1980-11-01', periods=len(X), freq='MS')
    for i, (train_idx, test_idx) in enumerate(splits, 1):
        train_start = dates_local[0]
        train_end = dates_local[train_idx[-1]]
        test_start = dates_local[test_idx[0]]
        test_end = dates_local[test_idx[-1]]
        print(f"\nFOLD {i}:")
        print(f"  Training:  {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')} ({len(train_idx):3d} months)")
        print(f"  Testing:   {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')} ({len(test_idx):3d} months)")
    print(f"\n✅ Total folds: {len(splits)}")
    print("=" * 70)

display_fold_info(X, expanding_splits)

## ================================================== McNemar Test Function ================================================== ##

def run_mcnemar(name, preds_A, preds_B, y_true):
    preds_A = np.array(preds_A)
    preds_B = np.array(preds_B)
    y_true = np.array(y_true)
    if len(preds_A) == 0 or len(preds_B) == 0:
        print(f"\nMcNemar test ({name}): not enough data.")
        return
    correct_A = (preds_A == y_true)
    correct_B = (preds_B == y_true)
    b = np.sum((correct_A == 1) & (correct_B == 0))
    c = np.sum((correct_A == 0) & (correct_B == 1))
    table = [[0, b],
             [c, 0]]
    result = mcnemar(table, exact=False, correction=True)
    print(f"\nMcNemar test ({name}):")
    print(f"b = {b}, c = {c}, p-value = {result.pvalue:.4f}")

## ================================================== GLOBAL STORAGE FOR PER-OBSERVATION PREDICTIONS ================================================== ##

# fold identity for each observation (NaN for training-only)
fold_id_global = np.full(n_samples, np.nan)

# For each model, store predictions / probabilities for every observation
logit_pred_global  = np.full(n_samples, np.nan)
logit_proba_global = np.full(n_samples, np.nan)

probit_pred_global  = np.full(n_samples, np.nan)
probit_proba_global = np.full(n_samples, np.nan)

gb_pred_global  = np.full(n_samples, np.nan)
gb_proba_global = np.full(n_samples, np.nan)

rf_pred_global  = np.full(n_samples, np.nan)
rf_proba_global = np.full(n_samples, np.nan)

all_y_logit, all_proba_logit = [], []
all_y_probit, all_proba_probit = [], []
all_y_gb, all_proba_gb = [], []
all_y_rf, all_proba_rf = [], []

## ================================================== Main Results: LOGIT ================================================== ##

clf = LogisticRegression(solver="liblinear", random_state=42)

f1s, precs, recs, aucs, pr_aucs = [], [], [], [], []
f1s_inv, precs_inv, recs_inv, aucs_inv, pr_aucs_inv = [], [], [], [], []

logit_y_all = []
logit_preds_slope2 = []
logit_preds_aug2 = []
logit_preds_slope3 = []
logit_preds_aug3 = []

for fold_id, (train_idx, test_idx) in enumerate(expanding_splits, start=1):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue

    logit_y_all.extend(y_te.values)
    fold_id_global[test_idx] = fold_id

    # main spec: T10Y3M
    clf.fit(X_tr[feature_cols], y_tr)
    proba = clf.predict_proba(X_te[feature_cols])[:, 1]
    preds = (proba >= threshold).astype(int)
    logit_preds_aug3.extend(preds)

    # store per-observation predictions globally
    logit_pred_global[test_idx]  = preds
    logit_proba_global[test_idx] = proba

    all_y_logit.append(y_te.values)
    all_proba_logit.append(proba)

    f1s.append(f1_score(y_te, preds, zero_division=0))
    precs.append(precision_score(y_te, preds, zero_division=0))
    recs.append(recall_score(y_te, preds, zero_division=0))
    aucs.append(roc_auc_score(y_te, proba))
    pr_aucs.append(average_precision_score(y_te, proba))

    inv_mask = X_te["Inverted"] == 1
    if inv_mask.sum() > 0 and len(np.unique(y_te[inv_mask])) == 2:
        f1s_inv.append(f1_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        precs_inv.append(precision_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        recs_inv.append(recall_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        aucs_inv.append(roc_auc_score(y_te[inv_mask], proba[inv_mask]))
        pr_aucs_inv.append(average_precision_score(y_te[inv_mask], proba[inv_mask]))

    # extra specs for McNemar (no need to store per-obs here)
    X_tr_slope3 = data1.iloc[train_idx][['T10Y3M']]
    X_te_slope3 = data1.iloc[test_idx][['T10Y3M']]
    clf.fit(X_tr_slope3, y_tr)
    proba_s3 = clf.predict_proba(X_te_slope3)[:, 1]
    preds_s3 = (proba_s3 >= threshold).astype(int)
    logit_preds_slope3.extend(preds_s3)

    X_tr_slope2 = data1.iloc[train_idx][['T10Y2Y']]
    X_te_slope2 = data1.iloc[test_idx][['T10Y2Y']]
    clf.fit(X_tr_slope2, y_tr)
    proba_s2 = clf.predict_proba(X_te_slope2)[:, 1]
    preds_s2 = (proba_s2 >= threshold).astype(int)
    logit_preds_slope2.extend(preds_s2)

    X_tr_aug2 = data1.iloc[train_idx][['T10Y2Y', 'BaaSpread', 'PERatioS&P']]
    X_te_aug2 = data1.iloc[test_idx][['T10Y2Y', 'BaaSpread', 'PERatioS&P']]
    clf.fit(X_tr_aug2, y_tr)
    proba_aug2 = clf.predict_proba(X_te_aug2)[:, 1]
    preds_aug2 = (proba_aug2 >= threshold).astype(int)
    logit_preds_aug2.extend(preds_aug2)

print("=" * 70)
print("\n {}-fold Time Series CV (Logistic Regression) @ threshold = {:.2f} (mean ± std)".format(len(expanding_splits), threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(precs), np.std(precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(recs),  np.std(recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(pr_aucs), np.std(pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(precs_inv), np.std(precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(recs_inv),  np.std(recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(pr_aucs_inv), np.std(pr_aucs_inv)))

run_mcnemar("Logit T10Y2Y slope vs augmented", logit_preds_slope2, logit_preds_aug2, logit_y_all)
run_mcnemar("Logit T10Y3M slope vs augmented", logit_preds_slope3, logit_preds_aug3, logit_y_all)
print("=" * 70)

## ================================================== PROBIT ================================================== ##

probit_f1s, probit_precs, probit_recs, probit_aucs, probit_pr_aucs = [], [], [], [], []
probit_f1s_inv, probit_precs_inv, probit_recs_inv, probit_aucs_inv, probit_pr_aucs_inv = [], [], [], [], []

probit_y_all = []
probit_preds_slope2 = []
probit_preds_aug2 = []
probit_preds_slope3 = []
probit_preds_aug3 = []

for train_idx, test_idx in expanding_splits:
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue

    probit_y_all.extend(y_te.values)

    X_tr_const = sm.add_constant(X_tr[feature_cols], has_constant='add')
    X_te_const = sm.add_constant(X_te[feature_cols], has_constant='add')

    probit_model = sm.Probit(y_tr, X_tr_const)
    probit_result = probit_model.fit(disp=0)

    probit_proba = probit_result.predict(X_te_const)
    probit_preds = (probit_proba >= threshold).astype(int)
    probit_preds_aug3.extend(probit_preds)

    # store per-observation predictions
    probit_pred_global[test_idx]  = probit_preds
    probit_proba_global[test_idx] = probit_proba

    all_y_probit.append(y_te.values)
    all_proba_probit.append(probit_proba)

    probit_f1s.append(f1_score(y_te, probit_preds, zero_division=0))
    probit_precs.append(precision_score(y_te, probit_preds, zero_division=0))
    probit_recs.append(recall_score(y_te, probit_preds, zero_division=0))
    probit_aucs.append(roc_auc_score(y_te, probit_proba))
    probit_pr_aucs.append(average_precision_score(y_te, probit_proba))

    inv_mask = X_te["Inverted"] == 1
    if inv_mask.sum() > 0 and len(np.unique(y_te[inv_mask])) == 2:
        probit_f1s_inv.append(f1_score(y_te[inv_mask], probit_preds[inv_mask], zero_division=0))
        probit_precs_inv.append(precision_score(y_te[inv_mask], probit_preds[inv_mask], zero_division=0))
        probit_recs_inv.append(recall_score(y_te[inv_mask], probit_preds[inv_mask], zero_division=0))
        probit_aucs_inv.append(roc_auc_score(y_te[inv_mask], probit_proba[inv_mask]))
        probit_pr_aucs_inv.append(average_precision_score(y_te[inv_mask], probit_proba[inv_mask]))

    X_tr_slope3 = data1.iloc[train_idx][['T10Y3M']]
    X_te_slope3 = data1.iloc[test_idx][['T10Y3M']]
    X_tr_slope3_const = sm.add_constant(X_tr_slope3, has_constant='add')
    X_te_slope3_const = sm.add_constant(X_te_slope3, has_constant='add')
    probit_model_s3 = sm.Probit(y_tr, X_tr_slope3_const)
    probit_result_s3 = probit_model_s3.fit(disp=0)
    probit_proba_s3 = probit_result_s3.predict(X_te_slope3_const)
    probit_preds_s3 = (probit_proba_s3 >= threshold).astype(int)
    probit_preds_slope3.extend(probit_preds_s3)

    X_tr_slope2 = data1.iloc[train_idx][['T10Y2Y']]
    X_te_slope2 = data1.iloc[test_idx][['T10Y2Y']]
    X_tr_slope2_const = sm.add_constant(X_tr_slope2, has_constant='add')
    X_te_slope2_const = sm.add_constant(X_te_slope2, has_constant='add')
    probit_model_s2 = sm.Probit(y_tr, X_tr_slope2_const)
    probit_result_s2 = probit_model_s2.fit(disp=0)
    probit_proba_s2 = probit_result_s2.predict(X_te_slope2_const)
    probit_preds_s2 = (probit_proba_s2 >= threshold).astype(int)
    probit_preds_slope2.extend(probit_preds_s2)

    X_tr_aug2 = data1.iloc[train_idx][['T10Y2Y', 'BaaSpread', 'PERatioS&P']]
    X_te_aug2 = data1.iloc[test_idx][['T10Y2Y', 'BaaSpread', 'PERatioS&P']]
    X_tr_aug2_const = sm.add_constant(X_tr_aug2, has_constant='add')
    X_te_aug2_const = sm.add_constant(X_te_aug2, has_constant='add')
    probit_model_aug2 = sm.Probit(y_tr, X_tr_aug2_const)
    probit_result_aug2 = probit_model_aug2.fit(disp=0)
    probit_proba_aug2 = probit_result_aug2.predict(X_te_aug2_const)
    probit_preds_aug2_fold = (probit_proba_aug2 >= threshold).astype(int)
    probit_preds_aug2.extend(probit_preds_aug2_fold)

print("\n {}-fold Time Series CV (Probit) @ threshold = {:.2f} (mean ± std)".format(len(expanding_splits), threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(probit_precs), np.std(probit_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(probit_recs),  np.std(probit_recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(probit_pr_aucs), np.std(probit_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(probit_precs_inv), np.std(probit_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(probit_recs_inv),  np.std(probit_recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(probit_pr_aucs_inv), np.std(probit_pr_aucs_inv)))

run_mcnemar("Probit T10Y2Y slope vs augmented", probit_preds_slope2, probit_preds_aug2, probit_y_all)
run_mcnemar("Probit T10Y3M slope vs augmented", probit_preds_slope3, probit_preds_aug3, probit_y_all)
print("=" * 70)

## ================================================== GRADIENT BOOSTING ================================================== ##

gb_f1s, gb_precs, gb_recs, gb_aucs, gb_pr_aucs = [], [], [], [], []
gb_f1s_inv, gb_precs_inv, gb_recs_inv, gb_aucs_inv, gb_pr_aucs_inv = [], [], [], [], []

gb_clf = GradientBoostingClassifier(
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=2
)

gb_y_all = []
gb_preds_slope2 = []
gb_preds_aug2 = []
gb_preds_slope3 = []
gb_preds_aug3 = []

for train_idx, test_idx in expanding_splits:
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue

    gb_y_all.extend(y_te.values)

    gb_clf.fit(X_tr[feature_cols], y_tr)
    gb_proba = gb_clf.predict_proba(X_te[feature_cols])[:, 1]
    gb_preds = (gb_proba >= threshold).astype(int)
    gb_preds_aug3.extend(gb_preds)

    gb_pred_global[test_idx]  = gb_preds
    gb_proba_global[test_idx] = gb_proba

    all_y_gb.append(y_te.values)
    all_proba_gb.append(gb_proba)

    gb_f1s.append(f1_score(y_te, gb_preds, zero_division=0))
    gb_precs.append(precision_score(y_te, gb_preds, zero_division=0))
    gb_recs.append(recall_score(y_te, gb_preds, zero_division=0))
    gb_aucs.append(roc_auc_score(y_te, gb_proba))
    gb_pr_aucs.append(average_precision_score(y_te, gb_proba))

    inv_mask = X_te["Inverted"] == 1
    if inv_mask.sum() > 0 and len(np.unique(y_te[inv_mask])) == 2:
        gb_f1s_inv.append(f1_score(y_te[inv_mask], gb_preds[inv_mask], zero_division=0))
        gb_precs_inv.append(precision_score(y_te[inv_mask], gb_preds[inv_mask], zero_division=0))
        gb_recs_inv.append(recall_score(y_te[inv_mask], gb_preds[inv_mask], zero_division=0))
        gb_aucs_inv.append(roc_auc_score(y_te[inv_mask], gb_proba[inv_mask]))
        gb_pr_aucs_inv.append(average_precision_score(y_te[inv_mask], gb_proba[inv_mask]))

    X_tr_slope3 = data1.iloc[train_idx][['T10Y3M']]
    X_te_slope3 = data1.iloc[test_idx][['T10Y3M']]
    gb_clf.fit(X_tr_slope3, y_tr)
    gb_proba_s3 = gb_clf.predict_proba(X_te_slope3)[:, 1]
    gb_preds_s3 = (gb_proba_s3 >= threshold).astype(int)
    gb_preds_slope3.extend(gb_preds_s3)

    X_tr_slope2 = data1.iloc[train_idx][['T10Y2Y']]
    X_te_slope2 = data1.iloc[test_idx][['T10Y2Y']]
    gb_clf.fit(X_tr_slope2, y_tr)
    gb_proba_s2 = gb_clf.predict_proba(X_te_slope2)[:, 1]
    gb_preds_s2 = (gb_proba_s2 >= threshold).astype(int)
    gb_preds_slope2.extend(gb_preds_s2)

    X_tr_aug2 = data1.iloc[train_idx][['T10Y2Y', 'BaaSpread', 'PERatioS&P']]
    X_te_aug2 = data1.iloc[test_idx][['T10Y2Y', 'BaaSpread', 'PERatioS&P']]
    gb_clf.fit(X_tr_aug2, y_tr)
    gb_proba_aug2 = gb_clf.predict_proba(X_te_aug2)[:, 1]
    gb_preds_aug2_fold = (gb_proba_aug2 >= threshold).astype(int)
    gb_preds_aug2.extend(gb_preds_aug2_fold)

print("\n {}-fold Time Series CV (Gradient Boosting) @ threshold = {:.2f} (mean ± std)".format(len(expanding_splits), threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(gb_precs), np.std(gb_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(gb_recs),  np.std(gb_recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(gb_pr_aucs), np.std(gb_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(gb_precs_inv), np.std(gb_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(gb_recs_inv),  np.std(gb_recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(gb_pr_aucs_inv), np.std(gb_pr_aucs_inv)))

run_mcnemar("Gradient Boosting T10Y2Y slope vs augmented", gb_preds_slope2, gb_preds_aug2, gb_y_all)
run_mcnemar("Gradient Boosting T10Y3M slope vs augmented", gb_preds_slope3, gb_preds_aug3, gb_y_all)
print("=" * 70)

## ================================================== RANDOM FOREST ================================================== ##

rf_f1s, rf_precs, rf_recs, rf_aucs, rf_pr_aucs = [], [], [], [], []
rf_f1s_inv, rf_precs_inv, rf_recs_inv, rf_aucs_inv, rf_pr_aucs_inv = [], [], [], [], []

rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf_y_all = []
rf_preds_slope2 = []
rf_preds_aug2 = []
rf_preds_slope3 = []
rf_preds_aug3 = []

for train_idx, test_idx in expanding_splits:
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue

    rf_y_all.extend(y_te.values)

    rf_clf.fit(X_tr[feature_cols], y_tr)
    rf_proba = rf_clf.predict_proba(X_te[feature_cols])[:, 1]
    rf_preds = (rf_proba >= threshold).astype(int)
    rf_preds_aug3.extend(rf_preds)

    rf_pred_global[test_idx]  = rf_preds
    rf_proba_global[test_idx] = rf_proba

    all_y_rf.append(y_te.values)
    all_proba_rf.append(rf_proba)

    rf_f1s.append(f1_score(y_te, rf_preds, zero_division=0))
    rf_precs.append(precision_score(y_te, rf_preds, zero_division=0))
    rf_recs.append(recall_score(y_te, rf_preds, zero_division=0))
    rf_aucs.append(roc_auc_score(y_te, rf_proba))
    rf_pr_aucs.append(average_precision_score(y_te, rf_proba))

    inv_mask = X_te["Inverted"] == 1
    if inv_mask.sum() > 0 and len(np.unique(y_te[inv_mask])) == 2:
        rf_f1s_inv.append(f1_score(y_te[inv_mask], rf_preds[inv_mask], zero_division=0))
        rf_precs_inv.append(precision_score(y_te[inv_mask], rf_preds[inv_mask], zero_division=0))
        rf_recs_inv.append(recall_score(y_te[inv_mask], rf_preds[inv_mask], zero_division=0))
        rf_aucs_inv.append(roc_auc_score(y_te[inv_mask], rf_proba[inv_mask]))
        rf_pr_aucs_inv.append(average_precision_score(y_te[inv_mask], rf_proba[inv_mask]))

    X_tr_slope3 = data1.iloc[train_idx][['T10Y3M']]
    X_te_slope3 = data1.iloc[test_idx][['T10Y3M']]
    rf_clf.fit(X_tr_slope3, y_tr)
    rf_proba_s3 = rf_clf.predict_proba(X_te_slope3)[:, 1]
    rf_preds_s3 = (rf_proba_s3 >= threshold).astype(int)
    rf_preds_slope3.extend(rf_preds_s3)

    X_tr_slope2 = data1.iloc[train_idx][['T10Y2Y']]
    X_te_slope2 = data1.iloc[test_idx][['T10Y2Y']]
    rf_clf.fit(X_tr_slope2, y_tr)
    rf_proba_s2 = rf_clf.predict_proba(X_te_slope2)[:, 1]
    rf_preds_s2 = (rf_proba_s2 >= threshold).astype(int)
    rf_preds_slope2.extend(rf_preds_s2)

    X_tr_aug2 = data1.iloc[train_idx][['T10Y2Y', 'BaaSpread', 'PERatioS&P']]
    X_te_aug2 = data1.iloc[test_idx][['T10Y2Y', 'BaaSpread', 'PERatioS&P']]
    rf_clf.fit(X_tr_aug2, y_tr)
    rf_proba_aug2 = rf_clf.predict_proba(X_te_aug2)[:, 1]
    rf_preds_aug2_fold = (rf_proba_aug2 >= threshold).astype(int)
    rf_preds_aug2.extend(rf_preds_aug2_fold)

print("\n {}-fold Time Series CV (Random Forest) @ threshold = {:.2f} (mean ± std)".format(len(expanding_splits), threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(rf_precs), np.std(rf_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(rf_recs),  np.std(rf_recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(rf_pr_aucs), np.std(rf_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(rf_precs_inv), np.std(rf_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(rf_recs_inv),  np.std(rf_recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(rf_pr_aucs_inv), np.std(rf_pr_aucs_inv)))

run_mcnemar("Random Forest T10Y2Y slope vs augmented", rf_preds_slope2, rf_preds_aug2, rf_y_all)
run_mcnemar("Random Forest T10Y3M slope vs augmented", rf_preds_slope3, rf_preds_aug3, rf_y_all)
print("=" * 70)

## ============================ PR–AUC Curves: mean over folds ============================ ##
plt.figure(figsize=(7, 6))

def plot_pr_curve(all_y, all_proba, label):
    if len(all_y) == 0:
        return
    y_concat = np.concatenate(all_y)
    proba_concat = np.concatenate(all_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_concat, proba_concat)
    pr_auc_val = average_precision_score(y_concat, proba_concat)
    plt.plot(rec_curve, prec_curve, label=f"{label} (PR AUC = {pr_auc_val:.3f})")

plot_pr_curve(all_y_logit, all_proba_logit, "Logit")
plot_pr_curve(all_y_probit, all_proba_probit, "Probit")
plot_pr_curve(all_y_gb, all_proba_gb, "Gradient Boosting")
plot_pr_curve(all_y_rf, all_proba_rf, "Random Forest")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves (Time-series CV, all folds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pr_curves.png", dpi=300, bbox_inches="tight")
plt.close()

## ================================================== SPF Comparison ================================================== ##

df_spf = pd.read_excel("Data.xlsx")
spf_data = df_spf[['R_12-18M', 'SPF']].dropna()
spf_precision = precision_score(spf_data['R_12-18M'], spf_data['SPF'], zero_division=0)

all_precisions = precs + probit_precs + gb_precs + rf_precs
all_precisions_inv = precs_inv + probit_precs_inv + gb_precs_inv + rf_precs_inv
model_avg_precision = np.mean(all_precisions)
model_avg_precision_inv = np.mean(all_precisions_inv)

print("\n======== Averages ========")
print(f"Models (avg): {model_avg_precision:.3f}")
print(f"Models (avg) YC inverted: {model_avg_precision_inv:.3f}")
print("=" * 70)

## ================================================== Per Fold Performance (Precision & Recall) ================================================== ##
print("\n Per-fold Precision & Recall (by model)")

def _print_per_fold(name, precs_list, recs_list):
    if len(precs_list) == 0 and len(recs_list) == 0:
        print(f"\n{name}: No valid folds (no recorded precision/recall).")
        return
    n = max(len(precs_list), len(recs_list))
    print(f"\n{name} (n_valid_folds = {n}):")
    for i in range(n):
        p = precs_list[i] if i < len(precs_list) else float('nan')
        r = recs_list[i] if i < len(recs_list) else float('nan')
        if np.isnan(p) and np.isnan(r):
            print(f"  Fold {i+1}: no metric")
        else:
            print(f"  Fold {i+1}: Precision = {p:.3f}, Recall = {r:.3f}")

_print_per_fold('Logistic Regression', precs, recs)
_print_per_fold('Probit', probit_precs, probit_recs)
_print_per_fold('Gradient Boosting', gb_precs, gb_recs)
_print_per_fold('Random Forest', rf_precs, rf_recs)
print("=" * 70)

## ================================================== Build per-observation table for all models ================================================== ##

mask_test = ~np.isnan(fold_id_global)   # True for test observations across all folds

pred_df_all = pd.DataFrame({
    "Fold": fold_id_global[mask_test].astype(int),
    "Date": dates[mask_test].strftime("%Y-%m"),
    "y_true": y.values[mask_test].astype(int),
    "Logit_pred":  logit_pred_global[mask_test].astype(int),
    "Logit_proba": logit_proba_global[mask_test],
    "Probit_pred":  probit_pred_global[mask_test].astype(int),
    "Probit_proba": probit_proba_global[mask_test],
    "GB_pred":  gb_pred_global[mask_test].astype(int),
    "GB_proba": gb_proba_global[mask_test],
    "RF_pred":  rf_pred_global[mask_test].astype(int),
    "RF_proba": rf_proba_global[mask_test],
})

print("\n=== All models: per-observation predictions in test folds ===")
print(pred_df_all.head(20).to_string(index=False))
print(f"\nTotal test observations in table: {len(pred_df_all)}")  # should be 312

pred_df_all.to_excel("all_models_fold_predictions.xlsx", index=False)
