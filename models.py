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

## ================================================== Data Handling and Threshold definition ================================================== ##

df = pd.read_excel("Data.xlsx")
print(df)

data1 = df[['R_12-18M', 'T10Y2Y', 'T10Y3M', 'BaaSpread', 'PERatioS&P']].dropna()
y = data1['R_12-18M']
X = data1[['T10Y2Y']].copy()

# SPF is quarterly (0/1) with NaNs in monthly data; align to the same index as data1
spf = df.loc[data1.index, 'SPF']

feature_cols = X.columns.tolist()
X["Inverted"] = (X.iloc[:, 0] < 0).astype(int)
threshold = 0.2

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
        dates = X.index
    else:
        dates = pd.date_range(start='1980-11-01', periods=len(X), freq='MS')
    for i, (train_idx, test_idx) in enumerate(splits, 1):
        train_start = dates[0]
        train_end = dates[train_idx[-1]]
        test_start = dates[test_idx[0]]
        test_end = dates[test_idx[-1]]
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

all_y_logit, all_proba_logit = [], []
all_y_probit, all_proba_probit = [], []
all_y_gb, all_proba_gb = [], []
all_y_rf, all_proba_rf = [], []

## ================================================== Main Results ================================================== ##

clf = LogisticRegression(solver="liblinear", random_state=42)

f1s, precs, recs, aucs, pr_aucs = [], [], [], [], []
f1s_inv, precs_inv, recs_inv, aucs_inv, pr_aucs_inv = [], [], [], [], []

# SPF per-fold precision (overall and when inverted), using only months where SPF is observed
spf_precs, spf_precs_inv = [], []

logit_y_all = []
logit_preds_slope2 = []
logit_preds_aug2 = []
logit_preds_slope3 = []
logit_preds_aug3 = []

for train_idx, test_idx in expanding_splits:
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue

    logit_y_all.extend(y_te.values)

    clf.fit(X_tr[feature_cols], y_tr)
    proba = clf.predict_proba(X_te[feature_cols])[:, 1]
    preds = (proba >= threshold).astype(int)
    logit_preds_aug3.extend(preds)

    all_y_logit.append(y_te.values)
    all_proba_logit.append(proba)

    f1s.append(f1_score(y_te, preds, zero_division=0))
    precs.append(precision_score(y_te, preds, zero_division=0))
    recs.append(recall_score(y_te, preds, zero_division=0))
    aucs.append(roc_auc_score(y_te, proba))
    pr_aucs.append(average_precision_score(y_te, proba))

    inv_mask = X_te["Inverted"] == 1

    # ===== SPF on the same test fold (quarterly, with NaNs) =====
    # Use only months where SPF is observed, both overall and when inverted.
    spf_fold = spf.iloc[test_idx]              # 0/1 or NaN for this fold
    mask_spf = spf_fold.notna()                # only months with SPF forecast

    if mask_spf.sum() > 0:
        y_spf = y_te[mask_spf]
        spf_preds = spf_fold[mask_spf].astype(int)
        spf_precs.append(precision_score(y_spf, spf_preds, zero_division=0))

        inv_spf_mask = mask_spf & inv_mask
        if inv_spf_mask.sum() > 0 and len(np.unique(y_te[inv_spf_mask])) == 2:
            spf_precs_inv.append(
                precision_score(y_te[inv_spf_mask],
                                spf_fold[inv_spf_mask].astype(int),
                                zero_division=0)
            )

    # ============================================================

    if inv_mask.sum() > 0 and len(np.unique(y_te[inv_mask])) == 2:
        f1s_inv.append(f1_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        precs_inv.append(precision_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        recs_inv.append(recall_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        aucs_inv.append(roc_auc_score(y_te[inv_mask], proba[inv_mask]))
        pr_aucs_inv.append(average_precision_score(y_te[inv_mask], proba[inv_mask]))

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

print("\n======== {}-fold Time Series CV (Logistic Regression) @ threshold = {:.2f} (mean ± std) ========".format(len(expanding_splits), threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(precs), np.std(precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(recs),  np.std(recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(pr_aucs), np.std(pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(precs_inv), np.std(precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(recs_inv),  np.std(recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(pr_aucs_inv), np.std(pr_aucs_inv)))

run_mcnemar("Logit T10Y2Y slope vs augmented", logit_preds_slope2, logit_preds_aug2, logit_y_all)
run_mcnemar("Logit T10Y3M slope vs augmented", logit_preds_slope3, logit_preds_aug3, logit_y_all)

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

print("\n======== {}-fold Time Series CV (Probit) @ threshold = {:.2f} (mean ± std) ========".format(len(expanding_splits), threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(probit_precs), np.std(probit_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(probit_recs),  np.std(probit_recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(probit_pr_aucs), np.std(probit_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(probit_precs_inv), np.std(probit_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(probit_recs_inv),  np.std(probit_recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(probit_pr_aucs_inv), np.std(probit_pr_aucs_inv)))

run_mcnemar("Probit T10Y2Y slope vs augmented", probit_preds_slope2, probit_preds_aug2, probit_y_all)
run_mcnemar("Probit T10Y3M slope vs augmented", probit_preds_slope3, probit_preds_aug3, probit_y_all)

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

print("\n======== {}-fold Time Series CV (Gradient Boosting) @ threshold = {:.2f} (mean ± std) ========".format(len(expanding_splits), threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(gb_precs), np.std(gb_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(gb_recs),  np.std(gb_recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(gb_pr_aucs), np.std(gb_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(gb_precs_inv), np.std(gb_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(gb_recs_inv),  np.std(gb_recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(gb_pr_aucs_inv), np.std(gb_pr_aucs_inv)))

run_mcnemar("Gradient Boosting T10Y2Y slope vs augmented", gb_preds_slope2, gb_preds_aug2, gb_y_all)
run_mcnemar("Gradient Boosting T10Y3M slope vs augmented", gb_preds_slope3, gb_preds_aug3, gb_y_all)

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

print("\n======== {}-fold Time Series CV (Random Forest) @ threshold = {:.2f} (mean ± std) ========".format(len(expanding_splits), threshold))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(rf_precs), np.std(rf_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(rf_recs),  np.std(rf_recs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(rf_pr_aucs), np.std(rf_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("Precision: {:.3f} ± {:.3f}".format(np.mean(rf_precs_inv), np.std(rf_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(rf_recs_inv),  np.std(rf_recs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(rf_pr_aucs_inv), np.std(rf_pr_aucs_inv)))

run_mcnemar("Random Forest T10Y2Y slope vs augmented", rf_preds_slope2, rf_preds_aug2, rf_y_all)
run_mcnemar("Random Forest T10Y3M slope vs augmented", rf_preds_slope3, rf_preds_aug3, rf_y_all)

def plot_per_model_pr_curves(all_y, all_proba, model_name, filename_prefix):
    if len(all_y) == 0:
        return
    plt.figure(figsize=(7, 6))
    for fold_idx, (y_fold, proba_fold) in enumerate(zip(all_y, all_proba), start=1):
        prec_curve, rec_curve, _ = precision_recall_curve(y_fold, proba_fold)
        ap = average_precision_score(y_fold, proba_fold)
        plt.plot(rec_curve, prec_curve, label=f"Fold {fold_idx} (PR AUC = {ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision–Recall Curves – {model_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_pr_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

plot_per_model_pr_curves(all_y_logit,  all_proba_logit,  "Logistic Regression", "logit")
plot_per_model_pr_curves(all_y_probit, all_proba_probit, "Probit",              "probit")
plot_per_model_pr_curves(all_y_gb,     all_proba_gb,     "Gradient Boosting",   "gb")
plot_per_model_pr_curves(all_y_rf,     all_proba_rf,     "Random Forest",       "rf")

#print("\n=== Statistical Analysis ===")
#X_with_const = sm.add_constant(X[feature_cols], has_constant='add')

#logit_model = sm.Logit(y, X_with_const)
#logit_results = logit_model.fit(disp=0)

#print("\nLogistic Regression (Logit) Coefficients:")
#print(logit_results.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])

#probit_model_full = sm.Probit(y, X_with_const)
#probit_results_full = probit_model_full.fit(disp=0)

#print("\nProbit Regression Coefficients:")
#print(probit_results_full.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])

#probit_margeff = probit_results_full.get_margeff(at='overall')
#print("\nProbit Average Marginal Effects (AME):")
#print(probit_margeff.summary())

#vif_data = pd.DataFrame()
#vif_data["feature"] = X_with_const.columns
#vif_data["VIF"] = [
#    variance_inflation_factor(X_with_const.values, i)
#    for i in range(X_with_const.shape[1])
#]

#print("\n=== Variance Inflation Factors (VIF) ===")
#print(vif_data[vif_data["feature"] != "const"])

## ================================================== SPF Comparison ================================================== ##

spf_precs_arr = np.array(spf_precs)

logit_prec_mean   = float(np.mean(precs))
logit_rec_mean    = float(np.mean(recs))
probit_prec_mean  = float(np.mean(probit_precs))
probit_rec_mean   = float(np.mean(probit_recs))
gb_prec_mean      = float(np.mean(gb_precs))
gb_rec_mean       = float(np.mean(gb_recs))
rf_prec_mean      = float(np.mean(rf_precs))
rf_rec_mean       = float(np.mean(rf_recs))
spf_prec_mean     = float(np.mean(spf_precs_arr))

print("\n======== Average per-fold Precision & Recall (by model) ========")
print(f"SPF (precision only): {spf_prec_mean:.3f}\n")

print("Logistic Regression:  Precision = {:.3f}, Recall = {:.3f}".format(logit_prec_mean,  logit_rec_mean))
print("Probit:               Precision = {:.3f}, Recall = {:.3f}".format(probit_prec_mean, probit_rec_mean))
print("Gradient Boosting:    Precision = {:.3f}, Recall = {:.3f}".format(gb_prec_mean,     gb_rec_mean))
print("Random Forest:        Precision = {:.3f}, Recall = {:.3f}".format(rf_prec_mean,     rf_rec_mean))

print("\n======== SPF vs Models – Precision (average over folds) ========")
print("SPF precision (avg per fold): {:.3f}".format(spf_prec_mean))
print("Logit  precision − SPF: {:+.3f}".format(logit_prec_mean  - spf_prec_mean))
print("Probit precision − SPF: {:+.3f}".format(probit_prec_mean - spf_prec_mean))
print("GB     precision − SPF: {:+.3f}".format(gb_prec_mean     - spf_prec_mean))
print("RF     precision − SPF: {:+.3f}".format(rf_prec_mean     - spf_prec_mean))

## ================================================== Per Fold Performance ================================================== ##

print("\n======== Per-fold Precision & Recall (by model) ========")

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
