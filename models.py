import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, TimeSeriesSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score
)
 
df = pd.read_excel("Data.xlsx")

data1 = df[['R_12-18M', 'T10Y2Y', 'T10Y3M', 'BaaSpread', 'PERatioS&P']].dropna()
y = data1['R_12-18M']
X = data1[['T10Y3M', 'BaaSpread']]

# Add inversion flag (1 if inverted)
X["Inverted"] = (X["T10Y3M"] < 0).astype(int)

threshold = 0.5 
tscv = TimeSeriesSplit(n_splits=5)

# Logit
clf = LogisticRegression(solver="liblinear", random_state=42)

f1s, precs, recs, aucs, pr_aucs = [], [], [], [], []
f1s_inv, precs_inv, recs_inv, aucs_inv, pr_aucs_inv = [], [], [], [], []

for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    clf.fit(X_tr[['T10Y3M', 'BaaSpread']], y_tr)
    proba = clf.predict_proba(X_te[['T10Y3M', 'BaaSpread']])[:, 1]
    preds = (proba >= threshold).astype(int)

    f1s.append(f1_score(y_te, preds, zero_division=0))
    precs.append(precision_score(y_te, preds, zero_division=0))
    recs.append(recall_score(y_te, preds, zero_division=0))
    aucs.append(roc_auc_score(y_te, proba))      
    pr_aucs.append(average_precision_score(y_te, proba))

    # When curve inverted
    inv_mask = X_te["Inverted"] == 1
    if inv_mask.sum() > 0 and len(np.unique(y_te[inv_mask])) == 2:
        f1s_inv.append(f1_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        precs_inv.append(precision_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        recs_inv.append(recall_score(y_te[inv_mask], preds[inv_mask], zero_division=0))
        aucs_inv.append(roc_auc_score(y_te[inv_mask], proba[inv_mask]))
        pr_aucs_inv.append(average_precision_score(y_te[inv_mask], proba[inv_mask]))

print("\n=== 5-fold Time Series CV (Logistic Regression) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("F1:        {:.3f} ± {:.3f}".format(np.mean(f1s),  np.std(f1s)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(precs), np.std(precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(recs),  np.std(recs)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(aucs),  np.std(aucs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(pr_aucs), np.std(pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("F1:        {:.3f} ± {:.3f}".format(np.mean(f1s_inv),  np.std(f1s_inv)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(precs_inv), np.std(precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(recs_inv),  np.std(recs_inv)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(aucs_inv),  np.std(aucs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(pr_aucs_inv), np.std(pr_aucs_inv)))

# Gradient Boosting Classifier
gb_f1s, gb_precs, gb_recs, gb_aucs, gb_pr_aucs = [], [], [], [], []
gb_f1s_inv, gb_precs_inv, gb_recs_inv, gb_aucs_inv, gb_pr_aucs_inv = [], [], [], [], []

gb_clf = GradientBoostingClassifier(
    random_state=42,
    n_estimators=300,     
    learning_rate=0.05,
    max_depth=2
)

for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx] 
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    gb_clf.fit(X_tr[['T10Y3M', 'BaaSpread']], y_tr)
    gb_proba = gb_clf.predict_proba(X_te[['T10Y3M', 'BaaSpread']])[:, 1]
    gb_preds = (gb_proba >= threshold).astype(int)

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

print("\n=== 5-fold Time Series CV (Gradient Boosting) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("F1:        {:.3f} ± {:.3f}".format(np.mean(gb_f1s),  np.std(gb_f1s)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(gb_precs), np.std(gb_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(gb_recs),  np.std(gb_recs)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(gb_aucs),  np.std(gb_aucs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(gb_pr_aucs), np.std(gb_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("F1:        {:.3f} ± {:.3f}".format(np.mean(gb_f1s_inv),  np.std(gb_f1s_inv)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(gb_precs_inv), np.std(gb_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(gb_recs_inv),  np.std(gb_recs_inv)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(gb_aucs_inv),  np.std(gb_aucs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(gb_pr_aucs_inv), np.std(gb_pr_aucs_inv)))

# Random Forest Classifier
rf_f1s, rf_precs, rf_recs, rf_aucs, rf_pr_aucs = [], [], [], [], []
rf_f1s_inv, rf_precs_inv, rf_recs_inv, rf_aucs_inv, rf_pr_aucs_inv = [], [], [], [], []

rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

for train_idx, test_idx in tscv.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    if len(np.unique(y_te)) < 2:
        continue
        
    rf_clf.fit(X_tr[['T10Y3M', 'BaaSpread']], y_tr)
    rf_proba = rf_clf.predict_proba(X_te[['T10Y3M', 'BaaSpread']])[:, 1]
    rf_preds = (rf_proba >= threshold).astype(int)

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

print("\n=== 5-fold Time Series CV (Random Forest) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("F1:        {:.3f} ± {:.3f}".format(np.mean(rf_f1s),  np.std(rf_f1s)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(rf_precs), np.std(rf_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(rf_recs),  np.std(rf_recs)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(rf_aucs),  np.std(rf_aucs)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(rf_pr_aucs), np.std(rf_pr_aucs)))

print("\nWhen Yield Curve Inverted (Slope < 0):")
print("F1:        {:.3f} ± {:.3f}".format(np.mean(rf_f1s_inv),  np.std(rf_f1s_inv)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(rf_precs_inv), np.std(rf_precs_inv)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(rf_recs_inv),  np.std(rf_recs_inv)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(rf_aucs_inv),  np.std(rf_aucs_inv)))
print("PR AUC:    {:.3f} ± {:.3f}".format(np.mean(rf_pr_aucs_inv), np.std(rf_pr_aucs_inv))) 

# Statistical significance analysis using statsmodels
print("\n=== Statistical Significance Analysis ===")
X_with_const = sm.add_constant(X[['T10Y3M', 'BaaSpread']], has_constant='add')
logit_model = sm.Logit(y, X_with_const)
logit_results = logit_model.fit(disp=0)

print("Logistic Regression Coefficients:")
print(logit_results.summary2().tables[1][['Coef.', 'Std.Err.', 'z', 'P>|z|']])

# Calculate odds ratios
odds_ratios = np.exp(logit_results.params)
print(f"\nOdds Ratios:")
for var, or_val in odds_ratios.items():
    if var != 'const':
        pval = logit_results.pvalues[var]
        significance = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"{var}: {or_val:.3f} {significance}")

# Testing for multicolinearity
vif_data = pd.DataFrame()
vif_data["feature"] = X_with_const.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_with_const.values, i)
    for i in range(X_with_const.shape[1])
]

print("\n=== Variance Inflation Factors (VIF) ===")
print(vif_data[vif_data["feature"] != "const"])