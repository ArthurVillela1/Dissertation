import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel("Data.xlsx")
#print(df)

df2 = df[['R12M', 'R6M', 'T10Y2Y', 'BaaSpread', 'E-Rule']].dropna()
y = df2['R12M']
X = df2[['T10Y2Y', 'BaaSpread']]

# Probit/Logit
# Stratified k-fold cross-validation (5-fold)
threshold = 0.5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(solver="liblinear", random_state=42)

f1s, precs, recs, aucs = [], [], [], []

for train_idx, test_idx in skf.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    #X_tr_sm = sm.add_constant(X_tr, has_constant='add')
    #X_te_sm = sm.add_constant(X_te, has_constant='add')
    #probit = sm.Probit(y_tr, X_tr_sm).fit(disp=False)
    #proba = probit.predict(X_te_sm)        
    preds = (proba >= threshold).astype(int)     

    f1s.append(f1_score(y_te, preds))
    precs.append(precision_score(y_te, preds, zero_division=0))
    recs.append(recall_score(y_te, preds))
    aucs.append(roc_auc_score(y_te, proba))      

print("\n=== 5-fold Stratified CV @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("F1:        {:.3f} ± {:.3f}".format(np.mean(f1s),  np.std(f1s)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(precs), np.std(precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(recs),  np.std(recs)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(aucs),  np.std(aucs)))

# Gradient Boosting Classifier
gb_f1s, gb_precs, gb_recs, gb_aucs = [], [], [], []

gb_clf = GradientBoostingClassifier(
    random_state=42,
    n_estimators=300,     
    learning_rate=0.05,
    max_depth=2
)

for train_idx, test_idx in skf.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    gb_clf.fit(X_tr, y_tr)
    gb_proba = gb_clf.predict_proba(X_te)[:, 1]
    gb_preds = (gb_proba >= threshold).astype(int)

    gb_f1s.append(f1_score(y_te, gb_preds))
    gb_precs.append(precision_score(y_te, gb_preds, zero_division=0))
    gb_recs.append(recall_score(y_te, gb_preds))
    gb_aucs.append(roc_auc_score(y_te, gb_proba))

print("\n=== 5-fold Stratified CV (Gradient Boosting) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("F1:        {:.3f} ± {:.3f}".format(np.mean(gb_f1s),  np.std(gb_f1s)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(gb_precs), np.std(gb_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(gb_recs),  np.std(gb_recs)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(gb_aucs),  np.std(gb_aucs)))

# Support Vector Classifier
svc_f1s, svc_precs, svc_recs, svc_aucs = [], [], [], []

svc_clf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,   # needed to get predict_proba
    random_state=42
)

for train_idx, test_idx in skf.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    svc_clf.fit(X_tr, y_tr)
    svc_proba = svc_clf.predict_proba(X_te)[:, 1]
    svc_preds = (svc_proba >= threshold).astype(int)

    svc_f1s.append(f1_score(y_te, svc_preds))
    svc_precs.append(precision_score(y_te, svc_preds, zero_division=0))
    svc_recs.append(recall_score(y_te, svc_preds))
    svc_aucs.append(roc_auc_score(y_te, svc_proba))

print("\n=== 5-fold Stratified CV (SVC) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("F1:        {:.3f} ± {:.3f}".format(np.mean(svc_f1s),  np.std(svc_f1s)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(svc_precs), np.std(svc_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(svc_recs),  np.std(svc_recs)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(svc_aucs),  np.std(svc_aucs)))

# Random Forest Classifier
rf_f1s, rf_precs, rf_recs, rf_aucs = [], [], [], []

rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
    # class_weight="balanced"  # optional if you want to emphasize class 1
)

for train_idx, test_idx in skf.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    rf_clf.fit(X_tr, y_tr)
    rf_proba = rf_clf.predict_proba(X_te)[:, 1]
    rf_preds = (rf_proba >= threshold).astype(int)

    rf_f1s.append(f1_score(y_te, rf_preds))
    rf_precs.append(precision_score(y_te, rf_preds, zero_division=0))
    rf_recs.append(recall_score(y_te, rf_preds))
    rf_aucs.append(roc_auc_score(y_te, rf_proba))

print("\n=== 5-fold Stratified CV (Random Forest) @ threshold = {:.2f} (mean ± std) ===".format(threshold))
print("F1:        {:.3f} ± {:.3f}".format(np.mean(rf_f1s),  np.std(rf_f1s)))
print("Precision: {:.3f} ± {:.3f}".format(np.mean(rf_precs), np.std(rf_precs)))
print("Recall:    {:.3f} ± {:.3f}".format(np.mean(rf_recs),  np.std(rf_recs)))
print("ROC AUC:   {:.3f} ± {:.3f}".format(np.mean(rf_aucs),  np.std(rf_aucs)))

# Testing for multicolinearity
# Add constant to X
X_with_const = sm.add_constant(X, has_constant='add')

vif_data = pd.DataFrame()
vif_data["feature"] = X_with_const.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_with_const.values, i)
    for i in range(X_with_const.shape[1])
]

# Drop the constant from the report
print("\n=== Variance Inflation Factors (VIF) ===")
print(vif_data[vif_data["feature"] != "const"])