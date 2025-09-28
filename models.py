import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np

df = pd.read_excel("Data.xlsx")
#print(df)

df2 = df[['R12M','R6M', 'T10Y2Y', 'BaaSpread', 'E-Rule']].dropna()
y = df2['R12M']
X = df2[['T10Y2Y', 'BaaSpread']]

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